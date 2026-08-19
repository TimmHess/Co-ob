from typing import TYPE_CHECKING, Dict, TypeVar
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR

from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.training.utils import unfreeze_everything, freeze_everything

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.models.utils import avalanche_forward



from src.benchmarks.visual_shortcuts import VisualShortcutInjectorPlugin
from src.utils.util import safe_index, get_plugin
from src.models.models import weight_reset
from src.models.models import _get_classifier

import copy
from tqdm import tqdm

TResult = TypeVar('TResult')

# NOTE: PluginMetric->GenericPluginMetric->AccuracyPluginMetric
    # ->'SpecificMetric' (in our case this will be the LinearProbingAccuracyMetric)
    # in avalnache this could be, e.g. MinibatchAccuracy...

class HeadPreAdaptationPlugin(SupervisedPlugin):
    def __init__(
        self, 
        train_stream, 
        train_all=False, 
        num_finetune_epochs=1,
        num_finetune_itrs=None,
        train_mb_size=32, 
        num_workers=1,
        skip_initial_eval=False, 
        buffer_lp_dataset=True,
    ):
        super().__init__()

        self.train_stream = train_stream
        self.train_all = train_all
        #self.test_stream = test_stream
        #self.train_stream_from_ER_buffer = train_stream_from_ER_buffer
        #self.criterion = criterion
        self.num_finetune_epochs = num_finetune_epochs
        self.num_finetune_itrs = num_finetune_itrs
        self.train_mb_size = train_mb_size
        self.num_workers = num_workers
        self.buffer_lp_dataset = buffer_lp_dataset

        self.head_copy = None  # NOTE: local copy of the model's head used for linear probing
        
        self.num_eval_exps = 0

        self.local_optim = None
        self.initial_out_features = None

        self.skip_initial_eval = skip_initial_eval
        self.is_initial_eval_run = False
        self._prev_state = None
        self._prev_training_modes = None  # NOTE: required to reset the training scheme after calling eval in train mode..

        self.visual_shortcut_plugin = None  # NOTE: reference to visual shortcuts (hacky but best I can do for now)
        return


    @torch.no_grad()
    def prepare_tensordataset(self, model, dataset, device):
        x_reprs = []
        ys = []
        ts = []

        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.train_mb_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            drop_last=False
        ) 

        print("\nPreparing dataset for pretraining the model head...")
        for _, mbatch in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

            x = x.to(device)
            y = y.to(device)

            if self.visual_shortcut_plugin is not None:
                x = self.visual_shortcut_plugin.apply_shortcut(
                    tensors=x,
                    targets=y,
                    task_id=tid.unique()[0].item(), 
                    probability=1.0
                )
            
            # Get representation from backbone
            x_rep = avalanche_forward(model, x, tid)  #NOTE: automatically handles potential MultiTaskModules
            # if self.normalize_features:
            #     x_rep = F.normalize(x_rep, dim=1)
            x_reprs.append(x_rep.cpu())
            ys.append(y.cpu())
            ts.append(tid)

        x_reprs = torch.concat(x_reprs)
        ys = torch.concat(ys)
        ts = torch.concat(ts)
        return x_reprs, ys, ts


    def before_training(self, strategy, **kwargs):
        if strategy.clock.train_exp_counter == 0:
            self.visual_shortcut_plugin = get_plugin(strategy, plugin_type=VisualShortcutInjectorPlugin)
        if self.visual_shortcut_plugin is not None:
            print("PretrainingHead: before_training: Using visual shortcut plugin!")
        return


    def before_training_exp(self, strategy, **kwargs):
        # Initialize and prepare the linear probing head
        print("\n========================Pretraining model head:==========================")
        # with torch.enable_grad():  # TODO: remove.. this is useless here..
        self.head_copy = strategy.model.train_classifier  # NOTE: this is a reference to the model's head
        
        #freeze_everything
        unfreeze_everything(self.head_copy)  # For safety reasons unfreeze everything in the head_copy
    
        # Check number of current heads against max numbre of heads possible  
        print("\nIs multi-headed:", isinstance(self.head_copy, MultiTaskModule))
        if isinstance(self.head_copy, MultiTaskModule):  #NOTE: this adds classifiers for every task possible
            if len(self.head_copy.classifiers) < len(self.train_stream):
                for exp in self.train_stream:
                    self.head_copy.adaptation(exp)

        # Prepare dataset and dataloader
        if self.train_all:  # NOTE: Override the number of experiences to use in each step with max value
            self.num_eval_exps = len(self.train_stream) -1 # -1 to make up for +1 in next step
            print("\nNum seen experiences is maxed out!")
        
        lp_datasets = []
        lp_dataset = None

        # Grab subset of the train_stream to use for training linear probes
        curr_exp_data_stream = self.train_stream[:(self.num_eval_exps+1)] # Grab the current subset of experiences from train_stream
        # Create a ConcatDataset and respective Dataloader
        for i, exp in enumerate(curr_exp_data_stream):
            lp_dataset = exp.dataset.eval()  # NOTE: needs to be set to eval to avoid data augmentation in training probes
            if self.buffer_lp_dataset:
                # Prepare tensor dataset to prevent massive compute overhead
                xs, ys, ts = self.prepare_tensordataset(
                    model=strategy.model.feature_extractor, 
                    dataset=lp_dataset, 
                    device=strategy.device
                )
                lp_dataset = torch.utils.data.TensorDataset(xs, ys, ts)
            lp_datasets.append(lp_dataset)
        print("PretrainingHead: num datasets", len(lp_datasets))

        # Check whether to use a single or multiple datasets for linear probing
        if not isinstance(self.head_copy, MultiTaskModule):
            lp_datasets = [torch.utils.data.ConcatDataset(lp_datasets)]  # reduce to a single dataset
        print("PretrainingHead: num lp datasets", len(lp_datasets))
        print("PretrainingHead: num samples in lp dataset", len(lp_datasets[0]))
        
        # Train the linear probes
        for lp_dataset in lp_datasets:
            lp_dataloader = torch.utils.data.DataLoader(
                lp_dataset, 
                batch_size=self.train_mb_size, #self.train_mb_size*len(self.train_stream), 
                shuffle=True, 
                num_workers=self.num_workers, 
                drop_last=True, #if not self.train_stream_from_ER_buffer else False, 
                pin_memory=True
            )

            # Initialize local optimizer for the new head
            self.local_optim = torch.optim.AdamW(
                self.head_copy.parameters(), 
                lr=1e-3, # 0.1
                weight_decay=5e-4, # 0.0 
            )

            num_iters = 0
            stop_training = False
            # Overwrite number of epochs to some arbitrary high value
            if self.num_finetune_itrs is not None:
                print("\nPretrainingHead: num_finetune_itrs", self.num_finetune_itrs)
                self.num_finetune_epochs = 10000 
            # Finetuning loop 
            tqdm_obj = tqdm(range(self.num_finetune_epochs))
            for _ in tqdm_obj:
                if stop_training:
                    break

                mean_loss = 0.0
                for _, mbatch in enumerate(lp_dataloader):
                    self.local_optim.zero_grad()

                    x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
                    x_rep = x.to(strategy.device)
                    if not self.buffer_lp_dataset:
                        x_rep = avalanche_forward(strategy.model.feature_extractor, x, tid)  # NOTE: automatically handles potential MultiTaskModules
                    y = y.to(strategy.device)
                    
                    # Forward representation through new head 
                    out = avalanche_forward(self.head_copy, x_rep, tid)  # TODO: is it the avalanche_forward?
                    loss = strategy._criterion(out, y)  # loss = self.criterion(out, y)
                    mean_loss += loss.item()
                    loss.backward()
                    self.local_optim.step()

                    num_iters += 1
                    # If using iterations - check if training continues
                    if self.num_finetune_itrs is not None:
                        if num_iters >= self.num_finetune_itrs:
                            stop_training = True
                            break

                print("\nDEBUG: head pre-training: mean_loss", mean_loss/len(lp_dataloader))
                if not stop_training:
                    tqdm_obj.set_postfix(loss=f"{mean_loss/len(lp_dataloader):.4f}")
            print("\nHead pre-training complete...")
            
            # Call a zero_grad on the entire model just to be sure
            strategy.model.zero_grad()
        return



    


    