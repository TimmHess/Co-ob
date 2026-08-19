#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from typing import TypeVar
import torch
import torch.nn.functional as F

from avalanche.models.dynamic_modules import MultiTaskModule
from avalanche.training.utils import unfreeze_everything

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import (
    Accuracy, 
    TaskAwareAccuracy,
)
from avalanche.evaluation.metrics.loss import (
    LossMetric,
    TaskAwareLoss
)
from avalanche.evaluation.metric_results import (
    MetricResult,
    MetricValue
)
from avalanche.evaluation.metric_utils import (
    phase_and_task,
    get_metric_name,
)
from avalanche.models.utils import avalanche_forward

from src.utils.util import safe_index, get_plugin
from src.models.models import weight_reset
from src.models.models import _get_classifier

from src.methods.ensemble import get_ensemble_model_buffer

import copy
from tqdm import tqdm

TResult = TypeVar('TResult')

class ConcatenatedLinearProbingAccuracyMetric(GenericPluginMetric[float, Accuracy]):
    def __init__(
        self, 
        train_stream, 
        test_stream, 
        criterion,
        eval_all=True, 
        num_finetune_epochs=1,
        train_mb_size=32, 
        num_workers=1,
        skip_initial_eval=False, 
        buffer_lp_dataset=True,
        normalize_features=False,
        force_mh_eval=False,
        record_loss=False,
        eval_train_stream=False
    ):
        self._accuracy = TaskAwareAccuracy() # metric calculation container
        self._accuracy_train = TaskAwareAccuracy()
        self._probe_loss = TaskAwareLoss()
        super(ConcatenatedLinearProbingAccuracyMetric, self).__init__(
            self._accuracy, 
            reset_at='stream', 
            emit_at='stream',
            mode='eval'
        )

        self.train_stream = train_stream
        self.test_stream = test_stream
        self.criterion = criterion
        self.num_fintune_epochs = num_finetune_epochs
        self.train_mb_size = train_mb_size
        self.num_workers = num_workers
        self.buffer_lp_dataset = buffer_lp_dataset
        self.normalize_features = normalize_features
        self.eval_all = eval_all  # flag to indicate forced evaluation on all experiences for each tasks (including yet unseed ones)
        self.force_mh_eval = force_mh_eval
        self.record_loss = record_loss
        self.eval_train_stream = eval_train_stream

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

    def __str__(self):
        if self.force_mh_eval:
            return "Top1_Concatenated_LP_MH_Exp_Acc"
        return "Top1_Concatenated_LP_Exp_Acc"

    def reset(self, strategy=None):
        print("DEBUG: concat_linear: self._reset_at", self._reset_at)
        self._metric.reset()
        self._accuracy_train.reset()
        self._probe_loss.reset()
        return
    
    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        else:
            if self.force_mh_eval:
                print(self._metric.result())
                import sys;sys.exit()
                result = self._metric.result(torch.div(strategy.mb_y, self.initial_out_features, rounding_mode='trunc')) 
                return result
            
            return self._metric.result(phase_and_task(strategy)[1])

    def _package_result(self, strategy):
        metric_value = self.result()
        add_exp = self._emit_at == "experience"
        plot_x_position = strategy.clock.train_iterations
 
        metrics = []

        # Accuracy Values
        if isinstance(metric_value, dict):
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k
                )
                metrics.append(MetricValue(self, metric_name, v, plot_x_position))
        else:
            metric_name = get_metric_name(
                self, strategy, add_experience=add_exp, add_task=True
            )
            metrics.append(MetricValue(self, metric_name, metric_value, plot_x_position))

        # Loss Values
        if self.record_loss:
            loss_metric_value = self._probe_loss.result()
            if isinstance(loss_metric_value, dict):
                for k, v in loss_metric_value.items():
                    metric_name = get_metric_name(
                        self, strategy, add_experience=add_exp, add_task=k
                    )
                    metric_name = metric_name.split("/")[0]+"_Loss/"+"/".join(metric_name.split("/")[1:])
                    metrics.append(MetricValue(self, metric_name, v, plot_x_position))
            else:
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=True
                )
                metric_name = metric_name.split("/")[0]+"_Loss/"+"/".join(metric_name.split("/")[1:])
                metrics.append(MetricValue(self, metric_name, loss_metric_value, plot_x_position))

        # Accuracy and Loss on Training Data
        if self.eval_train_stream:  #and task==0
            print("DEBUG: Adding accuracy and loss values for training data...")
            train_acc_metric_value = self._accuracy_train.result()
            # train_loss_metric_value = self._probe_loss_train.result()
            if isinstance(train_acc_metric_value, dict):
                for k, v in train_acc_metric_value.items():
                    metric_name = get_metric_name(
                        self, strategy, add_experience=False, add_task=k
                    )
                    metric_name_acc = metric_name.split("/")[0]+"_AccOnTrain/"+"/".join(metric_name.split("/")[1:])
                    metrics.append(MetricValue(self, metric_name_acc, train_acc_metric_value[k], plot_x_position))

        return metrics
    

    @torch.no_grad()
    def prepare_ensemble_tensordataset(self, models, dataset, device):
        x_reprs = {}
        ys = []
        ts = []
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.train_mb_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            drop_last=False
        ) 
        print("Preparing dataset for concatenated linear probing...")
        # Select the model to use for the linear probing
        for model_idx, model in enumerate(models):
            # Load the model to device
            model = model.to(device)
            model.eval()  # set to eval mode for safety
            # Recored features
            for mbatch_idx, mbatch in tqdm(enumerate(dataloader), total=len(dataloader)):
                x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
                
                x = x.to(device)
                y = y.to(device)

                # Apply shortcut if necessary
                if self.visual_shortcut_plugin is not None:
                    x = self.visual_shortcut_plugin.apply_shortcut(
                        tensors=x,
                        targets=y,
                        task_id=tid.unique()[0].item(), 
                        probability=1.0
                    )

                # Get representation from backbone
                x_rep = model(x).detach()  # detach() is most likely not necessary here but I want to be sure

                if mbatch_idx not in x_reprs:
                    x_reprs[mbatch_idx] = x_rep.cpu()
                else:
                    x_reprs[mbatch_idx] = torch.cat((x_reprs[mbatch_idx], x_rep.cpu()), dim=1)
                
                if model_idx == 0:  # NOTE: only record the target labels once
                    ys.append(y.cpu())
                    ts.append(tid)
                
            # Return model to cpu
            model = model.to("cpu")

        # Convert to tensor
        x_reprs = torch.cat(list(x_reprs.values()), dim=0)
        print("\nx_reprs.shape after concatenation:", x_reprs.shape)
        ys = torch.concat(ys)
        print("ys.shape after concatenation:", ys.shape)
        ts = torch.concat(ts)
        return x_reprs, ys, ts


    def before_training(self, strategy):
        if strategy.clock.train_exp_counter == 0:
            self.visual_shortcut_plugin = get_plugin(strategy, plugin_type=VisualShortcutInjectorPlugin)
        if self.visual_shortcut_plugin is not None:
            print("Using visual shortcut plugin!")
        return
    

    def forward_all_feats(self, x, models, device):
        x_repres = None
        for model in models:
            model = model.to(device)
            model.eval()  # set to eval mode for safety

            x_rep = model(x).detach()  # detach() is most likely not necessary here but I want to be sure
            
            if self.normalize_features:
                x_rep = F.normalize(x_rep, dim=1)
            if x_repres is None:
                x_repres = x_rep
            else:
                x_repres = torch.cat((x_repres, x_rep), dim=1)
            model = model.to("cpu")
        return x_repres
    

    def before_eval(self, strategy):
        models_buffer = get_ensemble_model_buffer(strategy)

        # Initialize and prepare the linear probing head
        print("\nLinearProbingAccuracyMetric: Preparing Linear Probe(s)")
        with torch.enable_grad():  # NOTE: This is necessary because avalanche has a hidden torch.no_grad() in eval context!
            # Initialize the probing head
            if self.force_mh_eval:
                # Get num initial classes
                initial_out_features = len(torch.tensor(self.train_stream[0].dataset.targets).unique())
                print("Forcing multi-headed linear-probe!")
                self.head_copy = _get_classifier(
                    classifier_type="linear", 
                    n_classes=None,  # NOTE: n_classes not used but initial_out_features
                    initial_out_features=initial_out_features,
                    feat_size=strategy.model.feature_extractor.feature_size*len(models_buffer), 
                    task_incr=True,
                ).to(strategy.device)
            else:
                unique_targets = torch.tensor(self.train_stream[0].dataset.targets).unique()
                for ds in self.train_stream[1:]:
                    unique_targets = torch.cat((unique_targets, torch.tensor(ds.dataset.targets).unique())).unique()
                num_classes = len(unique_targets)
                self.head_copy = _get_classifier(
                    classifier_type="linear", 
                    n_classes=num_classes,
                    initial_out_features=None,
                    feat_size=strategy.model.feature_extractor.feature_size*len(models_buffer), 
                    task_incr=False,
                ).to(strategy.device)
                print("\nHeadCopy:")
                print(self.head_copy)
                unfreeze_everything(self.head_copy)  # # For safety reasons unfreeze everything in the head_copy
            print("\nIs multi-headed:", isinstance(self.head_copy, MultiTaskModule))

            # Check number of current heads against max numbre of heads possible  
            if isinstance(self.head_copy, MultiTaskModule):  #NOTE: this adds classifiers for every task possible
                if len(self.head_copy.classifiers) < len(self.train_stream):
                    for exp in self.train_stream:
                        self.head_copy.adaptation(exp)
            
            # Prepare dataet and dataloader
            if self.eval_all: # NOTE: Override the number of experiences to use in each step with max value
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
                    xs, ys, ts = self.prepare_ensemble_tensordataset(
                        models=models_buffer,
                        dataset=lp_dataset, 
                        device=strategy.device
                    )
                    lp_dataset = torch.utils.data.TensorDataset(xs, ys, ts)
                lp_datasets.append(lp_dataset)
            print("\nLinearProbingMetric: num datasets", len(lp_datasets))

            # Check whether to use a single or multiple datasets for linear probing
            if not isinstance(self.head_copy, MultiTaskModule):
                lp_datasets = [torch.utils.data.ConcatDataset(lp_datasets)]  # reduce to a single dataset
            print("\nLinearProbingMetric: num lp datasets", len(lp_datasets))
            print("\nLinearProbingMetric: num samples in lp dataset", len(lp_datasets[0]))
            
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
                    lr=1e-3,
                    weight_decay=5e-4,
                )
            
                for _ in tqdm(range(self.num_fintune_epochs)):
                    mean_loss = 0.0
                    for _, mbatch in enumerate(lp_dataloader):
                        self.local_optim.zero_grad()

                        x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                        x_rep = x.to(strategy.device)
                        if not self.buffer_lp_dataset:
                            raise NotImplementedError("Buffering of linear probing dataset is required!")
                    
                        y = y.to(strategy.device)
                        
                        # Forward representation through new head 
                        out = avalanche_forward(self.head_copy, x_rep, tid) 

                        loss = self.criterion(out, y)
                        mean_loss += loss.item()
                        loss.backward()
                        self.local_optim.step()
            print("\nLinear Probe training complete...")

        super().before_eval(strategy) # NOTE: this will do the reset of results etc.
        return


    def do_eval(self, strategy):
        # Access the current dataloader -> this is the dataloader of the current eval experience
        lp_dataloader = strategy.dataloader

        models_buffer = get_ensemble_model_buffer(strategy)

        for _, mbatch in tqdm(enumerate(lp_dataloader), total=len(lp_dataloader)):
                x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                x = x.to(strategy.device)
                y = y.to(strategy.device)

                 # Apply shortcut if necessary
                if self.visual_shortcut_plugin is not None:
                    x = self.visual_shortcut_plugin.apply_shortcut(
                        tensors=x,
                        targets=y,
                        task_id=tid.unique()[0].item(), 
                        probability=1.0
                    )

                # Get representation from backbone
                x_rep_concat = self.forward_all_feats(x, models_buffer, strategy.device)
                out = avalanche_forward(self.head_copy, x_rep_concat, tid)
                self._accuracy.update(out, y, tid)
        return


    @torch.no_grad()
    def do_eval_train_stream(self, strategy):
        """
        Runs a separate evaluation loop on the training-data.
        Note: Training and eval is happening on the same data!
        This is merely meant as a surrogate in certain situations!
        """
        print("\nEvaluating linear probe on the training data...")

        models_buffer = get_ensemble_model_buffer(strategy)
        for train_exp in self.train_stream:
            print("\nLinearProbingAccuracyMetric: Evaluating on training data...")
            self.eval_set = train_exp.dataset.eval()
            
            # Init datalaoder for the down-stream task
            ds_dataloader = torch.utils.data.DataLoader(
                self.eval_set, 
                batch_size=self.train_mb_size, 
                shuffle=False, 
                num_workers=self.num_workers, 
                drop_last=False
            ) 

            mean_loss = 0.0
            for _, mbatch in tqdm(enumerate(ds_dataloader), total=len(ds_dataloader)):
                x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                x = x.to(strategy.device)
                y = y.to(strategy.device)
                
                # Get representation from backbone
                x_rep_concat = self.forward_all_feats(x, models_buffer, strategy.device)

                # Forward representation through new head 
                out = avalanche_forward(self.head_copy, x_rep_concat, tid)

                loss = self.criterion(out, y)
                mean_loss += loss.item()
                
                # Update the accuracy measure
                self._accuracy_train.update(out, y, tid)  # NOTE: Use TaskAwareAccuracy when wanting to use tid
            print("\nDEBUG: linear_probing_metric: mean_loss on train", mean_loss/len(ds_dataloader))
        return


    def after_eval_exp(self, strategy):
        # Run evaluation on the test data
        self.do_eval(strategy)
        # Run evaluation on the training data
        if self.eval_train_stream and not self.is_initial_eval_run:
            self.do_eval_train_stream(strategy)
        super().after_eval(strategy)
        return


    def after_eval(self, strategy) -> 'MetricResult':
        # Increase the counter on seen experiences
        self.num_eval_exps += 1 
        return super().after_eval(strategy)


    