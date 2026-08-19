#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from typing import TYPE_CHECKING, Dict, TypeVar
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR, CosineAnnealingLR

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy #, TaskAwareAccuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task

from src.models.models import _get_classifier, weight_reset
from src.utils.util import safe_index
from src.utils.named_accuracy import NameAwareAccuracy
from src.utils.corruption.corruption_handler import CorruptionHandler

from tqdm import tqdm


TResult = TypeVar('TResult')

class DownstreamLinearProbeAccuracyMetric(GenericPluginMetric[float, Accuracy]):
    """
    Evaluation plugin for down-stream tasks.

    Params:
        down_stream_task: The task to evaluate on
        scenario_loader: The scenario loader to use for the down-stream task, i.e. 'get_scenario' from helper.py # NOTE: this can't be importet due to cyclic dependece here..
        num_finetune_epochs: Number of epochs to finetune the model on the down-stream task
        batch_size: Batch size to use for the down-stream task
        num_workers: Number of workers to use for the down-stream task
        skip_initial_eval: If True, the initial evaluation on the down-stream task is skipped   
    """
    def __init__(
            self, 
            args, 
            downstream_task, 
            train_set, 
            eval_set, 
            n_classes, 
            criterion, 
            train_mb_size=32, 
            eval_mb_size=32,
            buffer_lp_dataset=True,
            normalize_features=False, # False,
        ):
        self._accuracy = Accuracy() # metric calculation container
        super(DownstreamLinearProbeAccuracyMetric, self).__init__(
            self._accuracy, reset_at='stream', emit_at='stream', mode='eval')

        self.args = args
        # Init the scenario for the downstream task
        self.downstream_task = downstream_task
        self.train_set = train_set
        self.eval_set = eval_set
        self.ds_n_classes = n_classes
        self.criterion = criterion
        self.buffer_lp_dataset = buffer_lp_dataset
        self.normalize_features = normalize_features
    
        self.train_mb_size = train_mb_size
        self.eval_mb_size = eval_mb_size
        self.num_workers = args.num_workers

        self.num_fintune_epochs = args.lp_finetune_epochs

        self.ds_head = None # NOTE: local copy of the model's head used for linear probing
        self.local_optim = None

        self.eval_complete = False
        self.initial_out_features = None

        self.is_initial_eval_run = False
        self._prev_state = None
        self._prev_training_modes = None # NOTE: required to reset the training scheme after calling eval in train mode..
        return

    def __str__(self):
        return "Top1_" + self.downstream_task + "_Acc"

    def _package_result(self, strategy) -> 'MetricResult':
        metric_value = self.result(strategy)
        
        add_exp = self._emit_at == 'experience'
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k)
                metrics.append(
                    MetricValue(
                        self, 
                        metric_name, 
                        v,
                        plot_x_position
                    )
                )
            return metrics
        
        metric_name = get_metric_name(
            self, 
            strategy,
            add_experience=add_exp,
            add_task=True
        )
        return [MetricValue(self, metric_name, metric_value,
                            plot_x_position)]

    def reset(self, strategy=None):
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()

        try: # NOTE: the try-except is a bit hacky, but necessary to avoid crash for initial eval
            self._metric.reset(phase_and_task(strategy)[1])
        except Exception:
            pass
        return
    
    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            print(type(self._metric))
            print("DEBUG: DownstreamLinear: results()", self._metric.result())
            #return self._metric.result(task_label=0) # HACK: for some reson there are other task labels as well which carry no values.. 
            return self._metric.result()
        #return self._metric.result(0)
        return self._metric.result()
        
    @torch.no_grad()
    def prepare_tensordataset(self, model, dataset, device):
        x_reprs = []
        ys = []
        ts = []
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.eval_mb_size, #self.train_mb_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            drop_last=False
        ) 
    
        for _, mbatch in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
            
            x = x.to(device)
            y = y.to(device)

            # Get representation from backbone
            x_rep = model(x).detach()  # detach() is most likely not necessary here but I want to be sure
            # if self.normalize_features:
            #     x_rep = F.normalize(x_rep, dim=1)

            x_reprs.append(x_rep.cpu())
            ys.append(y.cpu())
            ts.append(tid)

            # DEBUG
            # if len(x_reprs) > 1000:
            #     x_reprs = torch.concat(x_reprs)
            #     std = x_reprs.std(dim=0)
            #     print(f"Mean std per dim: {std.mean():.6f}")
            #     print(f"Dims with std < 0.01: {(std < 0.01).sum()} / {std.shape[0]}")

            #     # Check 2: effective rank
            #     cov = torch.cov(x_reprs.T)
            #     eigvals = torch.linalg.eigvalsh(cov)
            #     eigvals = eigvals[eigvals > 1e-8]
            #     p = eigvals / eigvals.sum()
            #     eff_rank = torch.exp(-torch.sum(p * torch.log(p))).item()
            #     print(f"Effective rank: {eff_rank:.1f} / {x_reprs.shape[1]}")
            #     import sys; sys.exit()

        x_reprs = torch.concat(x_reprs)
        ys = torch.concat(ys)
        ts = torch.concat(ts)
        return x_reprs, ys, ts
    
    @torch.no_grad()
    def eval(self, strategy):
        """
        Runs a custom evaluation loop on the down-stream scenario
        """
        # Init datalaoder for the down-stream task
        ds_dataloader = torch.utils.data.DataLoader(
            self.eval_set, 
            batch_size=self.eval_mb_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            drop_last=False
        ) 
        
        for _, mbatch in tqdm(enumerate(ds_dataloader), total=len(ds_dataloader)):
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

            x = x.to(strategy.device)
            y = y.to(strategy.device)
            
            # Get representation from backbone
            x_rep = strategy.model.feature_extractor(x).detach()  # detach() is most likely not necessary here but I want to be sure
            x_rep = x_rep.view(x_rep.shape[0], -1)
            
            out = self.ds_head(x_rep)
            
            # Update the accuracy measure
            self._accuracy.update(predicted_y=out, true_y=y) #, 0)  # NOTE: removed tid
        return  
    
    def before_eval(self, strategy):
        super().before_eval(strategy)
        return

    def after_eval(self, strategy) -> 'MetricResult':
        print("\nPreparing Down-Stream Linear-Probe for", self.downstream_task)
        # Initialize and prepare the linear probing head
        with torch.enable_grad():  # NOTE: This is necessary because avalanche has a hidden torch.no_grad() in eval context!
            self.ds_head = _get_classifier(
                classifier_type=self.args.classifier, 
                n_classes=self.ds_n_classes, 
                feat_size=strategy.model.feature_extractor.feature_size, 
                initial_out_features=None, 
                task_incr=False,  # NOTE: no needed because we will only have 1 task (the downstream task))
                #lin_bias=self.args.lin_bias
            )
            # self.ds_head.apply(weight_reset)
            print("\nDSHead:")
            print(self.ds_head)
            # Move novel probe head to common device and (re-)initialize
            self.ds_head = self.ds_head.to(strategy.device)
            self.ds_head.train() # set to train mode (for safety)
            
            # Initialize local optimizer for the new head
            # Controlled by --lp_optimizer: 'adamw' (default) | 'sgd' | 'sgd_cosine'
            _lp_optimizer = getattr(self.args, 'lp_optimizer', 'adamw')
            local_scheduler = None

            if _lp_optimizer == 'sgd':
                # SGD + MultiStepLR: mirrors cassle barlow_linear.sh (lr=0.1, wd=0, momentum=0.9, decay at 60%/80%)
                _milestones = [int(0.6 * self.num_fintune_epochs), int(0.8 * self.num_fintune_epochs)]
                self.local_optim = torch.optim.SGD(
                    self.ds_head.parameters(),
                    lr=0.1,
                    momentum=0.9,
                    weight_decay=0.0,
                )
                local_scheduler = MultiStepLR(self.local_optim, milestones=_milestones, gamma=0.1)
            elif _lp_optimizer == 'sgd_cosine':
                # SGD + CosineAnnealingLR
                self.local_optim = torch.optim.SGD(
                    self.ds_head.parameters(),
                    lr=0.1,
                    momentum=0.9,
                    weight_decay=0.0,
                )
                local_scheduler = CosineAnnealingLR(self.local_optim, T_max=self.num_fintune_epochs)
            else:
                # AdamW (default, existing behaviour)
                self.local_optim = torch.optim.AdamW(
                    self.ds_head.parameters(),
                    lr=1e-3,
                    weight_decay=5e-3,
                )

            # --- Previous commented-out attempts (kept for reference) ---
            # self.local_optim = torch.optim.SGD(
            #     self.ds_head.parameters(),
            #     lr=0.01, weight_decay=1e-4, momentum=0.9
            # )
            # local_scheduler = OneCycleLR(
            #     self.local_optim, max_lr=0.1, div_factor=25, final_div_factor=1000,
            #     pct_start=0.0, anneal_strategy='cos',
            #     total_steps=self.num_fintune_epochs*len(self.train_set)
            # )

            # Prepare dataet and dataloader
            train_set = self.train_set

            print("Prepare downstream dataset for linear-probe on task", self.downstream_task)
            if self.buffer_lp_dataset:
                xs, ys, _ = self.prepare_tensordataset(strategy.model.feature_extractor, train_set, strategy.device)
                tensor_train_set = torch.utils.data.TensorDataset(xs, ys)
                lp_dataloader = torch.utils.data.DataLoader(
                    tensor_train_set, 
                    batch_size=self.eval_mb_size,  #self.train_mb_size, 
                    shuffle=True, 
                    num_workers=self.num_workers, 
                    drop_last=True
                ) 
            else:
                lp_dataloader = torch.utils.data.DataLoader(
                    train_set, 
                    batch_size=self.train_mb_size, 
                    shuffle=True, 
                    num_workers=self.num_workers, 
                    drop_last=True
                ) 
                
            # Run local optimization
            for _ in tqdm(range(self.num_fintune_epochs)):
                mean_loss = 0.0
                for _, mbatch in enumerate(lp_dataloader):
                    self.local_optim.zero_grad()

                    x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                    x_rep = x.to(strategy.device)
                    if not self.buffer_lp_dataset:
                        # Get representation from backbone
                        x_rep = strategy.model.feature_extractor(x).detach()
                    y = y.to(strategy.device)

                    out = self.ds_head(x_rep)

                    loss = self.criterion(out, y)
                    mean_loss += loss.item()
                    loss.backward()
                    self.local_optim.step()
                # Step LR scheduler once per epoch (only for sgd / sgd_cosine)
                if local_scheduler is not None:
                    local_scheduler.step()
                print("\nDEBUG: mean_loss:", mean_loss/len(lp_dataloader))
            print("\nLinear Probe for downstream training complete...\n")

        # Evaluate 
        self.eval(strategy)
        return super().after_eval(strategy)



class CorruptedDownstreamLinearProbeAccuracyMetric(DownstreamLinearProbeAccuracyMetric):
    def __init__(
            self, 
            corruptions_set,
            severities,
            **kwargs
        ):
        super().__init__(**kwargs)
        self._accuracy = NameAwareAccuracy()
        self._metric = self._accuracy  # NOTE: overwrite both

        self.corruptions_set = corruptions_set
        self.severities = severities
        self.eval_corruption_handler = CorruptionHandler("eval")
        return
    
    def _package_result(self, strategy) -> 'MetricResult':
        metric_value = self.result(strategy)  # NOTE: dict[str, float]
        
        add_exp = self._emit_at == 'experience'  # NOTE: always False - remove?
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp)  # add_task_k
                if k != "none":
                    metric_name = metric_name + "_" + k +str(safe_index(self.severities, self.corruptions_set.index(k)))
                    print("DEBUG: metric_name:", metric_name)

                metrics.append(MetricValue(self, metric_name, 
                                           v, plot_x_position))
            return metrics
        print("\nDEBUG: should never reach here!")
        metric_name = get_metric_name(
            self, strategy,
            add_experience=add_exp,
            add_task=True
        )
        return [MetricValue(self, metric_name, metric_value,
                            plot_x_position)]
    

    def after_eval(self, strategy):
        print("DEBUG: CorruptedDownstreamEVAL: after_eval")
        # Discover the corruption pipeline
        self.eval_set.eval()  # NOTE: not sure this is neccesary but at least it worsk this way
        self.eval_corruption_handler.discover_corruption_pipeline(
            source_transform=self.eval_set._flat_data._transform_groups["eval"].transforms[0].transforms)
        # Deactiate the corruption pipeline for training the linear probe
        self.eval_corruption_handler.set_corruption("none")
        return super().after_eval(strategy)
        

    def eval(self, strategy):
        print("DEBUG: CorruptedDownstreamEVAL: eval")
        # Iterate on eval cycle for each corruption
        for c_idx, corruption_name in enumerate(self.corruptions_set):
            print("Evaluating down-stream linear-probe for", self.downstream_task, "with corruption", corruption_name)
            # Set corruption and severity for experience
            self.eval_corruption_handler.set_corruption(corruption_name)
            self.eval_corruption_handler.set_severity(safe_index(self.severities, c_idx))
            # Set name of current corruption to the accuracy metric
            self._accuracy.set_name(corruption_name)
            # Call the eval
            super().eval(strategy)
        return