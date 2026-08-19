#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

import copy
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task

from src.models.models import _get_classifier
from src.methods.ensemble import get_ensemble_model_buffer

from src.eval.continual_eval import ContinualEvaluationPhasePlugin # NOTE: used to call 'get_strategy_state' and 'restore_strategy_'



class ConcatDownstreamProbingAccuracyMetric(GenericPluginMetric[float, Accuracy]):
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
        train_set, 
        eval_set, 
        eval_mb_size,
        downstream_task_name, 
        n_classes,
        criterion,
        normalize_features=False, # False
        reduce_dim_in_head=None,
        target_dim=None
    ): 
        self._accuracy = Accuracy() # metric calculation container
        super(ConcatDownstreamProbingAccuracyMetric, self).__init__(
            self._accuracy, reset_at='stream', emit_at='stream', mode='eval')

        self.downstream_task_name = downstream_task_name
        self.train_set = train_set
        self.eval_set = eval_set
        self.eval_mb_size = eval_mb_size
        self.ds_n_classes = n_classes
        self.criterion = criterion
        self.normalize_features = normalize_features

        self.batch_size = args.bs
        self.num_workers = args.num_workers
        self.num_fintune_epochs = args.lp_finetune_epochs
        self.skip_initial_eval = args.skip_initial_eval
        self.lp_optimizer = getattr(args, 'lp_optimizer', 'adamw')  # default adamw for backward compat
        self.reduce_dim_in_head = reduce_dim_in_head
        self.target_dim = target_dim
        self.dim_reducer = None  # Will hold the fitted dimension reducer if used
        
        self.combined_head = None # NOTE: local copy of the model's head used for linear probing
        self.local_optim = None

        self.is_initial_eval_run = False
        self._prev_state = None
        self._prev_training_modes = None # NOTE: required to reset the training scheme after calling eval in train mode..
        return


    def __str__(self):
        return "Top1_Concatenated_"+self.downstream_task_name+"_Acc"


    def _package_result(self, strategy) -> 'MetricResult':
        metric_value = self.result(strategy)
        
        add_exp = self._emit_at == 'experience'
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = get_metric_name(
                    self, strategy, add_experience=add_exp, add_task=k)
                metrics.append(MetricValue(self, metric_name, v,
                                           plot_x_position))
            return metrics
        
        metric_name = get_metric_name(self, strategy,
                                        add_experience=add_exp,
                                        add_task=True)
        return [MetricValue(self, metric_name, metric_value,
                            plot_x_position)]


    @torch.no_grad()
    def prepare_ensemble_tensordataset( 
        self, 
        models, 
        dataset, 
        device,
        dim_reduction=None, 
        target_dim=None
    ):
        x_reprs = {}
        ys = []
        ts = []
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.eval_mb_size, #self.train_mb_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            drop_last=False
        ) 
        print("Preparing dataset for downstream eval...")
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

                # Get representation from backbone
                x_rep = model(x).detach()  # detach() is most likely not necessary here but I want to be sure
                
                # if self.normalize_features:
                #     x_rep = F.normalize(x_rep, dim=1)
                #x_reprs.append(x_rep.cpu())
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

        if dim_reduction is None:
            return x_reprs, ys, ts

        # Apply dimension reduction if specified
        if dim_reduction is not None:
            if target_dim is None:
                raise ValueError("target_dim must be specified when using dimension reduction")
            
            print(f"Applying {dim_reduction} dimension reduction from {x_reprs.shape[1]} to {target_dim}...")
            
            # Convert to numpy for sklearn
            x_numpy = x_reprs.numpy()
            
            if dim_reduction == 'pca':
                reducer = PCA(n_components=target_dim, whiten=False) # whiten=True
            elif dim_reduction == 'gaussian':
                if x_reprs.shape[1] <= target_dim:
                    print(f"Skipping Gaussian reduction: input dim {x_reprs.shape[1]} <= target_dim {target_dim}")
                    self.dim_reducer = None
                    return x_reprs, ys, ts
                reducer = GaussianRandomProjection(n_components=target_dim)
                #reducer = GaussianRandomProjection(n_components='auto', eps=0.1)  # this allows it to find the num_components itself
            else:
                raise ValueError("dim_reduction must be 'pca' or 'gaussian'")

            # Fit and transform
            print("Fitting reducer...")
            x_reduced_numpy = reducer.fit_transform(x_numpy)
            print("done...")
            
            # Convert back to tensor
            x_reprs = torch.from_numpy(x_reduced_numpy).float()
            print(f"Dimension reduction complete: shape is now {x_reprs.shape}")
            
            # Optionally return the fitted reducer for later use
            self.dim_reducer = reducer
        return x_reprs, ys, ts
    

    def forward_all_feats(self, x, models, device):
        x_repres = None
        for model in models:
            model = model.to(device)
            model.eval()  # set to eval mode for safety

            x_rep = model(x).detach()  # detach() is most likely not necessary here but I want to be sure
            
            # if self.normalize_features:
            #     x_rep = F.normalize(x_rep, dim=1)
            if x_repres is None:
                x_repres = x_rep
            else:
                x_repres = torch.cat((x_repres, x_rep), dim=1)
            model = model.to("cpu")
        return x_repres

    # DEPRACATED - replaced by get_ensemble_model_buffer
    # def update_models_buffer(self, strategy): 
    #     # Check if there is a models_buffer in the strategy
    #     if not hasattr(strategy, 'ds_models_buffer'):
    #         strategy.ds_models_buffer = {}
    #         print("DEBUG: Initializing ds_models_buffer in strategy:", strategy.ds_models_buffer)
    #         print("DEBUG: Type of strategy.ds_models_buffer:", type(strategy.ds_models_buffer))
    #     # Initialize the buffer for the current downstream task if not present
    #     if not self.downstream_task_name in strategy.ds_models_buffer:
    #         print("DEBUG: Type of strategy.ds_models_buffer:", type(strategy.ds_models_buffer))
    #         print("DEBUG: Initializing ds_models_buffer for downstream task:", self.downstream_task_name)
    #         strategy.ds_models_buffer[self.downstream_task_name] = [] #strategy.ds_models_buffer.append(copy.deepcopy(strategy.model.feature_extractor))
    #     # Append the current model to the buffer
    #     strategy.ds_models_buffer[self.downstream_task_name].append(copy.deepcopy(strategy.model.feature_extractor))
    #     return


    def before_eval(self, strategy):
        super().before_eval(strategy)

        # Store the current model locally
        #self.update_models_buffer(strategy)  #self.models.append(copy.deepcopy(strategy.model.feature_extractor))

        # Temporarily unload the current model from device
        strategy.model = strategy.model.to('cpu')  # NOTE: Reduce VRAM usage
        models_buffer = get_ensemble_model_buffer(strategy)

        xs, ys, _ = self.prepare_ensemble_tensordataset(
            models_buffer,  #strategy.ds_models_buffer[self.downstream_task_name], 
            self.train_set, 
            strategy.device,
            dim_reduction=self.reduce_dim_in_head,
            target_dim=self.target_dim
        )
        tensor_train_set = torch.utils.data.TensorDataset(xs, ys)
        print("DEBUG: tensor_train_set:", tensor_train_set)

        # Initialize and prepare the linear probing head
        with torch.enable_grad():  # NOTE: This is necessary because avalanche has a hidden torch.no_grad() in eval context!
            lp_dataloader = torch.utils.data.DataLoader(
                tensor_train_set,  #self.train_set, 
                batch_size=self.eval_mb_size, # self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers, 
                drop_last=True
            ) 
        
            # Prepare the linear-probing head
            if not self.reduce_dim_in_head is None:
                print(f"Using dimension reduction in linear probe head: {self.reduce_dim_in_head} to {self.target_dim}")
                num_input_features = self.target_dim
            else:
                #print("DEBUG: models_buffer:", len(strategy.ds_models_buffer[self.downstream_task_name]))
                #num_input_features = strategy.model.feature_extractor.feature_size * len(strategy.ds_models_buffer[self.downstream_task_name])  
                print("DEBUG: models_buffer:", len(models_buffer))
                num_input_features = strategy.model.feature_extractor.feature_size * len(models_buffer)
                # One full representation for each 
            
            print("DEBUG: num_input_features for combined representation:", num_input_features)
            self.combined_head = _get_classifier(
                classifier_type="linear", 
                n_classes=self.ds_n_classes, 
                feat_size=num_input_features,
                initial_out_features=None,  # NOTE: using n_classes as initial_out_features
                task_incr=False
            )
            print("combined_head:", self.combined_head)

            # Move novel probe head to common device and (re-)initialize
            self.combined_head = self.combined_head.to(strategy.device)
            self.combined_head.train() # set to train mode (for safety)
            
            # Initialize local optimizer for the new head
            # Controlled by --lp_optimizer: 'adamw' (default) | 'sgd' | 'sgd_cosine'
            local_scheduler = None

            if self.lp_optimizer == 'sgd':
                # SGD + MultiStepLR: mirrors cassle barlow_linear.sh (lr=0.1, wd=0, momentum=0.9, decay at 60%/80%)
                _milestones = [int(0.6 * self.num_fintune_epochs), int(0.8 * self.num_fintune_epochs)]
                self.local_optim = torch.optim.SGD(
                    self.combined_head.parameters(),
                    lr=0.1,
                    momentum=0.9,
                    weight_decay=0.0,
                )
                local_scheduler = MultiStepLR(self.local_optim, milestones=_milestones, gamma=0.1)
            elif self.lp_optimizer == 'sgd_cosine':
                # SGD + CosineAnnealingLR
                self.local_optim = torch.optim.SGD(
                    self.combined_head.parameters(),
                    lr=0.1,
                    momentum=0.9,
                    weight_decay=0.0,
                )
                local_scheduler = CosineAnnealingLR(self.local_optim, T_max=self.num_fintune_epochs)
            else:
                # AdamW (default, existing behaviour)
                self.local_optim = torch.optim.AdamW(
                    self.combined_head.parameters(),
                    lr=1e-3,
                    weight_decay=5e-3,
                )

            # Train the linear probe on the down-stream task
            for _ in tqdm(range(self.num_fintune_epochs)):
                mean_loss = 0.0
                for _, mbatch in enumerate(lp_dataloader):
                    self.local_optim.zero_grad()

                    x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

                    x = x.to(strategy.device)
                    y = y.to(strategy.device)

                    #x_rep_concat = self.forward_all_feats(x, device=strategy.device).detach()
                    out = self.combined_head(x)

                    loss = self.criterion(out, y) #strategy._criterion(out, y)
                    mean_loss += loss.item()
                    loss.backward()
                    self.local_optim.step()
                # Step LR scheduler once per epoch (only for sgd / sgd_cosine)
                if local_scheduler is not None:
                    local_scheduler.step()
                print("\nDEBUG: mean_loss:", mean_loss/len(lp_dataloader))
            print("\nLinear Probe for downstream training complete...\n")

        # Re-load the strategy.model to device
        strategy.model = strategy.model.to(strategy.device)
        return


    def do_eval(self, strategy):
        """
        Runs a custom evaluation loop on the down-stream scenario
        """
        # Init datalaoder for the down-stream task
        ds_dataloader = torch.utils.data.DataLoader(
            self.eval_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            drop_last=False
        ) 
        
        # Reduce VRAM usage
        strategy.model = strategy.model.to("cpu")

        print("\nEval concatenated representation on downstream task - ", self.downstream_task_name, "...")
        for _, mbatch in tqdm(enumerate(ds_dataloader), total=len(ds_dataloader)):
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]

            x = x.to(strategy.device)
            y = y.to(strategy.device)
            
            # Get representation from backbone
            x_rep_concat = self.forward_all_feats(
                x, 
                get_ensemble_model_buffer(strategy), #strategy.ds_models_buffer[self.downstream_task_name],
                device=strategy.device
            ).detach()

            if self.dim_reducer is not None:
                # Apply the fitted dimension reducer
                #print(f"Applying fitted dimension reducer to eval batch...")
                x_rep_concat_numpy = x_rep_concat.cpu().numpy()
                x_rep_concat_reduced = self.dim_reducer.transform(x_rep_concat_numpy)
                x_rep_concat = torch.from_numpy(x_rep_concat_reduced).float().to(strategy.device)
                #print(f"Dimension reduction complete: shape is now {x_rep_concat.shape}")
            out = self.combined_head(x_rep_concat)
            
            # Update the accuracy measure
            self._accuracy.update(out, y)  # NOTE: Use TaskAwareAccuracy when wanting to use tid
        
        # Return model to device
        strategy.model = strategy.model.to(strategy.device)
        return  
    
    
    def after_eval(self, strategy) -> 'MetricResult':
        # Evaluate 
        self.do_eval(strategy)

        # In case of initial evaluation, do not increase the exp_seen counter
        if self.is_initial_eval_run:
            # Reset the state of the continual learner
            assert(not self._prev_state is None)
            assert(not self._prev_training_modes is None)
            ContinualEvaluationPhasePlugin.restore_strategy_(strategy, self._prev_state, self._prev_training_modes)
            self.is_initial_eval_run = False # Reset flag
        return super().after_eval(strategy) # NOTE: This return is maximally necessary because otherwise it will not log porperly


    def reset(self, strategy=None):
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        else:
            self._metric.reset(phase_and_task(strategy)[1])
        return
    
    
    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            return self._metric.result()
        return self._metric.result()
        