from typing import TypeVar
import torch

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task

import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

from matplotlib import pyplot as plt
import seaborn as sns

import copy
from tqdm import tqdm

from src.methods.ensemble import get_ensemble_model_buffer

TResult = TypeVar('TResult')


def get_barplot(data, ylim=None, put_lines = None):
    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    sns.barplot(x=list(range(len(data))), y=data, ax=ax)
    #plt.locator_params(axis='x', nbins=len(data)//10)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks[::len(xticks) // 10])
    if ylim:
        ax.set_ylim(ylim)
    if put_lines:
        for k,v in put_lines.items():
            plt.axvline(x=v, color='red', linestyle='--')
            plt.text(v+0.1,0,k,rotation=90)
    plt.xlabel('Singular Values')
    plt.ylabel('Magnitude')
    fig.tight_layout()
    return fig


def compute_embeddings(
    dataloader, 
    models,
    device,
    do_normalize=False,
    dim_reduction=None,
    target_dim=None
):
    """
    Code adapted from: https://github.com/ivanpanshin/SupCon-Framework/blob/main/tools/losses.py
    """
    x_reprs = {}
    ys = []
    ts = []

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
            
            # if do_normalize:
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

    # Apply dimension reduction if specified
    if dim_reduction is not None:
        if target_dim is None:
            raise ValueError("target_dim must be specified when using dimension reduction")
        
        print(f"Applying {dim_reduction} dimension reduction from {x_reprs.shape[1]} to {target_dim}...")
        
        # Convert to numpy for sklearn
        x_numpy = x_reprs.numpy()
        
        reducer = None
        if dim_reduction == 'pca':
            reducer = PCA(n_components=target_dim, whiten=False)
        elif dim_reduction == 'gaussian':
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
        
    return x_reprs, ys, ts


class DownstreamConcatPCAProbing(GenericPluginMetric[float, Accuracy]):
    def __init__(
        self,
        args, 
        downstream_task,
        train_set,
        eval_set,
        on_train_set=False,
        reduce_dim_method=None,
        target_dim=None
    ):
        self._accuracy = Accuracy() # metric calculation container
        super(DownstreamConcatPCAProbing, self).__init__(
            self._accuracy, 
            reset_at='stream', 
            emit_at='stream', 
            mode='eval'
        )

        self.args = args
        self.downstream_task = downstream_task
        self.train_set = train_set
        self.eval_set = eval_set
        #self.pca_threshold = pca_threshold
        #self.tsne_perplexity = tsne_perplexity
        self.on_train_set = on_train_set  # TODO: integrate
        self.reduce_dim_method = reduce_dim_method
        self.target_dim = target_dim

        self.last_result_value = None

        self.batch_size = args.bs
        self.num_workers = args.num_workers

        self.DIR_NAME = "concat_PCA_plots"
        self.plt_counter = 0

        #self.is_initial_eval_run = False
        # self._prev_state = None
        # self._prev_training_modes = None  # NOTE: required to reset the training scheme after calling eval in train mode..
        return

    def __str__(self):
        return "Concat_PCA_"+self.downstream_task

    # def _package_result(self, strategy):
    #     #metric_value = self.result(strategy)
    #     metric_value = self.last_result_value

    #     add_exp = self._emit_at == 'experience'
    #     plot_x_position = strategy.clock.train_iterations

    #     if isinstance(metric_value, dict):
    #         metrics = []
    #         for k, v in metric_value.items():
    #             metric_name = get_metric_name(
    #                 self, strategy, add_experience=add_exp, add_task=k)
    #             metrics.append(MetricValue(self, metric_name, v,
    #                                        plot_x_position))
    #         return metrics
        
    #     metric_name = get_metric_name(self, strategy,
    #                                     add_experience=add_exp,
    #                                     add_task=True)
    #     print("metric_name: ", metric_name)
    #     return [MetricValue(self, metric_name, metric_value,
    #                         plot_x_position)]


    # def result(self, strategy=None):
    #     if self._emit_at == 'stream' or strategy is None:
    #         print(self._metric.result(task_label=0))
    #         return self._metric.result(task_label=0) # HACK: for some reson there are other task labels as well which carry no values.. 
    #     return self._metric.result(0)


    def reset(self, strategy=None):
        if self._reset_at == 'stream' or strategy is None:
            self._metric.reset()
        try:
            self._metric.reset(phase_and_task(strategy)[1])
        except Exception:
            pass
        return
    

    # def update_models_buffer(self, strategy):
    #     # Check if there is a models_buffer in the strategy
    #     if not hasattr(strategy, 'ds_models_buffer'):
    #         strategy.ds_models_buffer = {}
    #         print("DEBUG: Initializing ds_models_buffer in strategy:", strategy.ds_models_buffer)
    #         print("DEBUG: Type of strategy.ds_models_buffer:", type(strategy.ds_models_buffer))
    #     # Initialize the buffer for the current downstream task if not present
    #     if not self.downstream_task in strategy.ds_models_buffer:
    #         print("DEBUG: Type of strategy.ds_models_buffer:", type(strategy.ds_models_buffer))
    #         print("DEBUG: Initializing ds_models_buffer for downstream task:", self.downstream_task)
    #         strategy.ds_models_buffer[self.downstream_task] = [] #strategy.ds_models_buffer.append(copy.deepcopy(strategy.model.feature_extractor))
    #     # Append the current model to the buffer
    #     strategy.ds_models_buffer[self.downstream_task].append(copy.deepcopy(strategy.model.feature_extractor))
    #     return

    
    # def before_eval(self, strategy):
    #     super().before_eval(strategy)

    #     # Store the current model locally
    #     self.update_models_buffer(strategy)


    def eval(self, strategy):
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
        
        embeddings, _, _ = compute_embeddings(
            dataloader=ds_dataloader, 
            models=get_ensemble_model_buffer(strategy),  # strategy.ds_models_buffer[self.downstream_task],
            device=strategy.device,
            do_normalize=False,
            dim_reduction=self.reduce_dim_method,
            target_dim=self.target_dim
        )

        print("\nsklearn pca...")
        pca = PCA(n_components=None, whiten=True)
        pca.fit(embeddings)
        #self.last_result_value = len(pca.singular_values_)
        #print(self.last_result_value)
        self.last_result_value = {}
        # sum up pca.explained_variance_ratio_ until 90% variance is explained
        sum = 0
        put_lines = {}
        for i in range(len(pca.explained_variance_ratio_)):
            sum += pca.explained_variance_ratio_[i]
            if sum >= 0.9:
                if not "90%" in put_lines:
                    put_lines["90%"] = i
            if sum >= 0.95:
                if not "95%" in put_lines:
                    put_lines["95%"] = i
            if sum >= 0.99:
                if not "99%" in put_lines:
                    put_lines["99%"] = i
                break

        # eigenvector magnitudes
        self.last_result_value[0] = get_barplot(  # NOTE: returns a plt.fig
            data=pca.singular_values_,
            put_lines=put_lines
        )

        local_save_dir = self.args.results_path / self.DIR_NAME / self.downstream_task
        local_save_dir.mkdir(parents=True, exist_ok=True)
        self.last_result_value[0].savefig(local_save_dir / (f'pca_mag' + str(self.plt_counter) + '.png'))
        self.last_result_value[0].savefig(local_save_dir / (f'pca_mag' + str(self.plt_counter) + '.svg'), format="svg")
        torch.save(
            {
                'singular_values': pca.singular_values_,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                # 'variance_thresholds': put_lines,
            },
            local_save_dir / (f'pca_data' + str(self.plt_counter) + '.pt')
        )
        print("DEBUG: saved:")

        # explained_variance
        # self.last_result_value[1] = get_barplot(
        #     data=pca.explained_variance_ratio_,
        #     #ylim=(0, 0.11),
        #     put_lines=put_lines
        # )
        # DEBUG:
        #self.last_result_value[1].savefig('debug_pca_1.png')
        #print("DEBUG: saved debug_pca.png")

        # Increase plot_counter
        self.plt_counter += 1
        return  
 

    def after_eval(self, strategy):
        # Evaluate 
        self.eval(strategy)

        # In case of initial evaluation, do not increase the exp_seen counter
        # if self.is_initial_eval_run:
        #     # Reset the state of the continual learner
        #     assert(not self._prev_state is None)
        #     assert(not self._prev_training_modes is None)
        #     ContinualEvaluationPhasePlugin.restore_strategy_(strategy, self._prev_state, self._prev_training_modes)
        #     self.is_initial_eval_run = False # Reset flag
        return super().after_eval(strategy) # NOTE: This return is maximally necessary because otherwise it will not log porperly


    