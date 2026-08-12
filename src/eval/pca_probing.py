from typing import TYPE_CHECKING, Dict, TypeVar
from collections import deque, defaultdict
import torch

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task

import numpy as np
from sklearn.decomposition import PCA

from matplotlib import pyplot as plt
import seaborn as sns

from tqdm import tqdm

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


def compute_embeddings(loader, model, do_normalize=False):
    """
    Code adapted from: https://github.com/ivanpanshin/SupCon-Framework/blob/main/tools/losses.py
    """
    # TODO: keep in torch?
    print("Computing embeddings, with dim: ", model.feature_size)
    total_embeddings = []
    total_labels = []

    for idx, mbatch in tqdm(enumerate(loader), total=len(loader)):
        images = mbatch[0].cuda()
        labels = mbatch[1]
        #bsz = labels.shape[0]
        
        embed = model(images)
        if do_normalize:
            embed = torch.nn.functional.normalize(embed, dim=1)
        total_embeddings.append(embed.detach().cpu().numpy())
        total_labels.append(labels.detach().numpy())

        del images, labels, embed
        torch.cuda.empty_cache()
    total_embeddings = np.concatenate(total_embeddings, axis=0)
    total_labels = np.concatenate(total_labels, axis=0)
    return np.float32(total_embeddings), total_labels.astype(int)


class DownstreamPCAProbing(GenericPluginMetric[float, Accuracy]):
    def __init__(
        self,
        args, 
        downstream_task,
        train_set,
        eval_set, 
        #pca_threshold=0.99,
        on_train_set=False,
    ):
        self._accuracy = Accuracy() # metric calculation container
        super(DownstreamPCAProbing, self).__init__(
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

        self.last_result_value = None

        self.batch_size = args.bs
        self.num_workers = args.num_workers

        self.DIR_NAME = "PCA_plots"
        self.plt_counter = 0

        #self.is_initial_eval_run = False
        # self._prev_state = None
        # self._prev_training_modes = None  # NOTE: required to reset the training scheme after calling eval in train mode..
        return

    def __str__(self):
        return "PCA_"+self.downstream_task

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
        
        embeddings, _ = compute_embeddings(
            loader=ds_dataloader, 
            model=strategy.model.feature_extractor,
            do_normalize=False
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
                #'variance_thresholds': put_lines,
            },
            local_save_dir / (f'pca_data' + str(self.plt_counter) + '.pt')
        )
        print("DEBUG: saved to:", local_save_dir / (f'pca_mag' + str(self.plt_counter) + '.png'))
        print("DEBUG: saved to:", local_save_dir / (f'pca_data' + str(self.plt_counter) + '.pt'))

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


    