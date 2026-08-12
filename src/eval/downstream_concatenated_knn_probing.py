
import torch

from avalanche.evaluation.metric_definitions import GenericPluginMetric
from avalanche.evaluation.metrics.accuracy import Accuracy
from avalanche.evaluation.metric_results import MetricValue, MetricResult
from avalanche.evaluation.metric_utils import get_metric_name, phase_and_task

#from src.methods.replay import ERPlugin

from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

import numpy as np

from tqdm import tqdm

from src.eval.downstream_knn import KNNClassifier
from src.methods.ensemble import get_ensemble_model_buffer


def compute_embeddings(
    dataloader,
    models,
    device,
    do_normalize=False,
    dim_reduction=None,
    target_dim=None,
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
            
            if do_normalize:
                x_rep = torch.nn.functional.normalize(x_rep, dim=1)
            
            if mbatch_idx not in x_reprs:
                x_reprs[mbatch_idx] = x_rep.cpu()
            else:
                x_reprs[mbatch_idx] = torch.cat((x_reprs[mbatch_idx], x_rep.cpu()), dim=1)
                
                if do_normalize: # re-normalize after concatenation
                    x_reprs[mbatch_idx] = torch.nn.functional.normalize(x_reprs[mbatch_idx], dim=1)
            
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
    reducer = None
    if dim_reduction is not None:
        if target_dim is None:
            raise ValueError("target_dim must be specified when using dimension reduction")
        
        print(f"Applying {dim_reduction} dimension reduction from {x_reprs.shape[1]} to {target_dim}...")
        
        # Convert to numpy for sklearn
        x_numpy = x_reprs.numpy()
        
        if dim_reduction == 'pca':
            reducer = PCA(n_components=target_dim, whiten=True)
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

        # Re-normalize after reduction if needed
        if do_normalize:
            x_reprs = torch.nn.functional.normalize(x_reprs, dim=1)
    
    return x_reprs, ys, ts, reducer


class DownstreamConcatenatedKNNAccuracyMetric(GenericPluginMetric[float, Accuracy]):
    def __init__(
        self, 
        args, 
        downstream_task, 
        train_set, 
        eval_set, 
        k,
        do_normalize=True,
        #train_stream_from_ER_buffer=False
        reduce_dim_method=None,
        target_dim=None
    ):
        self._accuracy = Accuracy() # metric calculation container
        super(DownstreamConcatenatedKNNAccuracyMetric, self).__init__(
            self._accuracy, reset_at='stream', emit_at='stream', mode='eval'
        )

        self.args = args
        # Init the scenario for the downstream task
        self.downstream_task = downstream_task
        self.train_set = train_set
        #self.train_stream_from_ER_buffer = train_stream_from_ER_buffer
        self.eval_set = eval_set

        self.reduce_dim_method = reduce_dim_method
        self.target_dim = target_dim

        self.knn_classifier = None
        self.knn_classifiers_k = dict()
        self.knn_acc_k = dict()
        self.k = k # NOTE: list of k's

        self.do_normalize = do_normalize
        self.dim_reducer = None

        self.batch_size = args.bs
        self.num_workers = args.num_workers

        self.num_fintune_epochs = args.lp_finetune_epochs

        self.eval_complete = False
        self.initial_out_features = None

        self.skip_initial_eval = args.skip_initial_eval
        self.is_initial_eval_run = False
        self._prev_state = None
        self._prev_training_modes = None  # NOTE: required to reset the training scheme after calling eval in train mode..
        return


    def __str__(self):
        return "Top1_Concatenated_KNN_"+self.downstream_task+"_Acc"


    def _package_result(self, strategy) -> 'MetricResult':
        #metric_value = self.result(strategy)
        metric_value = self.knn_acc_k

        add_exp = self._emit_at == 'experience'
        plot_x_position = strategy.clock.train_iterations

        if isinstance(metric_value, dict):
            metrics = []
            for k, v in metric_value.items():
                metric_name = self.__str__() + "/eval_phase/test_stream/k_" + str(k)
                metrics.append(MetricValue(self, metric_name, v,
                                           plot_x_position))
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
            self.knn_acc_k = dict()
            
        try: # NOTE: the try-except is a bit hacky, but necessary to avoid crash for initial eval
            self._metric.reset(phase_and_task(strategy)[1])
        except Exception:
            pass
        return
    

    def result(self, strategy=None):
        if self._emit_at == 'stream' or strategy is None:
            print(self._metric.result(task_label=0))
            return self._metric.result(task_label=0) # HACK: for some reson there are other task labels as well which carry no values.. 
        return self._metric.result(0)
        

    def before_eval(self, strategy):
        super().before_eval(strategy)

        print("\nPreparing KNN classifier for", self.downstream_task)
        # Prepare dataet and dataloader
        train_set = self.train_set

        dataloader = torch.utils.data.DataLoader(
            train_set, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            drop_last=False
        ) 

        # Initialize and prepare KNN classifier
        for k_i in self.k:  # NOTE: supports multiple k's
            self.knn_classifiers_k[k_i] = KNNClassifier(k=k_i, use_inner_prduct=True)

        # Get the embeddings for the classifier
        embeddings, labels, _, reducer = compute_embeddings(
            dataloader=dataloader, 
            models=get_ensemble_model_buffer(strategy), #strategy.model.feature_extractor,
            device=strategy.device,
            do_normalize=self.do_normalize,
            dim_reduction=self.reduce_dim_method,
            target_dim=self.target_dim
        )
        self.dim_reducer = reducer  # save the reducer for later use on eval set
        print("Embeddings shape:", embeddings.shape)
        # Train the KNN classifier  
        for k_i in self.k:
            print("fitting with k=", k_i, "...")
            self.knn_classifiers_k[k_i].fit(embeddings, labels)
        print("KNNs prepared...")
        return
    
    
    def eval(self, strategy):
        """
        Runs a custom evaluation loop on the down-stream scenario
        """
        ds_dataloader = torch.utils.data.DataLoader(
            self.eval_set, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers, 
            drop_last=False
        ) 

        embeddings, labels, _, reducer = compute_embeddings(
            dataloader=ds_dataloader, 
            models=get_ensemble_model_buffer(strategy), #strategy.model.feature_extractor,
            device=strategy.device,
            do_normalize=self.do_normalize,
        )

        # Apply dim reduction
        if self.dim_reducer is not None:
            print("Applying dimension reduction to eval embeddings...")
            # Convert to numpy
            emb_numpy = embeddings.numpy()
            embeddings = self.dim_reducer.transform(emb_numpy)
            #embeddings = torch.from_numpy(emb_reduced_numpy).float()
            # Re-normalize after reduction if needed
            if self.do_normalize:
                embeddings = torch.nn.functional.normalize(torch.from_numpy(embeddings).float(), dim=1)

        print("Eval embeddings shape:", embeddings.shape)

        for k_i in self.k:
            out = self.knn_classifiers_k[k_i].predict(embeddings)
            # Compare out with labels for accuracy caluclation
            self.knn_acc_k[k_i] = (torch.sum(out == labels)/len(out)).item()
        return  
    

    def after_eval(self, strategy) -> 'MetricResult':
        # Evaluate 
        self.eval(strategy)
        return super().after_eval(strategy)

