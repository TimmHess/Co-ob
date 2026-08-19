import os
import yaml
import argparse
from distutils.util import strtobool

from pathlib import Path


def parse_tuple(s):
    return tuple(map(int, s.split(',')))


def get_arg_parser():
    PARSER_DEFAULTS = {
    }

    parser = argparse.ArgumentParser()

    # Meta hyperparams
    parser.add_argument('--exp_name', required=True, type=str, 
                        help='Name for the experiment.')
    parser.add_argument('--config_base', type=str, default="",
                        help='Base config path.')
    parser.add_argument('--config_path', nargs='*', type=str, default=None,
                        help='Yaml file with config for the args.')

    parser.add_argument('--exp_postfix', type=str, default='#now,#uid',
                        help='Extension of the experiment name. A static name enables continuing if checkpointing is define'
                            'Needed for fixes/gridsearches without needing to construct a whole different directory.'
                            'To use argument values: use # before the term, and for multiple separate with commas.'
                            'e.g. #cuda,#featsize,#now,#uid')
    parser.add_argument('--save_path', type=str, default='./results/', help='save eval results.')
    parser.add_argument('--num_workers', default=None, type=int, help='Number of workers for the dataloaders.')
    parser.add_argument('--use_max_workers', action='store_true', default=False,
                        help='Use all available CPU cores for the dataloaders.')
    parser.add_argument('--cuda', default=True, type=lambda x: bool(strtobool(x)), help='Enable cuda?')
    parser.add_argument('--disable_pbar', default=True, type=lambda x: bool(strtobool(x)), help='Disable progress bar')
    parser.add_argument('--seed', default=None, type=int, help='Run a specific seed.')
    parser.add_argument('--deterministic', default=False, type=lambda x: bool(strtobool(x)),
                        help='Set deterministic option for CUDNN backend.')
    parser.add_argument('--wandb', default=False, type=lambda x: bool(strtobool(x)), 
                        help="Use wandb for exp tracking.")

    parser.add_argument('--overwrite_results', action='store_true', default=False, 
                        help='Overwrite results directory.')
    parser.add_argument('--do_self_destruct', action='store_true', default=False,
                        help='Self destruct if run is not completing.')

    parser.add_argument('--requires_gpu', type=lambda x: bool(strtobool(x)), default=True, 
                        help='Marks the run to require a GPU. If no GPU is found, the run will not start.')
    parser.add_argument('--restrict_num_threads', default=None, type=int, 
                        help='Restrict number of threads for torch and numpy.')
    parser.add_argument('--restrict_num_interop_threads', default=None, type=int, 
                        help='Restrict number of threads for torch and numpy.')
    parser.add_argument('--use_ddp', action='store_true', default=False,
                        help='Use Distributed Data Parallel (DDP) training.')

    # Dataset
    parser.add_argument('--scenario', type=str, default='smnist',
        choices=['smnist', 
                'rotmnist',
                'incr_mnist',
                'static_corrupted_mnist',
                'cifar10', 
                'cifar100', 
                'incr_cifar100',
                'repeat_cifar100',
                'domain_cifar100', 
                'repeat_domain_cifar100',
                'incr_domain_cifar100',
                'imgnet1k_256x256',
                'repeat_dataset',  # Scenario crafted from (any) single dataset (len(scenario_datasets)) == 1)
                'split_dataset' 
            ]
        )
    
    parser.add_argument('--scenario_datasets', type=str, nargs='*', default=None,
                        help='Datasets to use for the scenario. (If applicable.)')
    parser.add_argument('--scenario_rootpaths', type=str, nargs='*', default=None,
                        help='Root paths of the datasets. (If applicable.)')

    parser.add_argument('--use_rand_class_ordering', default=False, type=lambda x: bool(strtobool(x)), 
                        help='Whether to use random task ordering.')
    parser.add_argument('--fixed_class_order', nargs='+', default=None, 
                        help='Fixed class order for the scenario.')
    parser.add_argument('--num_experiences', type=int, default=None, 
                        help='Number of experiences to use in the scenario.')
    parser.add_argument('--dset_rootpath', type=str, default='./data', 
                        help='Root path of the downloaded dataset for e.g. Mini-Imagenet')
    parser.add_argument('--partial_num_tasks', type=int, default=None,
                        help='Up to which task to include, e.g. to consider only first 2 tasks of 5-task Split-MNIST')
    parser.add_argument('--task_labels_per_exp', type=int, nargs='+', default=None,
                        help='Number of task labels that are combined per experience. Currently only used for DomainCifar100.\
                              If None, all tasks are separate. If [4,1], will have two experiences with\
                              the first spanning 4 tasks and the second 1 task.')
    
    parser.add_argument('--copy_results_on_end', action='store_true', default=False,
                        help='Keep and copy results from condor scratch dir.')

    # Feature extractor
    parser.add_argument('--featsize', type=int, default=400,
                        help='The feature size output of the feature extractor.'
                            'The classifier uses this embedding as input.')
    parser.add_argument('--backbone', type=str, default=None, 
                        choices=[None,
                                'input', 
                                'mlp', 'mlp_ln', 'mlp_bn', 'mlp_200',
                                'simple_cnn',
                                'VGG_DF',
                                'VGG_DF_BN',
                                'VGG11',
                                'VGG11_PT_ImgNet1k',
                                'SlimResNet18',
                                'SlimResNet18_nf64',
                                'SlimResNet34',
                                'ResNet18', 'ResNet18_rdim',
                                'ResNet18_PT',
                                'ViT_tiny',
                                'ViT-Lite-7-4',
                                'ViT_B_16_PT',
                                'ViT_tiny_PT_ImgNet21k',
                                'HF_ViT-B_16_PT_ImgNet1k',
                                'TM_ViT-B_16_augreg_in1k',
                                'ConvNext_Tiny',
                                'ConvNext_Tiny_PT',
                                ], 
                        help="Feature extractor backbone.")
    parser.add_argument('--use_GAP', action='store_true', default=False,
                        help="Use Global Avg Pooling after feature extractor (for Resnet18).")
    parser.add_argument('--backbone_weights', type=str, default=None, 
                        help='Path to backbone weights.')
    parser.add_argument('--pretrained_weights', type=str, default=None, 
                        help='Path to custom pretrained weights.')
    parser.add_argument('--allow_non_stric_load', action='store_true', default=False,
                        help='Allow non-strict loading of the model weights. Not recommended!')
    parser.add_argument('--force_repr_dim', type=int, default=None,
                        help='Force the representation dimension to this value. Only for certain backbones.')
    parser.add_argument('--force_compile_model', action='store_true', default=False,
                        help='Whether to force compile the model. NOT RECOMMENDED, can crash!')

    parser.add_argument('--classifier_zero_init', action='store_true', default=False, 
                        help='Whether to zero init the classifier.')
    parser.add_argument('--overwrite_input_size', type=int, nargs='+', default=None, 
                        help='Overwrite data input_size the backbone and add respective transform to match data.')
    parser.add_argument('--replace_BN_by_GN_groups', type=int, default=0,
                        help='Replace BatchNorm layers by GroupNorm layers where applicable, with given maximum number of groups.')
    parser.add_argument('--replace_BN', type=str, default='default', choices=['default', 'group_norm', 'layer_norm', 'instance_norm'],
                        help='Replace BatchNorm layers by given normalization layer.')

    # Classifier
    parser.add_argument('--classifier', type=str, default='linear', 
                        choices=['linear', 
                                'linear_dynamic',
                                'norm_embed',
                                'clip',
                                'identity'
                            ],  
                        help='linear classifier (prototype=weight vector for a class)'
                            'For feature-space classifiers, we output the embedding (identity) '
                            'or normalized embedding (norm_embed)')
    parser.add_argument('--lin_bias', default=True, type=lambda x: bool(strtobool(x)),
                        help="Use bias in Linear classifier")

    # Optimization
    parser.add_argument('--optim', type=str,
                        choices=['sgd', 'adam', 'adamw',
                                 'lars'],
                        default='sgd')
    parser.add_argument('--criterion', type=str, default=None,
                        help='Loss function to use. If None, cross entropy is used.')
    parser.add_argument('--bs', type=int, default=None, 
                        help='Minibatch size.')
    parser.add_argument('--bs_eval', type=int, default=None, 
                        help='Minibatch size.')
    parser.add_argument('--epochs', nargs='+', type=int, default=None, # default=[10], 
                        help='Number of epochs per experience. If len(epochs) != n_experiences, \
                            then the last epoch is used for the remaining experiences.')
    parser.add_argument('--iterations_per_task', nargs="+", type=int, default=None,
                        help='When this is defined, it overwrites the epochs per task.'
                            'This enables equal compute per task for imbalanced scenarios.')
    parser.add_argument('--lr', nargs="+", type=float, default=[0.1], 
                        help='Learning rate (for everything).')
    parser.add_argument('--head_lr', type=float, default=None, 
                        help='Learning rate for the head.')
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help='Momentum.')
    parser.add_argument('--weight_decay', type=float, default=0.0, 
                        help='Weight decay.')
    parser.add_argument('--lr_warmup', type=str, default=None, choices=['MAS_LR', 'MAS_FIX'],
                        help='Warmup learning rate mechanism.')
    parser.add_argument('--lr_scheduling', type=str, default="none",
                        choices=["none",
                                 "onecycle_cos",
                                 "inf_const_decay"],
                        help='Use learning rate scheduling.')
    parser.add_argument('--lr_milestones', nargs="+", type=int, default=None, 
                        help='Learning rate epoch decay milestones.')
    parser.add_argument('--lr_decay', type=float, default=None, 
                        help='Multiply factor on milestones.')
    parser.add_argument('--one_cycle_div_factor', nargs="+", type=float, default=25, 
                        help='Div factor determining the initial learning rate to warm-up from for OneCycleLR.')
    parser.add_argument('--one_cycle_final_lr', nargs="+", type=float, default=None, #default=1e-6,
                        help='Final learning rate that is annealed towards for OneCycleLR.')
    parser.add_argument('--one_cycle_pct_start', nargs="+", type=float, default=None, #default=0.3,
                        help='Percentage of the cycle (in number of steps) spent increasing the learning rate.')
    parser.add_argument('--lr_warmup_pct', type=float, default=0.1,
                        help='Percentage of total epochs to warmup the learning rate when using lr_warmup.')
    parser.add_argument('--lr_decay_pct', type=float, default=0.1,
                        help='Percentage of total epochs when using "inf"-scheduling-like to start decaying the lr.')

    parser.add_argument('--grad_clipping', type=float, default=None, 
                        help='Gradient clipping value.')

    parser.add_argument('--use_data_aug', nargs='+', default=[],
                        choices=['simclr',
                                'crop_flip',
                                'auto_cifar10',
                                'auto_imgnet',
                                'rand_augment',
                                'rand_crop'],
                        help='Define one or more data augmentations. This is especially helpful with corruptions')
    parser.add_argument('--overwrite_mean', nargs='+', type=float, default=None, help='Overwrite mean_norm of dataset.')
    parser.add_argument('--overwrite_stddev', nargs='+', type=float, default=None, help='Overwrite std_norm of dataset.')

    parser.add_argument('--use_corruptions', nargs='+', type=str, default=None,
                        help='Use sequence of corruptions for data augmentation per experience.')
    parser.add_argument('--corruption_severities', nargs='+', type=int, default=None,
                        help='Severity of corruptions for data augmentation per experience.')
 
    # Training
    parser.add_argument('--train_exps', nargs='+', type=int, default=None,
                        help='Experiences to train on. If None, train on all experiences.')

    # Evaluation
    parser.add_argument('--skip_eval', action='store_true', default=False,
                        help='Skip evaluation.')
    parser.add_argument('--eval_exps', nargs='+', type=int, default=None,
                        help='Experiences to evaluate on. If None, evaluate on all experiences.')
    parser.add_argument('--eval_every_during_training', type=int, default=None,
                        help='Evaluate every n iterations during training. If None, only evaluate at the end of each experience.')

    # Continual Evaluation
    parser.add_argument('--eval_with_test_data', default=True, type=lambda x: bool(strtobool(x)),
                        help="Continual eval with the test or train stream, default True for test data of datasets.")
    parser.add_argument('--enable_continual_eval', default=False, type=lambda x: bool(strtobool(x)),
                        help='Enable evaluation each eval_periodicity iterations.')
    parser.add_argument('--eval_periodicity', type=int, default=1,
                        help='Periodicity in number of iterations for continual evaluation. (None for no continual eval)')
    parser.add_argument('--eval_task_subset_size', type=int, default=1000,
                        help='Max nb of samples per evaluation task. (-1 if not applicable)')
    
    parser.add_argument('--eval_max_iterations', type=int, default=-1, help='Max nb of iterations for continual eval.\
                        After this number of iters is reached, no more continual eval is performed. Default value \
                        of -1 means no limit.')
    parser.add_argument('--skip_initial_eval', action='store_true', default=False, help='Skip initial eval.')
    parser.add_argument('--only_initial_eval', action='store_true', default=False, help='Only perform initial eval.')
    # parser.add_argument('--filter_experiences', nargs='+', type=int, default=None,
    #                     help='Train and evaluate only on these specific experiences. If None, evaluate on all experiences.')
    parser.add_argument('--only_prepare_data', action='store_true', default=False, 
                        help='Only prepare data.')
    parser.add_argument('--data_percentage_per_exp', nargs='+', type=float, default=None,
                        help='Percentage of data to use per experience. If None, no assumption is made.')
    parser.add_argument('--per_exp_classes_dict', type=str, default=None,
                        help='Path to a yaml file that contains a dictionary mapping experience ids to number of classes.')
    parser.add_argument('--use_single_test_set', action='store_true', default=False,
                        help='Use a single test set for all experiences.')
    parser.add_argument('--terminate_after_exp', type=int, default=None, help='Terminate training after this experience.')

    # Linear Probing
    parser.add_argument('--use_lp_eval', nargs="+", type=str, default=None, 
                        choices=['none',
                                 'linear',
                                 'linear_force_mh',
                                 'concat_linear',
                                 'concat_linear_force_mh',
                                 'knn'], 
                        help='Usa a probing evaluation metric')
    parser.add_argument('--lp_buffer_dataset', default=True, type=lambda x: bool(strtobool(x)), 
                        help='Buffer the linear probing dataset in memory.')
    parser.add_argument('--lp_finetune_epochs', type=int, default=None,
                        help='Number of epochs to finetune Linear Probing.')
    parser.add_argument('--lp_optimizer', type=str, default='adamw',
                        choices=['adamw', 'sgd', 'sgd_cosine'],
                        help='Optimizer for the downstream linear probe head. '
                             '"adamw" (default) keeps existing behaviour. '
                             '"sgd" uses SGD + MultiStepLR with decay at 60%%/80%% of epochs (cassle barlow_linear.sh style). '
                             '"sgd_cosine" uses SGD + CosineAnnealingLR.')
    parser.add_argument('--lp_finetune_itrs', type=int, default=None,
                        help='Number of iterations to finetune Linear Probing. Replaces lp_finetune_epochs when used.')
    parser.add_argument('--lp_record_loss', action='store_true', default=False,
                        help='Record the loss of the linear probing.')
    parser.add_argument('--lp_eval_train_stream', action='store_true', default=False,
                        help='Evaluate linear probing on the train stream.')
    parser.add_argument('--reduce_target_dim', type=int, default=None,
                        help='Reduce the target dimension of the concatenated features to this value.')
    #parser.add_argument('--force_mh_eval', action='store_true', default=False,
    #                    help='Force multi-head probe evaluation even if the model is single-headed.')
    #parser.add_argument('--lp_optim', type=str, choices=['sgd', 'adamW'], default='adamW', help='Optimizer for linear probing.')
    #parser.add_argument('--lp_lr', type=float, default=1e-3, help='Learning rate for linear probing.')
    
    #KNN
    parser.add_argument('--knn_k', nargs='+', type=int, default=[1], 
                       help='Number of nearest neighbors for KNN probing.')

    # Additional Evaluation
    parser.add_argument('--downstream_method', nargs='+', type=str, default=None,
                        choices=[
                            "linear",
                            "concat_linear",
                            "concat_linear_pca_reduce",
                            "concat_linear_gaussian_reduce",
                            "knn", "concat_knn",
                            "concat_knn_pca_reduce", "concat_knn_gaussian_reduce",
                            "linear_ER", "knn_ER",
                            "finetune",
                            "pca", "tsne",
                            "concat_pca", "concat_pca_gaussian_reduce", "concat_pca_pca_reduce",
                            "concat_tsne", "concat_tsne_gaussian_reduce"
                        ],
                        help='Method(s) to use for down-stream tasks.')
    
    parser.add_argument('--downstream_sets', nargs='+', type=str, default=None, 
                        help='Name of scenarios to take the down-stream tasks from.')
    parser.add_argument('--downstream_rootpaths', nargs='+', type=str, default=None, 
                        help='Root path of the down-stream tasks.')
    
    parser.add_argument('--ds_corruptions', nargs='+', type=str, default=None,
                        help='Use sequence of corruptions for data augmentation per down-stream task.')
    parser.add_argument('--ds_corruption_severities', nargs='+', type=int, default=None,
                        help='Severity of corruptions for data augmentation per down-stream task.')


    # Strategy
    parser.add_argument('--strategy', type=str, default='finetune',
        choices=['finetune', 'joint',
                'LwF', 'MAS', 'EWC',
                'ER', 'BiC', 'DER', 'DER++',
                'BarlowTwins', 'BarlowTwinsLT',
            ],
        help='Continual learning strategy.')
    parser.add_argument('--task_incr', action='store_true', default=False,
                        help="Give task ids during training to single out the head to the current task.")
    parser.add_argument('--class_from_zero_in_each_exp', action='store_true', default=False,
                        help="Whether to set the class ids from zero in each experience.")
    parser.add_argument('--domain_incr', action='store_true', default=False,
                        help="Rarely ever useful, but will make certain plugins behave like task-incremental\
                        which is desirable, e.g. for replay.")
    parser.add_argument('--eval_stored_weights', type=str, default=None,
                        help="When provided, will use the stored weights from the provided path to evaluate the model.")
    parser.add_argument('--weights_replace_key_dict', type=str, choices=["cassle", "cassle_ijepa"], default=None,
                        help="Name of the dict to use for replacing keys when loading stored weights. Default: None - no altering of the keys.")
    parser.add_argument('--rotations_list', type=float, nargs='+', default=None,
                        help='List of rotations to use for RotMNIST scenario.')
    parser.add_argument('--warmup_bn', type=str, choices=["statistics", "affine"], default=None,
                        help='Warmup BatchNorm layers with only statistics or inlcuding affine parameters.')
    parser.add_argument('--warmup_bn_iterations', type=int, default=0,
                        help='Number of warmup iterations for BatchNorm layers.')
    parser.add_argument('--freeze_bn', type=int, default=-1, 
                        help='Freeze BatchNorm layers after this experience. -1 to not freeze.')
    parser.add_argument('--freeze_model', type=str, default=None, 
                        choices=['backbone', 
                                 'backbone_but_norm',  # Do not freeze the backbone's BatchNorm layers
                                 'backbone_but_affine',  # Do not freeze the backbone's affine layers (but batch-statistics)
                                 'head', 
                                 'all'],
                        help='Freeze backbone (with or without batchnorm), head or all model parameters.')
    parser.add_argument('--freeze_model_exp', type=int, default=None,
                        help='Freeze model after this experience.')
    parser.add_argument('--drop_grad', type=float, default=0.0, 
                        help='DropGrad rate for the feature extractor.')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Use Automatic Mixed Precision (AMP) for training.')
    parser.add_argument('--use_accelerator', action='store_true', default=False,
                        help='Use HuggingFace Accelerator for training.')

    
    # Pretraining of the model head
    parser.add_argument('--pretrain_head', action='store_true', default=False,
                        help='Pretrain the head before training the model.')
    

    # Replay
    parser.add_argument('--use_replay', action='store_true', default=False,
                        help='Use replay mechanism.')
    parser.add_argument('--use_cumulative_replay', action='store_true', default=False,
                        help='Use cumulative replay, i.e. keep all samples from previous experiences in the replay memory.')
    parser.add_argument('--mem_size', default=1000, type=int, help='Total nb of samples in rehearsal memory.')
    parser.add_argument('--replay_storage_policy', type=str, default=None, 
                        choices=['class_balanced', 
                                 'experience_balanced',
                                 'class_task_balanced'
                        ],
                        help='Storage policy for the replay memory.')
    parser.add_argument('--replay_batch_handling', choices=['separate', 'combined'], default='separate',
                        help='How to handle the replay batch. Separate means that the replay batch is handled \
                            separately from the current batch. Combined means that the replay batch is \
                            combined with the current batch.')
    parser.add_argument('--replay_loss_weight', nargs="+", type=float, default=None,
                        help='Weight for the replay loss. If None, the weight is set to 1.')
    
    # Ensemble
    parser.add_argument('--use_ensemble', action='store_true', default=False,
                        help='Use ensemble of models. Requirement for concat probes.')

    # BiC
    parser.add_argument('--task_balanced_dataloader', action='store_true', default=False, 
                        help='Use task balanced dataloader for BiC method.')
    parser.add_argument('--val_percentage', type=float, default=0.1, 
                        help='Percentage of validation set for BiC method.')
    parser.add_argument('--second_stage_eps', type=int, default=10, 
                        help='Number of epochs for the second stage for BiC method.')
    parser.add_argument('--second_stage_lr', type=float, default=0.1, 
                        help='Learning rate for the second stage for BiC method.')

    # LWF
    parser.add_argument('--lwf', action='store_true', default=False, help='Use LwF strategy.')
    parser.add_argument('--lwf_alpha', type=float, default=1, help='Distillation loss weight')
    parser.add_argument('--lwf_softmax_t', type=float, default=2, help='Softmax temperature (division).')
    parser.add_argument('--add_feature_distill', action='store_true', default=False,
                        help='Add feature distillation loss to strategy.')

    # MAS
    parser.add_argument('--mas', action='store_true', default=False, help='Use MAS strategy.')

    # EWC
    parser.add_argument('--iw_strength', type=float, default=1, help="IW regularization strength.")

    # Self-supervised
    # Utility
    parser.add_argument('--use_two_view_transform', action='store_true', default=False,
                        help='Use two view transform for self-supervised methods.')
    parser.add_argument('--projection_sizes', nargs="+", type=int, default=None,
                        help='Size of the projector head for self-supervised methods.\
                              For example: [512,2048,2048,2048]')
    parser.add_argument('--predictor_sizes', nargs='+', type=int, default=None,
                        help='Size of the predictor head for self-supervised methods.\
                              For example: [512,128,512]')
    # BarlowTwins
    parser.add_argument('--barlow_twin', action='store_true', default=False, 
                        help='Use BarlowTwins.')
    parser.add_argument('--lambda_coeff', type=float, default=5e-4, # 3.9e-3
                        help='Lambda coefficient for BarlowTwins.')
    parser.add_argument('--barlowtwins_lt', action='store_true', default=False,
                        help='Use BarlowTwins similar to lightly lib.')


    # Store model every experience
    parser.add_argument('--store_models', default=True, type=lambda x: bool(strtobool(x)),
                        help='Store model after each experience.')
    parser.add_argument('--store_models_every_epoch', type=int, default=None,
                        help='If defined, store model every n epochs.')
    
    # Reinit model after experience
    parser.add_argument('--reinit_model_exp', type=int, default=-1,
                        help='Reinitialize model after this experience')
    parser.add_argument('--reinit_deterministic', action='store_true', default=False,
                        help='Reinitialize model deterministically.')
    
    return parser


def overwrite_args_with_config(args):
    """
    Directly overwrite the input args with values defined in config yaml file.
    Only if args.config_path is defined.
    """
    if args.config_path is None:
        return
    
    config_base = Path(args.config_base)
    arg_configs = {}
    for config_path in args.config_path:
        config_path = config_base / config_path  # implicit conversion to path
        if not config_path.suffix:
            config_path = config_path.with_suffix(".yaml")
        print("config_path", config_path)
        assert config_path.exists() and config_path.is_file(), f"Config file does not exist: {config_path}"
        with open(config_path, 'r') as stream:
            content = yaml.safe_load(stream)
            assert content is not None, f"Config file is empty: {config_path}"
            arg_configs.update(content)

    for arg_name, arg_val in arg_configs.items():  # Overwrite
        assert hasattr(args, arg_name), \
            f"'{arg_name}' defined in config is not specified in args, config: {args.config_path}"
        setattr(args, arg_name, arg_val)
    print(f"Overriden args with config: {args.config_path}")



def parse_args(parser):
    args = parser.parse_args()

    # Get the default values of the parser
    defaults = {action.dest: action.default for action in parser._actions if action.default != argparse.SUPPRESS}
    # Get the values of the user
    user_values = {key: getattr(args, key) for key in defaults}
    # Get the modified values
    modified_args = {key: user_values[key] for key in defaults if user_values[key] != defaults[key]}
            
    # Override args from yaml file
    overwrite_args_with_config(args)

    print(modified_args)
    # Reset user modified values - user values trump yaml values
    for key, value in modified_args.items():
        print(f"overwriting {key}: {value}")
        setattr(args, key, value)

    if args.per_exp_classes_dict is not None:
        import json
        args.per_exp_classes_dict = json.loads(args.per_exp_classes_dict)
        args.per_exp_classes_dict  = {int(k): v for k, v in args.per_exp_classes_dict.items()}

    if args.use_max_workers:
        args.num_workers = int(os.environ.get("PYTHON_CPU_COUNT", args.num_workers))
    return args