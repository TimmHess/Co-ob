from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


print("Loading libs...")
import traceback

import torch
from accelerate import Accelerator

from pathlib import Path
import random
import numpy
import copy
from copy import deepcopy

from datetime import datetime
from distutils.util import strtobool


import avalanche as avl
from avalanche.logging import TextLogger, TensorboardLogger
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin

from torchsummary import summary

# Custom
import helper
from src.models import get_model
from src.utils.util import safe_index
from cmd_parser import get_arg_parser, parse_args

from src.utils import data_utils 
print("loading libs done..")


def main():
    '''
    Patch Avalanche Functions
    '''
    from src.auxilliary.avalanche_patches import (
            patch_drop_last_dataloader,
            patch_feature_extractor_model,
            patch_avalanche_dataset_get_transforms,
    )
    print("")
    patch_drop_last_dataloader()
    patch_feature_extractor_model()
    patch_avalanche_dataset_get_transforms()
    print("Avalanche Patches done.")
    print("")


    '''
    Load CMD args
    '''
    # Capture the command string
    import sys
    command_string = 'python ' + ' '.join(sys.argv)
    # Parser the arguments
    args = parse_args(get_arg_parser())
    # Store the command string as argument
    args.cmd_string = command_string
    print("Command string:")
    print(args.cmd_string)


    '''
    Init device
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    accelerator = None
    if args.use_accelerator:
        accelerator = Accelerator()


    """
    Setup num threads
    """
    if args.restrict_num_threads:
        torch.set_num_threads(args.restrict_num_threads)
    if args.restrict_num_interop_threads:
        torch.set_num_interop_threads(args.restrict_num_interop_threads)


    """
    Setups seeds for reproducibility
    """
    def set_seed(seed, deterministic=False):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        numpy.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    if args.seed is None:
        args.seed = random.randint(0, 100000)
        print("No seed specified - using random seed:", args.seed)
    # Set the seed
    set_seed(args.seed, deterministic=True)


    '''
    Setup paths and results directory
    '''
    args.setupname = '_'.join([
        args.exp_name, 
        f"sd{args.seed}", 
        args.strategy, 
        args.backbone, 
        args.scenario, 
        args.optim, 
        f"e={str(args.epochs[0])}"
    ]) 

    # Initialize experiment workspace
    workspace = data_utils.ExperimentWorkspace(
        run_name=args.setupname,
        save_base_path=Path(args.save_path),
        use_condor=args.copy_results_on_end,
        overwrite=False #args.overwrite_results
    )

    args.results_path = workspace.working_dir
    args.eval_results_dir = workspace.eval_dir

    # Realize the working directory
    workspace_ok = workspace.setup()
    if not workspace_ok:
        print("Run already completed - nothing to do...")
        import sys; sys.exit()

    print("\nDEBUG: results_path", args.results_path)
    print("DEBUG: eval_results_dir", args.eval_results_dir)
    print("")

    if args.use_accelerator:
        accelerator.wait_for_everyone()

    '''
    Create Scenario
    '''
    scenario, data_transforms, input_size, initial_out_features = helper\
        .get_scenario(
            args=args, 
            scenario_name=args.scenario, 
            dset_rootpath=args.dset_rootpath, 
            num_experiences=args.num_experiences,
            use_data_aug=args.use_data_aug,
            seed=args.seed
        )

    '''
    Option to only load data
    '''
    if args.only_prepare_data:
        print("Data preparation only - exiting.")
        return


    '''
    Create Logger
    '''
    loggers = []
    def create_loggers():
        def args_to_tensorboard(writer, args):
            txt = ""
            for arg in sorted(vars(args)):
                txt += arg + ": " + str(getattr(args, arg)) + "<br/>"
            writer.add_text('command_line_parameters', txt, 0)
            return

        # Tensorboard
        tb_log_dir = os.path.join(args.results_path)
        tb_logger = TensorboardLogger(tb_log_dir=tb_log_dir)
        loggers.append(tb_logger)
        print(f"[Tensorboard] tb_log_dir={tb_log_dir}")
        args_to_tensorboard(tb_logger.writer, args)

        # Terminal
        print_logger = TextLogger() 
        if args.disable_pbar:
            print_logger = InteractiveLogger()  # print to stdout
        loggers.append(print_logger)
        return

    if args.use_accelerator:
        if accelerator.is_main_process:
            create_loggers()
        accelerator.wait_for_everyone()
    else:
        create_loggers()


    '''
    Init Evaluation
    '''
    metrics = helper.get_metrics(scenario, args, data_transforms=deepcopy(data_transforms))
    eval_plugin = EvaluationPlugin(*metrics, loggers=loggers)
    # If only prepareing data for later runs -> exit # NOTE: this is necessary to run multiple jobs on same GPU in parallel

    '''
    Init Model
    '''
    model = get_model(
        args=args, 
        n_classes=scenario.n_classes, 
        input_size=input_size, 
        initial_out_features=initial_out_features,
        backbone_weights=args.backbone_weights,
        model_weights=args.pretrained_weights,
    )
    if args.use_accelerator:
        print("\nUsing Accelerator - model will be moved to device by Accelerator")
    else:
        model = model.to(device)
    print(model)

    if args.force_compile_model:
        #torch.set_float32_matmul_precision('high')
        model = torch.compile(model)


    '''
    Init Strategy
    '''
    strategy = helper.get_strategy(
        args, 
        model, 
        eval_plugin, 
        scenario, 
        device, 
        plugins=[], 
        data_transforms=data_transforms
    )
    # If strategy has accelerator attribute
    if hasattr(strategy, 'accelerator'): 
        print("Strategy is using Accelerator")
        strategy.set_accelerator(accelerator)

    '''
    Train Loop
    '''
    try:
        print('Starting experiment...')
        if args.train_exps is None:
            args.train_exps = list(range(len(scenario.train_stream)))
        else:
            print("WARNING: Using only specified training experiences:", args.train_exps)
            
        for experience in [scenario.train_stream[i] for i in args.train_exps]:
        #for i, experience in enumerate(scenario.train_stream):
            # TRAIN
            print(f"\n{'-' * 40} TRAIN {'-' * 40}")
            print(f"Start training on experience {experience.current_experience}")
            print("exp_len:", len(experience.dataset))
            # Call train on the strategy
            strategy.train(
                experience, 
                num_workers=args.num_workers, 
                eval_streams=None
            )
            print(f"End training on experience {experience.current_experience}")

            # EVAL ALL TASKS (ON TASK TRANSITION)
            print(f"\n{'=' * 40} EVAL {'=' * 40}")
            print(f'Standard Continual Learning eval on entire test set on task transition.')
            task_results_file = args.eval_results_dir / f'seed={args.seed}' / f'task{experience.current_experience}_results.pt'
            task_results_file.parent.mkdir(parents=True, exist_ok=True)
            
            
            # Check whether eval is run
            do_eval = not args.skip_eval
            if hasattr(strategy, 'skip_eval'):
                if strategy.skip_eval:
                    print("Skipping eval for this experience")
                    do_eval = False

            # EVAL
            if do_eval:
                # Gathered by EvalLogger
                if args.eval_exps is not None:
                    res = strategy.eval([scenario.test_stream[i] for i in args.eval_exps])
                else:
                    res = strategy.eval(scenario.test_stream)

                # Store eval task results
                task_metrics = dict(strategy.evaluator.all_metric_results)
                torch.save(task_metrics, task_results_file)
                print(f"[FILE:TASK-RESULTS]: {task_results_file}")

            # Reset optimizer # NOTE: this is note needed because avalanche does this for us
            #if args.reset_optim_each_exp:
            #if True:
            exp_idx = strategy.clock.train_exp_counter
            lr = safe_index(args.lr, exp_idx)
            head_lr = args.head_lr  #safe_index(args.head_lr, exp_idx)
            strategy.optimizer = helper.get_optimizer(
                args.optim, 
                model, 
                lr,
                weight_decay=args.weight_decay, 
                momentum=args.momentum,
                head_lr=head_lr
            )
            print("\nRESET OPTIMIZER")
            if args.terminate_after_exp:
                if strategy.clock.train_exp_counter >= args.terminate_after_exp:
                    print("\n\nTERMINATING TRAINING AFTER EXPERIENCE", strategy.clock.train_exp_counter)
                    break
        
        # Extra code for when using ddp - kill all non-main processes
        if hasattr(strategy, "accelerator"):
            if not strategy.accelerator.is_main_process:
                strategy.accelerator.wait_for_everyone()
                strategy.accelerator.end_training()
        

        workspace.cleanup()
        if hasattr(strategy, "accelerator"):
            if strategy.accelerator.is_main_process:
                print("Main Process Waiting for everyone...")
                strategy.accelerator.wait_for_everyone()
                strategy.accelerator.end_training()

    except BaseException as e:
        print("Exception occured:", e)
        traceback.print_exc()
        try:
            print("\nDeleting results directory")
            # Delete the results directory
            if args.do_self_destruct:
                import shutil
                shutil.rmtree(args.results_path)
            import sys;sys.exit()
        except Exception as e2:
            import sys; sys.exit()

    except Exception as e:
        print("Exception occured:", e)
        traceback.print_exc()
        print("\nDeleting results directory")
        # Delete the results directory
        if args.do_self_destruct:
            import shutil
            shutil.rmtree(args.results_path)
        raise e



if __name__ == "__main__":
    main()