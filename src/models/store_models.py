import torch

from pathlib import Path

from avalanche.training.plugins.strategy_plugin import SupervisedPlugin


class StoreModelsPlugin(SupervisedPlugin):
    def __init__(
            self, 
            model_name, 
            model_store_path,
            store_on_intermediate_epoch=None
        ):
        super().__init__()

        self.model_name = model_name
        self.model_store_path = model_store_path
        self.store_on_intermediate_epoch = store_on_intermediate_epoch
        return

    def store_model(self, strategy, epoch=-1):
        # Store model to path
        dir_path = str(self.model_store_path) + "/model_weights/"
        file_name =  self.model_name + "_" + str(strategy.clock.train_exp_counter)
        if epoch > 0:
            file_name += "_e" + str(epoch)
        file_name += ".pth"
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Reference to model
        model = strategy.model
        # Extra code for when using ddp
        if hasattr(strategy, "accelerator"):
            try:
                # model.feature_extractor = strategy.accelerator.unwrap_model(strategy.model.feature_extractor)
                # model.train_classifier = strategy.accelerator.unwrap_model(strategy.model.train_classifier)
                # Overwrite "model" with reference to unwrapped model
                model = strategy.accelerator.unwrap_model(strategy.model)
                print("Sucessfully unwrapped model for storing.")
            except:
                pass

        if hasattr(model, "_orig_mod"):  # NOTE: for using compiled models
            torch.save(model._orig_mod.state_dict(), dir_path+file_name)
        else:
            torch.save(model.state_dict(), dir_path+file_name)
        print("\nStoring model to path: ", (dir_path+file_name))
        return


    def after_training_epoch(self, strategy, **kwargs):
        # Extra code for when using ddp
        if hasattr(strategy, "accelerator"):
            if not strategy.accelerator.is_main_process:
                strategy.accelerator.wait_for_everyone()
                return

        if not self.store_on_intermediate_epoch is None:
            if (strategy.clock.train_exp_epochs % self.store_on_intermediate_epoch) == 0:
               print("\nStoring model on intermediate epoch: ", strategy.clock.train_exp_epochs)
               self.store_model(strategy, epoch=strategy.clock.train_exp_epochs)

        if hasattr(strategy, "accelerator"):
            if strategy.accelerator.is_main_process:
                strategy.accelerator.wait_for_everyone()


    def after_training_exp(self, strategy, **kwargs):
        print("DEBUG: StoreModelsPlugin after_training_exp called!")
        # Extra code for when using ddp
        if hasattr(strategy, "accelerator"):
            if not strategy.accelerator.is_main_process:
                strategy.accelerator.wait_for_everyone()
                return
        
        print("DEBUG: Storeing model at the end of training experience.")
        self.store_model(strategy)

        print("DEBUG: StoreModelsPlugin after_training_exp finished storing model.")
        if hasattr(strategy, "accelerator"):
            if strategy.accelerator.is_main_process:
                print("DEBUG: Main Process Waiting for everyone...")
                strategy.accelerator.wait_for_everyone()
        return 

    # def after_eval(self, strategy, **kwargs):
    #     self.store_model(strategy)
    #     return



# class StoreCheckpointsPlugin(SupervisedPlugin):
#     def __init__(
#             self, 
#             model_name, 
#             model_store_path,
#             store_on_intermediate_epoch=1,
#             overwrite=True
#         ):
#         super().__init__()

#         self.model_name = model_name
#         self.model_store_path = model_store_path
#         self.store_on_intermediate_epoch = store_on_intermediate_epoch
#         self.overwrite = overwrite
#         return

#     def store_checkpoint(self, strategy, epoch=-1):
#         checkpoint = {
#             "epoch": epoch,
#             "model_state_dict": strategy.model.state_dict(),
#             "optimizer_state_dict": strategy.optimizer.state_dict(),
#             "scheduler_state_dict": strategy.scheduler.state_dict() if strategy.scheduler is not None else None,
#             "scaler_state_dict": strategy.scaler.state_dict() if strategy.scaler is not None else None,
#         }
       
#         # Store checkpoint to path
#         dir_path = str(self.model_store_path) + "/checkpoint/"
#         file_name =  self.model_name + "_" + str(strategy.clock.train_exp_counter)
#         if epoch > 0:
#             file_name += "_e" + str(epoch)
#         file_name += ".pth"
#         Path(dir_path).mkdir(parents=True, exist_ok=True)
#         torch.save(checkpoint, dir_path+file_name)
#         print(f"Checkpoint saved at {dir_path+file_name}")
#         return

#     def after_training_epoch(self, strategy, **kwargs):
#         if self.store_on_intermediate_epoch > 0:
#             if (strategy.clock.train_exp_epochs % self.store_on_intermediate_epoch) == 0:
#                print("\nStoring model on intermediate epoch: ", strategy.clock.train_exp_epochs)
#                self.store_model(strategy, epoch=strategy.clock.train_exp_epochs)

