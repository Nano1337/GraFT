import time
import os
import wandb
import optuna
from pathlib import Path

import torch 

import lr_schedulers
from train.base_trainer import Base_Trainer
from metrics.R1_mAP import R1_mAP

# FIXME: this code doesn't work and may be outdated. Check with trainer_rgbn_triplet_verb


def parse_string_list(string_list):
    """
    Parses a list of string identifiers to extract person IDs and camera IDs.
    
    :param string_list: List of strings where each string is expected to be an identifier containing a person ID and a camera ID.
    :return: Two lists: list of person IDs and list of camera IDs, extracted from the input strings.
    """

    ids = []
    camera_ids = []
    
    for string in string_list:
        parts = string.split("_")
        if len(parts) >= 3:
            try:
                id_num = int(parts[0])
                camera_id = int(parts[1][1:])
                ids.append(id_num)
                camera_ids.append(camera_id)
            except ValueError:
                # Skip invalid strings that cannot be parsed
                pass
    
    return ids, camera_ids

class Trainer_Visualize_Embedding_RNT(Base_Trainer): 
    """
    Trainer class for RGBN Triplet Verb model. This class inherits from the Base_Trainer class and
    provides functionalities for training and validating the RGBN Triplet Verb model.

    The trainer includes methods for training a single epoch, running validation, saving and loading models.
    """

    def __init__(self, cfgs, fabric, model, train_loader, val_loader, optimizer, criterion, unique_dir_name, trial=None):
        """
        Initializes the trainer with provided configurations, model, data loaders, optimizer, and loss function.

        :param cfgs: Configurations for the training process.
        :param fabric: The fabric object for handling distributed training.
        :param model: The model to train.
        :param train_loader: DataLoader for the training dataset.
        :param val_loader: DataLoader for the validation dataset.
        :param optimizer: The optimizer to use for training.
        :param criterion: The loss function to use for training.
        :param unique_dir_name: Unique directory name for saving checkpoints.
        :param trial: Optuna trial object for hyperparameter tuning. Default is None.
        """
        
        super().__init__(cfgs, fabric, model, train_loader, val_loader, optimizer, criterion)
        if cfgs.lr_scheduler_name: # Note: must add lr_scheduler to optimizer before wrapping with Fabric
            self.lr_scheduler = lr_schedulers.get_lr_scheduler(cfgs, self.optimizer, self.train_loader)
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
    
        self.accumulated_loss = []

        # load checkpoint if exists
        if not os.path.isdir(cfgs.ckpt_dir): 
            raise ValueError("Checkpoint directory does not exist")
        if cfgs.ckpt_full_path and os.path.isfile(cfgs.ckpt_full_path):
            self.load_networks(cfgs.ckpt_full_path)
        else:
            raise ValueError("ckpt_full_path Checkpoint file does not exist")
        
        # initialize validation metrics
        query_path = os.path.join(str(cfgs.dataroot), str(cfgs.dataset)) + "/rgbir/query"
        self.num_queries = len(os.listdir(query_path))
        self.metric = R1_mAP(self.fabric, self.num_queries, self.cfgs.max_rank)
        
        # initialize optuna trial if exists
        self.trial = trial if cfgs.use_optuna else None


    def train(self):
        """
        Trains the model for the specified number of epochs.

        :return: Highest validation accuracy.
        """

        validation_loss = self.validate()

        
        if self.fabric.is_global_zero: 
            print(f"Run completed. Final mAP: {validation_loss}")

            if self.cfgs.use_wandb:
                wandb.run.summary["validation_loss"] = validation_loss
                wandb.run.summary["state"] = "completed"
                wandb.finish(quiet=True)

        return validation_loss


    def validate(self): 
        """
        Validates the model.

        :return: Mean Average Precision (mAP) for this validation round.
        """

        start_time = time.time()  # start the timer

        # set model to evaluation mode
        self.model.eval()
        self.metric.reset()
        
        with torch.no_grad():

            
            for batch_input, batch_target in self.val_loader:

                pid, camid = parse_string_list(batch_target)

                inputs = {}
                for m, modality in enumerate(self.cfgs.model_modalities):
                    inputs[modality] = []

                    for b in range(len(batch_input)):
                        inputs[modality].append(batch_input[b][m])
        
                    inputs[modality] = self.fabric.to_device(torch.stack(inputs[modality], dim=0))

                # Forward pass
                embeddings = self.model(inputs, train_mode=False)

                # store data for val metric calc later
                self.metric.update(embeddings["z_reparamed_anchor"], pid, camid)

            # after all batches are processed, call metrics again to get final results
            cmc, mAP = self.metric.compute()

            # visualize embeddings
            if self.fabric.is_global_zero: 
                self.metric.compute_umap_plotly(save_path=self.cfgs.vis_save_path, dims=self.cfgs.vis_dim, reduction_method=self.cfgs.vis_reduction_method)

                # log to wandb 
                if self.cfgs.use_wandb:
                    wandb.log({"val/mAP": mAP, "val/rank1": cmc[0], "val/rank5": cmc[4], "val/rank10": cmc[9]})

                # log to terminal
                print("Validation Results")
                print("mAP: {:.4%}".format(mAP))
                print("CMC curve")
                for r in [1, 5, 10]:
                    print("Rank-{:<3}: {:.4%}".format(r, cmc[r - 1]))

                end_time = time.time()  # end the timer
                print('Validation Time: ', round(end_time - start_time, 2), 'sec')  # print the time taken for validation

                # log to optuna
                if self.cfgs.use_optuna:
                    self.trial.report(mAP, self.epoch)
                    if self.trial.should_prune():
                        print("Trial was pruned at epoch {}".format(self.epoch))
                        wandb.run.summary["state"] = "pruned"
                        wandb.finish(quiet=True)
                        raise optuna.exceptions.TrialPruned()
                    
        return mAP

    def load_networks(self, load_path): 
        """
        Loads the model, optimizer, and learning rate scheduler (if exists) from a specified file.

        Args:
            load_path: A string representing the path of the file from which the model state is to be loaded.

        Returns:
            None. This function operates by loading the state of the model, optimizer, and lr_scheduler from the disk to the Trainer_1 object.
        """
        
        state = {
            "mm_model": self.model,
            "optimizer": self.optimizer,
            "loss": self.accumulated_loss,
        }

        try:
            # load checkpoint to model in fabric in-place (model already setup)
            print("Loading in ckpt weights...")
            remainder = self.fabric.load(load_path, state)
        except Exception as e:
            print(f"Error loading network state: {e}")
            return

        try:
            if remainder['lr_scheduler'] is not None:
                self.lr_scheduler.load_state_dict(remainder['lr_scheduler'])
        except Exception as e:
            print(f"Error loading lr_scheduler: {e}")

        try:
            self.loaded_epoch = remainder['epoch']
        except Exception as e:
            print(f"Error loading loaded_epoch: {e}")


    def save_networks(save_dir, save_name):
        pass
    
    def train_epoch(epoch, end_epoch, total_iters, epoch_iter, iter_data_time, optimize_time):
        pass
