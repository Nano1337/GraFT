import time
import os
import wandb
import optuna
from pathlib import Path

import torch 

import lr_schedulers
from train.base_trainer import Base_Trainer
from metrics.R1_mAP import R1_mAP


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

class Trainer_RGBN_Triplet_Verb(Base_Trainer): 
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
        self.unique_dir_name = unique_dir_name
    
        self.accumulated_loss = []

        # load checkpoint if exists
        if not os.path.isdir(cfgs.ckpt_dir): 
            raise ValueError("Checkpoint directory does not exist")
        if cfgs.ckpt_full_path and os.path.isfile(cfgs.ckpt_full_path):
            self.load_networks(cfgs.ckpt_full_path)
        
        # Create checkpoint directory
        save_dir = Path(cfgs.ckpt_dir, unique_dir_name)
        save_dir.mkdir(parents=True, exist_ok=True)

        # initialize validation metrics
        query_path = os.path.join(str(cfgs.dataroot), str(cfgs.dataset)) + "/rgbir/query"
        self.num_queries = len(os.listdir(query_path))
        self.metric = R1_mAP(self.fabric, self.num_queries, self.cfgs.max_rank)
        
        # initialize optuna trial if exists
        self.trial = trial if cfgs.use_optuna else None

    def train_epoch(self, epoch, end_epoch, total_iters, epoch_iter, iter_data_time, optimize_time): 
        """
        Trains the model for one epoch.

        :param epoch: Current epoch number.
        :param end_epoch: Total number of epochs.
        :param total_iters: Total number of iterations.
        :param epoch_iter: Current epoch iteration.
        :param iter_data_time: Time taken for data loading in each iteration.
        :param optimize_time: Time taken for optimization in each iteration.
        """        
        
        # Initialize the running accuracy
        running_corrects = 0.0
        total_samples = 0

        for i, (batch_input, batch_target) in enumerate(self.train_loader):
            iter_start_time = time.time() # start timer for computation each iteration

            total_iters += self.cfgs.batch_size
            epoch_iter += self.cfgs.batch_size

            self.fabric.barrier() # synchronize all processes 

            # Retrieve input images
            if self.cfgs.non_generalizable_inputs:
                batched_rgb = []
                batched_ir = []
                for b in range(len(batch_input)):
                    batched_rgb.append(torch.stack(batch_input[b][0], dim=0))
                    batched_ir.append(torch.stack(batch_input[b][1], dim=0))
                batched_rgb = self.fabric.to_device(torch.stack(batched_rgb, dim=0))
                batched_ir = self.fabric.to_device(torch.stack(batched_ir, dim=0))

                output_class, embeddings = self.model(batched_rgb, batched_ir)

            else: 
                # NEW VERSION
                inputs = {}
                for m, modality in enumerate(self.cfgs.model_modalities):
                    inputs[modality] = []

                    for b in range(len(batch_input)):
                        inputs[modality].append(torch.stack(batch_input[b][m], dim=0))
        
                    inputs[modality] = self.fabric.to_device(torch.stack(inputs[modality], dim=0))

                # Forward pass
                output_class, embeddings = self.model(inputs)

            # Convert one-hot encoded output to class indices
            preds = torch.argmax(output_class, dim=-1)
            targets = torch.argmax(batch_target, dim=-1)

            # Access output embeddings
            anchor, pos, neg = embeddings["z_reparamed_anchor"], embeddings["z_reparamed_positive"], embeddings["z_reparamed_negative"]

            # Compare predictions to the truth
            running_corrects += torch.sum(preds == targets.data).item()
            total_samples += targets.size(0)

            loss = self.criterion(anchor, pos, neg, output_class, targets)
            self.accumulated_loss.append(loss.item())

            # Backward pass & step
            self.fabric.backward(loss)

            # # FIXME: remove this after debugging
            # for name, parameter in self.model.named_parameters():
            #     if parameter.grad is None:
            #         print(f"No gradient for {name}")

            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.fabric.barrier() # synchronize all processes
            optimize_time = (time.time() - iter_start_time) / self.cfgs.batch_size * 0.005 + optimize_time * 0.995

            if self.fabric.is_global_zero: 
                if i % self.cfgs.print_freq == 0:
                    # Calculate training accuracy for this batch
                    batch_accuracy = running_corrects / total_samples

                    if self.cfgs.use_wandb: 
                        current_lr = self.lr_scheduler.get_last_lr()[0] if self.cfgs.lr_scheduler_name else None
                        wandb.log({"train/loss": loss.item(), "train/acc": batch_accuracy, "train/lr": current_lr, "train/epoch": epoch})

                    self.fabric.print(
                        f"[{epoch}/{end_epoch}][{i}/{len(self.train_loader)}]\t"
                        f"Loss: {loss.item():.4f}\t"
                        f"Train Accuracy: {batch_accuracy:.4f}\t"
                        f"Time: {optimize_time:.4f}\t"
                        f"Data: {(iter_start_time - iter_data_time):.4f}\t"
                    )

            iter_data_time = time.time() # end timer for data loading per iteration

    def train(self):
        """
        Trains the model for the specified number of epochs.

        :return: Highest validation accuracy.
        """

        total_iters = 0 # total number of training iterations 
        optimize_time = 0.1 # set initial value to something small to avoid divide by zero error

        end_epoch = self.loaded_epoch + self.cfgs.num_epochs
        for epoch in range(self.loaded_epoch, end_epoch + 1): 
            self.epoch = epoch

            # set model to training mode
            self.model.train()
            
            epoch_start_time = time.time() # start time for epoch
            iter_data_time = time.time() # start timer for data loading per iteration
            epoch_iter = 0 # start epoch iteration counter, will be reset to 0 at end of epoch

            self.train_epoch(epoch, end_epoch, total_iters, epoch_iter, iter_data_time, optimize_time)
            
            if self.fabric.is_global_zero: 
                print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, end_epoch, time.time() - epoch_start_time))

            # run validation
            epoch_valid_acc = self.validate()

            # save checkpoint only if global mAP is higher than previous best
            if self.fabric.is_global_zero:
            
                if epoch_valid_acc > self.highest_val_acc:
                    print("Saving checkpoint at epoch {}".format(epoch))

                    self.save_networks(os.path.join(self.cfgs.ckpt_dir, self.unique_dir_name), 'best.pth', epoch)

                    self.highest_val_acc = epoch_valid_acc

        if self.fabric.is_global_zero: 
            print("Training completed. Highest validation acc: {}".format(self.highest_val_acc))
            wandb.run.summary["highest_val_acc"] = self.highest_val_acc
            wandb.run.summary["state"] = "completed"
            wandb.finish(quiet=True)

        return self.highest_val_acc


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

                if self.cfgs.non_generalizable_inputs:
                    batched_rgb = []
                    batched_ir = []
                    for b in range(len(batch_input)):
                        batched_rgb.append(batch_input[b][0])
                        batched_ir.append(batch_input[b][1])
                    batched_rgb = self.fabric.to_device(torch.stack(batched_rgb, dim=0))
                    batched_ir = self.fabric.to_device(torch.stack(batched_ir, dim=0))

                    output_class, embeddings = self.model(batched_rgb, batched_ir, train_mode=False)

                    # Forward pass
                    embeddings = self.model(batched_rgb, batched_ir, train_mode=False)

                else: 
                    # NEW VERSION
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

            if self.fabric.is_global_zero: 

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

    def save_networks(self, save_dir, save_name, epoch): 
        """
        Saves the model state.

        :param save_dir: Directory to save the model state.
        :param name: Name of the checkpoint file.
        :param epoch: The epoch number.
        """
        state = {
            "epoch": epoch,
            "mm_model": self.model,
            "optimizer": self.optimizer,
            "loss": self.accumulated_loss,
            "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }
        self.fabric.save(os.path.join(save_dir, save_name), state)

    def load_networks(self, load_path): 
        """
        Loads the model state from a checkpoint file.

        :param load_path: Path to the checkpoint file.
        """
        state = {
            "mm_model": self.model,
            "optimizer": self.optimizer,
            "loss": self.accumulated_loss,
        }

        try:
            # load checkpoint to model in fabric in-place (model already setup)
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
