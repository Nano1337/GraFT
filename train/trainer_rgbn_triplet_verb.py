from typing import Any, Dict, List, Optional, Type, Tuple
from pathlib import Path
import os
import time

import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.nn import Module
import wandb
import optuna

import lr_schedulers
from train.base_trainer import Base_Trainer
from metrics.R1_mAP import R1_mAP


def parse_string_list(string_list: List[str]) -> Tuple[List[int], List[int]]:
    """Parses a list of string identifiers to extract person IDs and camera IDs

    Args:
        string_list: List of strings where each string is expected to be an
                        identifier containing a person ID and a camera ID.

    Returns:
        Two lists: list of person IDs and list of camera IDs, extracted from 
                    the input strings.
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


class TrainerRGBNTripletVerb(Base_Trainer):
    """Trainer class for RGBN Triplet Verb model.

    This class inherits from the BaseTrainer class and provides functionalities
    for training and validating the RGBN Triplet Verb model. The trainer
    includes methods for training a single epoch, running validation, saving
    and loading models.
    """

    def __init__(self,
                 cfgs: Dict,
                 fabric: Any,
                 model: Type[Module],
                 train_loader: Type[DataLoader],
                 val_loader: Type[DataLoader],
                 optimizer: Type[optim.Optimizer],
                 criterion: Type[Module],
                 unique_dir_name: str,
                 trial: Optional[Any] = None) -> None:
        """Initializes the trainer with provided configurations, model,
            data loaders, optimizer, and loss function.

        Args:
            cfgs: Configurations for the training process.
            fabric: The fabric object for handling distributed training.
            model: The model to train.
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            optimizer: The optimizer to use for training.
            criterion: The loss function to use for training.
            unique_dir_name: Unique directory name for saving checkpoints.
            trial: Optuna trial object for hyperparameter tuning.

        Raises:
            ValueError: If checkpoint directory does not exist.
        """

        super().__init__(cfgs, fabric, model, train_loader, val_loader,
                         optimizer, criterion)

        if cfgs.lr_scheduler_name:
            self.lr_scheduler = lr_schedulers.get_lr_scheduler(
                cfgs, self.optimizer, self.train_loader)

        self.model, self.optimizer = self.fabric.setup(
            self.model, self.optimizer
            )
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
        query_path = os.path.join(str(cfgs.dataroot), str(
            cfgs.dataset)) + "/rgbir/query"
        self.num_queries = len(os.listdir(query_path))
        self.metric = R1_mAP(self.fabric, self.num_queries, self.cfgs.max_rank)

        # initialize optuna trial if exists
        self.trial = trial if cfgs.use_optuna else None

    def train_epoch(self, epoch: int, end_epoch: int, total_iters: int,
                    epoch_iter: int, iter_data_time: float,
                    optimize_time: float) -> None:
        """Trains the model for one epoch.

        Args:
            epoch: Current epoch number.
            end_epoch: Total number of epochs.
            total_iters: Total number of iterations.
            epoch_iter: Current epoch iteration.
            iter_data_time: Time taken for data loading in each iteration.
            optimize_time: Time taken for optimization in each iteration.
        """

        # Initialize the running accuracy
        running_corrects = 0.0
        total_samples = 0

        for i, (batch_input, batch_target) in enumerate(self.train_loader):
            iter_start_time = time.time()

            total_iters += self.cfgs.batch_size
            epoch_iter += self.cfgs.batch_size
            self.fabric.barrier()

            # Retrieve input images
            if self.cfgs.non_generalizable_inputs:
                batched_rgb = []
                batched_ir = []
                for b in range(len(batch_input)):
                    batched_rgb.append(torch.stack(batch_input[b][0], dim=0))
                    batched_ir.append(torch.stack(batch_input[b][1], dim=0))
                batched_rgb = self.fabric.to_device(
                    torch.stack(batched_rgb, dim=0))
                batched_ir = self.fabric.to_device(
                    torch.stack(batched_ir, dim=0))

                output_class, embeddings = self.model(batched_rgb, batched_ir)

            else:
                # NEW VERSION
                inputs = {}
                for m, modality in enumerate(self.cfgs.model_modalities):
                    inputs[modality] = []

                    for b in range(len(batch_input)):
                        inputs[modality].append(
                            torch.stack(batch_input[b][m], dim=0))

                    inputs[modality] = self.fabric.to_device(
                        torch.stack(inputs[modality], dim=0))

                # Forward pass
                output_class, embeddings = self.model(inputs)

            # Convert one-hot encoded output to class indices
            preds = torch.argmax(output_class, dim=-1)
            targets = torch.argmax(batch_target, dim=-1)

            # Access output embeddings
            anchor, pos, neg = embeddings["z_reparamed_anchor"], embeddings[
                "z_reparamed_positive"], embeddings["z_reparamed_negative"]

            # Compare predictions to the truth
            running_corrects += torch.sum(preds == targets.data).item()
            total_samples += targets.size(0)

            loss = self.criterion(anchor, pos, neg, output_class, targets)
            self.accumulated_loss.append(loss.item())

            # Backward pass & step
            self.fabric.backward(loss)
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.fabric.barrier()
            optimize_time = (
                time.time() - iter_start_time
            ) / self.cfgs.batch_size * 0.005 + optimize_time * 0.995

            if self.fabric.is_global_zero:
                if i % self.cfgs.print_freq == 0:
                    # Calculate training accuracy for this batch
                    batch_accuracy = running_corrects / total_samples

                    if self.cfgs.use_wandb:
                        current_lr = self.lr_scheduler.get_last_lr(
                        )[0] if self.cfgs.lr_scheduler_name else None
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/acc": batch_accuracy,
                            "train/lr": current_lr,
                            "train/epoch": epoch
                        })

                    self.fabric.print(
                        f"[{epoch}/{end_epoch}][{i}/{len(self.train_loader)}]\t"
                        f"Loss: {loss.item():.4f}\t"
                        f"Train Accuracy: {batch_accuracy:.4f}\t"
                        f"Time: {optimize_time:.4f}\t"
                        f"Data: {(iter_start_time - iter_data_time):.4f}\t")

            iter_data_time = time.time()

    def train(self) -> float:
        """Trains the model for the specified number of epochs.

        Returns:
            Highest validation accuracy.
        """

        total_iters = 0  # total number of training iterations
        optimize_time = 0.1  # initial value to avoid divide by zero error
        end_epoch = self.loaded_epoch + self.cfgs.num_epochs

        for epoch in range(self.loaded_epoch, end_epoch + 1):
            self.epoch = epoch
            self.model.train()

            epoch_start_time = time.time()  # start time for epoch
            iter_data_time = time.time()  # start timer for data loading/iter
            epoch_iter = 0  # start epoch iteration counter

            self.train_epoch(epoch, end_epoch, total_iters, epoch_iter, 
                             iter_data_time, optimize_time)

            if self.fabric.is_global_zero:
                print('End of epoch %d / %d \t Time Taken: %d sec' %
                      (epoch, end_epoch, time.time() - epoch_start_time))

            epoch_valid_acc = self.validate()  # run validation

            # save checkpoint only if global mAP is higher than previous best
            if self.fabric.is_global_zero and epoch_valid_acc > self.highest_val_acc:
                print("Saving checkpoint at epoch {}".format(epoch))
                self.save_networks(os.path.join(self.cfgs.ckpt_dir, self.unique_dir_name),
                                   'best.pth', epoch)
                self.highest_val_acc = epoch_valid_acc

        if self.fabric.is_global_zero:
            print("Training completed. Highest validation acc: {}".format(self.highest_val_acc))
            wandb.run.summary["highest_val_acc"] = self.highest_val_acc
            wandb.run.summary["state"] = "completed"
            wandb.finish(quiet=True)

        return self.highest_val_acc

    def validate(self) -> float:
        """Validates the model.

        Returns:
            Mean Average Precision (mAP) for this validation round.
        """

        start_time = time.time()  # start the timer
        self.model.eval()
        self.metric.reset()

        with torch.no_grad():
            for batch_input, batch_target in self.val_loader:
                pid, camid = parse_string_list(batch_target)

                if self.cfgs.non_generalizable_inputs:
                    batched_rgb, batched_ir = [], []
                    for b in range(len(batch_input)):
                        batched_rgb.append(batch_input[b][0])
                        batched_ir.append(batch_input[b][1])

                    batched_rgb = self.fabric.to_device(torch.stack(batched_rgb, dim=0))
                    batched_ir = self.fabric.to_device(torch.stack(batched_ir, dim=0))
                    embeddings = self.model(batched_rgb, batched_ir, train_mode=False)

                else:
                    # NEW VERSION
                    inputs = {}
                    for m, modality in enumerate(self.cfgs.model_modalities):
                        inputs[modality] = []
                        for b in range(len(batch_input)):
                            inputs[modality].append(batch_input[b][m])
                        inputs[modality] = self.fabric.to_device(torch.stack(inputs[modality], dim=0))
                    embeddings = self.model(inputs, train_mode=False)

                self.metric.update(embeddings["z_reparamed_anchor"], pid, camid)

            # after all batches are processed, get final results
            cmc, mAP = self.metric.compute()

            if self.fabric.is_global_zero:
                # log to wandb
                if self.cfgs.use_wandb:
                    wandb.log({
                        "val/mAP": mAP,
                        "val/rank1": cmc[0],
                        "val/rank5": cmc[4],
                        "val/rank10": cmc[9]
                    })
                # log to terminal
                print("Validation Results")
                print("mAP: {:.4%}".format(mAP))
                print("CMC curve")
                for r in [1, 5, 10]:
                    print("Rank-{:<3}: {:.4%}".format(r, cmc[r - 1]))
                print('Validation Time: ', round(time.time() - start_time, 2), 'sec')

                # log to optuna
                if self.cfgs.use_optuna:
                    self.trial.report(mAP, self.epoch)
                    if self.trial.should_prune():
                        print("Trial was pruned at epoch {}".format(self.epoch))
                        wandb.run.summary["state"] = "pruned"
                        wandb.finish(quiet=True)
                        raise optuna.exceptions.TrialPruned()

        return mAP

    def save_networks(self, save_dir: str, save_name: str, epoch: int) -> None:
        """Saves the model state.

        Args:
            save_dir: Directory to save the model state.
            save_name: Name of the checkpoint file.
            epoch: The epoch number.
        """

        state = {
            "epoch": epoch,
            "mm_model": self.model,
            "optimizer": self.optimizer,
            "loss": self.accumulated_loss,
            "lr_scheduler":
                self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }
        self.fabric.save(os.path.join(save_dir, save_name), state)

    def load_networks(self, load_path: str) -> None:
        """Loads the model state from a checkpoint file.

        Args:
            load_path: Path to the checkpoint file.
        """

        state = {
            "mm_model": self.model,
            "optimizer": self.optimizer,
            "loss": self.accumulated_loss,
        }

        try:
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
