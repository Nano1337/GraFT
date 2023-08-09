import time
import os
import wandb
import optuna

import torch

from train.base_trainer import Base_Trainer
from metrics.R1_mAP import R1_mAP


def parse_string_list(string_list):
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


class Trainer_Validation_Only(Base_Trainer):
    """Trainer class for RGBN Triplet Verb model.

    This class inherits from the BaseTrainer class and provides functionalities
    for training and validating the RGBN Triplet Verb model. The trainer
    includes methods for training a single epoch, running validation, saving
    and loading models.
    """

    def __init__(self, cfgs, fabric, model, val_loader):
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

        super().__init__(cfgs=cfgs,
                         fabric=fabric,
                         model=model,
                         val_loader=val_loader)
        self.model = self.fabric.setup(self.model)

        # load checkpoint if exists
        if not os.path.isdir(cfgs.ckpt_dir):
            raise ValueError("Checkpoint directory does not exist")
        if cfgs.ckpt_full_path and os.path.isfile(cfgs.ckpt_full_path):
            self.load_networks(cfgs.ckpt_full_path)
        else:
            raise ValueError("ckpt_full_path Checkpoint file does not exist")

        # initialize validation metrics
        query_path = os.path.join(str(cfgs.dataroot), str(
            cfgs.dataset)) + "/rgbir/query"
        self.num_queries = len(os.listdir(query_path))
        self.metric = R1_mAP(self.fabric, self.cfgs, self.num_queries, self.cfgs.max_rank)

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

                    inputs[modality] = self.fabric.to_device(
                        torch.stack(inputs[modality], dim=0))

                # Forward pass
                embeddings = self.model(inputs, train_mode=False)

                # store data for val metric calc later
                self.metric.update(embeddings["z_reparamed_anchor"], pid,
                                   camid)

            # after all batches are processed, call metrics again to get final results
            cmc, mAP = self.metric.compute()

            # visualize embeddings
            if self.fabric.is_global_zero:

                if self.cfgs.visualize_embeddings:
                    self.metric.compute_umap_plotly(
                        save_path=self.cfgs.vis_save_path,
                        dims=self.cfgs.vis_dim,
                        reduction_method=self.cfgs.vis_reduction_method)

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

                end_time = time.time()  # end the timer
                print('Validation Time: ', round(end_time - start_time, 2),
                      'sec')  # print the time taken for validation


        return mAP

    def load_networks(self, load_path):
        """
        Loads the model, optimizer, and learning rate scheduler (if exists) from a specified file.

        Args:
            load_path: A string representing the path of the file from which the model state is to be loaded.

        Returns:
            None. This function operates by loading the state of the model, optimizer,
                and lr_scheduler from the disk to the Trainer_1 object.
        """

        state = {
            "mm_model": self.model,
        }

        try:
            # load checkpoint to model in fabric in-place (model already setup)
            print("Loading in ckpt weights...")
            remainder = self.fabric.load(load_path, state)
        except Exception as e:
            print(f"Error loading network state: {e}")
            return

        try:
            self.loaded_epoch = remainder['epoch']
        except Exception as e:
            print(f"Error loading loaded_epoch: {e}")

    def train(self):
        pass

    def save_networks(save_dir, save_name):
        pass

    def train_epoch(epoch, end_epoch, total_iters, epoch_iter, iter_data_time,
                    optimize_time):
        pass
