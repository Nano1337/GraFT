import os
import time
import random
import logging
from pathlib import Path

import wandb
import optuna
from optuna.trial import Trial

import torch.utils.data
from torch.nn.parallel import DistributedDataParallel
from lightning.fabric import Fabric, seed_everything

import models
import data_utils
import train
import optimizers
import loss
from utils.opt import get_cfgs

torch.set_float32_matmul_precision("medium")
os.environ["WANDB_START_METHOD"] = "thread"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def objective(trial: Trial, cfgs: dict, fabric: Fabric, rand_seed: int) -> float:
    """Objective function for Optuna hyperparameter optimization."""

    # Removed hardcoding of hyperparameters
    hyperparams = cfgs.hyperparameters
    for hyperparam in hyperparams.keys():
        hyperparams[hyperparam] = trial.suggest_float(hyperparam, *cfgs.hyperparameters_range[hyperparam])

    for hyperparam, value in hyperparams.items():
        setattr(cfgs, hyperparam, value)

    trial_name = cfgs.wandb_trial_name + "_trial_" + str(trial.number) if cfgs.wandb_trial_name else "trial_" + str(trial.number)
    cfgs.trial_number = trial.number

    highest_val = main(cfgs, fabric, trial)

    return highest_val


def main(cfgs: dict, fabric: Fabric, trial: Trial = None) -> float:
    """Main function for the training/evaluation process."""

    train_dataset, val_dataset, train_loader, val_loader = data_utils.get_dataset_and_dataloader(cfgs)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    dataset_size = len(train_dataset) + len(val_dataset)
    logger.info("Dataset size total: {}".format(dataset_size))

    config_name = os.path.splitext(os.path.basename(cfgs.cfg_name))[0]
    unique_dir_name = time.strftime("%Y%m%d-%H%M%S-") + config_name
    output_dir = Path(cfgs.output_dir, unique_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: {}".format(output_dir))

    model = models.get_model(cfgs=cfgs, fabric=fabric)
    model = DistributedDataParallel(model, device_ids=fabric.devices)  # Ensure model is ready for multi-GPU

    criterion = loss.get_loss(cfgs, fabric)
    optimizer = optimizers.get_optim(cfgs, model)

    trainer = train.get_trainer(cfgs, fabric, model, train_loader, val_loader,
                                optimizer, criterion, unique_dir_name, trial)

    highest_val = trainer.train() if cfgs.phase == "train" else trainer.validate()  # Return the validation score

    return highest_val


if __name__ == "__main__":

    cfgs = get_cfgs()

    Path(cfgs.output_dir).mkdir(parents=True, exist_ok=True)

    rand_seed = random.randint(0, 2**32 - 1) if cfgs.new_random else cfgs.seed
    cfgs.seed = rand_seed
    seed_everything(cfgs.seed)

    fabric = Fabric(accelerator="auto", devices=cfgs.gpus)
    fabric.launch()

    sqlite_file = os.path.join(os.getcwd(), f'{cfgs.study_name}.db')
    storage_name = f"sqlite:///{sqlite_file}"

    sampler = optuna.samplers.QMCSampler(seed=cfgs.seed)
    try:
        study = optuna.load_study(study_name=cfgs.study_name, storage=storage_name)
        logger.info("Study {} loaded.".format(cfgs.study_name))
    except KeyError:
        logger.info("Study {} not found, creating a new one.".format(cfgs.study_name))
        study = optuna.create_study(
            direction="maximize",
            study_name=cfgs.study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=100),
            sampler=sampler,
            storage=storage_name,
        )

    try:
        study.optimize(lambda trial: objective(trial, cfgs, fabric, cfgs.seed),
                       n_trials=1,
                       n_jobs=cfgs.n_jobs)  # Parallelize trials across configurable number of processes
    except Exception as e:
        logger.error("Error: {}".format(e))

    logger.info("Best trial:")
    trial = study.best_trial
    logger.info("  Value: {}".format(trial.value))
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info("    {}: {}".format(key, value))

    # Cleanup SQLite database file
    if os.path.exists(sqlite_file):
        os.remove(sqlite_file)
