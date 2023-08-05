import os
import time
from pathlib import Path

import wandb
import optuna
from optuna.trial import Trial

import torch.utils.data
import torch.distributed as dist
from lightning.fabric import Fabric, seed_everything

import models
import data_utils
import train
import optimizers
import loss
from utils.opt import get_cfgs

torch.set_float32_matmul_precision("medium")
os.environ["WANDB_START_METHOD"] = "thread"


def objective(trial: Trial, cfgs: dict, fabric: Fabric, rand_seed: int, hyperparams_in: dict) -> float:
    """Objective function for Optuna hyperparameter optimization.

    Args:
        trial: A trial is a process of evaluating an objective function.
        cfgs: Configuration parameters for the model.
        fabric: Fabric object for distributed training.
        rand_seed: Random seed for reproducibility.
        hyperparams_in: Hyperparameters and their ranges for the model.

    Returns:
        Highest validation score.
    """

    if not hasattr(cfgs, "hyperparams"):
        cfgs.hyperparams = {}

    for hyperparam, value in hyperparams_in.items():
        if value['type'] == 'int':
            setattr(cfgs, hyperparam, trial.suggest_int(hyperparam, value['min'], value['max']))
        elif value['type'] == 'float':
            setattr(cfgs, hyperparam, trial.suggest_float(hyperparam, value['min'], value['max']))
        elif value['type'] == 'loguniform':
            setattr(cfgs, hyperparam, trial.suggest_loguniform(hyperparam, value['min'], value['max']))
        elif value['type'] == 'categorical':
            setattr(cfgs, hyperparam, trial.suggest_categorical(hyperparam, value['opt']))
        else:
            raise ValueError("Hyperparameter type not supported.")

    if cfgs.wandb_trial_name:
        trial_name = cfgs.wandb_trial_name + "_trial_" + str(trial.number)
    else:
        trial_name = "trial_" + str(trial.number)

    if fabric.is_global_zero:
        wandb.init(
            project=cfgs.wandb_project,
            config=cfgs,
            group=cfgs.study_name,
            name=trial_name,
            reinit=True,
        )
        wandb.run.save()

        for hyperparam in hyperparams_in.keys():
            wandb.log({"hyperparams/" + str(hyperparam): getattr(cfgs, hyperparam)})
        wandb.log({"hyperparams/seed": rand_seed})

    cfgs.trial_number = trial.number

    highest_val = main(cfgs, fabric, trial)

    return highest_val


def main(cfgs: dict, fabric: Fabric, trial: Trial = None) -> float:
    """Main function for the training/evaluation process.

    Args:
        cfgs: Configuration parameters for the model.
        fabric: Fabric object for distributed training.
        trial: A trial is a process of evaluating an objective function.

    Returns:
        Highest validation score.
    """
    train_dataset, val_dataset, train_loader, val_loader = data_utils.get_dataset_and_dataloader(cfgs)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)

    dataset_size = len(train_dataset) + len(val_dataset)

    if fabric.is_global_zero:
        print("Dataset size total:", dataset_size)
        print("Training set size:", len(train_dataset))
        print("Validation set size:", len(val_dataset))

    num_batches = 0
    for batch_input, batch_target in val_loader:
        num_batches += 1
    print("Number of batches for validation:", num_batches)
    print("Dataloader size total:", num_batches * cfgs.batch_size)

    config_name = os.path.splitext(os.path.basename(cfgs.cfg_name))[0]
    unique_dir_name = time.strftime("%Y%m%d-%H%M%S-") + config_name
    output_dir = Path(cfgs.output_dir, unique_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Output directory:", output_dir)

    process_group = None
    if len(cfgs.gpus) > 1:
        process_group = dist.new_group(
            ranks=list(range(fabric.world_size)))

    model = models.get_model(cfgs=cfgs, fabric=fabric, process_group=process_group)
    criterion = loss.get_loss(cfgs, fabric)
    optimizer = optimizers.get_optim(cfgs, model)

    trainer = train.get_trainer(cfgs, fabric, model, train_loader, val_loader,
                                optimizer, criterion, unique_dir_name, trial, process_group=process_group)

    if fabric.is_global_zero:
        trainer.print_networks()

    if cfgs.phase == "train":
        highest_val = trainer.train()
    elif cfgs.phase == "val":
        print(trainer.validate())

    return highest_val


if __name__ == "__main__":
    cfgs = get_cfgs()
    if cfgs.output_dir:
        Path(cfgs.output_dir).mkdir(parents=True, exist_ok=True)

    seed_everything(cfgs.seed)

    sampler = optuna.samplers.QMCSampler(seed=cfgs.seed)
    study = optuna.create_study(
        direction="maximize",
        study_name=cfgs.study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=100),
        sampler=sampler,
    )

    fabric = Fabric(accelerator="auto", devices=cfgs.gpus)
    fabric.launch()

    print("Random seed:", cfgs.seed)
    study.optimize(
        lambda trial: objective(trial, cfgs, fabric, cfgs.seed, cfgs.hyperparams),
        n_trials=cfgs.n_trials,
        n_jobs=1,
    )

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
