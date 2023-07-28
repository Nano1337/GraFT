from data_utils.datasets.rgbnt_dataset import build_rgbnt_dataset
from torch.utils.data import DataLoader, Subset, random_split

def split_dataset(shuffled_dataset, train_ratio=0.8):
    """
    Splits the dataset into training and validation subsets.

    Parameters:
    - shuffled_dataset: The dataset to split
    - train_ratio: The proportion of the dataset to use for training

    Returns:
    - train_dataset: A Subset instance representing the training data
    - val_dataset: A Subset instance representing the validation data
    """

    train_size = int(len(shuffled_dataset) * train_ratio)  # 80% for training
    val_size = len(shuffled_dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(shuffled_dataset, [train_size, val_size])

    return train_dataset, val_dataset


def get_dataset_and_dataloader(cfgs): 
    """
    Generates datasets and dataloaders for training and validation.

    Parameters:
    - cfgs: A configuration object with all the required parameters.

    Returns:
    - train_dataset: The training dataset
    - val_dataset: The validation dataset
    - train_dataloader: A DataLoader instance for the training data
    - val_dataloader: A DataLoader instance for the validation data
    """

    if "bifrost" in cfgs.dataset:
        # Building and splitting the "bifrost" dataset into training and validation datasets
        dataset = build_bifrost_dataset(cfgs)
        train_dataset, val_dataset = split_dataset(dataset, cfgs.train_ratio)

        # Creating dataloaders for the training and validation datasets
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfgs.batch_size,
            shuffle=True,
            num_workers=cfgs.workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfgs.batch_size,
            shuffle=False,
            num_workers=cfgs.workers,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )
    elif "RGBN" in cfgs.dataset:
        # Directly building the "RGBN" training and validation datasets
        train_dataset, val_dataset = build_rgbnt_dataset(cfgs)

        # Creating dataloaders for the training and validation datasets
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfgs.batch_size,
            shuffle=True,
            num_workers=cfgs.workers,
            pin_memory=True,
            collate_fn=train_dataset.collate_fn,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfgs.batch_size,
            shuffle=False,
            num_workers=cfgs.workers,
            pin_memory=True,
            collate_fn=val_dataset.collate_fn,
        )
    else: 
        # Raising an error if the specified dataset is not supported
        raise ValueError("Dataset not supported")

    return train_dataset, val_dataset, train_dataloader, val_dataloader
