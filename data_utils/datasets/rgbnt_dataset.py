from typing import Any, Dict, List, Tuple

import os
import random

import torchvision.transforms.functional as TF

from torchvision import transforms

from data_utils.datasets.data_constants import *
from data_utils.datasets.rgbnt_data_folders import RGBNT_MultimodalImageFolder


class DataAugmentation_RGBNT(object):
    """A class used to handle the data augmentation process for a given dataset.

    Args:
        cfgs: Configuration dictionary.

    Attributes:
        num_workers: Number of worker threads for data loading.
        rgb_mean: The global mean for RGB channels.
        rgb_std: The global standard deviation for RGB channels.
        nir_mean: The global mean for infrared channel.
        nir_std: The global standard deviation for infrared channel.
        tir_mean: The global mean for the thermal infrared channel.
        tir_std: The global standard deviation for the thermal infrared channel.
        input_size: Size of the input image.
        hflip: Boolean flag for whether to apply horizontal flip.
        resize_range: Range of sizes for random resize.
        augment: Boolean flag for whether to apply data augmentation.
        augment_type: Type of data augmentation to apply.
        random_erase: Random erasing data augmentation.
        modality_list: List of modalities used in the model.
    """

    def __init__(self, cfgs: Dict[str, Any]):
        self.num_workers = cfgs.workers
        self.rgb_mean = globals()[cfgs.dataset_name + "_R_MEAN"]
        self.rgb_std = globals()[cfgs.dataset_name + "_R_STD"]
        self.nir_mean = globals()[cfgs.dataset_name + "_N_MEAN"]
        self.nir_std = globals()[cfgs.dataset_name + "_N_STD"]
        self.tir_mean = globals()[cfgs.dataset_name + "_T_MEAN"]
        self.tir_std = globals()[cfgs.dataset_name + "_T_STD"]
        self.input_size = cfgs.input_size
        self.hflip = cfgs.hflip
        self.resize_range = (cfgs.resize_min, cfgs.resize_max)
        self.augment = cfgs.data_augmentation
        self.augment_type = cfgs.augment_type
        self.random_erase = transforms.RandomErasing(p=0.5,
                                                     scale=(0.02, 0.33),
                                                     ratio=(0.3, 3.3),
                                                     value=0,
                                                     inplace=False)
        self.modality_list = cfgs.model_modalities

    def process_modality(self, input_sample: List[Any], phase: str) -> List[Any]:
        """Performs data augmentation on a single data sample.
        
        Args:
            input_sample: List of images of different modalities.
            phase: Phase of the data processing (train or validate).

        Returns:
            input_sample: Augmented input sample.
        """
        if phase == "train":
            flip = {
                x: random.random() < self.hflip
                for x in range(3)
            }  # Stores whether to flip all images or not
            ijhw = {
                x: None
                for x in range(3)
            }  # Stores crop coordinates used for all modalities

            for modality_i in range(len(input_sample)):
                for triplet in range(len(input_sample[modality_i])):
                    if self.modality_list[modality_i] not in MODALITY_LIST:
                        continue

                    if self.augment:
                        if "random_resize_crop" in self.augment_type:
                            if ijhw[triplet] is None:
                                ijhw[triplet] = transforms.RandomResizedCrop.get_params(
                                    input_sample[modality_i][triplet],
                                    scale=self.resize_range,
                                    ratio=(0.75, 1.3333))
                            i, j, h, w = ijhw[triplet]
                            input_sample[modality_i][triplet] = TF.crop(
                                input_sample[modality_i][triplet], i, j, h, w)
                            if flip[triplet]:
                                input_sample[modality_i][triplet] = TF.hflip(
                                    input_sample[modality_i][triplet])

                    input_sample[modality_i][triplet] = input_sample[modality_i][triplet].resize(
                        (self.input_size[0], self.input_size[1])
                    )
                    img = TF.to_tensor(input_sample[modality_i][triplet])

                    if self.augment and "random_erasure" in self.augment_type:
                        img = self.random_erase(img)

                    if self.modality_list[modality_i] == 'R':
                        img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
                    elif self.modality_list[modality_i] == 'N':
                        img = TF.normalize(img, mean=self.nir_mean, std=self.nir_std)
                    elif self.modality_list[modality_i] == 'T':
                        img = TF.normalize(img, mean=self.tir_mean, std=self.tir_std)

                    input_sample[modality_i][triplet] = img
        elif phase == "validate":
            for modality_i in range(len(input_sample)):
                if self.modality_list[modality_i] not in MODALITY_LIST:
                    continue

                input_sample[modality_i] = input_sample[modality_i].resize((self.input_size[0], self.input_size[1]))
                img = TF.to_tensor(input_sample[modality_i])

                if self.modality_list[modality_i] == 'R':
                    img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
                elif self.modality_list[modality_i] == 'N':
                    img = TF.normalize(img, mean=self.nir_mean, std=self.nir_std)
                elif self.modality_list[modality_i] == 'T':
                    img = TF.normalize(img, mean=self.tir_mean, std=self.tir_std)

                input_sample[modality_i] = img

        return input_sample

    def __call__(self, batch_inputs: List[Any], phase: str) -> List[Any]:
        """Allows the class to be callable, so that data augmentation can be applied like a function.

        Args:
            batch_inputs: List of input samples.
            phase: Phase of the data processing (train or validate).

        Returns:
            Augmented samples.
        """
        return list(map(lambda x: self.process_modality(x, phase), batch_inputs))

    def __repr__(self) -> str:
        repr = "(DataAugmentationForGraFT,\n"
        repr += ")"
        return repr


def build_rgbnt_dataset(cfgs: Dict[str, Any]) -> Tuple[RGBNT_MultimodalImageFolder, RGBNT_MultimodalImageFolder]:
    """Builds a dataset using the DataAugmentation_RGBNT class for data augmentation.

    Args:
        cfgs: The configurations for the data augmentation and the dataset.

    Returns:
        A dataset with applied data augmentation.
    """
    transform = DataAugmentation_RGBNT(cfgs)
    data_path = os.path.join(cfgs.dataroot, cfgs.dataset)
    return (RGBNT_MultimodalImageFolder(data_path, "train", cfgs, cfgs.model_modalities, transform=transform),
            RGBNT_MultimodalImageFolder(data_path, "validate", cfgs, cfgs.model_modalities, transform=transform))
