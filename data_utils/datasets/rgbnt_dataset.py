import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.multiprocessing as mp

from torchvision import datasets, transforms
from torchvision.transforms.functional import crop


from data_utils.datasets.data_constants import *
from data_utils.datasets.rgbnt_data_folders import RGBNT_MultimodalImageFolder



class DataAugmentation_RGBNT(object):

    """
    A class used to handle the data augmentation process for a given dataset.
    
    Attributes:
    -----------
    - num_workers: Number of worker threads for data loading
    - rgb_mean: The global mean for RGB channels
    - rgb_std: The global standard deviation for RGB channels
    - ir1_mean: The global mean for infrared channel
    - ir1_std: The global standard deviation for infrared channel
    - patch: Boolean flag for whether to create patched tokens
    - input_size: Size of the input image
    - hflip: Boolean flag for whether to apply horizontal flip
    - patch_size: Size of the image patches
    - modality_list: List of modalities used in the model
    
    Methods:
    --------
    - process_modality: Applies data augmentation to a given input sample
    - create_patched_tokens: Creates patched tokens from a given image
    - __call__: Makes the class callable so the augmentation can be applied like a function
    """
    
    def __init__(self, cfgs):
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
        self.random_erase = transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)

        self.modality_list = cfgs.model_modalities

    def process_modality(self, input_sample, phase): 
        """
        Performs data augmentation on a single data sample.
        
        Parameters:
        - input_sample: List of images of different modalities
        
        Returns:
        - input_sample: Augmented input sample
        """
        # TODO: GET RID OF THIS AFTER ABLATION!!!!!!!
        
        # Crop and flip all modalitys randomly, but consistently for all modalitys
        if phase == "train":
            flip = {x: random.random() < self.hflip for x in range(3)} # Stores whether to flip all images or not
            ijhw = {x: None for x in range(3)} # Stores crop coordinates used for all modalities

            for modality_i in range(len(input_sample)):
                for triplet in range(len(input_sample[modality_i])):
                    if self.modality_list[modality_i] not in MODALITY_LIST:
                        continue

                    if self.augment:
                        if "random_resize_crop" in self.augment_type:
                            if ijhw[triplet] is None:
                                # Official MAE code uses (0.2, 1.0) for scale and (0.75, 1.3333) for ratio
                                ijhw[triplet] = transforms.RandomResizedCrop.get_params(
                                    input_sample[modality_i][triplet], scale=self.resize_range, ratio=(0.75, 1.3333)
                                )
                            i, j, h, w = ijhw[triplet]
                            input_sample[modality_i][triplet] = TF.crop(input_sample[modality_i][triplet], i, j, h, w)
                            if flip[triplet]:
                                input_sample[modality_i][triplet] = TF.hflip(input_sample[modality_i][triplet])
                        
                    # resize cropped region
                    input_sample[modality_i][triplet] = input_sample[modality_i][triplet].resize((self.input_size[0], self.input_size[1]))
                    # Convert to Tensor
                    img = input_sample[modality_i][triplet]
                    img = TF.to_tensor(input_sample[modality_i][triplet]) 

                    if self.augment:
                        if "random_erasure" in self.augment_type:
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

                # resize cropped region
                input_sample[modality_i] = input_sample[modality_i].resize((self.input_size[0], self.input_size[1]))
                # Convert to Tensor
                img = input_sample[modality_i]
                img = TF.to_tensor(input_sample[modality_i])
                if self.modality_list[modality_i] == 'R':
                    img = TF.normalize(img, mean=self.rgb_mean, std=self.rgb_std)
                elif self.modality_list[modality_i] == 'N': 
                    img = TF.normalize(img, mean=self.nir_mean, std=self.nir_std) 
                elif self.modality_list[modality_i] == 'T':
                    img = TF.normalize(img, mean=self.tir_mean, std=self.tir_std)

                input_sample[modality_i] = img
                
        return input_sample

    
    # TODO: Figure out how to use concurrency libraries
    def __call__(self, batch_inputs, phase):
        """
        Allows the class to be callable, so that data augmentation can be applied like a function.

        Parameters:
        - batch_inputs: List of input samples
        
        Returns:
        - Augmented samples
        """
        return list(map(lambda x: self.process_modality(x, phase), batch_inputs))
        
    def __repr__(self):
        repr = "(DataAugmentationForMultiMAE,\n"
        #repr += "  transform = %s,\n" % str(self.transform)
        repr += ")"
        return repr


def build_rgbnt_dataset(cfgs): 
    """
    Builds a dataset using the DataAugmentation_Bifrost class for data augmentation.

    Parameters:
    - cfgs: The configurations for the data augmentation and the dataset

    Returns:
    - A dataset with applied data augmentation
    """
    transform = DataAugmentation_RGBNT(cfgs)
    data_path = os.path.join(cfgs.dataroot, cfgs.dataset)
    return RGBNT_MultimodalImageFolder(data_path, "train", cfgs, cfgs.model_modalities, transform=transform), RGBNT_MultimodalImageFolder(data_path, "validate", cfgs, cfgs.model_modalities, transform=transform) 