import os
import os.path
import random
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')

def pil_loader(path: str, convert_rgb=True) -> Image.Image:
    ''' Opens image from path using Python Imaging Library (PIL)'''
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # with open(path, 'rb') as f:
    img = Image.open(path)
    return img.convert('RGB') if convert_rgb else img

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    :param filename (string): path to a file
    :param extensions (tuple of strings): extensions to consider (lowercase)
    
    :return bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        cfgs: Dict[str, Any],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        train = True,
        root_dir = None
) -> List[Tuple[str, int]]:
    ''' Creates a list of all images in directory with their corresponding class index.

    :param directory (string): root directory path
    :param class_to_idx (Dict[str, int]): dictionary mapping class name to class index
    :param extensions (optional): A list of allowed extensions.
        Either extensions or is_valid_file should be passed. Defaults to None.
    :param is_valid_file (optional): A function that takes path of a file
        and checks if the file is a valid file
        (used to check of corrupt files) both extensions and
        is_valid_file should not be passed. Defaults to None.

    :return List[Tuple[str, int]]: List of (image path, class_index) tuples    
    '''
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    if train == False:
        for query in os.listdir(root_dir + "/rgbir/query"):
            target_q = query.split("_")[0]
            target_dir = os.path.join(directory, target_q, query)
            instances.append((target_dir, query.split(".")[0]))

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):

            #  filter directory for specified number of image duplicates per pose
            if train == True:
                filter_fn = lambda x: len(x.split("_")) == 3 and int(x.split(".")[0].split("_")[2]) < cfgs.image_num_for_reid_train
                fnames = list(filter(filter_fn, fnames))
            else:
                filter_fn = lambda x: len(x.split("_")) == 3 and int(x.split(".")[0].split("_")[2]) < cfgs.image_num_for_reid_validate
                fnames = list(filter(filter_fn, fnames))

            #  go through remaining files and get a positive image for each one
            for fname in sorted(fnames):
                anchor = os.path.join(root, fname)
                if is_valid_file(anchor):
                    item = anchor, class_index
                    if not train:
                        item = anchor, fname.split(".")[0]
                    instances.append(item)

    return instances

class RGBNT_MultimodalDatasetFolder(VisionDataset):
    """A generic multi-modality dataset loader where the samples are arranged in this way: ::
        root/modality_a/class_x/xxx.ext
        root/modality_a/class_y/xxy.ext
        root/modality_a/class_z/xxz.ext
        root/modality_b/class_x/xxx.ext
        root/modality_b/class_y/xxy.ext
        root/modality_b/class_z/xxz.ext
    :param root (string): Root directory path.
    :param modality_list (list): List of modalitys as strings
    :param loader (callable): A function to load a sample given its path.
    :param extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
    :param transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
    :param target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
    :MultimodalDatasetFolderparam is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt logs)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            mode: str,
            cfgs: Dict[str, Any],
            modality_list: List[str],
            loader: Callable[[str], Any],
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None
    ) -> None:
        super(RGBNT_MultimodalDatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.modality_list = modality_list
        self.cfgs = cfgs
        self.mode = mode

        if len(self.modality_list) < 1:
            raise ValueError('MultimodalImageFolder requires at least one modality')
        
        for modality in self.modality_list:
            if not os.path.exists(os.path.join(root, self.mode, modality)):
                raise ValueError(f"Modality directory does not exist: {modality}")
    

        classes, class_to_idx = self._find_classes(os.path.join(self.root, self.mode, self.modality_list[0]))

        prefixes = {} if prefixes is None else prefixes
        prefixes.update({modality: '' for modality in modality_list if modality not in prefixes})
        
        if self.mode == "train":
            samples = {
                modality: make_dataset(os.path.join(self.root, self.mode, f'{prefixes[modality]}{modality}'), class_to_idx, self.cfgs, extensions, is_valid_file)
                for modality in self.modality_list
            }

            # select a positive instance from the same class, and a negative instance from a diff class
            all_items = np.asarray(list(samples.values())[0])

            
            new_samples = {modality: [] for modality in self.modality_list}
            for i, item in enumerate(all_items):
                for _ in range(self.cfgs.num_triplet_samples):
                    np.random.seed(0)
                    matching_class = list([x for x, y in all_items if y == item[1] and x != item[0]])
                    non_matching_class = list([x for x, y in all_items if y != item[1]])

                    # Randomly select an item from the filtered array
                    random_pos = np.random.choice(range(len(matching_class)))
                    random_neg = np.random.choice(range(len(non_matching_class)))
                    
                    # Add that img for all three modalities
                    for modality in samples:
                        new_samples[modality].append((samples[modality][i][0], 
                                                matching_class[random_pos], 
                                                non_matching_class[random_neg], 
                                                samples[modality][i][1]))
                
            samples = new_samples
            print("Length of sample set:", len(list(samples.values())))

        elif self.mode == "validate":
            samples = {
                modality: make_dataset(os.path.join(self.root, self.mode, f'{prefixes[modality]}{modality}'), class_to_idx, self.cfgs, extensions, is_valid_file, train=False, root_dir=self.root)
                for modality in self.modality_list
            }

        
        for modality, modality_samples in samples.items():
            if len(modality_samples) == 0:
                msg = "Found 0 logs in subfolders of: {}\n".format(os.path.join(self.root, modality))
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        # self.targets = [s[1] for s in list(samples.values())[0]]

        # Select random subset of dataset if so specified
        if isinstance(max_images, int):
            total_samples = len(list(self.samples.values())[0])
            np.random.seed(0)
            permutation = np.random.permutation(total_samples)
            for modality in samples:
                self.samples[modality] = [self.samples[modality][i] for i in permutation][:max_images]
        
        self.cache = {}

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


    # NOTE: use collate_fn instead if using batch transformations
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, pos, neg, target) where target is class_index of the target class.
        """
        if index in self.cache:
            sample_dict, target = deepcopy(self.cache[index])
        else:
            sample_dict = {}
            for modality in self.modality_list:
                if self.mode ==  "train":
                    path, pos, neg, target = self.samples[modality][index]
                    sample = [pil_loader(x, convert_rgb=(modality=='R' or modality=="T")) for x in [path, pos, neg]]
                else:
                    path, target = self.samples[modality][index]
                    sample = pil_loader(path, convert_rgb=(modality=='R' or modality=="T"))
                sample_dict[modality] = sample

        # commented out since we use batch transformations in collate_fn
        #     self.cache[index] = deepcopy((sample_dict, target))
        # if self.transform is not None:
        #     sample_dict = self.transform(sample_dict)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

              # [RGB_vec, IR_vec] in the order of self.modality_list
        
        return list(sample_dict.values()), target
    
    def collate_fn(self, batch):
        batch_unzipped = list(zip(*batch))
        if self.transform is not None:
            batch_inputs = self.transform(batch_unzipped[0], self.mode)
        else:
            batch_inputs = list(batch_unzipped[0])
        
        if self.target_transform is not None:
            batch_target = self.transform(batch_unzipped[1])
        else:
            batch_target = list(batch_unzipped[1])
            if self.cfgs.one_hot and self.mode == "train": 
                batch_target = torch.nn.functional.one_hot(torch.tensor(batch_target), num_classes=len(self.classes)).float()

        

        return batch_inputs, batch_target

    def __len__(self) -> int:
        return len(list(self.samples.values())[0])

class RGBNT_MultimodalImageFolder(RGBNT_MultimodalDatasetFolder):
    """A generic multi-modality dataset loader where the images are arranged in this way: ::
        root/modality_a/class_x/xxx.ext
        root/modality_a/class_y/xxy.ext
        root/modality_a/class_z/xxz.ext
        root/modality_b/class_x/xxx.ext
        root/modality_b/class_y/xxy.ext
        root/modality_b/class_z/xxz.ext
    Args:
        root (string): Root directory path.
        cfgs (dict): Dictionary of configurations
        modality_list (list): List of modalities to load
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt logs)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root: str,
            mode: str,
            cfgs: Dict[str, Any],
            modality_list: List[str],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            prefixes: Optional[Dict[str,str]] = None,
            max_images: Optional[int] = None
    ):
        super(RGBNT_MultimodalImageFolder, self).__init__(root, mode, cfgs, modality_list, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          prefixes=prefixes,
                                          max_images=max_images)

        self.imgs = self.samples


