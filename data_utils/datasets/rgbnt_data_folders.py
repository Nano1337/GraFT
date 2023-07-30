import os
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def pil_loader(path: str, convert_rgb: bool = True) -> Image.Image:
    """Opens image from path using Python Imaging Library (PIL)."""
    img = Image.open(path)
    return img.convert('RGB') if convert_rgb else img


def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename: Path to a file.
        extensions: Extensions to consider (lowercase).

    Returns:
        True if the filename ends with one of the given extensions.
    """
    return filename.lower().endswith(extensions)


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
    cfgs: Dict[str, Any],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    train: bool = True,
    root_dir: Optional[str] = None
) -> List[Tuple[str, int]]:
    """Creates a list of all images in directory with their corresponding class index.

    Args:
        directory: Root directory path.
        class_to_idx: Dictionary mapping class name to class index.
        extensions: A list of allowed extensions. Either extensions or is_valid_file should be passed.
        is_valid_file: A function that takes path of a file and checks if the file is a valid file.
        train: Whether to prepare for training or not.
        root_dir: Root directory.

    Returns:
        List of (image path, class_index) tuples.
    """
    instances = []
    directory = os.path.expanduser(directory)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError(
            "Both extensions and is_valid_file cannot be None or not None at the same time"
        )
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))
    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    if train is False:
        for query in os.listdir(root_dir + "/rgbir/query"):
            target_q = query.split("_")[0]
            target_dir = os.path.join(directory, target_q, query)
            instances.append((target_dir, query.split(".")[0]))

    #  filter directory for specified number of image duplicates per pose
    def filter_train_images(x, cfgs):
        return len(x.split("_")) == 3 and int(x.split(".")[0].split("_")[2]) < cfgs.image_num_for_reid_train

    def filter_validate_images(x, cfgs):
        return len(x.split("_")) == 3 and int(x.split(".")[0].split("_")[2]) < cfgs.image_num_for_reid_validate

    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):

            if train:
                fnames = list(filter(lambda x: filter_train_images(x, cfgs), fnames))
            else:
                fnames = list(filter(lambda x: filter_validate_images(x, cfgs), fnames))

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
    """A generic multi-modality dataset loader.

    The samples are arranged in this way:
        root/modality_a/class_x/xxx.ext
        root/modality_a/class_y/xxy.ext
        root/modality_a/class_z/xxz.ext
        root/modality_b/class_x/xxx.ext
        root/modality_b/class_y/xxy.ext
        root/modality_b/class_z/xxz.ext

    Args:
        root: Root directory path.
        mode: Mode of operation (train or validate).
        cfgs: Configuration dictionary.
        modality_list: List of modalities as strings.
        loader: A function to load a sample given its path.
        extensions: A list of allowed extensions. Either extensions or is_valid_file should be passed.
        transform: A function/transform that takes in a sample and returns a transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
        is_valid_file: A function that takes path of a file and check if the file is a valid file.
        prefixes: Dictionary of modality prefixes.
        max_images: Maximum number of images to use from the dataset.

    Attributes:
        classes: List of the class names sorted alphabetically.
        class_to_idx: Dict with items (class_name, class_index).
        samples: List of (sample path, class_index) tuples.
        targets: The class_index value for each image in the dataset.
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
        prefixes: Optional[Dict[str, str]] = None,
        max_images: Optional[int] = None
    ) -> None:
        super(RGBNT_MultimodalDatasetFolder, self).__init__(
            root, transform=transform, target_transform=target_transform)
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
                modality: make_dataset(
                    os.path.join(self.root, self.mode, f'{prefixes[modality]}{modality}'),
                    class_to_idx, self.cfgs, extensions, is_valid_file)
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
                        new_samples[modality].append(
                            (samples[modality][i][0],
                             matching_class[random_pos],
                             non_matching_class[random_neg],
                             samples[modality][i][1]))

            samples = new_samples
            print("Length of sample set:", len(list(samples.values())))

        elif self.mode == "validate":
            samples = {
                modality: make_dataset(
                    os.path.join(self.root, self.mode, f'{prefixes[modality]}{modality}'),
                    class_to_idx, self.cfgs, extensions, is_valid_file, train=False, root_dir=self.root)
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
        self.cache = {}

    def _find_classes(self, dir: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        Args:
            dir: Root directory path.

        Returns:
            Tuple containing list of classes and dictionary mapping class name to class index.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Fetches an item by index.

        Args:
            index: Index of item to fetch.

        Returns:
            Tuple containing sample and target.
        """
        if index in self.cache:
            sample_dict, target = deepcopy(self.cache[index])
        else:
            sample_dict = {}
            for modality in self.modality_list:
                if self.mode == "train":
                    path, pos, neg, target = self.samples[modality][index]
                    sample = [
                        pil_loader(x, convert_rgb=(modality == 'R' or modality == "T"))
                        for x in [path, pos, neg]
                    ]
                else:
                    path, target = self.samples[modality][index]
                    sample = pil_loader(path, convert_rgb=(modality == 'R' or modality == "T"))
                sample_dict[modality] = sample
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
                batch_target = torch.nn.functional.one_hot(
                    torch.tensor(batch_target),
                    num_classes=len(self.classes)).float()

        return batch_inputs, batch_target

    def __len__(self) -> int:
        return len(list(self.samples.values())[0])


class RGBNT_MultimodalImageFolder(RGBNT_MultimodalDatasetFolder):
    """A generic multi-modality dataset loader.

    The images are arranged in this way:
        root/modality_a/class_x/xxx.ext
        root/modality_a/class_y/xxy.ext
        root/modality_a/class_z/xxz.ext
        root/modality_b/class_x/xxx.ext
        root/modality_b/class_y/xxy.ext
        root/modality_b/class_z/xxz.ext

    Args:
        root: Root directory path.
        mode: Mode of operation (train or validate).
        cfgs: Configuration dictionary.
        modality_list: List of modalities to load.
        transform: A function/transform that takes in a PIL image and returns a transformed version.
        target_transform: A function/transform that takes in the target and transforms it.
        loader: A function to load an image given its path.
        is_valid_file: A function that takes path of an Image file and check if the file is a valid file.
        prefixes: Dictionary of modality prefixes.
        max_images: Maximum number of images to use from the dataset.

    Attributes:
        classes: List of the class names sorted alphabetically.
        class_to_idx: Dict with items (class_name, class_index).
        imgs: List of (image path, class_index) tuples.
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
        prefixes: Optional[Dict[str, str]] = None,
        max_images: Optional[int] = None
    ):
        super(RGBNT_MultimodalImageFolder, self).__init__(
            root,
            mode,
            cfgs,
            modality_list,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            prefixes=prefixes,
            max_images=max_images)

        self.imgs = self.samples
