"""Dataset Class for loading YOLO format datasets where the source data dir has the image and labels subdirs where each
image must have a corresponding label file with the same name.
"""

import glob
import os.path as osp
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTNS = {".png", ".jpg", ".jpeg"}


class YOLODataset(Dataset):
    """
    Create a dataset for adversarial-yolt.

    Attributes:
        image_dir: Directory containing the images of the YOLO format dataset.
        label_dir: Directory containing the labels of the YOLO format dataset.
        max_labels: max number labels to use for each image
        model_in_sz: model input image size (height, width)
        use_even_odd_images: optionally load a data subset based on the last numeric char of the img filename [all, even, odd]
        filter_class_id: np.ndarray class id(s) to get. Set None to get all classes
        min_pixel_area: min pixel area below which all boxes are filtered out. (Out of the model in size area)
        shuffle: Whether or not to shuffle the dataset.
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        max_labels: int,
        model_in_sz: Tuple[int, int],
        use_even_odd_images: str = "all",
        transform: Optional[torch.nn.Module] = None,
        filter_class_ids: Optional[np.array] = None,
        min_pixel_area: Optional[int] = None,
        shuffle: bool = True,
    ):
        assert use_even_odd_images in {"all", "even", "odd"}, "use_even_odd param can only be all, even or odd"
        image_paths = glob.glob(osp.join(image_dir, "*"))
        label_paths = glob.glob(osp.join(label_dir, "*"))
        image_paths = sorted([p for p in image_paths if osp.splitext(p)[-1] in IMG_EXTNS])
        label_paths = sorted([p for p in label_paths if osp.splitext(p)[-1] in {".txt"}])

        # if use_even_odd_images is set, use images with even/odd numbers in the last char of their filenames
        if use_even_odd_images in {"even", "odd"}:
            rem = 0 if use_even_odd_images == "even" else 1
            image_paths = [p for p in image_paths if int(osp.splitext(p)[0][-1]) % 2 == rem]
            label_paths = [p for p in label_paths if int(osp.splitext(p)[0][-1]) % 2 == rem]
        assert len(image_paths) == len(label_paths), "Number of images and number of labels don't match"
        # all corresponding image and labels must exist
        for img, lab in zip(image_paths, label_paths):
            if osp.basename(img).split(".")[0] != osp.basename(lab).split(".")[0]:
                raise FileNotFoundError(f"Matching image {img} or label {lab} not found")
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.model_in_sz = model_in_sz
        self.shuffle = shuffle
        self.max_n_labels = max_labels
        self.transform = transform
        self.filter_class_ids = np.asarray(filter_class_ids) if filter_class_ids is not None else None
        self.min_pixel_area = min_pixel_area

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        assert idx <= len(self), "Index range error"
        img_path = self.image_paths[idx]
        lab_path = self.label_paths[idx]
        image = Image.open(img_path).convert("RGB")
        # check to see if label file contains any annotation data
        label = np.loadtxt(lab_path) if osp.getsize(lab_path) else np.zeros([1, 5])
        if label.ndim == 1:
            label = np.expand_dims(label, axis=0)
        # sort in reverse by bbox area
        label = np.asarray(sorted(label, key=lambda annot: -annot[3] * annot[4]))
        # selectively get classes if filter_class_ids is not None
        if self.filter_class_ids is not None:
            label = label[np.isin(label[:, 0], self.filter_class_ids)]
            label = label if len(label) > 0 else np.zeros([1, 5])

        label = torch.from_numpy(label).float()
        image, label = self.pad_and_scale(image, label)
        if self.transform:
            image = self.transform(image)
            if np.random.random() < 0.5:  # rand horizontal flip
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if label.shape:
                    label[:, 1] = 1 - label[:, 1]
        # filter boxes by bbox area pixels compared to the model in size (640x640 by default)
        if self.min_pixel_area is not None:
            label = label[
                (label[:, 3] * label[:, 4]) >= (self.min_pixel_area / (self.model_in_sz[0] * self.model_in_sz[1]))
            ]
            label = label if len(label) > 0 else torch.zeros([1, 5])
        image = transforms.ToTensor()(image)
        label = self.pad_label(label)
        return image, label

    def pad_and_scale(self, img, lab):
        """Pad image and adjust label img is a PIL image lab is of fmt class x_center y_center width height with
        normalized coords.
        """
        img_w, img_h = img.size
        if img_w == img_h:
            padded_img = img
        else:
            if img_w < img_h:
                padding = (img_h - img_w) / 2
                padded_img = Image.new("RGB", (img_h, img_h), color=(127, 127, 127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * img_w + padding) / img_h
                lab[:, [3]] = lab[:, [3]] * img_w / img_h
            else:
                padding = (img_w - img_h) / 2
                padded_img = Image.new("RGB", (img_w, img_w), color=(127, 127, 127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * img_h + padding) / img_w
                lab[:, [4]] = lab[:, [4]] * img_h / img_w
        padded_img = transforms.Resize(self.model_in_sz)(padded_img)

        return padded_img, lab

    def pad_label(self, label: torch.Tensor) -> torch.Tensor:
        """Pad labels with zeros if fewer labels than max_n_labels present."""
        pad_size = self.max_n_labels - label.shape[0]
        if pad_size > 0:
            padded_lab = F.pad(label, (0, 0, 0, pad_size), value=0)
        else:
            padded_lab = label[: self.max_n_labels]
        return padded_lab
