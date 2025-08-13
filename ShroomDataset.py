from pathlib import Path
from typing import Optional, Dict, List
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode


class ShroomDataset(Dataset):
    def __init__(self, df, base_path, transform=None, label2idx=None):
        self.base_path = Path(base_path)
        self.paths  = [self.base_path / p for p in df["image_path"].tolist()]
        self.labels_str = df["label"].tolist()
        if label2idx is None:
            classes = sorted(set(self.labels_str))
            self.label2idx = {c:i for i,c in enumerate(classes)}
        else:
            self.label2idx = label2idx
        self.labels = [self.label2idx[s] for s in self.labels_str]
        self.transform = transform

    def __getitem__(self, index):
        # uint8 tensor [C,H,W] in RGB in [0..255]
        img = read_image(str(self.paths[index]), mode=ImageReadMode.RGB)
        if self.transform:
            img = self.transform(img)      # <-- OK because it’s Tensor here
        return img, self.labels[index]

    def __len__(self):
        return len(self.paths)