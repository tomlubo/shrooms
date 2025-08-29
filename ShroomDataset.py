from pathlib import Path
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Sampler
from torchvision.io import read_image, ImageReadMode
from collections import defaultdict
import pandas as pd


class ShroomDataset(Dataset):
    """
    image classification dataset that:
    - reads RGB images from disk using torchvision.io.read_image
    - builds/uses a label→index mapping (string labels → int ids)
    - (optionally) caches decoded uint8 CHW tensors in a RAM cache
    - applies a torchvision v2 transform pipeline (expects uint8 CHW → float + normalize)

    Args:
        df (pd.DataFrame): Must contain columns:
            - ``image_path`` (relative to ``base_path``)
            - ``label`` (string species name or similar)
        base_path (str | Path): Root folder containing image files referenced in ``df``.
        transform (callable | None): A torchvision v2 transform callable applied to the
            uint8 CHW tensor (e.g., RandomResizedCrop → ToDtype → Normalize).
        label2idx (dict[str,int] | None): Optional external mapping to keep class ids consistent
            across splits. If None, a new mapping is created from the unique labels in ``df``.
        cache (bool): If True, keep a small in-memory cache of decoded images (uint8 CHW).
        cache_max (int): Max entries to keep in the RAM cache (simple FIFO/LRU-ish eviction).
    """
    
    def __init__(self, df, base_path, transform=None, label2idx=None,
                 cache=False, cache_max=2048):

        self.base_path = Path(base_path)
        paths = [self.base_path / p for p in df["image_path"].tolist()]
        # string paths for torchvision.io.read_image, do it all before to make it faster
        self.path_strs = [str(p) for p in paths]

        # build or reuse label mapping
        self.labels_str = df["label"].tolist()
        if label2idx is None:
            classes = sorted(set(self.labels_str))
            self.label2idx = {c: i for i, c in enumerate(classes)}
        else:
            self.label2idx = label2idx

        # vectorized class-id array
        self.labels = np.asarray([self.label2idx[s] for s in self.labels_str], dtype=np.int64)
        self.transform = transform

        # DO NOT USE THIS UNLESS YOUR MACHINE IS BEEFY,
        # it has caused me a lot of pain, didn't even know you could BSOD a mac
        self.cache = bool(cache)
        self.cache_max = int(cache_max)
        self._ram = {}          # idx -> tensor(uint8, CHW)
        self._ram_keys = []     # maintain insertion order for simple eviction

    def __len__(self):
        return len(self.path_strs)

    def _cache_put(self, idx, img):
        # again this seems useful at first, but it will kill your ram if you arent careful
        if not self.cache: return
        if idx in self._ram: return
        self._ram[idx] = img
        self._ram_keys.append(idx)
        if len(self._ram_keys) > self.cache_max:
            old = self._ram_keys.pop(0)
            self._ram.pop(old, None)

    def __getitem__(self, index):
        # Fast path: RAM cache hit
        if self.cache and index in self._ram:
            img = self._ram[index]
        else:
            # uint8 tensor [C,H,W] in RGB [0..255]
            img = read_image(self.path_strs[index], mode=ImageReadMode.RGB)
            self._cache_put(index, img)

        if self.transform:
            img = self.transform(img)  # v2 transforms expect CHW uint8 → float+normalize
        return img, int(self.labels[index])
    

    
class TwoCropsTransform(nn.Module):
    """
    Compose two correlated 'views' of the same input image for contrastive learning:

        output = heavy(x) → q = light(output),  k = light(output.clone())

    - ``heavy``: geometry-heavy ops (crop/resize/affine) applied once
    - ``light``: photometric/lightweight ops applied twice *independently*

    This is like the "two augmented views" recipe from SimCLR,
    creates invariance across augmentations while keeping 'views' correlated.


    from: Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations"
    """
    def __init__(self, heavy: nn.Module, light: nn.Module):
        super().__init__()
        self.heavy = heavy
        self.light = light
    def forward(self, x):
        base = self.heavy(x)
        # two independent light branches (clone to avoid getting a pointer exception in python lol)
        q = self.light(base)
        k = self.light(base.clone())
        return q, k
    
class ContrastiveWrapper(Dataset):
    r"""
    Wrap a base (image, label) dataset to return two augmented views and the label, for contrastive:
        returns: (q, k, y)

    this also means you can use the same dataset for both online and pre-training.

    """

    def __init__(self, base_ds: Dataset, transform_twice: nn.Module):
        self.base = base_ds
        self.t2 = transform_twice
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        q,k = self.t2(x)
        return q, k , y
    
class HierSupConLoss(nn.Module):
    r"""
    Hierarchical Supervised Contrastive Loss

    Because im working with species but also genus dataset this extends supervised contrastive loss
    - same-species pairs as *full* positives
    - same-genus (but different species) pairs as *weak* positives, weighted by ``gamma``

    Given embeddings for two views per sample (V=2), this computes a (2B x 2B) similarity matrix,
    mask self-similarities, and average the log-softmax over positives with appropriate weights.

    args:
        temperature (float): Softmax temperature τ (lower → sharper similarities).
        gamma (float): Weight for genus-only positives relative to species positives.


    from: Khosla et al., "Supervised Contrastive Learning"
    """

    def __init__(self, temperature=0.07, gamma=0.3):
        super().__init__()
        self.t = float(temperature)
        self.gamma = float(gamma)

    def forward(self, z, y_species, y_genus):
        device = z.device
        B, V, D = z.shape

        # sanitize & cast
        z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
        if z.dtype != torch.float32:
            z = z.float()

        # pairwise logits over 2B embeddings    
        z = z.reshape(B * V, D).contiguous()
        logits = torch.matmul(z, z.T) / self.t
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        # remove self-similarities (diagonal) from consideration
        self_mask = torch.eye(B * V, device=device, dtype=torch.bool)
        logits = logits.masked_fill(self_mask, float("-inf"))

        # stabilize per row (log-sum-exp trick)
        row_max = logits.max(dim=1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        logits = logits - row_max
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)

        # build positive masks at species and genus levels
        # broadcast to 2B × 2B
        ys = y_species.reshape(B, 1)
        yg = y_genus.reshape(B, 1)
        sp = (ys == ys.T)
        gn = (yg == yg.T)

        sp = sp.repeat_interleave(V,0).repeat_interleave(V,1) & (~self_mask)
        gn = gn.repeat_interleave(V,0).repeat_interleave(V,1) & (~self_mask)
        # genus-only positives = same genus but different species
        gn_only = gn & (~sp)
        pos_w = sp.float() + self.gamma * gn_only.float()
        # log-softmax over rows
        log_denom = torch.logsumexp(logits, dim=1, keepdim=True)
        log_denom = torch.where(torch.isfinite(log_denom), log_denom, torch.zeros_like(log_denom))
        log_prob = logits - log_denom
        # weighted average of positive log-probs per anchor row
        pos_weight_sum = pos_w.sum(dim=1)
        safe = pos_weight_sum.clamp(min=1.0)
        loss = -(pos_w * log_prob).sum(dim=1) / safe
        #if a row had no positives (shouldn’t happen in supervised batches), drop it
        valid = (pos_weight_sum > 0).float()
        denom = valid.sum().clamp(min=1.0)
        out = (loss * valid).sum() / denom

        # one last numeric guard
        if not torch.isfinite(out):
            out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out
    
class HierBatchSampler(Sampler):
    """
    Each batch groups images by genus → species to ensure
    multiple same-species and same-genus pairs per step. sampling is with replacement
    produces batches of size = n_genus * species_per_genus * imgs_per_species
    """
    def __init__(self, species_ids, genus_ids, n_genus=16, species_per_genus=4, imgs_per_species=2, steps=800):
        self.s = np.asarray(species_ids, dtype=np.int64)
        self.g = np.asarray(genus_ids,   dtype=np.int64)
        # build nested dict: genus → species → list(indices)
        g2s = defaultdict(lambda: defaultdict(list))
        for i, (si, gi) in enumerate(zip(self.s, self.g)):
            g2s[int(gi)][int(si)].append(i)

        # freeze to numpy arrays for fast sampling
        self.g2s = {g: {s: np.asarray(ix, dtype=np.int64) for s, ix in sdict.items()} for g, sdict in g2s.items()}
        self.genera = np.fromiter(self.g2s.keys(), dtype=np.int64, count=len(self.g2s))

        # precompute species key arrays per genus
        self._species_keys = {g: np.fromiter(sdict.keys(), dtype=np.int64, count=len(sdict))
                              for g, sdict in self.g2s.items()}

        self.nG, self.Sg, self.M, self.steps = int(n_genus), int(species_per_genus), int(imgs_per_species), int(steps)

    def __iter__(self):
        rng = np.random.default_rng()
        G = len(self.genera)
        replace_g = (G < self.nG)
        for _ in range(self.steps):
            # choose genus (with replacement if not enough unique genera)
            g_chosen = rng.choice(self.genera, size=self.nG, replace=replace_g)
            batch = []
            for g in g_chosen:
                skeys = self._species_keys[g]
                replace_s = (len(skeys) < self.Sg)
                # choose species within this genus
                sp_chosen = rng.choice(skeys, size=self.Sg, replace=replace_s)
                for s in sp_chosen:
                    pool = self.g2s[g][s]
                    replace_m = (pool.shape[0] < self.M)
                    # choose images for that species
                    pick = rng.choice(pool, size=self.M, replace=replace_m)
                    batch.extend(pick.tolist())
            yield batch

    def __len__(self):
        return self.steps


class ClassBalancedBatchSampler(Sampler):
    """
    Sampler for supervised classification: each batch has C classes x M images/class = batch size.
    works with replacement so rare classes are fine.
    """
    def __init__(self, labels, classes_per_batch=32, images_per_class=4, steps=1000, seed=42):
        super().__init__(None)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.classes = np.unique(self.labels)
        self.C = int(classes_per_batch)
        self.M = int(images_per_class)
        self.steps = int(steps)
        self.rng = np.random.default_rng(seed)

        # Precompute index lists per class
        cls2ix = defaultdict(list)
        for i, c in enumerate(self.labels):
            cls2ix[int(c)].append(i)
        self.cls2ix = {c: np.asarray(ix, dtype=np.int64) for c, ix in cls2ix.items()}

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            # sample C classes (with replacement if needed)
            replace_c = (len(self.classes) < self.C)
            chosen = self.rng.choice(self.classes, size=self.C, replace=replace_c)
            batch = []
            for c in chosen:
                pool = self.cls2ix[int(c)]
                replace_m = (len(pool) < self.M)
                # Sample M images from this class (with replacement if needed)
                pick = self.rng.choice(pool, size=self.M, replace=replace_m)
                batch.extend(pick.tolist())
            yield batch