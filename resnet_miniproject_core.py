import torch
from torch import nn
from torch.utils.data import random_split, Dataset
import torch.nn.functional as F
import torchvision as tv
from torchvision import models
from torchvision import transforms as T, datasets as tvds
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import os, inspect, importlib, shutil
from typing import Optional, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger
import random
from torchmetrics.classification import (
    MulticlassConfusionMatrix,
    MulticlassAccuracy,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial
import modal

print = partial(print, flush=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)

seed_everything(42)
device = 'cuda' if torch.cuda.is_available() else "cpu"

# from PIL import Image
# import requests
# from io import BytesIO

# import matplotlib.pyplot as plt


# Ensures compatibility with most datasets
# transforms and small helpers

DATASET_CONFIG = {
    "tv:CIFAR10": {"in_channels": 3, "num_classes": 10},
    "tv:CIFAR100": {"in_channels": 3, "num_classes": 100},
    "tv:STL10": {"in_channels": 3, "num_classes": 10},
    "tv:MNIST": {"in_channels": 1, "num_classes": 10},
}

# Decide if images are grayscale (1 channel) or RGB (3 channels) so CNN first conv sized correctly
def infer_in_channels(sample):
    x, _ = sample
    return x.shape[0] if isinstance(x, torch.Tensor) else 3


# figures out how many classes dataset has (uses ds.classes if available; otherwise scans labels)
def infer_num_classes(ds: Dataset):
    # best case: dataset exposes classes explicitly
    if hasattr(ds, "classes"): 
        return len(ds.classes)
    
    # scan all labels once
    ys = torch.tensor([int(ds[i][1]) for i in range(len(ds))])
    unique = torch.unique(ys)
    K = int(unique.max().item() + 1)
    
    # safety check: labels should be 0..K-1
    if len(unique) != K:
        raise ValueError(
            f"Cannot safely infer num_classes: labels are {unique.tolist()}, "
            f"but that does not cover all indices from 0 to {K-1}. "
            "Please specify num_classes explicitly for this dataset"
        )
    
    return K

# builds train and eval transforms that handle resizing, optional augmentation, and normalization
def default_transforms(img_size: int, in_channels: int, augment: bool, mean=None, std=None):
    
    # forcing ResNet-compatible size
    target = 224 if img_size < 224 else img_size
    
    # prefer ImageNet stats + 3-channel inputs
    imnet_mean = (0.485, 0.456, 0.406)
    imnet_std = (0.229, 0.224, 0.225)
    mean = imnet_mean if (mean is None) else mean
    std = imnet_std if (std is None) else std

    ensure_rgb = []
    if in_channels == 1:
        # map grayscale to 3-channel. Do before ToTensor/Normalize
        ensure_rgb = [T.Grayscale(num_output_channels=3)]

    if augment:
        train_tfms = [
            T.Resize(256 if target >= 224 else target),
            T.RandomResizedCrop(target) if target >= 224 else T.Resize((target, target)),
            T.RandomHorizontalFlip(),
            *ensure_rgb,
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    else:
        train_tfms = [
            T.Resize(256 if target >= 224 else target),
            T.CenterCrop(target) if target >= 224 else T.Resize((target, target)),
            *ensure_rgb,
            T.ToTensor(),
            T.Normalize(mean, std),
        ]

    eval_tfms = [
        T.Resize(256 if target >= 224 else target),
        T.CenterCrop(target) if target >= 224 else T.Resize((target, target)),
        *ensure_rgb,
        T.ToTensor(),
        T.Normalize(mean, std),
    ]

    return T.Compose(train_tfms), T.Compose(eval_tfms)

# torchvision auto-instantiation (no per-dataset code)
# robustly tries multiple constructor signatures to build torchvision dataset split
# (because different datasets use train=... vs split=...)
def _instantiate_tv_split(cls, split: str, root: str, transform, download: bool):
    """
    Try common constructor patterns across torchvision datasets *without* knowing each dataset's quirks.
    """
    attempts = [
        lambda: cls(root=root, train=(split=='train'), transform=transform, download=download),
        lambda: cls(root=root, train=(split!='test'), transform=transform, download=download),
        lambda: cls(root=root, split=split, transform=transform, download=download),
        lambda: cls(root=root, split=split.upper(), transform=transform, download=download),
        lambda: cls(root=root, split=split.capitalize(), transform=transform, download=download),
    ]
    errors = []
    for fn in attempts:
        try:
            return fn()
        except Exception as e:
            errors.append(repr(e))
    raise RuntimeError(f"Could not construct {cls.__name__} for split='{split}'. Tried common signatures.\n" +
                       "Last errors:\n" + "\n- ".join(errors[-3:]))

# the dispatcher
def load_dataset_any(
        dataset_spec: str,
        root: str = "./data",
        img_size: int = 224,
        augment: bool = True,
        val_split: int = 5000,
        download: bool = True,
        mean = None,
        std = None,
        in_channels: int | None = None,
        num_classes: int | None = None,
):
    """
    dataset_spec:
     - 'tv:<ClassName>' for torchvision.datasets.<ClassName>
     - 'folder:/abs/or/relative/path' for ImageFolder-style dirs
     - 'hf:<dataset_name>' for huggingface datasets (optional dependency)
    Returns: (train_set, val_set, test_set, in_channels, num_classes)
    """
    # fill from DATASET_CONFIG if not provided
    cfg = DATASET_CONFIG.get(dataset_spec, {})
    if in_channels is None:
        in_channels = cfg.get("in_channels")
    if num_classes is None:
        num_classes = cfg.get("num_classes")

    if dataset_spec.startswith("folder:"):
        data_dir = dataset_spec.split("folder:", 1)[1]
        train_dir, val_dir, test_dir = [os.path.join(data_dir, d) for d in ("train", "val", "test")]
                                                                            
        # probe channels
        probe = tvds.ImageFolder(train_dir, transform=T.ToTensor())
        C_inferred = infer_in_channels(probe[0])
        C = in_channels if in_channels is not None else C_inferred

        tr_tfms, ev_tfms = default_transforms(img_size, C, augment)
        train_set = tvds.ImageFolder(train_dir, transform=tr_tfms)
        val_set = tvds.ImageFolder(val_dir, transform=ev_tfms) if os.path.isdir(val_dir) else None
        test_set = tvds.ImageFolder(test_dir, transform=ev_tfms) if os.path.isdir(test_dir) else None

        if val_set is None:
            n_val = min(val_split, max(1, len(train_set)//10))
            train_set, val_set = random_split(
                train_set, [len(train_set)-n_val, n_val],
                generator=torch.Generator().manual_seed(42)
            )
            val_set.dataset.transform = ev_tfms

        K_inferred = len(getattr(train_set, "dataset", train_set).classes)
        K = num_classes if num_classes is not None else K_inferred
        
        return train_set, val_set, test_set, C, K

    if dataset_spec.startswith("tv:"):
        name = dataset_spec.split("tv:", 1)[1]
        # Get class object from torchvision.datasets dynamically
        try:
            cls = getattr(tvds, name)
        except AttributeError:
            raise ValueError(f"torchvision.datasets has no class '{name}'.")
        
        gen = torch.Generator().manual_seed(42)

        # Probe channels with a minimal ToTensor transform
        probe = _instantiate_tv_split(cls, "train", root, T.ToTensor(), download)
        C_inferred = infer_in_channels(probe[0])
        C = in_channels if in_channels is not None else C_inferred
        tr_tfms, ev_tfms = default_transforms(img_size, C, augment)

        # create a "no-transform" dataset just to sample indices deterministically
        base_no_tfm = _instantiate_tv_split(cls, "train", root, transform=None, download=False)
        n_val = min(val_split, max(1, len(base_no_tfm)//10))
        perm = torch.randperm(len(base_no_tfm), generator=gen)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        # create two separate train datasets with different transforms
        train_ds = _instantiate_tv_split(cls, "train", root, transform=tr_tfms, download=False)
        val_ds = _instantiate_tv_split(cls, "train", root, transform=ev_tfms, download=False)

        train_set = Subset(train_ds, train_idx)
        val_set = Subset(val_ds, val_idx)

        # try to get test set; if it fails, None
        try:
            test_set = _instantiate_tv_split(cls, "test", root, transform=ev_tfms, download=False)
        except Exception:
            test_set = None

        K_inferred = infer_num_classes(train_ds)
        K = num_classes if num_classes is not None else K_inferred
        
        return train_set, val_set, test_set, C, K

    # Hugging Face datasets
    if dataset_spec.startswith("hf:"):
        # Optional path using Hugging Face 'datasets'
        try:
            import datasets as hfd
            from PIL import Image
        except ImportError as e:
            raise RuntimeError("Hugging Face 'datasets' not installed. Run 'pip install datasets pillow'.") from e
                               
        ds_name = dataset_spec.split("hf:", 1)[1]
        ds = hfd.load_dataset(ds_name)
        # expect splits named 'train' and 'test'. If no 'test', create val split from train
        has_test = "test" in ds
        train_hf = ds["train"]
        test_hf = ds["test"] if has_test else None

        # Probe channel count by loading one image
        sample = train_hf[0]["image"]
        C_inferred = 1 if (
            getattr(sample, "mode", None) in ("L", "1") or 
            (hasattr(sample, "getbands") and len(sample.getbands())==1)
        ) else 3
        C = in_channels if in_channels is not None else C_inferred

        tr_tfms, ev_tfms = default_transforms(img_size, C, augment)

        full_len = len(train_hf)
        n_val = min(5000, max(1, full_len//10))
        perm = torch.randperm(full_len, generator=torch.Generator().manual_seed(42))
        val_idx = perm[:n_val].tolist()
        train_idx = perm[n_val:].tolist()

        def to_pt(ds_split, tfm, indices=None):
            class HFWrapper(Dataset):
                def __init__(self, split, tfm, idxs=None): 
                    self.split = split
                    self.tfm = tfm
                    self.idxs = list(range(len(split))) if idxs is None else idxs
                def __len__(self): 
                    return len(self.idxs)
                def __getitem__(self, i):
                    ex = self.split[self.idxs[i]]
                    img = ex["image"] if isinstance(ex["image"], Image.Image) else Image.fromarray(ex["image"])
                    x = self.tfm(img)
                    y = int(ex["label"])
                    return x, y
            return HFWrapper(ds_split, tfm, indices)
        
        train_set = to_pt(train_hf, tr_tfms, train_idx)
        val_set = to_pt(train_hf, ev_tfms, val_idx)
        test_set = to_pt(ds["test"], ev_tfms) if "test" in ds else None

        # num_classes handling for HF datasets
        label_feat = train_hf.features["label"]

        if num_classes is not None:
            K = num_classes
        else:
            if hasattr(label_feat, "num_classes") and label_feat.num_classes is not None:
                K = label_feat.num_classes
            elif hasattr(label_feat, "names") and label_feat.names is not None:
                K = len(label_feat.names)
            else:
                # fallback to safe inference from wrapped PyTorch dataset
                K = infer_num_classes(train_set)

        return train_set, val_set, test_set, C, K

    raise ValueError("dataset_spec must start with one of 'tv:', 'folder:' or 'hf:'")

def mixup_data(x, y, alpha: float):
    """
    Classic mixup for single-label classfication (y is class indices, LongTensor).
    Returns mixed_x, y_a, y_b, lam.
    """
    if alpha is None or alpha <= 0:
        return x, y, y, 1.0 # no-op
    
    # sample from Beta distribution safely on GPU
    lam = float(np.random.beta(alpha, alpha))
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    # mixup images
    mixed_x = lam * x + (1 - lam) * x[index, :]

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, preds, y_a, y_b, lam: float):
    """Computes mixup loss between two sets of targets."""
    if lam == 1.0:
        return criterion(preds, y_a)
    return lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)

class AnyImageDM(pl.LightningDataModule):
    def __init__(self, dataset_spec: str, data_dir="./data", batch_size=64, num_workers=2,
                 img_size = 224, augment=True, val_split=5000, download=True):
        super().__init__()
        self.dataset_spec = dataset_spec
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # kwargs passed into auto-adapter
        self.kw = dict(
            root=data_dir,
            img_size=img_size,
            augment=augment,
            val_split=val_split,
            download=download,
        )
        
        # set in setup()
        self._C = None
        self._K = None
        self.train_set = None
        self.val_set = None
        self.test_set = None

    @property
    def in_channels(self):
        return self._C
    
    @property
    def num_classes(self): 
        return self._K
    
    # keep prepare_data empty because the auto-adapter may split the data;
    # we don't want to do that in prepare_data (which should be stateless & global)
    def prepare_data(self):
        pass

    ###############
    def setup(self, stage = None):
        if self.train_set is None:
            tr, va, te, C, K = load_dataset_any(self.dataset_spec, **self.kw)
            self.train_set, self.val_set, self.test_set = tr, va, te
            self._C, self._K = C, K
            # ensure we have validation set
            if self.val_set is None:
                raise RuntimeError("Validation split is None. Ensure load_dataset_any carves a val split.")
            
    def _loader(self, ds, shuffle: bool, drop_last: bool):
        return DataLoader(
            ds,
            batch_size = self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
            drop_last=drop_last,
            prefetch_factor=2 if self.num_workers > 0 else None
        )

    def train_dataloader(self):
        return self._loader(self.train_set, shuffle=True, drop_last=True)
        
    def val_dataloader(self):
        return self._loader(self.val_set, shuffle=False, drop_last=False)
                          
    def test_dataloader(self):
        if self.test_set is None:
            return None
        return self._loader(self.test_set, shuffle=False, drop_last=False)

class ImageClassifier(pl.LightningModule):
    def __init__(self, in_channels: int, num_classes: int,
                 lr: float = 1e-3, weight_decay: float = 5e-4,
                 mixup_alpha: float = 0.4, cosine_tmax_epochs: int = 20):
        super().__init__()
        self.save_hyperparameters(ignore=['model']) # e.g., lr=1e-3 for full fine-tuning; 1e-2 for final layer; weight_decay = 1e-4
        
        #Backbone
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        # load ResNet18
        net = models.resnet18(weights=weights)
        # replace final classification layer
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

        # Make inputs safe if not RGB
        self.input_adapter = nn.Identity() if in_channels == 3 else nn.Conv2d(in_channels, 3, kernel_size=1, bias=False)
        
        # loss
        self.criterion = nn.CrossEntropyLoss()
        
        # metrics
        self.train_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.train_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.val_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="macro")
        self.test_precision = MulticlassPrecision(num_classes=num_classes, average="macro")
        self.test_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.test_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.test_cm = MulticlassConfusionMatrix(num_classes=num_classes)
        
        # scheduler hyperparam
        self.cosine_tmax_epochs = cosine_tmax_epochs

    def forward(self, x):
        x = self.input_adapter(x)
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # apply mixup only if enabled
        if self.hparams.mixup_alpha and self.hparams.mixup_alpha > 0:
            x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=self.hparams.mixup_alpha)
            logits = self(x_mix)
            loss = mixup_criterion(self.criterion, logits, y_a, y_b, lam)
        else:
            logits = self(x)
            loss = self.criterion(logits, y)

        # predictions for metrics (using original hard labels y)
        preds = torch.argmax(logits, dim=1)

        # update training metrics
        self.train_precision(preds, y)
        self.train_recall(preds, y)
        self.train_f1(preds, y)

        # logging
        self.log(
            "train_loss", 
            loss,
            on_step=True,
            on_epoch=True, 
            prog_bar=True,
            batch_size = x.size(0),
            sync_dist=True,
        )
        
        # log metrics only on_epoch
        self.log(
            "train_precision",
            self.train_precision,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
            sync_dist=True,
        )

        self.log(
            "train_recall",
            self.train_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=x.size(0),
            sync_dist=True,
        )

        self.log(
            "train_f1",
            self.train_f1,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
            sync_dist=True,
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(1)

        # accumulate metrics
        self.val_acc.update(preds, y)
        self.val_precision.update(preds, y)
        self.val_recall.update(preds, y)
        self.val_f1.update(preds, y)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
            sync_dist=True,
        )
        
        return{"val_loss": loss}
        
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        prec = self.val_precision.compute()
        rec = self.val_recall.compute()
        f1 = self.val_f1.compute()

        # log
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        self.log("val_precision", prec, prog_bar=False, sync_dist=True)
        self.log("val_recall", rec, prog_bar=False, sync_dist=True)
        self.log("val_f1", f1, prog_bar=True, sync_dist=True)

        # reset for next epoch
        self.val_acc.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(1)

        #accumulate metrics
        self.test_acc.update(preds, y)
        self.test_precision.update(preds, y)
        self.test_recall.update(preds, y)
        self.test_f1.update(preds, y)
        self.test_cm.update(preds, y)

        self.log(
            "test_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            on_step=False,
            batch_size=x.size(0),
            sync_dist=True,
        )
        
        return {"test_loss": loss}
    
    def on_test_epoch_end(self):
        acc = self.test_acc.compute()
        cm = self.test_cm.compute()
        prec = self.test_precision.compute()
        rec = self.test_recall.compute()
        f1 = self.test_f1.compute()

        self.log("test_acc", acc, prog_bar=True, sync_dist=True)
        self.log("test_precision", prec, prog_bar=False, sync_dist=True)
        self.log("test_recall", rec, prog_bar=False, sync_dist=True)
        self.log("test_f1", f1, prog_bar=True, sync_dist=True)

        self.print(f"Test Accuracy: {acc:.4f}")
        self.print(f"Test Precision: {prec:.4f}")
        self.print(f"Test Recall: {rec:.4f}")
        self.print(f"Test F1: {f1:.4f}")
        self.print(f"Confusion Matrix Shape: {tuple(cm.shape)}")
        
        try:
            fig = plt.figure()
            plt.imshow(cm.cpu().numpy(), interpolation="nearest")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.colorbar()
            if self.logger is not None:
                self.logger.experiment.add_figure("confusion_matrix", fig, global_step = self.current_epoch)
            plt.close(fig)
        except Exception:
            pass

        self.test_acc.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_cm.reset()
            
    def configure_optimizers(self):
        opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cosine_tmax_epochs)
        return {"optimizer": opt, "lr_scheduler": sch}
    
def train_and_export_checkpoint(
        dataset_spec: str,
        mixup_alpha: float,
        run_tag: str,
        max_epochs: int = 20,
        batch_size: int = 128,
        img_size: int = 224,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        export_dir: str = "user_interface",
):
    """
    Train ImageClassifier on a given dataset with a given mixup_alpha, 
    then copy the best checkpoint into user_interface/ as <run_tag>.ckpt.

    Args:
        dataset_spec: e.g. "tv:CIFAR10" or "tv:CIFAR100"
        mixup_alpha: 0.0 for no mixup, >0 for mixup (e.g., 0.4)
        run_tag: short name for the run, used to name the exported ckpt
                e.g., "cifar10_nomix", "cifar10_mixup"
    """

    # datamodule
    dm = AnyImageDM(
        dataset_spec=dataset_spec,
        data_dir="./data",
        batch_size=batch_size,
        img_size=img_size,
        augment=True,
        val_split=5000,
        download=True,
    )
    dm.prepare_data()
    dm.setup()

    # model
    model = ImageClassifier(
        in_channels=dm.in_channels,
        num_classes=dm.num_classes,
        mixup_alpha=mixup_alpha,
        lr=lr,
        weight_decay=weight_decay,
        cosine_tmax_epochs=max_epochs,
    )

    # callbacks
    ckpt_cb = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    early_cb = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=5,
    )
    lrmon_cb = LearningRateMonitor(logging_interval="epoch")

    # Logger
    logger = CSVLogger(
        save_dir=".",
        name=f"lightning_logs_{run_tag}",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        callbacks=[ckpt_cb, early_cb, lrmon_cb],
        logger=logger,
        enable_progress_bar=True,
        enable_model_summary=False,
        log_every_n_steps=10,
        deterministic=False,
    )

    # fit
    trainer.fit(model, datamodule=dm)

    # find best checkpoint
    best_path = ckpt_cb.best_model_path
    print(f"[{run_tag}] Best checkpoint (original):", best_path)

    # copy to user_interface/<run_tag>.ckpt
    os.makedirs(export_dir, exist_ok=True)

    export_path = os.path.join(export_dir, f"{run_tag}.ckpt")
    shutil.copy(best_path, export_path)

    print(f"[{run_tag}] Exported checkpoint to: {export_path}")
    return export_path

