import os
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# For plotting validation reconstructions
import torchvision.utils as vutils
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
import torch.nn as nn

from mae import vit_base_patch16, vit_large_patch16, vit_huge_patch14  # your class
from pytorch_lightning.loggers import WandbLogger

# =============================================================================
# PART 1: USER-PROVIDED DATASET CLASS
# (I've included it here for a single-file runnable script)
# =============================================================================

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None,
                 with_labels=True, use_no_finding=True):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.with_labels = with_labels

        # Label columns
        self.label_cols = [
            "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
            "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
            "Atelectasis", "Pneumothorax", "Pleural Effusion",
            "Pleural Other", "Fracture", "Support Devices", "No Finding",
        ]
        if not use_no_finding:
            self.label_cols.remove("No Finding")

        self.df[self.label_cols] = self.df[self.label_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        self.df[self.label_cols] = self.df[self.label_cols].fillna(0.0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["Path"])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if not self.with_labels:
            # For MAE, we only need the image
            return image

        labels_np = row[self.label_cols].to_numpy(dtype="float32")
        labels = torch.from_numpy(labels_np)

        return image, labels

# =============================================================================
# PART 2: THE PYTORCH LIGHTNING DATAMODULE
# =============================================================================

class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, val_csv, root_dir, 
                 batch_size=64, num_workers=4, image_size=224):
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        
        # MAE uses simple augmentations: Resize, Horizontal Flip, ToTensor, Normalize
        self.transform = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        # IMPORTANT: For SSL (MAE), we set with_labels=False
        self.train_dataset = ChestXrayDataset(
            csv_file=self.train_csv,
            root_dir=self.root_dir + "/train",
            transform=self.transform,
            with_labels=False 
        )
        self.val_dataset = ChestXrayDataset(
            csv_file=self.val_csv,
            root_dir=self.root_dir + "/valid",
            transform=self.transform,
            with_labels=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            persistent_workers=True,
            pin_memory=True
        )

batch_size = 256
data_module = CheXpertDataModule(
    train_csv="./data/train_split.csv",
    val_csv="./data/val_split.csv",
    root_dir="/users/yliu802/.cache/kagglehub/datasets/ashery/chexpert/versions/1",
    batch_size=batch_size,
    num_workers=12,
    image_size=224
)

# test the datamodule
data_module.setup()
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()

print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")

# Fetch a batch to inspect
images = next(iter(train_loader))
print(f"Batch shape: {images.shape}")

def build_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias"):  # layernorm/embedding/bias => no decay
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

class MAELightningModule(pl.LightningModule):
    def __init__(self,
                 size = "base",
                 norm_pix_loss=False,
                 mask_ratio: float = 0.75,
                 lr: float = 1.5e-4,
                 weight_decay: float = 0.05,
                 betas: tuple = (0.9, 0.95),
                 warmup_epochs: int = 10,
                 log_images_every_n_epochs: int = 10,
                 log_max_images: int = 8):
        super().__init__()
        self.save_hyperparameters()

        if size == "base":
            self.model = vit_base_patch16(norm_pix_loss=norm_pix_loss)
        elif size == "large":
            self.model = vit_large_patch16(norm_pix_loss=norm_pix_loss)
        elif size == "huge":
            self.model = vit_huge_patch14(norm_pix_loss=norm_pix_loss)
        else:
            raise ValueError(f"Unknown MAE size: {size}")

    def forward_loss(self, imgs, pred, mask):
        target = self.model.patchify(imgs)
        if self.hparams.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6).sqrt()
        loss = (pred - target).pow(2).mean(dim=-1)               # [N, L]
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)         # masked-only
        return loss

    def forward(self, imgs):
        return self.model(imgs, mask_ratio=self.hparams.mask_ratio)

    def training_step(self, batch, batch_idx):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        _, pred, mask = self(imgs)
        loss = self.forward_loss(imgs, pred, mask)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
        _, pred, mask = self(imgs)
        loss = self.forward_loss(imgs, pred, mask)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.size(0), sync_dist=True)

        # optional: log reconstructions on first batch every N epochs
        if (
            batch_idx == 0
            and self.hparams.log_images_every_n_epochs > 0
            and (self.current_epoch % self.hparams.log_images_every_n_epochs == 0)
        ):
            self._log_reconstructions(imgs, pred)
        return loss

    # ----- helpers -----
    @torch.no_grad()
    def _log_reconstructions(self, imgs: torch.Tensor, pred: torch.Tensor, tag: str = "reconstructions"):
        """
        Log a grid: top = ground truth, bottom = MAE reconstruction.
        Works with TensorBoard or Weights & Biases if attached as Lightning loggers.

        Args:
            imgs: [B,3,H,W] input batch in normalized space (e.g., ImageNet stats).
            pred: MAE decoder outputs in the same normalized space as the training target.
            tag:  image tag/name for the logger.
        """
        if imgs.ndim != 4 or imgs.size(1) != 3:
            return  # nothing to do; only handle RGB 4D tensors

        # MAE outputs normalized pixels; clamp gently to avoid extreme values in display
        rec = self.model.unpatchify(pred).clamp(-3, 3)

        n = int(getattr(self.hparams, "log_max_images", 8))
        n = max(1, min(n, imgs.size(0)))

        # de-normalization toward [0,1]; keep on same device then cast/move right before grid
        def denorm(x: torch.Tensor) -> torch.Tensor:
            mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
            return (x * std + mean).clamp(0, 1)

        gt_vis  = denorm(imgs[:n])
        rec_vis = denorm(rec[:n])

        # stack into two rows: first row GT, second row REC
        grid_src = torch.cat([gt_vis, rec_vis], dim=0).detach().float().cpu()
        # ensure uint8 for image loggers when converting to numpy
        grid = vutils.make_grid(grid_src, nrow=n, padding=2, pad_value=0.5)

        logger = getattr(self, "logger", None)
        if logger is None or getattr(logger, "experiment", None) is None:
            return

        exp = logger.experiment
        step = int(getattr(self, "global_step", 0))

        # TensorBoard
        if hasattr(exp, "add_image"):
            exp.add_image(tag, grid, global_step=step)
            return

        # Weights & Biases
        if hasattr(exp, "log"):
            try:
                import numpy as np
                np_img = (grid.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
                import wandb
                exp.log({tag: wandb.Image(np_img)}, step=step)
            except Exception:
                pass

    def configure_optimizers(self):
        optimizer = AdamW(
            build_param_groups(self.model, self.hparams.weight_decay),
            lr=self.hparams.lr, betas=self.hparams.betas
        )

        # Warmup + cosine to zero across epochs
        # Use LambdaLR so T_max = total_epochs is implicit
        def lr_lambda(current_epoch):
            # linear warmup
            if current_epoch < self.hparams.warmup_epochs:
                return float(current_epoch + 1) / float(max(1, self.hparams.warmup_epochs))
            # cosine
            total = max(self.trainer.max_epochs - self.hparams.warmup_epochs, 1)
            progress = (current_epoch - self.hparams.warmup_epochs) / total
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

logger = WandbLogger(project="mae-cxr")

model = MAELightningModule(
    size = "base",
    mask_ratio = 0.75,
    lr = 1.5e-4,
    weight_decay = 0.05,
    betas = (0.9, 0.95),
    warmup_epochs = 10,
    log_images_every_n_epochs = 10,
    log_max_images = 8
)

trainer = pl.Trainer(
    max_epochs=300,
    precision="16-mixed",   # AMP
    logger=logger,
    gradient_clip_val=0.0,
    deterministic=False,
    check_val_every_n_epoch=1,
    accelerator="gpu",
    devices=2,
    num_nodes=1,
    strategy="auto",
)

trainer.fit(model, datamodule=data_module)
trainer.save_checkpoint("./checkpoints/mae_cxr_base.ckpt")