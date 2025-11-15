import pytorch_lightning as pl

# For plotting validation reconstructions
import torchvision.utils as vutils
import torch
import pytorch_lightning as pl
from torch.optim import AdamW

from Utils.utils import build_param_groups
from Models.mae import vit_base_patch16, vit_large_patch16, vit_huge_patch14

import math
from typing import Optional, List

import torch
import torch.nn as nn
import lightning as pl
from torch.optim import AdamW

from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    MultilabelAUROC,
    MultilabelAccuracy,
    MultilabelPrecision,
    MultilabelRecall,
    MultilabelF1Score,
)

from Models.models import CXRModel

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

class ClassificationLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_mode: str = "imagenet",               # "imagenet" or "mae"
        model_weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
        pos_weight: Optional[torch.Tensor] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10,
        betas=(0.9, 0.95),
        class_names: Optional[List[str]] = None,
        backbone_name: str = "vit_base_patch16_224",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.is_binary = num_classes == 1

        # backbone + head
        self.model = CXRModel(
            num_classes=num_classes,
            mode=model_mode,
            backbone_name=backbone_name,
            model_weights=model_weights_path,
            freeze_backbone=freeze_backbone,
        )

        # Loss (same for binary & multilabel)
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        else:
            self.pos_weight = None
            self.criterion = nn.BCEWithLogitsLoss()

        # Class names
        if class_names is None:
            if self.is_binary:
                self.class_names = ["class_0"]
            else:
                self.class_names = [f"class_{i}" for i in range(num_classes)]
        else:
            assert len(class_names) == num_classes
            self.class_names = class_names

        # ---------------- METRICS ----------------
        if self.is_binary:
            # Binary metrics: scalar only (no per-class concept)
            # train
            self.train_auroc = BinaryAUROC()
            self.train_acc = BinaryAccuracy()
            self.train_prec = BinaryPrecision()
            self.train_rec = BinaryRecall()
            self.train_f1 = BinaryF1Score()
            # val
            self.val_auroc = BinaryAUROC()
            self.val_acc = BinaryAccuracy()
            self.val_prec = BinaryPrecision()
            self.val_rec = BinaryRecall()
            self.val_f1 = BinaryF1Score()
            # test
            self.test_auroc = BinaryAUROC()
            self.test_acc = BinaryAccuracy()
            self.test_prec = BinaryPrecision()
            self.test_rec = BinaryRecall()
            self.test_f1 = BinaryF1Score()
        else:
            # Multilabel global (macro)
            # train
            self.train_auroc_macro = MultilabelAUROC(
                num_labels=num_classes, average="macro"
            )
            self.train_acc_macro = MultilabelAccuracy(
                num_labels=num_classes, average="macro"
            )
            self.train_prec_macro = MultilabelPrecision(
                num_labels=num_classes, average="macro"
            )
            self.train_rec_macro = MultilabelRecall(
                num_labels=num_classes, average="macro"
            )
            self.train_f1_macro = MultilabelF1Score(
                num_labels=num_classes, average="macro"
            )
            # train local (per class)
            self.train_auroc_local = MultilabelAUROC(
                num_labels=num_classes, average=None
            )
            self.train_acc_local = MultilabelAccuracy(
                num_labels=num_classes, average=None
            )
            self.train_prec_local = MultilabelPrecision(
                num_labels=num_classes, average=None
            )
            self.train_rec_local = MultilabelRecall(
                num_labels=num_classes, average=None
            )
            self.train_f1_local = MultilabelF1Score(
                num_labels=num_classes, average=None
            )

            # val global
            self.val_auroc_macro = MultilabelAUROC(
                num_labels=num_classes, average="macro"
            )
            self.val_acc_macro = MultilabelAccuracy(
                num_labels=num_classes, average="macro"
            )
            self.val_prec_macro = MultilabelPrecision(
                num_labels=num_classes, average="macro"
            )
            self.val_rec_macro = MultilabelRecall(
                num_labels=num_classes, average="macro"
            )
            self.val_f1_macro = MultilabelF1Score(
                num_labels=num_classes, average="macro"
            )
            # val local
            self.val_auroc_local = MultilabelAUROC(
                num_labels=num_classes, average=None
            )
            self.val_acc_local = MultilabelAccuracy(
                num_labels=num_classes, average=None
            )
            self.val_prec_local = MultilabelPrecision(
                num_labels=num_classes, average=None
            )
            self.val_rec_local = MultilabelRecall(
                num_labels=num_classes, average=None
            )
            self.val_f1_local = MultilabelF1Score(
                num_labels=num_classes, average=None
            )

            # test global
            self.test_auroc_macro = MultilabelAUROC(
                num_labels=num_classes, average="macro"
            )
            self.test_acc_macro = MultilabelAccuracy(
                num_labels=num_classes, average="macro"
            )
            self.test_prec_macro = MultilabelPrecision(
                num_labels=num_classes, average="macro"
            )
            self.test_rec_macro = MultilabelRecall(
                num_labels=num_classes, average="macro"
            )
            self.test_f1_macro = MultilabelF1Score(
                num_labels=num_classes, average="macro"
            )
            # test local
            self.test_auroc_local = MultilabelAUROC(
                num_labels=num_classes, average=None
            )
            self.test_acc_local = MultilabelAccuracy(
                num_labels=num_classes, average=None
            )
            self.test_prec_local = MultilabelPrecision(
                num_labels=num_classes, average=None
            )
            self.test_rec_local = MultilabelRecall(
                num_labels=num_classes, average=None
            )
            self.test_f1_local = MultilabelF1Score(
                num_labels=num_classes, average=None
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        imgs, targets = batch
        logits = self(imgs)

        if self.is_binary:
            # logits: [B,1] or [B] -> [B]
            logits = logits.view(-1)
            targets = targets.float().view(-1)
        else:
            # multilabel: [B,C]
            targets = targets.float()

        loss = self.criterion(logits, targets)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        # -------- metrics update --------
        if self.is_binary:
            if stage == "train":
                self.train_auroc.update(probs, targets.int())
                self.train_acc.update(preds, targets.int())
                self.train_prec.update(preds, targets.int())
                self.train_rec.update(preds, targets.int())
                self.train_f1.update(preds, targets.int())
            elif stage == "val":
                self.val_auroc.update(probs, targets.int())
                self.val_acc.update(preds, targets.int())
                self.val_prec.update(preds, targets.int())
                self.val_rec.update(preds, targets.int())
                self.val_f1.update(preds, targets.int())
            elif stage == "test":
                self.test_auroc.update(probs, targets.int())
                self.test_acc.update(preds, targets.int())
                self.test_prec.update(preds, targets.int())
                self.test_rec.update(preds, targets.int())
                self.test_f1.update(preds, targets.int())
        else:
            if stage == "train":
                self.train_auroc_macro.update(probs, targets.int())
                self.train_acc_macro.update(preds, targets.int())
                self.train_prec_macro.update(preds, targets.int())
                self.train_rec_macro.update(preds, targets.int())
                self.train_f1_macro.update(preds, targets.int())

                self.train_auroc_local.update(probs, targets.int())
                self.train_acc_local.update(preds, targets.int())
                self.train_prec_local.update(preds, targets.int())
                self.train_rec_local.update(preds, targets.int())
                self.train_f1_local.update(preds, targets.int())
            elif stage == "val":
                self.val_auroc_macro.update(probs, targets.int())
                self.val_acc_macro.update(preds, targets.int())
                self.val_prec_macro.update(preds, targets.int())
                self.val_rec_macro.update(preds, targets.int())
                self.val_f1_macro.update(preds, targets.int())

                self.val_auroc_local.update(probs, targets.int())
                self.val_acc_local.update(preds, targets.int())
                self.val_prec_local.update(preds, targets.int())
                self.val_rec_local.update(preds, targets.int())
                self.val_f1_local.update(preds, targets.int())
            elif stage == "test":
                self.test_auroc_macro.update(probs, targets.int())
                self.test_acc_macro.update(preds, targets.int())
                self.test_prec_macro.update(preds, targets.int())
                self.test_rec_macro.update(preds, targets.int())
                self.test_f1_macro.update(preds, targets.int())

                self.test_auroc_local.update(probs, targets.int())
                self.test_acc_local.update(preds, targets.int())
                self.test_prec_local.update(preds, targets.int())
                self.test_rec_local.update(preds, targets.int())
                self.test_f1_local.update(preds, targets.int())

        return loss

    # ---------- TRAIN ----------
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="train")
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        if self.is_binary:
            auroc = self.train_auroc.compute()
            acc = self.train_acc.compute()
            prec = self.train_prec.compute()
            rec = self.train_rec.compute()
            f1 = self.train_f1.compute()

            self.log("train/auroc", auroc, prog_bar=True, sync_dist=True)
            self.log("train/acc", acc, sync_dist=True)
            self.log("train/precision", prec, sync_dist=True)
            self.log("train/recall", rec, sync_dist=True)
            self.log("train/f1", f1, sync_dist=True)

            self.train_auroc.reset()
            self.train_acc.reset()
            self.train_prec.reset()
            self.train_rec.reset()
            self.train_f1.reset()
        else:
            # global
            auroc = self.train_auroc_macro.compute()
            acc = self.train_acc_macro.compute()
            prec = self.train_prec_macro.compute()
            rec = self.train_rec_macro.compute()
            f1 = self.train_f1_macro.compute()

            self.log("train/auroc_macro", auroc, prog_bar=True, sync_dist=True)
            self.log("train/acc_macro", acc, sync_dist=True)
            self.log("train/precision_macro", prec, sync_dist=True)
            self.log("train/recall_macro", rec, sync_dist=True)
            self.log("train/f1_macro", f1, sync_dist=True)

            # per-class
            auroc_local = self.train_auroc_local.compute()
            acc_local = self.train_acc_local.compute()
            prec_local = self.train_prec_local.compute()
            rec_local = self.train_rec_local.compute()
            f1_local = self.train_f1_local.compute()

            for i, cls in enumerate(self.class_names):
                self.log(f"train/auroc_{cls}", auroc_local[i], sync_dist=True)
                self.log(f"train/acc_{cls}", acc_local[i], sync_dist=True)
                self.log(f"train/precision_{cls}", prec_local[i], sync_dist=True)
                self.log(f"train/recall_{cls}", rec_local[i], sync_dist=True)
                self.log(f"train/f1_{cls}", f1_local[i], sync_dist=True)

            # reset
            self.train_auroc_macro.reset()
            self.train_acc_macro.reset()
            self.train_prec_macro.reset()
            self.train_rec_macro.reset()
            self.train_f1_macro.reset()

            self.train_auroc_local.reset()
            self.train_acc_local.reset()
            self.train_prec_local.reset()
            self.train_rec_local.reset()
            self.train_f1_local.reset()

    # ---------- VAL ----------
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="val")
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_validation_epoch_end(self):
        if self.is_binary:
            auroc = self.val_auroc.compute()
            acc = self.val_acc.compute()
            prec = self.val_prec.compute()
            rec = self.val_rec.compute()
            f1 = self.val_f1.compute()

            self.log("val/auroc", auroc, prog_bar=True, sync_dist=True)
            self.log("val/acc", acc, sync_dist=True)
            self.log("val/precision", prec, sync_dist=True)
            self.log("val/recall", rec, sync_dist=True)
            self.log("val/f1", f1, sync_dist=True)

            self.val_auroc.reset()
            self.val_acc.reset()
            self.val_prec.reset()
            self.val_rec.reset()
            self.val_f1.reset()
        else:
            # global
            auroc = self.val_auroc_macro.compute()
            acc = self.val_acc_macro.compute()
            prec = self.val_prec_macro.compute()
            rec = self.val_rec_macro.compute()
            f1 = self.val_f1_macro.compute()

            self.log("val/auroc_macro", auroc, prog_bar=True, sync_dist=True)
            self.log("val/acc_macro", acc, sync_dist=True)
            self.log("val/precision_macro", prec, sync_dist=True)
            self.log("val/recall_macro", rec, sync_dist=True)
            self.log("val/f1_macro", f1, sync_dist=True)

            # per-class
            auroc_local = self.val_auroc_local.compute()
            acc_local = self.val_acc_local.compute()
            prec_local = self.val_prec_local.compute()
            rec_local = self.val_rec_local.compute()
            f1_local = self.val_f1_local.compute()

            for i, cls in enumerate(self.class_names):
                self.log(f"val/auroc_{cls}", auroc_local[i], sync_dist=True)
                self.log(f"val/acc_{cls}", acc_local[i], sync_dist=True)
                self.log(f"val/precision_{cls}", prec_local[i], sync_dist=True)
                self.log(f"val/recall_{cls}", rec_local[i], sync_dist=True)
                self.log(f"val/f1_{cls}", f1_local[i], sync_dist=True)

            self.val_auroc_macro.reset()
            self.val_acc_macro.reset()
            self.val_prec_macro.reset()
            self.val_rec_macro.reset()
            self.val_f1_macro.reset()

            self.val_auroc_local.reset()
            self.val_acc_local.reset()
            self.val_prec_local.reset()
            self.val_rec_local.reset()
            self.val_f1_local.reset()

    # ---------- TEST ----------
    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="test")
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        return loss

    def on_test_epoch_end(self):
        if self.is_binary:
            auroc = self.test_auroc.compute()
            acc = self.test_acc.compute()
            prec = self.test_prec.compute()
            rec = self.test_rec.compute()
            f1 = self.test_f1.compute()

            self.log("test/auroc", auroc, sync_dist=True)
            self.log("test/acc", acc, sync_dist=True)
            self.log("test/precision", prec, sync_dist=True)
            self.log("test/recall", rec, sync_dist=True)
            self.log("test/f1", f1, sync_dist=True)

            self.test_auroc.reset()
            self.test_acc.reset()
            self.test_prec.reset()
            self.test_rec.reset()
            self.test_f1.reset()
        else:
            # global
            auroc = self.test_auroc_macro.compute()
            acc = self.test_acc_macro.compute()
            prec = self.test_prec_macro.compute()
            rec = self.test_rec_macro.compute()
            f1 = self.test_f1_macro.compute()

            self.log("test/auroc_macro", auroc, sync_dist=True)
            self.log("test/acc_macro", acc, sync_dist=True)
            self.log("test/precision_macro", prec, sync_dist=True)
            self.log("test/recall_macro", rec, sync_dist=True)
            self.log("test/f1_macro", f1, sync_dist=True)

            # per-class
            auroc_local = self.test_auroc_local.compute()
            acc_local = self.test_acc_local.compute()
            prec_local = self.test_prec_local.compute()
            rec_local = self.test_rec_local.compute()
            f1_local = self.test_f1_local.compute()

            for i, cls in enumerate(self.class_names):
                self.log(f"test/auroc_{cls}", auroc_local[i], sync_dist=True)
                self.log(f"test/acc_{cls}", acc_local[i], sync_dist=True)
                self.log(f"test/precision_{cls}", prec_local[i], sync_dist=True)
                self.log(f"test/recall_{cls}", rec_local[i], sync_dist=True)
                self.log(f"test/f1_{cls}", f1_local[i], sync_dist=True)

            self.test_auroc_macro.reset()
            self.test_acc_macro.reset()
            self.test_prec_macro.reset()
            self.test_rec_macro.reset()
            self.test_f1_macro.reset()

            self.test_auroc_local.reset()
            self.test_acc_local.reset()
            self.test_prec_local.reset()
            self.test_rec_local.reset()
            self.test_f1_local.reset()

    # ---------- OPTIMIZER ----------
    def configure_optimizers(self):
        # If you have custom param groups, swap this out:
        # params = build_param_groups(self.model, self.hparams.weight_decay)
        params = self.model.parameters()

        optimizer = AdamW(
            params,
            lr=self.hparams.lr,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(current_epoch):
            # linear warmup
            if current_epoch < self.hparams.warmup_epochs:
                return float(current_epoch + 1) / float(
                    max(1, self.hparams.warmup_epochs)
                )
            # cosine decay
            total = max(self.trainer.max_epochs - self.hparams.warmup_epochs, 1)
            progress = (current_epoch - self.hparams.warmup_epochs) / total
            return 0.5 * (1.0 + math.cos(progress * math.pi))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
