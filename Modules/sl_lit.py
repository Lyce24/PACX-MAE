
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

import torch
import lightning as pl
from torch.optim import AdamW

import math
from typing import Optional, List
import torch.nn as nn

class ClassificationLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        model_mode: str = "imagenet",
        model_weights_path: Optional[str] = None,
        freeze_backbone: bool = True,
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
            model_checkpoints=model_weights_path,
            freeze_backbone=freeze_backbone,
        )

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
            if stage == "val":
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
            if stage == "val":
                self.val_auroc_macro.update(probs, targets.int())
                self.val_acc_macro.update(preds, targets.int())
                self.val_prec_macro.update(preds, targets.int())
                self.val_rec_macro.update(preds, targets.int())
                self.val_f1_macro.update(preds, targets.int())

            elif stage == "test":
                self.test_auroc_macro.update(probs, targets.int())
                self.test_acc_macro.update(preds, targets.int())
                self.test_prec_macro.update(preds, targets.int())
                self.test_rec_macro.update(preds, targets.int())
                self.test_f1_macro.update(preds, targets.int())

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

            self.val_auroc_macro.reset()
            self.val_acc_macro.reset()
            self.val_prec_macro.reset()
            self.val_rec_macro.reset()
            self.val_f1_macro.reset()

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

            self.test_auroc_macro.reset()
            self.test_acc_macro.reset()
            self.test_prec_macro.reset()
            self.test_rec_macro.reset()
            self.test_f1_macro.reset()

    # ---------- OPTIMIZER ----------
    def configure_optimizers(self):
        # only the linear head should be trainable anyway
        params = [p for p in self.model.parameters() if p.requires_grad]

        optimizer = AdamW(
            params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=self.hparams.betas,
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

