"""
Cross-Modal Knowledge Distillation for CXR Encoder

Goal: Fine-tune MAE ViT (CXR) so its embeddings capture physiological 
information from frozen Labs and ECG encoders.

Training: CXR + ECG + Labs available
Inference: CXR only → embeddings contain "phantom" physiology knowledge

Architecture:
- MAE ViT (768 dim) - TRAINABLE
- Labs Encoder (256 dim) - FROZEN
- ECG Encoder (1024 dim) - FROZEN

Losses:
- CLIP loss - align CXR with ECG/Labs in shared space
- Regression loss - predict ECG/Labs embeddings from CXR
- Regularization Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchvision.transforms as T
from typing import Optional, Dict, Tuple, List
from torchmetrics import MeanMetric, CosineSimilarity
import math
import torch.distributed as dist  # <-- added (no new functions)
from peft import LoraConfig, get_peft_model

# Assuming this import exists in your codebase
from Models.models import MAECXREncoder
import numpy as np

# ============================================================================
# Projection Heads (Discarded at Inference)
# ============================================================================

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.mlp(x)

# ============================================================================
# Loss Module
# ============================================================================

class CLIPRegressionLoss(nn.Module):
    """
    Combined CLIP + Regression loss for cross-modal distillation.
    
    CLIP Loss: Contrastive alignment in shared embedding space
    Regression Loss: Direct prediction of target embeddings
    """
    
    def __init__(
        self,
        clip_weight: float = 1.0,
        regression_weight: float = 0.1,
        temperature: float = 0.1,
        use_cosine_regression: bool = True,
        label_smoothing: float = 0.1,

    ):
        super().__init__()
        self.clip_weight = clip_weight
        self.regression_weight = regression_weight
        self.temperature = temperature
        self.use_cosine_regression = use_cosine_regression
        self.label_smoothing=label_smoothing
        
    #     # Learnable temperature (optional, often helps)
    #     self.log_temp = nn.Parameter(torch.log(torch.tensor(temperature)))
    
    # @property
    # def current_temperature(self) -> float:
    #     return self.log_temp.exp().clamp(min=0.01, max=1.0)
    
    def clip_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Symmetric InfoNCE with Soft Targets (Label Smoothing).
        """
        z1 = F.normalize(z1, p=2, dim=-1, eps=1e-6)
        z2 = F.normalize(z2, p=2, dim=-1, eps=1e-6)
        
        # Gather negatives if DDP
        if dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            rank = dist.get_rank()
            
            # Gather z2
            z2_list = [torch.zeros_like(z2) for _ in range(world)]
            dist.all_gather(z2_list, z2)
            z2_all = torch.cat(z2_list, dim=0) # [world*B, D]
            
            # Gather z1
            z1_list = [torch.zeros_like(z1) for _ in range(world)]
            dist.all_gather(z1_list, z1)
            z1_all = torch.cat(z1_list, dim=0) # [world*B, D]
        else:
            rank = 0
            z2_all = z2
            z1_all = z1

        # Compute Logits
        # z1 -> [B, D], z2_all -> [Total_B, D] = [B, Total_B]
        logits_1 = (z1 @ z2_all.t()) / self.temperature
        logits_2 = (z2 @ z1_all.t()) / self.temperature

        B = z1.size(0)
        Total_B = z2_all.size(0)
        device = z1.device

        # ---- Create Soft Targets ----
        # The positive match for local row 'i' is at index 'rank*B + i' in the gathered buffer
        targets = torch.arange(B, device=device) + rank * B
        
        # If label smoothing is on, we manually construct the target distribution
        if self.label_smoothing > 0:
            # Create a matrix of [B, Total_B]
            # 1. Fill with smoothing factor distributed across non-targets
            # Note: A strict mathematical implementation distributes smoothing over K-1 classes.
            # Here simplified to uniform noise over all + spike on target.
            
            # Standard CrossEntropy with label_smoothing support (PyTorch > 1.10)
            loss_1 = F.cross_entropy(logits_1, targets, label_smoothing=self.label_smoothing)
            loss_2 = F.cross_entropy(logits_2, targets, label_smoothing=self.label_smoothing)
        else:
            loss_1 = F.cross_entropy(logits_1, targets)
            loss_2 = F.cross_entropy(logits_2, targets)

        return 0.5 * (loss_1 + loss_2)

    def regression_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_norm = F.normalize(pred, p=2, dim=-1, eps=1e-6)
        target_norm = F.normalize(target.detach(), p=2, dim=-1, eps=1e-6)
        cosine_sim = (pred_norm * target_norm).sum(dim=-1).mean()
        return 1.0 - cosine_sim
    
    def forward(
        self,
        cxr_shared: torch.Tensor,      # CXR projected to shared space
        ecg_shared: torch.Tensor,      # ECG projected to shared space
        labs_shared: torch.Tensor,     # Labs projected to shared space
        cxr_to_ecg_pred: torch.Tensor, # CXR regression prediction for ECG
        cxr_to_labs_pred: torch.Tensor,# CXR regression prediction for Labs
        ecg_target: torch.Tensor,      # Frozen ECG embeddings (target)
        labs_target: torch.Tensor,     # Frozen Labs embeddings (target)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Returns:
            total_loss: Combined loss for backprop
            loss_dict: Individual components for logging
        """
        # ---- CLIP Losses (large weight) ----
        clip_cxr_ecg = self.clip_loss(cxr_shared, ecg_shared)
        clip_cxr_labs = self.clip_loss(cxr_shared, labs_shared)
        clip_total = (clip_cxr_ecg + clip_cxr_labs) / 2
        
        # ---- Regression Losses (small weight) ----
        reg_ecg = self.regression_loss(cxr_to_ecg_pred, ecg_target)
        reg_labs = self.regression_loss(cxr_to_labs_pred, labs_target)
        reg_total = (reg_ecg + reg_labs) / 2
        
        # ---- Total ----
        total_loss = (
            self.clip_weight * clip_total + 
            self.regression_weight * reg_total
        )
        
        loss_dict = {
            "total": total_loss.item(),
            "clip_total": clip_total.item(),
            "clip_cxr_ecg": clip_cxr_ecg.item(),
            "clip_cxr_labs": clip_cxr_labs.item(),
            "reg_total": reg_total.item(),
            "reg_ecg": reg_ecg.item(),
            "reg_labs": reg_labs.item(),
            # "temperature": self.current_temperature.item(),
        }
        
        return total_loss, loss_dict


# ============================================================================
# Main Model
# ============================================================================

class CrossModalCXRDistillation(pl.LightningModule):
    """
    Cross-modal distillation: Train CXR encoder to capture ECG/Labs knowledge.
    
    Training: All three modalities available
    Inference: CXR only, but embeddings contain physiological information
    """
    
    def __init__(
        self,
        # Encoders
        mae_checkpoint_path: str,

        # Dimensions
        cxr_dim: int = 768,
        ecg_dim: int = 1024,
        labs_dim: int = 256,
        shared_dim: int = 256,

        tuning_strategy: str = "lora",  # "lora", "last_n_blocks", "full"
        
        # LoRA Params (used if tuning_strategy="lora")
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,

        # Last-N Params (used if tuning_strategy="last_n_blocks")
        train_last_n_blocks: int = 4,

        # Loss weights
        clip_weight: float = 1.0,
        regression_weight: float = 0.1,
        temperature: float = 0.1,
        label_smoothing: float = 0.1, # Added param
        mixup_alpha: float = 0.0,     # Added param (0.0 disables it)

        # Optimizer
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        max_epochs: int = 50,
        warmup_epochs = 5,
        batch_size = 128
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Store dimensions
        self.cxr_dim = cxr_dim
        self.ecg_dim = ecg_dim
        self.labs_dim = labs_dim
        self.shared_dim = shared_dim
        
        # ---- CXR Encoder (TRAINABLE) ----
        self.cxr_encoder = self._build_mae_encoder(mae_checkpoint_path)
        # Unfreeze ALL layers
        for param in self.cxr_encoder.parameters():
            param.requires_grad = True

        self._apply_tuning_strategy()

        # ---- Projection Heads for CLIP (all trainable, discarded at inference) ----
        self.cxr_to_shared = ProjectionHead(cxr_dim, shared_dim)
        self.ecg_to_shared = ProjectionHead(ecg_dim, shared_dim)
        self.labs_to_shared = ProjectionHead(labs_dim, shared_dim)
        
        # ---- Regression Heads (trainable, discarded at inference) ----
        self.cxr_to_ecg = ProjectionHead(cxr_dim, ecg_dim)
        self.cxr_to_labs = ProjectionHead(cxr_dim, labs_dim)
        
        # ---- Loss Module ----
        self.loss_module = CLIPRegressionLoss(
            clip_weight=clip_weight,
            regression_weight=regression_weight,
            temperature=temperature,
            label_smoothing=label_smoothing
        )

        self.val_cos_ecg = CosineSimilarity(reduction='mean')
        self.val_cos_labs = CosineSimilarity(reduction='mean')
        self.train_loss_tracker = MeanMetric()
        
        self._print_model_info()
    
    # -------------------------------------------------------------------------
    # Initialization Helpers
    # -------------------------------------------------------------------------
    
    def _build_mae_encoder(self, checkpoint_path: str):
        """Load MAE-pretrained encoder."""
        encoder = MAECXREncoder(embed_dim=768, depth=12, num_heads=12)
        print(f"Loading MAE checkpoint from: {checkpoint_path}")
        
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        
        encoder_state = {}
        for k, v in state.items():
            if k.startswith("model.") and not k.startswith("model.decoder"):
                new_key = k.replace("model.", "")
                encoder_state[new_key] = v
        
        missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
        print(f"✓ MAE loaded - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        return encoder

    def _apply_tuning_strategy(self):
        strategy = self.hparams.tuning_strategy
        print(f"\nApplying Fine-Tuning Strategy: {strategy.upper()}")

        # Strategy A: LoRA
        if strategy == "lora":
            config = LoraConfig(
                r=self.hparams.lora_r,
                lora_alpha=self.hparams.lora_alpha,
                target_modules=["qkv", "fc1", "fc2"],  # correct for your encoder
                # OPTIONAL (often useful): include attention output proj too
                # target_modules=["qkv", "proj", "fc1", "fc2"],
                lora_dropout=self.hparams.lora_dropout,
                bias="none",
            )

            self.cxr_encoder = get_peft_model(self.cxr_encoder, config)

            # HARD guarantee: only LoRA params trainable
            for n, p in self.cxr_encoder.named_parameters():
                p.requires_grad = ("lora_" in n)

            for n, p in self.cxr_encoder.named_parameters():
                if n.startswith("base_model.model.norm.") or n.endswith(".norm.weight") or n.endswith(".norm.bias"):
                    p.requires_grad = True

            self.cxr_encoder.print_trainable_parameters()

        # Strategy B: Last N Blocks (Partial Freezing)
        elif strategy == "last_n_blocks":
            total = len(self.cxr_encoder.blocks)
            n_train = int(self.hparams.train_last_n_blocks)
            n_train = max(0, min(n_train, total))
            cutoff = total - n_train

            print(f"Freezing first {cutoff} blocks. Training last {n_train}.")

            # Freeze everything
            for p in self.cxr_encoder.parameters():
                p.requires_grad = False

            # Unfreeze last N blocks
            for i in range(cutoff, total):
                for p in self.cxr_encoder.blocks[i].parameters():
                    p.requires_grad = True

            # Unfreeze final norm
            for p in self.cxr_encoder.norm.parameters():
                p.requires_grad = True

            # OPTIONAL: also let cls_token adapt (sometimes helps; not required)
            # self.cxr_encoder.cls_token.requires_grad = True

            trainable = sum(p.numel() for p in self.cxr_encoder.parameters() if p.requires_grad)
            totalp = sum(p.numel() for p in self.cxr_encoder.parameters())
            print(f"Trainable Params: {trainable:,} ({100*trainable/totalp:.2f}%)")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _print_model_info(self):
        def count_params(m, trainable=False):
            if trainable:
                return sum(p.numel() for p in m.parameters() if p.requires_grad)
            return sum(p.numel() for p in m.parameters())
        
        print("\n" + "=" * 70)
        print("Cross-Modal CXR Distillation Model")
        print("=" * 70)
        print(f"  CXR Encoder (trainable):     {count_params(self.cxr_encoder, True):>12,}")
        print("-" * 70)
        print(f"  CLIP Projection Heads:       {count_params(self.cxr_to_shared) + count_params(self.ecg_to_shared) + count_params(self.labs_to_shared):>12,}")
        print(f"  Regression Heads:            {count_params(self.cxr_to_ecg) + count_params(self.cxr_to_labs):>12,}")
        print("-" * 70)
        print(f"  Loss weights: CLIP={self.loss_module.clip_weight}, Regression={self.loss_module.regression_weight}")
        # print(f"  Learnable Temperature: {self.loss_module.current_temperature:.4f}")
        print(f"  Tuning Strategy: {self.hparams.tuning_strategy} (LoRA r={self.hparams.lora_r} alpha={self.hparams.lora_alpha} dropout={self.hparams.lora_dropout})" if self.hparams.tuning_strategy=="lora" else f"  Tuning Strategy: {self.hparams.tuning_strategy} (Last {self.hparams.train_last_n_blocks} blocks)")
        print(f"  Total Params: {count_params(self):>12,}")
        print("=" * 70 + "\n")
    
    # -------------------------------------------------------------------------
    # Forward Pass
    # -------------------------------------------------------------------------
    
    def encode_cxr(self, cxr: torch.Tensor) -> torch.Tensor:
        """
        Encode CXR images. Use this at inference time.
        Returns enriched embeddings (768 dim).
        """
        return self.cxr_encoder(cxr)

    def forward(
        self,
        cxr: torch.Tensor,
        ecg_emb: Optional[torch.Tensor] = None,
        labs_emb: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            cxr: Raw CXR images [B, 3, H, W]
            is_training: Whether in training mode
        
        Returns:
            Dictionary with all embeddings and projections
        """
        # ---- Encode CXR (trainable) ----
        cxr_emb = self.cxr_encoder(cxr)  # [B, 768]
        
        outputs = {"cxr_emb": cxr_emb}
        
        # ---- CLIP projections ----
        outputs["cxr_shared"] = self.cxr_to_shared(cxr_emb)  # [B, shared_dim]
        
        # ---- Regression predictions ----
        outputs["cxr_to_ecg_pred"] = self.cxr_to_ecg(cxr_emb)   # [B, ecg_dim]
        outputs["cxr_to_labs_pred"] = self.cxr_to_labs(cxr_emb) # [B, labs_dim]
        
        # ---- Encode ECG/Labs (frozen) ----
        # inference-safe: only compute these when provided
        if ecg_emb is not None:
            if ecg_emb.ndim != 2 or ecg_emb.size(-1) != self.ecg_dim:
                raise ValueError(f"ecg_emb must be [B,{self.ecg_dim}], got {tuple(ecg_emb.shape)}")
            ecg_emb = ecg_emb.detach()
            outputs["ecg_emb"] = ecg_emb
            outputs["ecg_shared"] = self.ecg_to_shared(ecg_emb)

        if labs_emb is not None:
            if labs_emb.ndim != 2 or labs_emb.size(-1) != self.labs_dim:
                raise ValueError(f"labs_emb must be [B,{self.labs_dim}], got {tuple(labs_emb.shape)}")
            labs_emb = labs_emb.detach()
            outputs["labs_emb"] = labs_emb
            outputs["labs_shared"] = self.labs_to_shared(labs_emb)

        return outputs
    
    # -------------------------------------------------------------------------
    # MixUp Helper
    # -------------------------------------------------------------------------
    def do_mixup(self, x, ecg, labs):
        """
        Applies MixUp to Input Images and Target Embeddings.
        Returns mixed inputs, mixed targets, and lambda.
        """
        if self.hparams.mixup_alpha <= 0:
            return x, ecg, labs, 1.0

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.hparams.mixup_alpha, self.hparams.mixup_alpha)
        
        # Generate permutation
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        # Mix Inputs
        mixed_x = lam * x + (1 - lam) * x[index, :]
        
        # Mix Targets (Embeddings)
        # We mix the embeddings so the regression loss learns the interpolated physiology
        mixed_ecg = lam * ecg + (1 - lam) * ecg[index, :]
        mixed_labs = lam * labs + (1 - lam) * labs[index, :]

        return mixed_x, mixed_ecg, mixed_labs, lam

    # -------------------------------------------------------------------------
    # Training & Validation Steps
    # -------------------------------------------------------------------------   
    def training_step(self, batch, batch_idx):
        cxr, ecg_emb, labs_emb = batch
        B = cxr.size(0)
        
        cxr_mixed, ecg_mixed, labs_mixed, _ = self.do_mixup(cxr, ecg_emb, labs_emb)

        # Forward
        outputs = self(cxr_mixed, ecg_mixed, labs_mixed)
        
        # Compute loss
        loss, loss_dict = self.loss_module(
            cxr_shared=outputs["cxr_shared"],
            ecg_shared=outputs["ecg_shared"],
            labs_shared=outputs["labs_shared"],
            cxr_to_ecg_pred=outputs["cxr_to_ecg_pred"],
            cxr_to_labs_pred=outputs["cxr_to_labs_pred"],
            ecg_target=outputs["ecg_emb"],
            labs_target=outputs["labs_emb"],
        )
        
        # Logging
        self.log("train/loss_total", loss_dict["total"], prog_bar=True, batch_size=B, sync_dist=True)
        self.log("train/loss_clip", loss_dict["clip_total"], batch_size=B, sync_dist=True)
        self.log("train/loss_clip_ecg", loss_dict["clip_cxr_ecg"], batch_size=B, sync_dist=True)
        self.log("train/loss_clip_labs", loss_dict["clip_cxr_labs"], batch_size=B, sync_dist=True)
        self.log("train/loss_reg", loss_dict["reg_total"], batch_size=B, sync_dist=True)
        self.log("train/loss_reg_ecg", loss_dict["reg_ecg"], batch_size=B, sync_dist=True)
        self.log("train/loss_reg_labs", loss_dict["reg_labs"], batch_size=B, sync_dist=True)
        # self.log("train/temperature", loss_dict["temperature"], batch_size=B, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        cxr, ecg_emb, labs_emb = batch
        B = cxr.size(0)

        outputs = self(cxr, ecg_emb, labs_emb)

        loss, loss_dict = self.loss_module(
            cxr_shared=outputs["cxr_shared"],
            ecg_shared=outputs["ecg_shared"],
            labs_shared=outputs["labs_shared"],
            cxr_to_ecg_pred=outputs["cxr_to_ecg_pred"],
            cxr_to_labs_pred=outputs["cxr_to_labs_pred"],
            ecg_target=outputs["ecg_emb"],
            labs_target=outputs["labs_emb"],
        )

        with torch.no_grad():
            # ---- CLIP retrieval metrics (within-batch) ----
            cxr_norm = F.normalize(outputs["cxr_shared"], dim=-1, eps=1e-6)
            ecg_norm = F.normalize(outputs["ecg_shared"], dim=-1, eps=1e-6)
            labs_norm = F.normalize(outputs["labs_shared"], dim=-1, eps=1e-6)

            targets = torch.arange(B, device=cxr_norm.device)

            sim_cxr_ecg = cxr_norm @ ecg_norm.t()
            sim_cxr_labs = cxr_norm @ labs_norm.t()

            top1_cxr_ecg = sim_cxr_ecg.argmax(1).eq(targets).float().mean()
            top1_cxr_labs = sim_cxr_labs.argmax(1).eq(targets).float().mean()

            k = min(5, B)
            top5_cxr_ecg = sim_cxr_ecg.topk(k, dim=1).indices.eq(targets[:, None]).any(1).float().mean()
            top5_cxr_labs = sim_cxr_labs.topk(k, dim=1).indices.eq(targets[:, None]).any(1).float().mean()

            # pos/neg similarity diagnostics (detect collapse / shortcutting)
            pos_ecg = sim_cxr_ecg.diag().mean()
            pos_labs = sim_cxr_labs.diag().mean()
            if B > 1:
                neg_ecg = (sim_cxr_ecg.sum() - sim_cxr_ecg.diag().sum()) / (B * (B - 1))
                neg_labs = (sim_cxr_labs.sum() - sim_cxr_labs.diag().sum()) / (B * (B - 1))
            else:
                neg_ecg = sim_cxr_ecg.new_tensor(0.0)
                neg_labs = sim_cxr_labs.new_tensor(0.0)

            # ---- Regression cosine (epoch-aggregated) ----
            pred_ecg_norm = F.normalize(outputs["cxr_to_ecg_pred"], dim=-1, eps=1e-6)
            tgt_ecg_norm = F.normalize(outputs["ecg_emb"], dim=-1, eps=1e-6)
            cosine_ecg = (pred_ecg_norm * tgt_ecg_norm).sum(dim=-1).mean()

            pred_labs_norm = F.normalize(outputs["cxr_to_labs_pred"], dim=-1, eps=1e-6)
            tgt_labs_norm = F.normalize(outputs["labs_emb"], dim=-1, eps=1e-6)
            cosine_labs = (pred_labs_norm * tgt_labs_norm).sum(dim=-1).mean()

            self.val_cos_ecg(outputs["cxr_to_ecg_pred"], outputs["ecg_emb"]) 
            self.val_cos_labs(outputs["cxr_to_labs_pred"], outputs["labs_emb"])

            # embedding health checks
            cxr_emb_norm = outputs["cxr_emb"].norm(dim=-1).mean()
            cxr_shared_norm = outputs["cxr_shared"].norm(dim=-1).mean()
            ecg_shared_norm = outputs["ecg_shared"].norm(dim=-1).mean()
            labs_shared_norm = outputs["labs_shared"].norm(dim=-1).mean()

            cxr_shared_std = outputs["cxr_shared"].std(dim=0).mean()

        # ---- Logging (epoch-level for val) ----
        self.log("val/loss_total", loss_dict["total"], prog_bar=True, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/loss_clip", loss_dict["clip_total"], batch_size=B, sync_dist=True)
        self.log("val/loss_clip_ecg", loss_dict["clip_cxr_ecg"], batch_size=B, sync_dist=True)
        self.log("val/loss_clip_labs", loss_dict["clip_cxr_labs"], batch_size=B, sync_dist=True)
        self.log("val/loss_reg", loss_dict["reg_total"], batch_size=B, sync_dist=True)
        self.log("val/loss_reg_ecg", loss_dict["reg_ecg"], batch_size=B, sync_dist=True)
        self.log("val/loss_reg_labs", loss_dict["reg_labs"], batch_size=B, sync_dist=True)
        self.log("val/top1_cxr_ecg", top1_cxr_ecg, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/top1_cxr_labs", top1_cxr_labs, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/top5_cxr_ecg", top5_cxr_ecg, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/top5_cxr_labs", top5_cxr_labs, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)

        self.log("val/pos_sim_ecg", pos_ecg, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/neg_sim_ecg", neg_ecg, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/pos_sim_labs", pos_labs, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/neg_sim_labs", neg_labs, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)

        self.log("val/cxr_emb_norm", cxr_emb_norm, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/cxr_shared_norm", cxr_shared_norm, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/ecg_shared_norm", ecg_shared_norm, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/labs_shared_norm", labs_shared_norm, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        self.log("val/cxr_shared_std", cxr_shared_std, batch_size=B, sync_dist=True, on_step=False, on_epoch=True)

        # self.log("val/temperature", loss_dict["temperature"], batch_size=B, sync_dist=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        # epoch-aggregated regression cosines
        self.log("val/cosine_sim_ecg", self.val_cos_ecg.compute(), prog_bar=True, sync_dist=True)
        self.log("val/cosine_sim_labs", self.val_cos_labs.compute(), prog_bar=True, sync_dist=True)
        self.val_cos_ecg.reset()
        self.val_cos_labs.reset()

    def configure_optimizers(self):
        """
        Optimizer with linear warmup + cosine decay.
        """
        trainable_params = filter(lambda p: p.requires_grad, self.parameters())
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        
        # Warmup + Cosine Decay scheduler
        warmup_epochs = self.hparams.warmup_epochs
        max_epochs = self.hparams.max_epochs
        
        def lr_lambda(epoch):
            # Linear warmup
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            # Cosine decay after warmup
            progress = float(epoch - warmup_epochs) / float(max(1, max_epochs - warmup_epochs))
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }