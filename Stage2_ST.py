"""
PACX Teacher-Student Pipeline
==============================
Robust teacher-student distillation for physiology-aware CXR encoding.

Architecture:
- Teacher: Fusion([CXR, ECG, Labs]) → Rich multi-modal representation
- Student: CXR-only → Matches teacher's representation

Training:
- Stage 1 (Teacher): Learn multi-modal fusion with reconstruction
- Stage 2 (Student): Distill physiology knowledge to CXR encoder
"""

import os
import sys
import math
import time
import warnings
import itertools
from typing import Optional, Tuple, Dict, List, Literal
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader, TensorDataset
import timm
from torchvision import transforms as T
from torchvision import models
from functools import partial

from timm.models.vision_transformer import Block, PatchEmbed

pl.seed_everything(42, workers=True)
warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================================
# Data Module
# ============================================================================

class SymileMIMICRetrievalDataset(Dataset):
    """Retrieval dataset for validation"""
    def __init__(self, data_dir: Path, split: str):
        self.data_dir = Path(data_dir)
        self.cxr = torch.load(self.data_dir / f"{split}/cxr_{split}.pt")
        self.ecg = torch.load(self.data_dir / f"{split}/ecg_{split}.pt")
        self.labs_percentiles = torch.load(self.data_dir / f"{split}/labs_percentiles_{split}.pt")
        self.labs_missingness = torch.load(self.data_dir / f"{split}/labs_missingness_{split}.pt")
        self.hadm_id = torch.load(self.data_dir / f"{split}/hadm_id_{split}.pt")
        self.label_hadm_id = torch.load(self.data_dir / f"{split}/label_hadm_id_{split}.pt")
        self.label = torch.load(self.data_dir / f"{split}/label_{split}.pt")

    def __len__(self):
        return len(self.ecg)

    def __getitem__(self, idx):
        return {
            "cxr": self.cxr[idx],
            "ecg": self.ecg[idx],
            "labs_percentiles": self.labs_percentiles[idx],
            "labs_missingness": self.labs_missingness[idx],
            "hadm_id": self.hadm_id[idx],
            "label_hadm_id": self.label_hadm_id[idx],
            "label": self.label[idx]
        }


class SymileMIMICDataModule(pl.LightningDataModule):
    """Lightning DataModule for Symile-MIMIC"""
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 256,
        num_workers: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        
        if num_workers is None:
            try:
                self.num_workers = len(os.sched_getaffinity(0))
            except AttributeError:
                self.num_workers = 4
        else:
            self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            # Load training data
            cxr_train = torch.load(self.data_dir / "train/cxr_train.pt")
            ecg_train = torch.load(self.data_dir / "train/ecg_train.pt")
            labs_pct_train = torch.load(self.data_dir / "train/labs_percentiles_train.pt")
            labs_miss_train = torch.load(self.data_dir / "train/labs_missingness_train.pt")
            
            # Load validation data
            cxr_val = torch.load(self.data_dir / "val/cxr_val.pt")
            ecg_val = torch.load(self.data_dir / "val/ecg_val.pt")
            labs_pct_val = torch.load(self.data_dir / "val/labs_percentiles_val.pt")
            labs_miss_val = torch.load(self.data_dir / "val/labs_missingness_val.pt")
            
            self.ds_train = TensorDataset(cxr_train, ecg_train, labs_pct_train, labs_miss_train)
            self.ds_val = TensorDataset(cxr_val, ecg_val, labs_pct_val, labs_miss_val)
            
            print(f"✓ Loaded {len(self.ds_train)} training samples")
            print(f"✓ Loaded {len(self.ds_val)} validation samples")

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True if self.num_workers > 0 else False,
        )


# ============================================================================
# Encoder Modules
# ============================================================================

class MAECXREncoder(nn.Module):
    """MAE-pretrained ViT encoder"""
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        norm_layer = nn.LayerNorm,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)[:, 0]


class ECGEncoder(nn.Module):
    """ResNet18-based ECG encoder"""
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        x = self.resnet(x)
        return self.projection(x)


class LabsEncoder(nn.Module):
    """MLP-based Labs encoder"""
    def __init__(self, output_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(100, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class CXRAnchoredFusion(nn.Module):
    """CXR queries ECG/Labs - asymmetric design for better student transfer"""
    def __init__(self, cxr_dim, ecg_dim, labs_dim, hidden_dim, output_dim, dropout=0.1, num_heads=8):
        super().__init__()
        self.cxr_proj = nn.Linear(cxr_dim, hidden_dim)
        self.ecg_proj = nn.Linear(ecg_dim, hidden_dim)
        self.labs_proj = nn.Linear(labs_dim, hidden_dim)
        
        # CXR attends to ECG/Labs (asymmetric)
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_dim, 
                                                num_heads=num_heads, 
                                                dropout=dropout, 
                                                batch_first=True)        
        # Gated fusion to control contribution
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, output_dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, z_c, z_e, z_l, return_attn_weights=False):
        """
        Args:
            modality_mask: dict with 'ecg': bool tensor, 'labs': bool tensor
                          True = keep, False = mask out (for modality dropout)
        """
        B = z_c.size(0)
        
        h_c = self.cxr_proj(z_c).unsqueeze(1)  # (B, 1, H)
        h_e = self.ecg_proj(z_e).unsqueeze(1)  # (B, 1, H)
        h_l = self.labs_proj(z_l).unsqueeze(1)  # (B, 1, H)
        
        physio = torch.cat([h_e, h_l], dim=1)  # (B, 2, H)
        
        # CXR queries ECG and Labs separately
        h_physio, attn_weights = self.cross_attn(h_c, physio, physio)
        
        x = self.norm1(h_c + h_physio)   # inject physio into CXR space
        x = self.norm2(x).squeeze(1)
        out = self.mlp(x)              # (B, output_dim)

        if return_attn_weights:
            return out, attn_weights
        return out

class ReconstructionHeads(nn.Module):
    """
    Auxiliary heads to reconstruct modality features from fused representation.
    Ensures the fused embedding preserves information from all modalities.
    """
    def __init__(self, fused_dim: int, ecg_dim: int, labs_dim: int, hidden_dim: int = 512):
        super().__init__()
        
        # ECG reconstruction
        self.ecg_recon = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, ecg_dim),
        )
        
        # Labs reconstruction
        self.labs_recon = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, labs_dim),
        )
        
    def forward(self, z_fused: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_fused: (B, fused_dim) fused representation
        Returns:
            ecg_pred: (B, ecg_dim) predicted ECG features
            labs_pred: (B, labs_dim) predicted Labs features
        """
        return self.ecg_recon(z_fused), self.labs_recon(z_fused)

# ============================================================================
# Teacher Module
# ============================================================================

class TeacherLossModule(nn.Module):
    """
    Minimal, robust loss design.
    
    Core insight: Only 2 losses are truly necessary:
    1. Reconstruction - ensures physiology is encoded
    2. Alignment - ensures semantic structure
    
    Everything else is a METRIC for monitoring, not a loss to optimize.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        recon_weight: float = 1.0,
        align_ecg_weight: float = 1.0,
        align_labs_weight: float = 1.0,
        align_cxr_weight: float = 0.3,  # smaller!
    ):
        super().__init__()
        self.temperature = temperature
        self.recon_weight = recon_weight
        self.align_ecg_weight = align_ecg_weight
        self.align_labs_weight = align_labs_weight
        self.align_cxr_weight = align_cxr_weight
    
    def forward(
        self,
        z_fused: torch.Tensor,      # (B, D) fused representation
        z_c_proj: torch.Tensor,     # (B, D) CXR projected
        z_e_proj: torch.Tensor,     # (B, D) ECG projected
        z_l_proj: torch.Tensor,     # (B, D) Labs projected
        z_e: torch.Tensor,          # (B, d_e) ECG raw features
        z_l: torch.Tensor,          # (B, d_l) Labs raw features
        ecg_pred: torch.Tensor,     # (B, d_e) reconstructed from fused
        labs_pred: torch.Tensor,    # (B, d_l) reconstructed from fused
        modality_mask: dict = None,
    ) -> dict:
        
        B = z_fused.size(0)
        device = z_fused.device
        
        if modality_mask is None:
            ecg_mask = torch.ones(B, dtype=torch.bool, device=device)
            labs_mask = torch.ones(B, dtype=torch.bool, device=device)
        else:
            ecg_mask = modality_mask["ecg"]
            labs_mask = modality_mask["labs"]
        
        losses = {}
        
        # ================================================================
        # LOSS 1: Reconstruction (Primary)
        # ================================================================
        # PURPOSE: z_fused must contain enough info to recover physiology.
        #
        # This is the ONLY direct constraint that physiology is encoded.
        # Without it, z_fused could align well in CLIP space but lose
        # actual physiological detail.
        #
        # CRITICAL: Detach targets to prevent encoders from learning
        #           "easy to predict" features.
        # ================================================================
        
        loss_recon_ecg = self._masked_mse(ecg_pred, z_e.detach(), ecg_mask)
        loss_recon_labs = self._masked_mse(labs_pred, z_l.detach(), labs_mask)
        loss_recon = loss_recon_ecg + loss_recon_labs
        
        losses["recon_total"] = loss_recon
        losses["recon_ecg"] = loss_recon_ecg
        losses["recon_labs"] = loss_recon_labs
        
        # ================================================================
        # LOSS 2: Contrastive Alignment
        # ================================================================
        # PURPOSE: Create structured embedding space.
        #
        # We align fused with ALL modalities including CXR:
        #   - fused ↔ ECG: semantic link to cardiac signal
        #   - fused ↔ Labs: semantic link to metabolic signal  
        #   - fused ↔ CXR: keeps z_fused "reachable" from CXR space
        #
        # WHY include CXR alignment?
        #   The student only has CXR. If z_fused is orthogonal to z_cxr,
        #   distillation becomes impossible. We WANT them related.
        #   The reconstruction loss ensures physiology is still there.
        # ================================================================
        
        loss_align_ecg = self._masked_clip_loss(z_fused, z_e_proj, ecg_mask)
        loss_align_labs = self._masked_clip_loss(z_fused, z_l_proj, labs_mask)
        loss_align_cxr = self._clip_loss(z_fused, z_c_proj)  # Always available
        
        loss_align = (
            self.align_ecg_weight  * loss_align_ecg +
            self.align_labs_weight * loss_align_labs +
            self.align_cxr_weight  * loss_align_cxr
        )
        
        losses["align_total"] = loss_align
        losses["align_ecg"] = loss_align_ecg
        losses["align_labs"] = loss_align_labs
        losses["align_cxr"] = loss_align_cxr
        
        # ================================================================
        # TOTAL LOSS
        # ================================================================
        
        loss_total = (
            self.recon_weight * loss_recon +
            loss_align  # already weighted components
        )
        losses["total"] = loss_total
        
        # ================================================================
        # METRICS (for monitoring, NOT backpropagated)
        # ================================================================
        
        return losses
    
    def _masked_mse(self, pred, target, mask):
        if mask.sum() == 0:
            return torch.zeros((), device=pred.device)
        mse = (pred - target).pow(2).mean(dim=-1)
        return (mse * mask.float()).sum() / mask.float().sum()
    
    def _clip_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (F.cross_entropy(logits, labels) + 
                      F.cross_entropy(logits.t(), labels))
    
    def _masked_clip_loss(self, z1, z2, mask):
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            return torch.zeros((), device=z1.device)
        return self._clip_loss(z1[idx], z2[idx])

class TeacherModule(pl.LightningModule):
    """
    Teacher Module: Learns to fuse CXR + ECG + Labs via CLIP-style alignment

    Training objective:
    - Cross-modal contrastive alignment:
        L_align = L_fused-cxr + L_fused-ecg + L_fused-labs
      where each term is an InfoNCE loss between the fused embedding and
      the corresponding modality embedding.

    Validation metrics:
    - Contrastive losses
    - Cosine similarity between fused and each modality
    - Basic feature stats (mean, std)
    """

    def __init__(
        self,
        # Architecture
        mae_checkpoint_path: str,
        cxr_dim: int = 768,
        ecg_dim: int = 256,
        labs_dim: int = 256,
        fusion_hidden_dim: int = 1024,
        fusion_output_dim: int = 768,
        # Contrastive
        temperature: float = 0.07,
        # Optimizer
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 200,
        cxr_unfreeze: str = "none",   # "none", "last_n", or "all"
        cxr_unfreeze_last_n: int = 4,
        recon_weight: float = 1.0,
        align_ecg_weight: float = 1.0,
        align_labs_weight: float = 1.0,
        align_cxr_weight: float = 0.3,  # smaller!
        modality_dropout_prob: float = 0.3, # Probability to zero-out ECG/Labs
    ):
        super().__init__()
        self.save_hyperparameters()

        self.cxr_dim = cxr_dim
        self.ecg_dim = ecg_dim
        self.labs_dim = labs_dim
        self.fusion_output_dim = fusion_output_dim

        self.temperature = temperature
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        self.modality_dropout_prob = modality_dropout_prob

        self.recon_weight = recon_weight
        self.align_ecg_weight = align_ecg_weight
        self.align_labs_weight = align_labs_weight
        self.align_cxr_weight = align_cxr_weight

        # -----------------
        # Encoders
        # -----------------
        self.cxr_encoder = self._build_mae_encoder(mae_checkpoint_path)
        self.ecg_encoder = ECGEncoder(output_dim=ecg_dim)
        self.labs_encoder = LabsEncoder(output_dim=labs_dim)

        # CXR unfreezing strategy
        self.cxr_unfreeze = cxr_unfreeze
        self.cxr_unfreeze_last_n = cxr_unfreeze_last_n
        self._set_cxr_trainability()

        # -----------------
        # Fusion module
        # -----------------
        self.fusion = CXRAnchoredFusion(
            cxr_dim=cxr_dim,
            ecg_dim=ecg_dim,
            labs_dim=labs_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim
        )

        # -----------------
        # Projection heads for contrastive alignment
        # All mapped into fusion_output_dim
        # -----------------
        self.cxr_proj = nn.Sequential(
            nn.Linear(cxr_dim, fusion_output_dim),
            nn.LayerNorm(fusion_output_dim),
        )
        self.ecg_proj = nn.Sequential(
            nn.Linear(ecg_dim, fusion_output_dim),
            nn.LayerNorm(fusion_output_dim),
        )
        self.labs_proj = nn.Sequential(
            nn.Linear(labs_dim, fusion_output_dim),
            nn.LayerNorm(fusion_output_dim),
        )

        # The fused embedding must be able to regenerate the physiology features
        self.recon_heads = ReconstructionHeads(
            fused_dim=fusion_output_dim,
            ecg_dim=ecg_dim,
            labs_dim=labs_dim,
        )
        
        self.loss_module = TeacherLossModule(
            temperature=temperature,
            recon_weight=recon_weight,
            align_ecg_weight=align_ecg_weight,
            align_labs_weight=align_labs_weight,
            align_cxr_weight=align_cxr_weight,
        )
        
        # Transforms
        self.val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        self._print_model_info()

    # ---------------------------------------------------------------------
    # Init helpers
    # ---------------------------------------------------------------------

    def _print_model_info(self):
        def count_params(m, trainable=False):
            if trainable:
                return sum(p.numel() for p in m.parameters() if p.requires_grad)
            return sum(p.numel() for p in m.parameters())
        
        print("\n" + "=" * 60)
        print("Teacher Module Initialized")
        print("=" * 60)
        print(f"  CXR encoder (trainable):  {count_params(self.cxr_encoder, True):>10,}")
        print(f"  ECG encoder:              {count_params(self.ecg_encoder):>10,}")
        print(f"  Labs encoder:             {count_params(self.labs_encoder):>10,}")
        print(f"  Fusion:                   {count_params(self.fusion):>10,}")
        proj_params = count_params(self.cxr_proj) + count_params(self.ecg_proj) + count_params(self.labs_proj)
        print(f"  Projections:              {proj_params:>10,}")
        print(f"  Reconstruction heads:     {count_params(self.recon_heads):>10,}")
        print("-" * 60)
        print(f"  Loss weights: recon={self.loss_module.recon_weight}, align_ecg={self.loss_module.align_ecg_weight}, align_labs={self.loss_module.align_labs_weight}, align_cxr={self.loss_module.align_cxr_weight}")
        print(f"  Modality dropout: {self.modality_dropout_prob}")
        print("=" * 60 + "\n")

    def _build_mae_encoder(self, checkpoint_path: str):
        """Load MAE-pretrained encoder"""
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
    
    def _set_cxr_trainability(self):
        # freeze everything first
        for p in self.cxr_encoder.parameters():
            p.requires_grad = False

        if self.cxr_unfreeze == "all":
            for p in self.cxr_encoder.parameters():
                p.requires_grad = True

        elif self.cxr_unfreeze == "last_n":
            total_blocks = len(self.cxr_encoder.blocks)
            start = max(0, total_blocks - self.cxr_unfreeze_last_n)

            # unfreeze last N transformer blocks
            for i in range(start, total_blocks):
                for p in self.cxr_encoder.blocks[i].parameters():
                    p.requires_grad = True

            # usually also unfreeze final norm + cls_token
            for p in self.cxr_encoder.norm.parameters():
                p.requires_grad = True
            self.cxr_encoder.cls_token.requires_grad = True

        # eval mode is still fine; gradients will flow if requires_grad=True
        self.cxr_encoder.eval()

        trainable = sum(p.numel() for p in self.cxr_encoder.parameters() if p.requires_grad)
        print(f"✓ Teacher CXR trainable params: {trainable:,}")

    # ---------------------------------------------------------------------
    # Modality dropout
    # ---------------------------------------------------------------------

    def _sample_modality_mask(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Returns bool masks where True = keep, False = drop.

        We never drop BOTH ECG and Labs for a given sample.
        """
        p = self.modality_dropout_prob
        if p <= 0.0:
            return {
                "ecg": torch.ones(batch_size, dtype=torch.bool, device=device),
                "labs": torch.ones(batch_size, dtype=torch.bool, device=device),
            }

        drop_ecg = (torch.rand(batch_size, device=device) < p)
        drop_labs = (torch.rand(batch_size, device=device) < p)

        # Prevent dropping both: if both would be dropped, keep ECG
        both = drop_ecg & drop_labs
        drop_ecg = drop_ecg & ~both  # keep labs-only when collision

        return {
            "ecg": ~drop_ecg,
            "labs": ~drop_labs,
        }

   
    def forward(
        self,
        cxr: torch.Tensor,
        ecg: torch.Tensor,
        labs: torch.Tensor,
        modality_mask: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Forward pass through teacher

        Args:
            cxr: (B, 3, H, W) - raw CXR, will be transformed to 224x224
            ecg: (B, 1, H_ecg, W_ecg) or similar
            labs: (B, 100)

        Returns:
            z_fused: (B, fusion_output_dim) - teacher fused representation
            z_e:     (B, ecg_dim) - ECG features (unprojected)
            z_l:     (B, labs_dim) - Labs features (unprojected)
            z_c:     (B, cxr_dim) - CXR features (unprojected)
        """
        B = cxr.size(0)

        # Transform CXR
        cxr = torch.stack([self.val_transform(img) for img in cxr])

        # Encode
        z_c = self.cxr_encoder(cxr)   # (B, cxr_dim)
        z_e = self.ecg_encoder(ecg)   # (B, ecg_dim)
        z_l = self.labs_encoder(labs) # (B, labs_dim)

        # Apply modality dropout (zero features) *only* on physio inputs
        if modality_mask is not None:
            ecg_keep = modality_mask["ecg"].view(B, 1).float()
            labs_keep = modality_mask["labs"].view(B, 1).float()
            z_e = z_e * ecg_keep
            z_l = z_l * labs_keep

        # Fuse (using unprojected features)
        z_fused = self.fusion(z_c, z_e, z_l)  # (B, fusion_output_dim)

        return z_fused, z_e, z_l, z_c
    
    # ---------------------------------------------------------------------
    # Lightning hooks
    # ---------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        cxr, ecg, labs_pct, labs_miss = batch
        labs = torch.cat([labs_pct, labs_miss], dim=1)
        B = cxr.size(0)
        device = cxr.device

        # Sample modality mask for this batch
        modality_mask = self._sample_modality_mask(B, device)

        # Forward
        z_fused, z_e, z_l, z_c = self(cxr, ecg, labs, modality_mask)
        z_c_proj = self.cxr_proj(z_c)
        z_e_proj = self.ecg_proj(z_e)
        z_l_proj = self.labs_proj(z_l)

        # 1) CLIP-style contrastive
        ecg_pred, labs_pred = self.recon_heads(z_fused)

        losses = self.loss_module(
            z_fused=z_fused,
            z_c_proj=z_c_proj,
            z_e_proj=z_e_proj,
            z_l_proj=z_l_proj,
            z_e=z_e,
            z_l=z_l,
            ecg_pred=ecg_pred,
            labs_pred=labs_pred,
            modality_mask=modality_mask,
        )

        # ----------------- Logging (TRAIN) -----------------
        # Main objective
        self.log("teacher/train/loss_total", losses["total"],
                 on_step=True, on_epoch=True, prog_bar=True, batch_size=B)

        # Decomposed losses
        self.log("teacher/train/loss_recon_total", losses["recon_total"],
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/loss_recon_ecg", losses["recon_ecg"],
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/loss_recon_labs", losses["recon_labs"],
                 on_epoch=True, batch_size=B)

        self.log("teacher/train/loss_align_total", losses["align_total"],
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/loss_align_cxr", losses["align_cxr"],
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/loss_align_ecg", losses["align_ecg"],
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/loss_align_labs", losses["align_labs"],
                 on_epoch=True, batch_size=B)

        # Modality dropout stats
        self.log("teacher/train/modality_keep_ecg",
                 modality_mask["ecg"].float().mean(),
                 on_epoch=True, batch_size=B)
        self.log("teacher/train/modality_keep_labs",
                 modality_mask["labs"].float().mean(),
                 on_epoch=True, batch_size=B)

        return losses["total"]

    def validation_step(self, batch, batch_idx):
        cxr, ecg, labs_pct, labs_miss = batch
        labs = torch.cat([labs_pct, labs_miss], dim=1)
        B = cxr.size(0)
        device = cxr.device
        
        # No modality dropout in validation
        z_fused, z_e, z_l, z_c = self(cxr, ecg, labs, modality_mask=None)
        
        # Project
        z_c_proj = self.cxr_proj(z_c)
        z_e_proj = self.ecg_proj(z_e)
        z_l_proj = self.labs_proj(z_l)
        
        # Reconstruct from fused
        ecg_pred, labs_pred = self.recon_heads(z_fused)
        
        # ─────────────────────────────────────────────────────────────
        # Compute losses
        # ─────────────────────────────────────────────────────────────
        losses = self.loss_module(
            z_fused=z_fused,
            z_c_proj=z_c_proj,
            z_e_proj=z_e_proj,
            z_l_proj=z_l_proj,
            z_e=z_e,
            z_l=z_l,
            ecg_pred=ecg_pred,
            labs_pred=labs_pred,
            modality_mask=None,
        )
        
        # ─────────────────────────────────────────────────────────────
        # Key Metrics: Physiological Gain
        # ─────────────────────────────────────────────────────────────
        # Does z_fused reconstruct physiology better than z_cxr alone?
        # This is THE metric that tells us if fusion adds value.
        
        with torch.no_grad():
            ecg_pred_cxr, labs_pred_cxr = self.recon_heads(z_c_proj)
            
            # MSE from fused (should be lower)
            mse_fused_ecg = F.mse_loss(ecg_pred, z_e)
            mse_fused_labs = F.mse_loss(labs_pred, z_l)
            
            # MSE from CXR only (should be higher)
            mse_cxr_ecg = F.mse_loss(ecg_pred_cxr, z_e)
            mse_cxr_labs = F.mse_loss(labs_pred_cxr, z_l)
            
            # Gain = improvement of fused over CXR (positive = good)
            gain_ecg = mse_cxr_ecg - mse_fused_ecg
            gain_labs = mse_cxr_labs - mse_fused_labs
            
            # Relative gain (percentage improvement)
            rel_gain_ecg = gain_ecg / (mse_cxr_ecg + 1e-8)
            rel_gain_labs = gain_labs / (mse_cxr_labs + 1e-8)
        
        # ─────────────────────────────────────────────────────────────
        # Retrieval Metrics
        # ─────────────────────────────────────────────────────────────
        
        z_f_norm = F.normalize(z_fused, dim=-1)
        z_c_norm = F.normalize(z_c_proj, dim=-1)
        z_e_norm = F.normalize(z_e_proj, dim=-1)
        z_l_norm = F.normalize(z_l_proj, dim=-1)
        
        targets = torch.arange(B, device=device)
        
        # Top-1 retrieval accuracy
        top1_fused_ecg = (z_f_norm @ z_e_norm.t()).argmax(1).eq(targets).float().mean()
        top1_fused_labs = (z_f_norm @ z_l_norm.t()).argmax(1).eq(targets).float().mean()
        top1_fused_cxr = (z_f_norm @ z_c_norm.t()).argmax(1).eq(targets).float().mean()
        top1_cxr_fused = (z_c_norm @ z_f_norm.t()).argmax(1).eq(targets).float().mean()
        
        # ─────────────────────────────────────────────────────────────
        # Representation Quality Metrics
        # ─────────────────────────────────────────────────────────────
        
        with torch.no_grad():
            # CXR-fused similarity (want 0.6-0.9: related but not identical)
            cxr_similarity = (z_f_norm * z_c_norm).sum(dim=-1).mean()
            
            # Collapse detection
            gram = z_f_norm @ z_f_norm.t()
            off_diag = gram - torch.eye(B, device=device)
            collapse_score = off_diag.abs().mean()
            
            # Feature stats
            fused_std = z_fused.std()
            fused_mean = z_fused.mean()
            
            # Attention weights distribution
            _, attn_weights = self.fusion(z_c, z_e, z_l, return_attn_weights=True)
            attn = attn_weights.squeeze(1)  # (B, 2)
            attn_entropy = -(attn * (attn + 1e-8).log()).sum(dim=-1).mean()
            attn_to_ecg = attn[:, 0].mean()
            attn_to_labs = attn[:, 1].mean()
        
        # ─────────────────────────────────────────────────────────────
        # Logging
        # ─────────────────────────────────────────────────────────────
        self.log("teacher/val/loss_total", losses["total"],
                 prog_bar=True, batch_size=B)

        # Decomposed losses
        self.log("teacher/val/loss_recon_total", losses["recon_total"],
                 batch_size=B)
        self.log("teacher/val/loss_align_total", losses["align_total"],
                 batch_size=B)

        # Physiological gain – key metrics
        self.log("teacher/val/physio_gain_ecg", gain_ecg,
                 prog_bar=True, batch_size=B)
        self.log("teacher/val/physio_gain_labs", gain_labs,
                 prog_bar=True, batch_size=B)
        self.log("teacher/val/physio_rel_gain_ecg", rel_gain_ecg,
                 batch_size=B)
        self.log("teacher/val/physio_rel_gain_labs", rel_gain_labs,
                 batch_size=B)

        self.log("teacher/val/mse_fused_ecg", mse_fused_ecg, batch_size=B)
        self.log("teacher/val/mse_cxr_ecg", mse_cxr_ecg, batch_size=B)
        self.log("teacher/val/mse_fused_labs", mse_fused_labs, batch_size=B)
        self.log("teacher/val/mse_cxr_labs", mse_cxr_labs, batch_size=B)

        # Retrieval
        self.log("teacher/val/top1_fused_ecg", top1_fused_ecg, batch_size=B)
        self.log("teacher/val/top1_fused_labs", top1_fused_labs, batch_size=B)
        self.log("teacher/val/top1_fused_cxr", top1_fused_cxr, batch_size=B)
        self.log("teacher/val/top1_cxr_fused", top1_cxr_fused, batch_size=B)

        # Representation quality
        self.log("teacher/val/cxr_similarity", cxr_similarity, batch_size=B)
        self.log("teacher/val/collapse_score", collapse_score, batch_size=B)
        self.log("teacher/val/fused_std", fused_std, batch_size=B)
        self.log("teacher/val/fused_mean", fused_mean, batch_size=B)

        # Attention
        self.log("teacher/val/attn_entropy", attn_entropy, batch_size=B)
        self.log("teacher/val/attn_to_ecg", attn_to_ecg, batch_size=B)
        self.log("teacher/val/attn_to_labs", attn_to_labs, batch_size=B)

        return losses["total"]

    # ---------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------

    def configure_optimizers(self):
        base_params = (
            list(self.ecg_encoder.parameters())
            + list(self.labs_encoder.parameters())
            + list(self.fusion.parameters())
            + list(self.cxr_proj.parameters())
            + list(self.ecg_proj.parameters())
            + list(self.labs_proj.parameters())
            + list(self.recon_heads.parameters())
        )

        param_groups = [{"params": base_params, "lr": self.lr}]

        cxr_trainable = [p for p in self.cxr_encoder.parameters() if p.requires_grad]
        if len(cxr_trainable) > 0:
            param_groups.append({"params": cxr_trainable, "lr": self.lr * 0.1})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = (current_step - self.warmup_steps) / float(
                max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ---------------------------------------------------------------------
    # Public API for student
    # ---------------------------------------------------------------------

    @torch.no_grad()
    def get_distillation_targets(
        self, 
        cxr: torch.Tensor, 
        ecg: torch.Tensor, 
        labs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get targets for student distillation.
        
        Args:
            cxr, ecg, labs: Input data
            
        Returns:
            z_fused: Target representation (contains CXR + physiology)
            z_c_proj: CXR-only projection (for comparison/debugging)
        """
        self.eval()
        z_fused, _, _, z_c = self(cxr, ecg, labs, modality_mask=None)
        z_c_proj = self.cxr_proj(z_c)
        return z_fused, z_c_proj
    
    @torch.no_grad()
    def encode_cxr_only(self, cxr: torch.Tensor) -> torch.Tensor:
        """
        Encode CXR without fusion (for student baseline comparison).
        """
        self.eval()
        cxr = torch.stack([self.val_transform(img) for img in cxr])
        z_c = self.cxr_encoder(cxr)
        return self.cxr_proj(z_c)


# ============================================================================
# Student Module
# ============================================================================


class StudentLossModule(nn.Module):
    """
    Loss for CXR-only student distilling a multimodal teacher.

    Components:
      - L_mse:    MSE between student and teacher fused embeddings
      - L_clip:   CLIP-style cosine-contrastive distillation
      - L_anchor: small MSE to teacher's CXR-only projection (stability)
      - L_mae:    optional MAE-style reconstruction loss on patches

    Only L_mse + L_clip are required. Others are optional / small-weight.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        mse_weight: float = 1.0,
        clip_weight: float = 1.0,
        anchor_weight: float = 0.1,   # small, can set 0.0 to disable
    ):
        super().__init__()
        self.temperature = temperature
        self.mse_weight = mse_weight
        self.clip_weight = clip_weight
        self.anchor_weight = anchor_weight

    def forward(
        self,
        z_s: torch.Tensor,             # (B, D) student embedding
        z_fused_t: torch.Tensor,       # (B, D) teacher fused embedding
        z_cxr_t: torch.Tensor = None
    ) -> dict:
        """
        Returns:
            dict with 'total' and individual loss components / metrics.
        """
        losses = {}

        # ---------------------------------------------
        # 1) Distillation to teacher fused embedding
        # ---------------------------------------------
        z_t = z_fused_t.detach()

        loss_mse = F.mse_loss(z_s, z_t)
        loss_clip = self._clip_loss(z_s, z_t)

        losses["distill_mse"] = loss_mse
        losses["distill_clip"] = loss_clip

        # ---------------------------------------------
        # 2) CXR anchor (optional, small weight)
        # ---------------------------------------------
        if z_cxr_t is not None and self.anchor_weight > 0.0:
            z_c = z_cxr_t.detach()
            loss_anchor = F.mse_loss(z_s, z_c)
        else:
            loss_anchor = torch.zeros((), device=z_s.device)

        losses["distill_anchor"] = loss_anchor

        # ---------------------------------------------
        # Total
        # ---------------------------------------------
        loss_total = (
            self.mse_weight * loss_mse +
            self.clip_weight * loss_clip +
            self.anchor_weight * loss_anchor
        )
        losses["distill_total"] = loss_total

        return losses

    # ---------------------------------------------
    # Helper: CLIP-style loss
    # ---------------------------------------------
    def _clip_loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = z1 @ z2.t() / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return 0.5 * (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)
        )


class StudentModule(pl.LightningModule):
    """
    Student Module: CXR-only encoder that matches teacher
    
    Training objective:
    - Distillation: Match teacher's fused representation using only CXR
    - This injects physiology knowledge into CXR encoder
    
    Validation metrics:
    - Distillation loss (MSE + cosine)
    - Alignment with teacher
    - Retrieval accuracy (if data_dir provided)
    """
    
    def __init__(
        self,
        # Architecture
        mae_checkpoint_path: str,
        teacher_checkpoint_path: str,
        cxr_dim: int = 768,
        student_hidden_dim: int = 768,
        
        # Unfreezing strategy
        unfreeze_strategy: Literal["all", "last_n_blocks"] = "last_n_blocks",
        unfreeze_last_n: int = 4,
        
        # Loss weights
        mse_weight: float = 1.0,
        cos_weight: float = 1.0,
        
        # Optimizer
        lr: float = 1e-4,
        weight_decay: float = 0.05,
        warmup_steps: int = 500,
        
        # Validation
        eval_retrieval_every_n_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.cxr_dim = cxr_dim
        self.unfreeze_strategy = unfreeze_strategy
        self.unfreeze_last_n = unfreeze_last_n
        
        self.mse_weight = mse_weight
        self.cos_weight = cos_weight
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        
        self.eval_retrieval_every_n_epochs = eval_retrieval_every_n_epochs
        
        # Build student encoder
        self.cxr_encoder = self._build_mae_encoder(mae_checkpoint_path)
        
        # Unfreeze strategy
        self._apply_unfreezing()
        
        # Student prediction head
        self.student_head = nn.Sequential(
            nn.Linear(cxr_dim, student_hidden_dim),
            nn.GELU(),
            nn.Linear(student_hidden_dim, cxr_dim)
        )
        
        # Load teacher (frozen)
        print(f"Loading teacher from: {teacher_checkpoint_path}")
        self.teacher = TeacherModule.load_from_checkpoint(teacher_checkpoint_path)
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
        print(f"✓ Teacher loaded and frozen")
        
        # Transforms
        self.train_transform = T.Compose([
            T.Resize(256),
            T.RandomCrop(224),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _build_mae_encoder(self, checkpoint_path: str):
        """Load MAE-pretrained encoder"""
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
    
    def _apply_unfreezing(self):
        """Apply unfreezing strategy to CXR encoder"""
        if self.unfreeze_strategy == "all":
            # Unfreeze everything
            for param in self.cxr_encoder.parameters():
                param.requires_grad = True
            trainable = sum(p.numel() for p in self.cxr_encoder.parameters())
            print(f"✓ Unfroze entire CXR encoder ({trainable:,} params)")
        
        elif self.unfreeze_strategy == "last_n_blocks":
            for p in self.cxr_encoder.parameters():
                p.requires_grad = False

            total_blocks = len(self.cxr_encoder.blocks)
            start = max(0, total_blocks - self.unfreeze_last_n)
            for i in range(start, total_blocks):
                for p in self.cxr_encoder.blocks[i].parameters():
                    p.requires_grad = True
            for p in self.cxr_encoder.norm.parameters():
                p.requires_grad = True

            num_trainable = sum(p.numel() for p in self.cxr_encoder.parameters()
                                if p.requires_grad)
            print(f"✓ Unfroze last {self.unfreeze_last_n}/{total_blocks} blocks "
                  f"({num_trainable:,} params)")
            
    def forward(self, cxr, transform: bool = True):
        """
        Forward through student
        
        Args:
            cxr: (B, 3, H, W)
            transform: If True, apply augmentation/normalization
        
        Returns:
            z_student: (B, cxr_dim) - student's prediction
        """
        if transform:
            if self.training:
                cxr = torch.stack([self.train_transform(img) for img in cxr])
            else:
                cxr = torch.stack([self.val_transform(img) for img in cxr])
        
        z = self.cxr_encoder(cxr)
        z_student = self.student_head(z)
        return z_student
    
    def training_step(self, batch, batch_idx):
        """Training step with distillation objective"""
        cxr, ecg, labs_pct, labs_miss = batch
        labs = torch.cat([labs_pct, labs_miss], dim=1)
        
        B = cxr.size(0)
        
        # Get teacher's target (frozen)
        # Teacher targets
        with torch.no_grad():
            z_fused_t, _, _, z_c_t = self.teacher(cxr, ecg, labs, modality_mask=None)
            z_cxr_t = self.teacher.cxr_proj(z_c_t)
        
        z_s = self(cxr, transform=True)

        # Losses
        losses = self.loss_module(
            z_s=z_s,
            z_fused_t=z_fused_t,
            z_cxr_t=z_cxr_t,
        )
        loss = losses["distill_total"]

        # Main distillation objective
        self.log("student/train/distill_loss_total", loss,
                 on_step=True, on_epoch=True, prog_bar=True, batch_size=B)

        # Components
        self.log("student/train/distill_loss_mse", losses["distill_mse"],
                 on_epoch=True, batch_size=B)
        self.log("student/train/distill_loss_clip", losses["distill_clip"],
                 on_epoch=True, batch_size=B)
        self.log("student/train/distill_loss_anchor", losses["distill_anchor"],
                 on_epoch=True, batch_size=B)
        return loss
    
    def validation_step(self, batch, batch_idx):
        cxr, ecg, labs_pct, labs_miss = batch
        labs = torch.cat([labs_pct, labs_miss], dim=1)
        B = cxr.size(0)

        with torch.no_grad():
            z_fused_t, _, _, z_c_t = self.teacher(cxr, ecg, labs, modality_mask=None)
            z_cxr_t = self.teacher.cxr_proj(z_c_t)

        z_s = self(cxr, transform=True)

        losses = self.loss_module(
            z_s=z_s,
            z_fused_t=z_fused_t,
            z_cxr_t=z_cxr_t,
        )
        loss = losses["distill_total"]

        with torch.no_grad():
            cos_align = F.cosine_similarity(z_s, z_fused_t).mean()

        self.log("student/val/distill_loss_total", loss,
                 on_epoch=True, prog_bar=True, batch_size=B)
        self.log("student/val/distill_loss_mse", losses["distill_mse"],
                 on_epoch=True, batch_size=B)
        self.log("student/val/distill_loss_clip", losses["distill_clip"],
                 on_epoch=True, batch_size=B)
        self.log("student/val/distill_loss_anchor", losses["distill_anchor"],
                 on_epoch=True, batch_size=B)
        self.log("student/val/distill_alignment_cos", cos_align,
                 on_epoch=True, prog_bar=True, batch_size=B)
        return loss
    
    # ========================================================================
    # Optimizer
    # ========================================================================
    def configure_optimizers(self):
        cxr_params = [p for p in self.cxr_encoder.parameters() if p.requires_grad]
        params = cxr_params + list(self.student_head.parameters())

        n_trainable = sum(p.numel() for p in params if p.requires_grad)
        print(f"✓ Student trainable params: {n_trainable:,}")

        optimizer = torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            progress = (current_step - self.warmup_steps) / float(
                max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    # ========================================================================
    # Public API
    # ========================================================================
    
    def get_encoder(self):
        """Return physiology-aware CXR encoder"""
        return self.cxr_encoder


# ============================================================================
# Training Pipeline
# ============================================================================
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
try:
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    WandbLogger = None

def build_logger(
    project_name: str,
    save_dir: Path,
    run_name: Optional[str] = None,
):
    """
    Build a single (optional) WandbLogger instance shared by teacher + student.

    If wandb is not installed, returns None and Lightning will just use CSV/progress bar.
    """
    if WandbLogger is None:
        print("⚠ wandb not available, running without external logger.")
        return None

    if run_name is None:
        run_name = f"pacx_teacher_student_{time.strftime('%Y%m%d_%H%M%S')}"

    logger = WandbLogger(
        project=project_name,
        name=run_name,
        save_dir=str(save_dir),
        log_model=False,   # we're managing checkpoints ourselves
    )
    print(f"✓ Using WandbLogger run: {project_name}/{run_name}")
    return logger

def train_teacher(
    data_module: SymileMIMICDataModule,
    mae_checkpoint: str,
    save_dir: Path,
    max_epochs: int = 10,
    logger = None,
):
    """Stage 1: Train Teacher (multi-modal fusion with CLIP-style loss)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    model = TeacherModule(mae_checkpoint_path=mae_checkpoint)

    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="teacher-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision="16-mixed",
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        default_root_dir=str(save_dir),
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("STAGE 1: Training Teacher (Multi-Modal Fusion / CLIP-style)")
    print("=" * 80 + "\n")

    trainer.fit(model, datamodule=data_module)

    # Prefer best checkpoint if available, else fallback to last
    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt is None or best_ckpt == "":
        best_ckpt = os.path.join(str(save_dir), "last.ckpt")

    print(f"\n✓ Teacher training complete")
    print(f"✓ Teacher checkpoint: {best_ckpt}\n")

    return Path(best_ckpt)


def train_student(
    data_module: SymileMIMICDataModule,
    mae_checkpoint: str,
    teacher_checkpoint: Path,
    save_dir: Path,
    max_epochs: int = 20,
    logger = None,
):
    """Stage 2: Train Student (CXR-only distillation from teacher fused embedding)."""
    save_dir.mkdir(parents=True, exist_ok=True)

    model = StudentModule(
        mae_checkpoint_path=mae_checkpoint,
        teacher_checkpoint_path=str(teacher_checkpoint),
        data_dir=str(data_module.data_dir),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=save_dir,
        filename="student-{epoch:02d}-{val/loss:.4f}",
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        precision="16-mixed",
        accelerator="auto",
        devices=1,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        default_root_dir=str(save_dir),
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("STAGE 2: Training Student (Feature-level Distillation)")
    print("=" * 80 + "\n")

    trainer.fit(model, datamodule=data_module)

    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt is None or best_ckpt == "":
        best_ckpt = os.path.join(str(save_dir), "last.ckpt")

    print(f"\n✓ Student training complete")
    print(f"✓ Student checkpoint: {best_ckpt}\n")

    return Path(best_ckpt)

import argparse

def main(args):
    # ------------------------------------------------------------------
    # Paths / dirs
    # ------------------------------------------------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("✓ Initializing Symile-MIMIC data module...")
    data_module = SymileMIMICDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    data_module.setup("fit")

    # ------------------------------------------------------------------
    # Logger (single run for both stages)
    # ------------------------------------------------------------------
    logger = build_logger(
        project_name=args.project_name,
        save_dir=save_dir,
    )

    # ------------------------------------------------------------------
    # Stage 1: Teacher (optional)
    # ------------------------------------------------------------------
    if args.teacher_ckpt is not None and Path(args.teacher_ckpt).is_file():
        # Use existing teacher checkpoint
        teacher_ckpt = Path(args.teacher_ckpt)
        print("\n" + "=" * 80)
        print("USING EXISTING TEACHER CHECKPOINT")
        print("=" * 80)
        print(f"✓ teacher_ckpt: {teacher_ckpt}")
        print("=" * 80 + "\n")
    else:
        # Train teacher from MAE checkpoint
        print("\n" + "=" * 80)
        print("STAGE 1: TRAINING TEACHER")
        print("=" * 80 + "\n")

        teacher_ckpt = train_teacher(
            data_module=data_module,
            mae_checkpoint=args.mae_checkpoint,
            save_dir=save_dir / "teacher",
            max_epochs=args.max_epochs_teacher,
            logger=logger,
        )

        print("\n" + "=" * 80)
        print("TEACHER TRAINING COMPLETE")
        print("=" * 80)
        print(f"✓ Best teacher checkpoint: {teacher_ckpt}")
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Stage 2: Student
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STAGE 2: TRAINING STUDENT (DISTILLATION)")
    print("=" * 80 + "\n")

    student_ckpt = train_student(
        data_module=data_module,
        mae_checkpoint=args.mae_checkpoint,
        teacher_checkpoint=str(teacher_ckpt),
        save_dir=save_dir / "student",
        max_epochs=args.max_epochs_student,
        logger=logger,
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"✓ Final student checkpoint: {student_ckpt}")
    print("\nTo extract encoder:")
    print("  from train_teacher_student import StudentModule")
    print(f"  model = StudentModule.load_from_checkpoint('{student_ckpt}')")
    print("  encoder = model.get_encoder()")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PACX teacher+student")

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./src/symile-mimic-a-multimodal-clinical-dataset-of-chest-x-rays-electrocardiograms-and-blood-labs-from-mimic-iv-1.0.0/data_npy",
        help="Root directory of Symile-MIMIC npy files",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for teacher & student training",
    )

    # Checkpoints
    parser.add_argument(
        "--mae_checkpoint",
        type=str,
        default="./checkpoints/mae/last.ckpt",
        help="Path to MAE pretraining checkpoint for CXR encoder",
    )
    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        default=None,
        help="Optional: existing teacher checkpoint. "
             "If provided, teacher training is skipped.",
    )

    # Training config
    parser.add_argument(
        "--max_epochs_teacher",
        type=int,
        default=40,
        help="Number of training epochs for teacher",
    )
    parser.add_argument(
        "--max_epochs_student",
        type=int,
        default=30,
        help="Number of training epochs for student",
    )

    # Logging / saving
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./checkpoints/teacher_student",
        help="Base directory to save teacher and student checkpoints",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="PACX-Teacher-Student",
        help="W&B / logger project name",
    )

    args = parser.parse_args()
    main(args)
