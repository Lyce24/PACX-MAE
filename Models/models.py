import timm
import torch
import torch.nn as nn

from Scripts.MAE.mae_to_vit import get_vit_from_mae

class CXRModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        mode: str = "imagenet",               # "imagenet" or "mae"
        backbone_name: str = "vit_base_patch16_224",
        model_weights: str | None = None,     # only used if mode == "mae"
        freeze_backbone: bool = True,
    ):
        super().__init__()

        # --------- 1. Build backbone inside the class ----------
        if mode == "imagenet":
            backbone = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,   # feature extractor
            )
        elif mode == "mae":
            if model_weights is None:
                raise ValueError("model_weights must be provided when mode='mae'")
            mae_ckpt = torch.load(model_weights, map_location="cpu", weights_only=False)
            backbone = get_vit_from_mae(mae_ckpt, global_pool=False)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Remove classifier head if any
        if hasattr(backbone, "reset_classifier"):
            backbone.reset_classifier(0)
        elif hasattr(backbone, "head"):
            backbone.head = nn.Identity()

        self.backbone = backbone

        # --------- 2. Freeze backbone if linear probing ----------
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # --------- 3. New classifier head ----------
        in_dim = getattr(self.backbone, "num_features", None)
        if in_dim is None:
            in_dim = getattr(self.backbone, "embed_dim")
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        feats = self.backbone(x)     # [B, C]
        logits = self.head(feats)    # [B, num_classes]
        return logits
