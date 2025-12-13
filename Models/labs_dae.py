import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW

class LabsDenoisingAE(nn.Module):
    """
    Mask-aware denoising autoencoder for lab percentiles.
    Encoder sees (x, m); decoder reconstructs x only.
    """

    def __init__(self, input_dim: int, latent_dim: int = 256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.GELU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),
        )

    def forward(self, x, m):
        z = self.encoder(torch.cat([x, m], dim=1))
        x_hat = self.decoder(z)
        return z, x_hat
    
    def encode(self, x, m):
        return self.encoder(torch.cat([x, m], dim=1))

class LabsDAELightning(pl.LightningModule):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 256,
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        warmup_steps: int = 200,
        corruption_prob: float = 0.15,  # denoising strength
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = LabsDenoisingAE(input_dim, latent_dim)

    def forward(self, x, m):
        return self.model(x, m)

    def _corrupt(self, x, m):
        """
        m: observed mask (1=present, 0=missing)
        Returns: x_corrupt, m_corrupt, drop_mask (bool)
        """
        x = x.float()
        m = m.float()

        # don't let any placeholder at missing positions leak in
        x = x * m

        rand = torch.rand_like(m)
        drop_mask = (rand < self.hparams.corruption_prob) & (m > 0.5)  # only drop observed

        x_corrupt = x.clone()
        m_corrupt = m.clone()

        x_corrupt[drop_mask] = 0.0
        m_corrupt[drop_mask] = 0.0

        return x_corrupt, m_corrupt, drop_mask

    def _masked_mse(self, x_hat, x, m):
        diff = (x_hat - x) ** 2
        return (diff * m).sum() / (m.sum() + 1e-6)

    def training_step(self, batch, batch_idx):
        x, m = batch

        x_corrupt, m_corrupt, _ = self._corrupt(x, m)
        _, x_hat = self(x_corrupt, m_corrupt)

        loss = self._masked_mse(x_hat, x, m)

        self.log("train/loss", loss, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, m = batch
        
        # --- Task A: Clean Reconstruction (Autoencoder) ---
        # Checks if model can compress/decompress without losing info
        z_clean, x_hat_clean = self(x, m)
        clean_loss = self._masked_mse(x_hat_clean, x, m)
        
        # --- Task B: Denoising Performance (Imputation) ---
        # Checks if model understands correlations (e.g., inferring missing values)
        x_corrupt, m_corrupt, drop_mask = self._corrupt(x, m)
        _, x_hat_denoise = self(x_corrupt, m_corrupt)
        
        # Loss 1: Overall Denoising Error (Global structure)
        denoise_loss = self._masked_mse(x_hat_denoise, x, m)
        
        # Loss 2: Imputation Specific Error (Did it guess the missing parts right?)
        # We look specifically at the indices where drop_mask is True
        if drop_mask.sum() > 0:
            imputation_error = (x_hat_denoise - x) ** 2
            imputation_mse = imputation_error[drop_mask].mean()
        else:
            imputation_mse = torch.tensor(0.0, device=x.device)

        # --- Task C: Latent Space Health ---
        # Check for posterior collapse (if std is near 0, the encoder is dead)
        z_std = z_clean.std()

        # Logging
        self.log_dict({
            "val/clean_recon_mse": clean_loss,
            "val/denoising_mse": denoise_loss,
            "val/imputation_mse": imputation_mse, # The most important metric for you
            "val/z_std": z_std
        }, prog_bar=True)
        
        return denoise_loss
    
    def test_step(self, batch, batch_idx):
        # Mirror validation logic for testing
        x, m = batch
        
        # Clean
        _, x_hat_clean = self(x, m)
        clean_loss = self._masked_mse(x_hat_clean, x, m)
        
        # Denoising
        x_corrupt, m_corrupt, drop_mask = self._corrupt(x, m)
        _, x_hat_denoise = self(x_corrupt, m_corrupt)
        
        if drop_mask.sum() > 0:
            imputation_error = (x_hat_denoise - x) ** 2
            imputation_mse = imputation_error[drop_mask].mean()
        else:
            imputation_mse = torch.tensor(0.0, device=x.device)

        self.log_dict({
            "test/clean_recon_mse": clean_loss,
            "test/imputation_mse": imputation_mse
        })

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        def lr_lambda(step):
            if step < self.hparams.warmup_steps:
                return step / max(1, self.hparams.warmup_steps)
            progress = (
                step - self.hparams.warmup_steps
            ) / max(1, self.trainer.estimated_stepping_batches - self.hparams.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

# Load tensors
labs_pct_train = torch.load("../../../scratch/physionet.org/files/symile-mimic/1.0.0/data_npy/train/labs_percentiles_train.pt").float()
labs_miss_train = torch.load("../../../scratch/physionet.org/files/symile-mimic/1.0.0/data_npy/train/labs_missingness_train.pt").float()

labs_pct_val = torch.load("../../../scratch/physionet.org/files/symile-mimic/1.0.0/data_npy/val/labs_percentiles_val.pt").float()
labs_miss_val = torch.load("../../../scratch/physionet.org/files/symile-mimic/1.0.0/data_npy/val/labs_missingness_val.pt").float()

labs_pct_test = torch.load("../../../scratch/physionet.org/files/symile-mimic/1.0.0/data_npy/test/labs_percentiles_test.pt").float()
labs_miss_test = torch.load("../../../scratch/physionet.org/files/symile-mimic/1.0.0/data_npy/test/labs_missingness_test.pt").float()
# Datasets
train_ds = TensorDataset(labs_pct_train, labs_miss_train)
val_ds   = TensorDataset(labs_pct_val, labs_miss_val)
test_ds  = TensorDataset(labs_pct_test, labs_miss_test)

train_loader = DataLoader(
    train_ds,
    batch_size=256,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

val_loader = DataLoader(
    val_ds,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

test_loader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

model = LabsDAELightning(
    input_dim=labs_pct_train.shape[1],
    latent_dim=256,
    corruption_prob=0.15,
)

# add wandb logger
from lightning.pytorch.loggers import WandbLogger
wandb_logger = WandbLogger(project="Labs_Denoising_AE", log_model="all")

trainer = pl.Trainer(
    accelerator="auto",
    devices=1,
    precision="16-mixed",
    max_epochs=50,
    logger=wandb_logger,
    log_every_n_steps=50,
    default_root_dir="../src/labs_dae",
)

trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)