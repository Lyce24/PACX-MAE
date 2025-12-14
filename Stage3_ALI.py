import torch
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Import your datamodule
from Models.models import CXRModel

import os
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Callable

"""

*** Make sure you run Data/create_test_uni.ipynb first before this***

"""
class SymileMIMICDatasetV3(Dataset):
    def __init__(
        self, 
        data_dir: Path, 
        split: str, 
        transform: Optional[Callable] = None
    ):
        self.split = split
        # Using mmap_mode="r" keeps memory usage low
        self.cxr = np.load(data_dir / split / f"cxr_{split}.npy", mmap_mode="r")
        self.ecg = np.load(data_dir / split / "ecg_features.npy", mmap_mode="r")
        self.labs = np.load(data_dir / split / "labs_features.npy", mmap_mode="r")
        
        self.transform = transform
        self.N = len(self.cxr)

        assert len(self.cxr) == len(self.ecg) == len(self.labs)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # 1. Load data
        # Note: Copying avoids stride issues with negative indexing in transforms
        cxr_img = self.cxr[idx].copy() 
        
        # 2. Apply Transforms (Augmentation/Normalization)
        if self.transform:
            # Transforms usually expect Tensor [C, H, W] or PIL
            cxr_tensor = torch.from_numpy(cxr_img).float()
            if cxr_tensor.ndim == 2:
                cxr_tensor = cxr_tensor.unsqueeze(0) # Add channel dim if missing
            
            cxr = self.transform(cxr_tensor)
        else:
            # Fallback if no transform provided
            cxr = torch.from_numpy(cxr_img).float()

        # 3. Load Features (Frozen embeddings)
        # --- FIX: ADD .copy() TO MAKE ARRAYS WRITABLE ---
        ecg_features = torch.from_numpy(self.ecg[idx].copy()).float()   # [1024]
        labs_features = torch.from_numpy(self.labs[idx].copy()).float() # [256]

        return cxr, ecg_features, labs_features

class SymileMIMICDataModuleV3(pl.LightningDataModule):
    def __init__(
        self, 
        data_dir: str, 
        batch_size: int = 256, 
        num_workers: Optional[int] = None,
        img_size: int = 224,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.img_size = img_size

        if num_workers is None:
            try:
                self.num_workers = len(os.sched_getaffinity(0))
            except AttributeError:
                self.num_workers = 4
        else:
            self.num_workers = num_workers
            
    def get_val_transforms(self):
        return T.Compose([
            T.Resize((self.img_size, self.img_size), antialias=True),
        ])

    def setup(self, stage=None):
        if stage in ("fit", None):
            self.ds_train = SymileMIMICDatasetV3(
                self.data_dir, 
                split="train", 
                transform=self.get_val_transforms()
            )
            self.ds_val = SymileMIMICDatasetV3(
                self.data_dir, 
                split="val", 
                transform=self.get_val_transforms()
            )
        
        if stage in ("test", None):
            self.ds_test = SymileMIMICDatasetV3(
                self.data_dir, 
                split="test_uni", 
                transform=self.get_val_transforms()
            )

    def train_dataloader(self):
        return DataLoader(
            self.ds_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.ds_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
        )

def extract_features(model, dataloader, device):
    """
    Runs the full test set through the model to collect:
    1. Frozen CXR features (from your model)
    2. Ground Truth ECG/Lab features (from the dataset)
    """
    model.eval()
    model.to(device)
    
    cxr_embeddings = []
    ecg_targets = []
    lab_targets = []
    
    print(f"Extracting features on {device}...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            # Unpack batch according to your Dataset.__getitem__
            cxr_imgs, ecg_feats, lab_feats = batch
            
            cxr_imgs = cxr_imgs.to(device)
            z_cxr = model.backbone(cxr_imgs)
            
            # Handle ViT output: if [B, N, D], take CLS token [B, 0, :] or Mean [B, :]
            if z_cxr.ndim == 3:
                z_cxr = z_cxr[:, 0, :] 
                
            cxr_embeddings.append(z_cxr.cpu().numpy())
            ecg_targets.append(ecg_feats.detach().cpu().numpy())
            lab_targets.append(lab_feats.detach().cpu().numpy())
            
    # Concatenate into large matrices
    X_cxr = np.vstack(cxr_embeddings)
    Y_ecg = np.vstack(ecg_targets)
    Y_lab = np.vstack(lab_targets)
    
    return X_cxr, Y_ecg, Y_lab

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def _check_array(name, A):
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {A.shape}")
    if not np.isfinite(A).all():
        bad = np.where(~np.isfinite(A))
        raise ValueError(f"{name} contains NaN/Inf at (first 5): {list(zip(bad[0][:5], bad[1][:5]))}")
    return A.astype(np.float32, copy=False)


def _row_norm(x, eps=1e-8):
    return np.linalg.norm(x, axis=1, keepdims=True) + eps


def _cosine_rows(A, B, eps=1e-8):
    A_n = A / _row_norm(A, eps)
    B_n = B / _row_norm(B, eps)
    return np.sum(A_n * B_n, axis=1)


def _row_hash_keys(Y):
    # exact-row hash (works for exact float duplicates)
    Yc = np.ascontiguousarray(Y)
    return Yc.view(np.dtype((np.void, Yc.dtype.itemsize * Yc.shape[1]))).reshape(-1)

def evaluate_physiology(
    X_train, Y_train,
    X_test, Y_test,
    task_name="ECG",
    alpha=10.0,
    retrieval_ks=(1, 5, 10),
    seed=0,
):
    print(f"\n=== Evaluating {task_name} Alignment ===")

    X_train = _check_array("X_train", X_train)
    Y_train = _check_array("Y_train", Y_train)
    X_test  = _check_array("X_test",  X_test)
    Y_test  = _check_array("Y_test",  Y_test)

    n_tr, d_x = X_train.shape
    n_te, d_y = Y_test.shape
    print(f"[Shapes] X_train={X_train.shape}, Y_train={Y_train.shape}, X_test={X_test.shape}, Y_test={Y_test.shape}")

    # --------------------------
    # Duplication stats on Y_test
    # --------------------------
    keys = _row_hash_keys(Y_test)
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    print(f"[Y dups] unique_rows={len(uniq)}/{n_te}, avg_count={counts.mean():.3f}, max_count={counts.max()}")

    # --------------------------
    # Train stable ridge probe
    # --------------------------
    print(f"Training Linear Probe (Standardize+Ridge(svd)) on {n_tr} samples... alpha={alpha}")
    probe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        Ridge(alpha=alpha, solver="svd")
    )
    probe.fit(X_train, Y_train)

    print(f"Predicting on {n_te} test samples...")
    Y_pred = probe.predict(X_test).astype(np.float32, copy=False)

    # --------------------------
    # Metrics: R2 + cosine
    # --------------------------
    r2 = r2_score(Y_test, Y_pred, multioutput="uniform_average")

    # 1. Raw Cosine Similarity (Prediction vs Truth)
    cos_raw = _cosine_rows(Y_pred, Y_test)
    avg_cos_raw = float(cos_raw.mean())

    # 2. Centered Cosine Similarity (Prediction vs Truth)
    # Removing the mean prevents inflation from the "average physiological profile"
    mu = Y_test.mean(axis=0, keepdims=True)
    Y_pred_c = Y_pred - mu
    Y_test_c = Y_test - mu

    cos_ctr = _cosine_rows(Y_pred_c, Y_test_c)
    avg_cos_ctr = float(cos_ctr.mean())

    # --------------------------
    # Baselines: Random Pair Similarity (Noise Floor)
    # --------------------------
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_te)

    # A. Raw Random Baseline: Average cosine between two random patients
    # This tells you: "How similar are two random people's ECGs/Labs?"
    Y_test_n = Y_test / _row_norm(Y_test)
    rand_cos_raw = float(np.sum(Y_test_n * Y_test_n[perm], axis=1).mean())

    # B. Centered Random Baseline: Average centered cosine between two random patients
    # This should be close to 0.0. If not, your data has residual structure or duplicates.
    Y_test_c_n = Y_test_c / _row_norm(Y_test_c)
    rand_cos_ctr = float(np.sum(Y_test_c_n * Y_test_c_n[perm], axis=1).mean())

    # --------------------------
    # Print Results
    # --------------------------
    print(f"[Baseline] Random Pair (Raw):      {rand_cos_raw:.4f}  (High due to common mean)")
    print(f"[Baseline] Random Pair (Centered): {rand_cos_ctr:.4f}  (Should be approx 0.0)")
    print("-" * 30)
    print(f"[Results]  R2 Score:               {r2:.4f}")
    print(f"[Results]  Cosine (Raw):           {avg_cos_raw:.4f}")
    print(f"[Results]  Cosine (Centered):      {avg_cos_ctr:.4f}")

    # --------------------------
    # Retrieval: similarity matrix
    # --------------------------
    Y_pred_n = Y_pred / _row_norm(Y_pred)
    sim = Y_pred_n @ Y_test_n.T  # [N, N]

    ks = list(retrieval_ks)

    # (A) Identity retrieval (your old metric) — keep but label as flawed
    id_hits = {k: 0 for k in ks}
    for i in range(n_te):
        scores = sim[i]
        for k in ks:
            topk = np.argpartition(scores, -k)[-k:]
            if i in topk:
                id_hits[k] += 1

    for k in ks:
        print(f"    R@{k}: {id_hits[k] / n_te * 100:.2f}%")

# 1. Load Data
model_mode = "imagenet"  # or "pacx", "imagenet"
backbone_name = "vit_base_patch16_224"
data_dir = "../../scratch/physionet.org/files/symile-mimic/1.0.0/data_npy"
model_weights_path = "./checkpoints/mae/last.ckpt"  if model_mode == "mae" else "./checkpoints/mae_vit_lora_merged_v3.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dm = SymileMIMICDataModuleV3(
    data_dir=data_dir, 
    batch_size=256,
)

dm.setup()

train_loader = dm.train_dataloader()
test_loader = dm.test_dataloader() # or val_dataloader depending on protocol

# 2. Load Model
print("Loading model...")
# INSERT YOUR MODEL LOADING LOGIC HERE
model = CXRModel(
            num_classes=100,
            mode=model_mode,
            model_checkpoints=model_weights_path,
        )
model.eval()

# 3. Extract Features (TRAIN)
print("Extracting TRAIN features...")
X_train, Y_ecg_train, Y_lab_train = extract_features(model, train_loader, device)

# 4. Extract Features (TEST)
print("Extracting TEST features...")
X_test, Y_ecg_test, Y_lab_test = extract_features(model, test_loader, device)


# Task A: Verify ECG Alignment
evaluate_physiology(X_train, Y_ecg_train, X_test, Y_ecg_test, task_name="ECG")

# Task B: Verify Labs Alignment
evaluate_physiology(X_train, Y_lab_train, X_test, Y_lab_test, task_name="Labs")