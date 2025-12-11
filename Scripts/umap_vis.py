import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Models.models import CXRModel
from Data.data_modules import CXRDataModule

import argparse
import numpy as np
import matplotlib.pyplot as plt
import umap
import torch

def plot_umap_all_targets(latent_embeddings, y, target_names=None, save_path=None):
    """
    Plots one UMAP projection of latent_embeddings, overlaying all targets in different colors.
    (Reference: https://github.com/Lyce24/Surv_MAC/blob/main/Surv_MAC.ipynb)
    
    Args:
      latent_embeddings (np.ndarray): [n_samples, latent_dim]
      y (np.ndarray): [n_samples, n_targets], e.g. binary indicators per target
      target_names (list of str]): optional names length n_targets
      save_path (str): optional path to save the figure (e.g. "umap_all_targets.png")
    """
    # Ensure numpy
    y_np = y if isinstance(y, np.ndarray) else y.numpy()
    n_targets = y_np.shape[1]

    # 1. Compute UMAP once
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_emb = reducer.fit_transform(latent_embeddings)

    # 2. Choose a qualitative colormap with enough distinct colors
    #    'tab10' has 10 colors; for more targets try 'tab20' or a custom palette
    cmap = plt.get_cmap('tab10')
    
    plt.figure(figsize=(10, 8))
    
    # 3. Plot each target's points
    for i in range(n_targets):
        mask = y_np[:, i] == 1     # or whatever criterion defines membership
        color = cmap(i % cmap.N)
        label = target_names[i] if target_names else f"Target {i}"
        
        plt.scatter(
            umap_emb[mask, 0],
            umap_emb[mask, 1],
            c=[color],
            label=label,
            s=10,
            alpha=0.7,
            edgecolors='none'
        )
    
    plt.title("UMAP of Latents: All Targets Overlayed")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -------------------------
    # Data module
    # -------------------------
    data_module = CXRDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        task=args.task,
    )

    data_module.setup()
    test_loader = data_module.test_dataloader()

    save_path = args.output_dir + f"/{args.task}_umap.png"
    
    label_col = data_module.train_df.columns.difference(["Path"])[0]
    num_classes = int(data_module.train_df[label_col].nunique())

    if args.task == "COVIDQU":
        label_map = {
            0: "Normal",
            1: "COVID-19",
            2: "Non-COVID",
        }
    elif args.task == "TB":
        label_map = {
            0: "Normal",
            1: "Abnormal",
        }
    elif args.task == "PNE":
        label_map = {
            0: "Normal", 
            1: "Pneumonia",
        }
    elif args.task == "CHESTX6":
        label_map = {
            0: "Covid-19", 
            1: "Emphysema",
            2: "Normal",
            3: "Pneumonia-Bacterial",
            4: "Pneumonia-Viral",
            5: "Tuberculosis"
        }
    else:
        label_map = {i: f"Class {i}" for i in range(num_classes)}
    target_names = [label_map[i] for i in sorted(data_module.train_df[label_col].unique())]

    # -------------------------
    # Load MAE encoder model
    # -------------------------
    encoder = CXRModel(
        num_classes=num_classes,
        mode="mae",
        backbone_name="vit_base_patch16_224",
        model_checkpoints=args.ckpt_path,
        unfreeze_backbone=False,
        task="sl",
    ).to(device)

    encoder.eval()

    # -------------------------
    # Extract embeddings + labels
    # -------------------------
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            feats = encoder.backbone(x) 

            all_embeddings.append(feats.cpu().numpy())

            if y is not None:
                y = y.to(device)

                if y.ndim == 1 or y.shape[1] == 1:
                    y = torch.nn.functional.one_hot(
                        y.view(-1).long(),
                        num_classes=num_classes
                    ).float()

                all_labels.append(y.cpu().numpy())

    latent_embeddings = np.concatenate(all_embeddings, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print("Latent shape:", latent_embeddings.shape)
    print("Label shape:", y.shape)

    # -------------------------
    # UMAP
    # -------------------------
    plot_umap_all_targets(
        latent_embeddings=latent_embeddings,
        y=y,
        target_names=target_names,
        save_path=save_path
    )

    print("UMAP fig saved at:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="COVIDQU")
    parser.add_argument("--train_csv", type=str, default="/users/kmaeda2/scratch/QUEx/covidqu_train_split.csv")
    parser.add_argument("--val_csv", type=str, default="/users/kmaeda2/scratch/QUEx/covidqu_val_split.csv")
    parser.add_argument("--test_csv", type=str, default="/users/kmaeda2/scratch/QUEx/covidqu_test_split.csv")
    parser.add_argument("--root_dir", type=str, default="/users/kmaeda2/scratch/QUEx/Lung Segmentation Data/Lung Segmentation Data")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--ckpt_path", type=str, default="/users/kmaeda2/scratch/last.ckpt")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./umap_plots",
    )

    args = parser.parse_args()
    
    main(args)