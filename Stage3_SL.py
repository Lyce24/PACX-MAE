import torch

# --- Existing code ---
from Modules.data_modules import CheXpertDataModule

import argparse
from Scripts.MAE.mae_to_vit import get_vit_from_mae

from Modules.lightning_modules import ClassificationLightningModule

def main(args):
    data_module = CheXpertDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        task=args.task
    )

    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    vit_model = get_vit_from_mae(ckpt, global_pool=False)

    model = ClassificationLightningModule(
        model=vit_model,
        num_classes=1,
        model_weights=None,
        freeze_backbone=True,
        pos_weight=None,
        lr=1e-3,
        weight_decay=0.05,
        warmup_epochs=10,
        class_names=["COVID"]
    )

    # ---------- Test the Model ----------
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    batch = next(iter(train_loader))  # CheXpertDataModule returns only images
    imgs, labels = batch
    imgs = imgs.to(device)

    with torch.no_grad():
        out = model(imgs)

    print("Forward output type:", type(out))
    if isinstance(out, tuple) or isinstance(out, list):
        print("Tuple length:", len(out))
        for i, t in enumerate(out):
            if torch.is_tensor(t):
                print(f"  out[{i}] shape:", t.shape)
    else:
        if torch.is_tensor(out):
            print("Output shape:", out.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--train_csv", type=str, default="./data/covid_train_split.csv")
    parser.add_argument("--val_csv", type=str, default="./data/covid_val_split.csv")
    parser.add_argument("--test_csv", type=str, default="./data/covid_test_split.csv")
    parser.add_argument("--root_dir", type=str, default="../../.cache/kagglehub/datasets/andyczhao/covidx-cxr2/versions/9")
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--task", type=str, default="COVID")
    parser.add_argument("--ckpt_path", type=str, default="../../scratch/model_checkpoints/mae/mae_cxr_final.ckpt")

    args = parser.parse_args()
    
    main(args)