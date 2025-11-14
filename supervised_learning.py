import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm

###############################
# Config
###############################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 14          # e.g., CheXpert-style 14 labels
IMAGE_SIZE = (3, 224, 224)
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4

CKPT_PATH = "vit_pretrained.ckpt"  # <- put your local backbone weights here
BEST_MODEL_PATH = "best_linear_probe_vit.pth"

torch.manual_seed(42)


###############################
# Create Dataset + Dataloader
###############################

class RandomXrayDataset(Dataset):
    """
    Dummy dataset: returns random images and random multi-label targets.
    Replace this with your real X-ray dataset later.
    """
    def __init__(self, num_samples: int, num_classes: int, image_size=(3, 224, 224)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.c, self.h, self.w = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random "image"
        img = torch.randn(self.c, self.h, self.w)

        # Random multi-label target: 0/1 per class
        # (you can adjust the probability of positives if needed)
        target = torch.randint(low=0, high=2, size=(self.num_classes,)).float()
        return img, target


def create_dataloaders(batch_size: int, num_classes: int, image_size=(3, 224, 224)):
    train_ds = RandomXrayDataset(num_samples=2000, num_classes=num_classes, image_size=image_size)
    val_ds   = RandomXrayDataset(num_samples=500,  num_classes=num_classes, image_size=image_size)
    test_ds  = RandomXrayDataset(num_samples=500,  num_classes=num_classes, image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


###############################
# Model Definition
###############################

class LinearProbeViT(nn.Module):
    """
    ViT backbone from timm with a new trainable linear head on top.
    Backbone is frozen -> linear probing.
    """
    def __init__(self, num_classes: int, ckpt_path: str = None):
        super().__init__()

        # Create ViT backbone with no classification head (num_classes=0)
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
            num_classes=0  # remove classifier head
        )

        # Load backbone weights from local checkpoint (if provided)
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print(f"Loading backbone weights from: {ckpt_path}")
            state = torch.load(ckpt_path, map_location="cpu")
            # If your checkpoint is a dict with 'state_dict', adjust accordingly.
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
                # If keys are prefixed (e.g., 'model.'), you may need to strip those.
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print("Backbone loaded. Missing keys:", missing)
            print("Unexpected keys:", unexpected)
        else:
            print("No valid checkpoint provided, using random init for backbone.")

        # Freeze backbone params for linear probing
        for p in self.backbone.parameters():
            p.requires_grad = False

        # New trainable linear head
        in_features = self.backbone.num_features
        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # timm ViT with num_classes=0 returns the CLS embedding
        feats = self.backbone(x)          # [B, num_features]
        logits = self.head(feats)         # [B, num_classes]
        return logits


###############################
# Training + Validation Loop
###############################

def multi_label_accuracy(logits, targets, threshold=0.5):
    """
    Simple multi-label accuracy:
    - logits: [B, C]
    - targets: [B, C] with 0/1
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        correct = (preds == targets).float().mean()  # average over batch & classes
    return correct.item()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    for imgs, targets in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        acc = multi_label_accuracy(logits, targets)
        running_loss += loss.item()
        running_acc += acc
        num_batches += 1

    avg_loss = running_loss / num_batches
    avg_acc = running_acc / num_batches
    return avg_loss, avg_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, targets)

            acc = multi_label_accuracy(logits, targets)
            running_loss += loss.item()
            running_acc += acc
            num_batches += 1

    avg_loss = running_loss / num_batches
    avg_acc = running_acc / num_batches
    return avg_loss, avg_acc


###############################
# Main Training Loop
###############################

def run_training():
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        image_size=IMAGE_SIZE,
    )

    model = LinearProbeViT(num_classes=NUM_CLASSES, ckpt_path=CKPT_PATH)
    model.to(DEVICE)

    # Multi-label classification -> BCEWithLogitsLoss (no sigmoid in model)
    criterion = nn.BCEWithLogitsLoss()

    # Only train the linear head
    optimizer = torch.optim.Adam(
        model.head.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # Save best model based on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New best model saved to {BEST_MODEL_PATH}")

    # Return test_loader so we can evaluate after training
    return test_loader


###############################
# Main Test Loop
###############################

def test(model_path: str, test_loader, device):
    model = LinearProbeViT(num_classes=NUM_CLASSES, ckpt_path=None)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()

    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = model(imgs)
            loss = criterion(logits, targets)

            acc = multi_label_accuracy(logits, targets)
            running_loss += loss.item()
            running_acc += acc
            num_batches += 1

    avg_loss = running_loss / num_batches
    avg_acc = running_acc / num_batches

    print(f"Test Loss: {avg_loss:.4f} | Test Acc: {avg_acc:.4f}")


###############################
# Entry Point
###############################

if __name__ == "__main__":
    test_loader = run_training()
    if os.path.isfile(BEST_MODEL_PATH):
        test(BEST_MODEL_PATH, test_loader, DEVICE)
    else:
        print("No best model found, skipping test.")
