import os
import random
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import precision_score, recall_score, f1_score

# === Seeding ===
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# === Folder Flattening ===
def flatten_distortion_folders(root_dir):
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        distortion_path = os.path.join(class_path, "distortion")
        if os.path.isdir(distortion_path):
            for img_file in os.listdir(distortion_path):
                src = os.path.join(distortion_path, img_file)
                dst = os.path.join(class_path, img_file)
                if os.path.isfile(src):
                    shutil.move(src, dst)
            os.rmdir(distortion_path)

# === Dataset ===
class SiameseDistortionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.image_paths = {
            cls: sorted([
                os.path.join(root_dir, cls, img)
                for img in os.listdir(os.path.join(root_dir, cls))
                if os.path.isfile(os.path.join(root_dir, cls, img))
            ]) for cls in self.classes
        }

        self.pairs = []
        self.labels = []
        for cls in self.classes:
            same_class_imgs = self.image_paths[cls]
            for i in range(len(same_class_imgs) - 1):
                self.pairs.append((same_class_imgs[i], same_class_imgs[i + 1]))
                self.labels.append(1)
                other_classes = [c for c in self.classes if c != cls]
                random.shuffle(other_classes)
                for diff_cls in other_classes:
                    if self.image_paths[diff_cls]:
                        diff_img = self.image_paths[diff_cls][i % len(self.image_paths[diff_cls])]
                        self.pairs.append((same_class_imgs[i], diff_img))
                        self.labels.append(0)
                        break

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        img1 = np.array(Image.open(img1_path).convert('RGB'))
        img2 = np.array(Image.open(img2_path).convert('RGB'))
        if self.transform:
            img1 = self.transform(image=img1)['image']
            img2 = self.transform(image=img2)['image']
        return img1, img2, label

# === Model ===
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model('convnext_atto', pretrained=True, num_classes=0)
        self.projection = nn.Sequential(
            nn.Linear(self.encoder.num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        feat = self.encoder(x)
        feat = self.projection(feat)
        feat = F.normalize(feat, p=2, dim=1)
        return feat

    def forward(self, x1, x2):
        return self.forward_once(x1), self.forward_once(x2)

# === Loss ===
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, feat1, feat2, label):
        distance = F.pairwise_distance(feat1, feat2)
        loss = label * distance.pow(2) + (1 - label) * (torch.clamp(self.margin - distance, min=0.0).pow(2))
        return loss.mean()

# === Metrics ===
def compute_all_metrics(preds, labels):
    preds_np = np.array(preds)
    labels_np = np.array(labels)
    acc = (preds_np == labels_np).mean()
    precision = precision_score(labels_np, preds_np, zero_division=0)
    recall = recall_score(labels_np, preds_np, zero_division=0)
    f1 = f1_score(labels_np, preds_np, zero_division=0)
    return acc, precision, recall, f1

# === Main training code ===
def main():
    seed_everything(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    train_dir = "data/Task_B/train"
    val_dir = "data/Task_B/val"
    flatten_distortion_folders(train_dir)
    flatten_distortion_folders(val_dir)

    # Transforms
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.HueSaturationValue(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    # Datasets and Dataloaders
    g = torch.Generator()
    g.manual_seed(42)

    train_dataset = SiameseDistortionDataset(root_dir=train_dir, transform=train_transform)
    val_dataset = SiameseDistortionDataset(root_dir=val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    # Model setup
    model = SiameseNet().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler()

    best_acc = 0
    patience = 5
    counter = 0

    for epoch in range(50):
        model.train()
        all_train_preds, all_train_labels = [], []
        total_train_loss = 0

        for img1, img2, label in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/50"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            optimizer.zero_grad()
            with autocast():
                feat1, feat2 = model(img1, img2)
                loss = criterion(feat1, feat2, label)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            with torch.no_grad():
                distance = F.pairwise_distance(feat1, feat2)
                preds = (distance < 0.5).float()
                all_train_preds.extend(preds.cpu().numpy())
                all_train_labels.extend(label.cpu().numpy())

        train_acc, train_prec, train_recall, train_f1 = compute_all_metrics(all_train_preds, all_train_labels)
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        all_val_preds, all_val_labels = [], []
        total_val_loss = 0
        with torch.no_grad():
            for img1, img2, label in tqdm(val_loader, desc=f"[Val] Epoch {epoch+1}/50"):
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                feat1, feat2 = model(img1, img2)
                loss = criterion(feat1, feat2, label)
                total_val_loss += loss.item()

                distance = F.pairwise_distance(feat1, feat2)
                preds = (distance < 0.5).float()
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(label.cpu().numpy())

        val_acc, val_prec, val_recall, val_f1 = compute_all_metrics(all_val_preds, all_val_labels)
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}, Precision={train_prec:.4f}, Recall={train_recall:.4f}, F1={train_f1:.4f}")
        print(f"Epoch {epoch+1}: Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}, Precision={val_prec:.4f}, Recall={val_recall:.4f}, F1={val_f1:.4f}\n")

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs("comsys-hackathon-taskB/weights", exist_ok=True)
            torch.save(model.state_dict(), 'comsys-hackathon-taskB/weights/best_siamese_convnext.pt')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

# === Entry Point ===
if __name__ == "__main__":
    main()
