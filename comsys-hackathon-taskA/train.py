import os
import random
from PIL import Image
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FaceGenderDataset(Dataset):
    def __init__(self, root_dir, transform=None, undersample=False):
        self.samples = []
        self.transform = transform

        male_paths = []
        female_paths = []

        for gender_str in ['male', 'female']:
            label = 0 if gender_str == 'male' else 1
            folder = os.path.join(root_dir, gender_str)
            if not os.path.exists(folder):
                print(f"Warning: Directory not found -> {folder}")
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(folder, fname)
                    if label == 0:
                        male_paths.append((path, label))
                    else:
                        female_paths.append((path, label))

        if undersample and male_paths and female_paths:
            min_count = min(len(male_paths), len(female_paths))
            male_paths = random.sample(male_paths, min_count)
            female_paths = random.sample(female_paths, min_count)

        self.samples = male_paths + female_paths
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image, torch.tensor(label).float()


class FocalLossWithSmoothing(nn.Module):
    def __init__(self, smoothing=0.1, alpha=1, gamma=2):
        super().__init__()
        self.smoothing = smoothing
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma * bce)
        return focal_loss.mean()


class ConvNeXtGenderClassifier(nn.Module):
    def __init__(self, backbone='convnext_tiny'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.classifier = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(1)


def main():
    seed_everything()

    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_dataset = FaceGenderDataset("data/Task_A/train", transform=train_transform, undersample=True)
    val_dataset = FaceGenderDataset("data/Task_A/val", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvNeXtGenderClassifier().to(device)

    criterion = FocalLossWithSmoothing(smoothing=0.1)

    param_optimizer = [
        {'params': model.backbone.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ]

    optimizer = optim.AdamW(param_optimizer)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)
    scaler = GradScaler()

    best_val_acc = 0.0
    best_model_state = None
    epochs = 50
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    patience = 7
    patience_counter = 0

    print("\n" + "=" * 20 + " Training Upgraded ConvNeXt with Early Stopping " + "=" * 20 + "\n")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        all_train_preds = []
        all_train_labels = []


        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
            all_train_preds.extend(preds.cpu().numpy())
            all_train_labels.extend(labels.cpu().numpy())


        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_precision = precision_score(all_train_labels, all_train_preds, zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)



        model.eval()
        val_running_loss = 0.0
        val_correct, val_total = 0, 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(outputs) > 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
                val_total += labels.size(0)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())


        val_loss = val_running_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_precision = precision_score(all_val_labels, all_val_preds, zero_division=0)
        val_recall = recall_score(all_val_labels, all_val_preds, zero_division=0)
        val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)


        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Prec: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"✅ Best model updated (val_acc: {best_val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⚠️ No improvement in val_loss for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print(f"\n🛑 Stopping early! No improvement in validation loss for {patience} consecutive epochs.")
                break

    if best_model_state is not None:
        os.makedirs("weights", exist_ok=True)
        torch.save(best_model_state, "weights/best_convnext_gender_model.pth")
        print("\n" + "=" * 50)
        print("✅ Best model saved as 'best_convnext_gender_model.pth'")
        print(f"🏆 Final Best Validation Accuracy: {best_val_acc:.2%}")
        print("=" * 50)
    else:
        print("⚠️ No best model state was saved!")


if __name__ == "__main__":
    main()
