import os
import random
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    # Seeding made fixed
    def seed_everything(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    seed_everything(42)

    # Dataset with optional undersampling
    class FaceGenderDataset(Dataset):
        def __init__(self, root_dir, transform=None, undersample=False):
            self.samples = []
            self.transform = transform

            male_paths = []
            female_paths = []

            for gender_str in ['male', 'female']:
                label = 0 if gender_str == 'male' else 1
                folder = os.path.join(root_dir, gender_str)
                for fname in os.listdir(folder):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        path = os.path.join(folder, fname)
                        if label == 0:
                            male_paths.append((path, label))
                        else:
                            female_paths.append((path, label))

            if undersample:
                min_count = min(len(male_paths), len(female_paths))
                male_paths = random.sample(male_paths, min_count)
                female_paths = random.sample(female_paths, min_count)

            self.samples = male_paths + female_paths
            random.shuffle(self.samples)

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor(label).float()

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Data Loaders
    g = torch.Generator()
    g.manual_seed(42)

    train_dataset = FaceGenderDataset("/kaggle/working/Comys_Hackathon5/Task_A/train", transform=train_transform, undersample=True)
    val_dataset = FaceGenderDataset("/kaggle/working/Comys_Hackathon5/Task_A/train", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Focal Loss with Label Smoothing
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

    # ConvNeXt Model
    class ConvNeXtGenderClassifier(nn.Module):
        def __init__(self, backbone='convnext_atto'):
            super().__init__()
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            self.classifier = nn.Linear(self.backbone.num_features, 1)

        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features).squeeze(1)

    # Model
    model = ConvNeXtGenderClassifier().to(device)

    # Optimizer, Loss, Scheduler, AMP
    criterion = FocalLossWithSmoothing(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler()

    # Training Setup
    best_val_acc = 0.0
    best_model_state = None

    epochs = 20
    print("\n" + "="*20 + " Training ConvNeXt with Undersampling " + "="*20 + "\n")

    for epoch in range(epochs):
        # === Train ===
        model.train()
        train_loss = 0.0
        total = 0
        all_preds, all_labels = [], []

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

            train_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.long().cpu().numpy())
            total += labels.size(0)

        avg_train_loss = train_loss / total
        train_acc = accuracy_score(all_labels, all_preds)
        train_prec = precision_score(all_labels, all_preds, zero_division=0)
        train_rec = recall_score(all_labels, all_preds, zero_division=0)
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)

        print(f"\n[Epoch {epoch+1}] TRAINING METRICS")
        print(f"Loss: {avg_train_loss:.4f}")
        print(f"Accuracy: {train_acc:.4f} | Precision: {train_prec:.4f} | Recall: {train_rec:.4f} | F1 Score: {train_f1:.4f}")

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()

                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.long().cpu().numpy())
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total
        val_acc = accuracy_score(val_labels_list, val_preds)
        val_prec = precision_score(val_labels_list, val_preds, zero_division=0)
        val_rec = recall_score(val_labels_list, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)

        print(f"\n[Epoch {epoch+1}] VALIDATION METRICS")
        print(f"Loss: {avg_val_loss:.4f}")
        print(f"Accuracy: {val_acc:.4f} | Precision: {val_prec:.4f} | Recall: {val_rec:.4f} | F1 Score: {val_f1:.4f}")

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"\n✅ Best model updated (val_acc: {best_val_acc:.4f})")

    # === Save Best Model ===
    if best_model_state is not None:
        torch.save(best_model_state, "best_convnext_gender_model.pth")
        print("\n" + "="*50)
        print("✅ Best model saved as 'best_convnext_gender_model.pth'")
        print(f"Final Best Validation Accuracy: {best_val_acc:.2%}")
        print("="*50)
    else:
        print("⚠️ No best model state was saved!")


# === Entry Point Guard ===
if __name__ == "__main__":
    main()
