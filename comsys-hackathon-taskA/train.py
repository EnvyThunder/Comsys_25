import os

import numpy as np
import random
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

    # Dataset class with undersampling done
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

    # Setting random seed for reproducibility
    g = torch.Generator()
    g.manual_seed(42)

    # Train and val loaders
    train_dataset = FaceGenderDataset("data/Task_A/train", transform=train_transform, undersample=True)
    val_dataset = FaceGenderDataset("data/Task_A/train", transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, generator=g, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Cuda initialisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Focal Loss with Label Smoothing ===
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

    # ConvNext_Atto Model
    class ConvNeXtGenderClassifier(nn.Module):
        def __init__(self, backbone='convnext_atto'):
            super().__init__()
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            self.classifier = nn.Linear(self.backbone.num_features, 1)  # Binary classification

        def forward(self, x):
            features = self.backbone(x)
            return self.classifier(features).squeeze(1)

    model = ConvNeXtGenderClassifier().to(device)

    # === Loss, Optimizer, Scheduler, Scaler ===
    criterion = FocalLossWithSmoothing(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    scaler = GradScaler()

    # Setting up training
    best_val_acc = 0.0
    best_model_state = None

    epochs = 20
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    print("\n" + "="*20 + " Training ConvNeXt with Undersampling " + "="*20 + "\n")

    # Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

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

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Val loop
        model.eval()
        val_running_loss = 0.0
        val_correct, val_total = 0, 0

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

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step()

        # Showing the output
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"✅ Best model updated (val_acc: {best_val_acc:.4f})")

    # Saving the best model and its weights
    if best_model_state is not None:
        torch.save(best_model_state, "comsys-hackathon-taskA/weights/best_convnext_gender_model1.pt")
        print("\n" + "="*50)
        print("✅ Best model saved as 'best_convnext_gender_model.pt'")
        print(f"Final Best Validation Accuracy: {best_val_acc:.2%}")
        print("="*50)
    else:
        print("⚠️ No best model state was saved!")

# === Entry Point Guard ===
if __name__ == "__main__":
    main()
