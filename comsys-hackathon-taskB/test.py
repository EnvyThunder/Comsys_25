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

# ============ Seeding for Reproducibility ============
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============ Flatten Distortion Folders ============
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

# ============ Siamese Dataset ============
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

# ============ Siamese Network ============
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

# ============ Contrastive Loss ============
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, feat1, feat2, label):
        distance = F.pairwise_distance(feat1, feat2)
        loss = label * distance.pow(2) + (1 - label) * torch.clamp(self.margin - distance, min=0.0).pow(2)
        return loss.mean()

# ============ Evaluation Metrics ============
def compute_all_metrics(preds, labels):
    preds_np = np.array(preds)
    labels_np = np.array(labels)

    acc = (preds_np == labels_np).mean()

    # Binary metrics (class 1)
    precision_bin = precision_score(labels_np, preds_np, average='binary', zero_division=0)
    recall_bin = recall_score(labels_np, preds_np, average='binary', zero_division=0)
    f1_bin = f1_score(labels_np, preds_np, average='binary', zero_division=0)

    # Macro-averaged metrics (across both classes)
    precision_macro = precision_score(labels_np, preds_np, average='macro', zero_division=0)
    recall_macro = recall_score(labels_np, preds_np, average='macro', zero_division=0)
    f1_macro = f1_score(labels_np, preds_np, average='macro', zero_division=0)

    return acc, precision_bin, recall_bin, f1_bin, precision_macro, recall_macro, f1_macro

# ============ Main ============
def main():
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    test_dir = "test_data"
    flatten_distortion_folders(test_dir)

    test_dataset = SiameseDistortionDataset(root_dir=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = SiameseNet().to(device)
    model.load_state_dict(torch.load('weights/best_siamese_convnext.pt', map_location=device))
    model.eval()

    criterion = ContrastiveLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for img1, img2, label in tqdm(test_loader, desc="[Test]"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            feat1, feat2 = model(img1, img2)
            loss = criterion(feat1, feat2, label)
            total_loss += loss.item()

            distance = F.pairwise_distance(feat1, feat2)
            preds = (distance < 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    acc, prec_bin, rec_bin, f1_bin, prec_macro, rec_macro, f1_macro = compute_all_metrics(all_preds, all_labels)

    print(f"\nTest Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    print(f"Binary Precision: {prec_bin:.4f}, Recall: {rec_bin:.4f}, F1: {f1_bin:.4f}")
    print(f"Macro  Precision: {prec_macro:.4f}, Recall: {rec_macro:.4f}, F1: {f1_macro:.4f}")

if __name__ == "__main__":
    main()