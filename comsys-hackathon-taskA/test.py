import os
import random
from PIL import Image
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import precision_score, recall_score, f1_score

# Dataset class with optional undersampling
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

# ConvNeXt Model
class ConvNeXtGenderClassifier(nn.Module):
    def __init__(self, backbone='convnext_atto'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.classifier = nn.Linear(self.backbone.num_features, 1)  # Binary classification

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(1)

# Test code entry point
def main():
    test_dir = "test_data"  # Change this if needed
    model_path = "best_convnext_gender_model1.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test transforms
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Load test dataset and loader
    test_dataset = FaceGenderDataset(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Load model
    model = ConvNeXtGenderClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Test loop
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            with autocast():
                outputs = model(images)

            preds = (torch.sigmoid(outputs) > 0.5).long()
            test_correct += (preds == labels.long()).sum().item()
            test_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    test_acc = test_correct / test_total
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    # Print results
    print("\n" + "="*50)
    print(f"🧪 Test Accuracy : {test_acc:.4f}")
    print(f"📌 Precision     : {precision:.4f}")
    print(f"📌 Recall        : {recall:.4f}")
    print(f"📌 F1 Score      : {f1:.4f}")
    print("="*50)

# Safe CLI entry point
if __name__ == "__main__":
    main()
