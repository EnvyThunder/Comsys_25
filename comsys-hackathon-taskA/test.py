import os
import random
from PIL import Image
import gdown

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score





if not os.path.exists("weights/best_convnext_gender_model.pth"):
    url = "https://drive.google.com/uc?id=1Q97l7ROkH5MrO3hc-5D6ugAo2sOyXaMQ"
    gdown.download(url, "weights/best_convnext_gender_model.pth", quiet=False)



class FaceGenderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        for gender_str in ['male', 'female']:
            label = 0 if gender_str == 'male' else 1
            folder = os.path.join(root_dir, gender_str)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(folder, fname)
                    self.samples.append((path, label))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label).float()


class ConvNeXtGenderClassifier(nn.Module):
    def __init__(self, backbone='convnext_tiny'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.classifier = nn.Linear(self.backbone.num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features).squeeze(1)


def main():
    test_path = "data/Task_A"
    model_path = "weights/best_convnext_gender_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = FaceGenderDataset(test_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = ConvNeXtGenderClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            with autocast(device_type=device.type):
                outputs = model(images)

            preds = (torch.sigmoid(outputs) > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.long().cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n" + "="*50)
    print(f"🧪 Test Accuracy : {acc:.4f}")
    print(f"📌 Precision     : {prec:.4f}")
    print(f"📌 Recall        : {rec:.4f}")
    print(f"📌 F1 Score      : {f1:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
