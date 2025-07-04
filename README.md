# 🧠 COMSYS 5 Hackathon

## 📁 Repository Structure

```
├── comsys-hackathon-taskA/
│   ├── test_data/                       # hold-out test images
│   ├── weights/
│   │   └── best_convnext_gender_model.pth
│   ├── download_data.py                 # download & unpack Task A data
│   ├── gender_classification_convnext.ipynb  # notebook (exploration)
│   ├── train.py                         # training script
│   ├── test.py                          # inference & submission script
│   └── requirements.txt
│
├── comsys-hackathon-taskB/
│   ├── test_data/                       # hold-out test images
│   ├── weights/
│   │   └── best_siamese_convnext.pt
│   ├── download_data.py                 # download & unpack Task B data
│   ├── face_recognition_siamese.ipynb   # notebook (exploration)
│   ├── train.py                         # training script
│   ├── test.py                          # inference script
│   └── requirements.txt
│
└── README.md                            # technical overview
```

---

# 🧠 Task A: Gender Classification

This repository implements a **ConvNeXt‑Atto**‑based classifier to predict gender (male/female) from face images. Class imbalance is handled via random undersampling of the majority (male) class and focal loss with label smoothing.

---

## 📂 Dataset Layout (after download)

```
Comys_Hackathon5/
└── Task_A/
    ├── train/
    │   ├── male/
    │   │   ├── img_001.jpg
    │   │   └── ...
    │   └── female/
    │       ├── img_777.jpg
    │       └── ...
    └── val/
        ├── male/
        └── female/
```

---

## 🧹 Data Pre‑processing

| Step                     | Purpose                                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------------------------- |
| **Random undersampling** | Reduce 1623 male images down to 303 to match female count; mitigates class bias.                     |
| **Augmentations**        | Resize → 224×224, `HorizontalFlip`, `ColorJitter`, `RandomRotation(±15°)` to improve generalisation. |

All transforms implemented with **torchvision**.

---

## 🏗️ Model Architecture

```
ConvNeXt‑Atto (pre‑trained, features only)
       ↓
Linear(num_features → 1)
```

* **Backbone**: ConvNeXt‑Atto from [`timm`](https://github.com/huggingface/pytorch-image-models) with frozen weights.
* **Head**: Single linear layer ➔ sigmoid via `BCEWithLogits`.

---

## 📉 Loss Function

### Focal Loss + Label Smoothing

```
FL = α · (1 − p_t)^γ · CE_smooth
```

* Smoothing ε = 0.1
* γ = 2, α = 1

Provides extra focus on hard / minority samples.

---

## 🚀 Training Configuration

| Setting         | Value                    |
| --------------- | ------------------------ |
| Optimizer       | AdamW                    |
| Base LR         | 1e‑4                     |
| Scheduler       | CosineAnnealingLR (T=10) |
| Epochs          | 50                       |
| Batch Size      | 32                       |
| Mixed Precision | ✅ (`torch.cuda.amp`)     |
| Model Selection | Highest validation acc   |

---

## ✅ Results

| Metric                       | Value       |
| ---------------------------- | ----------- |
| **Best Validation Accuracy** | **96.37 %** |

The best checkpoint is saved to `weights/best_convnext_gender_model.pth` and loaded automatically by `test.py`.

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

* **Key Libraries:** PyTorch ≥ 2.0, `timm`, `torchvision`, `Pillow`

---

## 🔄 Training & Evaluation

```bash
# Train
python train.py --data_dir /path/to/Comys_Hackathon5/Task_A --epochs 50

# Inference on organiser test set
python test.py  --weights weights/best_convnext_gender_model.pth \
                --img_dir  test_data/ \
                --output   submission.csv
```

---

## ✍️ Notes

* Class imbalance is tackled via **undersampling + focal loss** rather than oversampling to keep training stable.
* ConvNeXt‑Atto offers an excellent speed/accuracy trade‑off, ideal for limited GPU quotas.
* Further gains could come from class‑balanced loss or larger ConvNeXt variants if compute allows.

---

# 🧠 Task B: Face Recognition (Siamese Network)

This repository implements a Siamese Neural Network using a ConvNeXt-Atto encoder for face verification under distortion. The network learns to embed similar faces closer together while pushing dissimilar pairs apart using Contrastive Loss.

---

## ⚙️ Setup Instructions

Follow these steps to set up your environment and run the project:

```bash
# Step 1: Clone the repository
git clone https://github.com/EnvyThunder/Comsys_25
cd comsys-hackathon-taskB

# Step 2: Create a virtual environment
python3 -m venv venv

# Step 3: Activate the virtual environment
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Step 4: Install all dependencies
pip install -r requirements.txt

# Step 5: Download dataset (Google Drive link embedded in script)
python3 download_data.py  # Downloads and sets up dataset locally

# Step 6: Run training (adjust path if needed)
python3 train.py --data_dir /path/to/Comys_Hackathon5/Task_B --epochs 50

# Step 7: Place test dataset inside the `test_data/` directory

# Step 8: Run inference
python3 test.py --weights weights/best_siamese_convnext.pt                 --img_dir  test_data/                 --output   submission.csv
```

---

## 📂 Dataset Structure

```
Comys_Hackathon5/
└── Task_B/
    ├── train/
    │   ├── person_1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── distortion/        # flattened automatically
    │   └── ...
    └── val/
        ├── person_2/
        │   ├── image1.jpg
        │   └── distortion/
        └── ...
```

Each class folder may contain a `distortion/` subfolder with additional samples. This is flattened at runtime.

---

## 📦 Components

### 🧹 Data Preprocessing

- **Distortion Flattening:** Moves images from `distortion/` subfolders into their parent identity class folders.
- **Pair Generation:**
  - Positive Pairs: Two images from the same identity.
  - Negative Pairs: One image from the current identity and one from a different random identity.

### 🖼️ Data Augmentation

Implemented using **Albumentations**:

- **Training Transforms:**
  - Resize(224, 224)
  - HorizontalFlip, BrightnessContrast, HueSaturation, ShiftScaleRotate
  - Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
  - ToTensorV2

- **Validation Transforms:**
  - Resize(224, 224)
  - Normalize
  - ToTensorV2

---

## 🏗️ Model Architecture

```
ConvNeXt-Atto (pretrained, no classifier head)
        ↓
    Linear → ReLU → Linear → L2 Normalization
```

- **Backbone:** ConvNeXt-Atto from `timm`
- **Projection Head:**  
  - Linear(encoder_out → 512) → ReLU → Linear(512 → 128)
- **Output:** Normalized embeddings using L2 norm
- **Shared encoder** is used for both image branches

---

## 📉 Loss Function

**Contrastive Loss**

Minimizes distance for similar pairs, maximizes for dissimilar pairs.

```
Loss = y * d^2 + (1 - y) * max(0, margin - d)^2
```

Where:
- `y ∈ {0, 1}` is the label
- `d` is the Euclidean distance between the embeddings
- `margin` is a hyperparameter (default = 1.0)

---

## ✅ Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  

**Threshold-based decision:** If distance < 0.5 ➔ positive match.

---

## 🚀 Training Configuration

| Setting           | Value               |
|-------------------|---------------------|
| Model Backbone     | ConvNeXt-Atto       |
| Loss Function      | Contrastive Loss    |
| Optimizer          | AdamW               |
| Learning Rate      | 1e-4                |
| Weight Decay       | 1e-5                |
| LR Scheduler       | CosineAnnealingLR   |
| Epochs             | 50                  |
| Batch Size         | 32                  |
| Mixed Precision    | ✅ (torch.cuda.amp) |
| Model Saving       | Based on Val Accuracy |

---

## ✅ Results

| Metric              | Value     |
|---------------------|-----------|
| Best Val Accuracy   | 97.73 %   |

Best checkpoint saved to:

```
weights/best_siamese_convnext.pt
```

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

---

## 🔄 Training & Inference

```bash
# Train
python train.py --data_dir /path/to/Comys_Hackathon5/Task_B --epochs 50

# Inference on test pairs
python test.py --weights weights/best_siamese_convnext.pt                --img_dir  test_data/                --output   submission.csv
```

