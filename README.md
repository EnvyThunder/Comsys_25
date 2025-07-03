# Comsys

# 🧠 Task B: Face Recognition (Multi-Class Classification)

This repository implements a **Siamese Neural Network** using a **ConvNeXt-Atto** encoder for face verification under distortion. The network learns to embed similar faces closer together while pushing dissimilar pairs apart using **Contrastive Loss**.


## 📂 Dataset Structure

The dataset should follow this structure:

<pre>
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
</pre>

Each class folder may contain a `distortion/` subfolder with additional samples. This is flattened at runtime.

---

## 📦 Components

### 🧹 Data Preprocessing

- **Distortion Flattening**: Moves images from `distortion/` subfolders into their parent identity class folders.
- **Pair Generation**:
  - **Positive Pairs**: Two images from the same identity.
  - **Negative Pairs**: One image from current identity and one from a different random identity.

---

### 🖼️ Data Augmentation

Implemented using **Albumentations**:

#### Training Transforms:
- `Resize(224, 224)`
- `HorizontalFlip`, `BrightnessContrast`, `HueSaturation`, `ShiftScaleRotate`
- `Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))`
- `ToTensorV2`

#### Validation Transforms:
- `Resize(224, 224)`
- `Normalize`
- `ToTensorV2`

---

## 🏗️ Model Architecture

ConvNeXt-Atto (pretrained, no classifier head)
↓
Linear → ReLU → Linear → L2 Normalization


- **Backbone**: ConvNeXt-Atto from [timm](https://github.com/huggingface/pytorch-image-models)
- **Projection Head**:  
  `Linear(encoder_out → 512) → ReLU → Linear(512 → 128)`
- **Output**: Normalized embeddings using L2 norm
- Shared encoder is used for both image branches

---

## 📉 Loss Function

### Contrastive Loss

Minimizes distance for similar pairs, maximizes distance for dissimilar pairs.

Loss = y * d^2 + (1 - y) * max(0, margin - d)^2


Where:
- `y ∈ {0, 1}` is the label
- `d` is the Euclidean distance between the two embeddings
- `margin` is a hyperparameter (default = `1.0`)

---

## ✅ Accuracy Metric

- Computes pairwise Euclidean distance between embeddings.
- If distance < threshold (default `0.5`), it is predicted as a "same person" pair.
- Accuracy = proportion of correct similarity predictions.

---

## 🚀 Training Configuration

| Setting              | Value              |
|----------------------|--------------------|
| Model Backbone       | ConvNeXt-Atto      |
| Loss Function        | Contrastive Loss   |
| Optimizer            | AdamW              |
| Learning Rate        | 1e-4               |
| Weight Decay         | 1e-5               |
| LR Scheduler         | CosineAnnealingLR  |
| Epochs               | 10                 |
| Batch Size           | 32                 |
| Mixed Precision      | ✅ (`torch.cuda.amp`) |
| Model Saving         | Based on Val Accuracy |

---


## 🧪 Output

- Best model is saved at:

/kaggle/working/best_siamese_convnext.pt

