"""
Phase 5 Training & Validation Script for Car Condition CNN Deep Learning Model.
Verifies PyTorch CUDA GPU availability, creates synthetic image samples for 4 condition tiers,
trains MobileNetV3 CNN transfer learning model, and saves weights checkpoint to models/car_condition_cnn.pt.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision_model import CarConditionCNN, CarConditionScorer, CONDITION_TIERS


class SyntheticCarDataset(Dataset):
    """
    Synthetic Car Condition Dataset Generator for fast, robust PyTorch CV training & testing.
    Generates images representing 4 condition tiers:
    - Tier 0: Clean blue/white image (Pristine)
    - Tier 1: Minor scratch lines
    - Tier 2: Moderate dent patches
    - Tier 3: Major dark damage blotches
    """

    def __init__(self, num_samples: int = 100, transform=None):
        self.num_samples = num_samples
        self.transform = transform
        self.samples = []
        self.labels = []

        for i in range(num_samples):
            tier = i % 4
            img = Image.new("RGB", (224, 224), color=(200, 220, 240) if tier == 0 else (180, 180, 180))
            draw = ImageDraw.Draw(img)

            if tier == 1:  # Minor scratch
                draw.line([(30, 50), (180, 70)], fill=(50, 50, 50), width=2)
            elif tier == 2:  # Moderate dent
                draw.rectangle([(60, 60), (140, 140)], fill=(100, 100, 100))
            elif tier == 3:  # Major damage
                draw.ellipse([(20, 20), (200, 200)], fill=(20, 20, 20))

            self.samples.append(img)
            self.labels.append(tier)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.samples[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def train_vision_model():
    print("=" * 80)
    print("PHASE 5: PYTORCH DEEP LEARNING (CV) CONDITION ASSESSMENT ENGINE")
    print("=" * 80)

    # 1. Check GPU / CUDA hardware availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Step 1/4] PyTorch Hardware Device Check: {device.type.upper()}")
    if torch.cuda.is_available():
        print(f"  |-- GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("  |-- GPU Not Detected / Using CPU fallback")

    # 2. Data Preparation
    print("\n[Step 2/4] Initializing Synthetic Car Condition Dataset (100 samples across 4 tiers)...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SyntheticCarDataset(num_samples=100, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. Model & Optimizer Setup
    print("\n[Step 3/4] Training MobileNetV3 Transfer Learning Model (5 Epochs)...")
    model = CarConditionCNN(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(5):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        print(f"  Epoch {epoch+1}/5 | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.2f}%")

    # 4. Save Checkpoint & Evaluate
    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.abspath("models/car_condition_cnn.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"\n[Step 4/4] Saved trained CNN model weights checkpoint to: {checkpoint_path}")

    # Test Inference Engine
    print("\nEvaluating Condition Scorer on Test Image...")
    scorer = CarConditionScorer(model_path=checkpoint_path)
    test_img = Image.new("RGB", (224, 224), color=(200, 220, 240))
    res = scorer.predict_image_condition(test_img)
    print("Inference Output:")
    print(f"  |-- Predicted Tier        : Tier {res['predicted_tier']} ({res['condition_label']})")
    print(f"  |-- Visual Condition Score: {res['visual_condition_score']}")
    print(f"  \\-- Probabilities         : {res['confidence_probabilities']}")
    print("=" * 80)


if __name__ == "__main__":
    train_vision_model()
