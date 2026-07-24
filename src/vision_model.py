"""
PyTorch Deep Learning Computer Vision (CV) Module for Used Car Condition Assessment (Phase 5).
Uses a MobileNetV3 transfer learning CNN backbone to classify vehicle condition into 4 tiers
and output a normalized visual_condition_score (0.0 to 1.0).
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

CONDITION_TIERS = {
    0: ("Pristine / Showroom Condition", 1.0),
    1: ("Minor Scratches / Wear", 0.8),
    2: ("Moderate Dents / Paint Fading", 0.5),
    3: ("Major Body Damage / Collision", 0.2)
}


class CarConditionCNN(nn.Module):
    """
    MobileNetV3 Transfer Learning CNN for Vehicle Exterior Condition Classification.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super(CarConditionCNN, self).__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v3_small(weights=weights)

        # Replace classification head
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class CarConditionScorer:
    """
    Inference Engine for scoring car condition from an image file or PIL Image object.
    """

    def __init__(self, model_path: str = "models/car_condition_cnn.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = os.path.abspath(model_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = CarConditionCNN(num_classes=4, pretrained=False)

        if os.path.exists(self.model_path):
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def predict_image_condition(self, image_input) -> dict:
        """
        Accepts a file path string or PIL.Image object and returns predicted condition details.
        """
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image not found at {image_input}")
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError("Input must be a file path string or a PIL.Image object.")

        tensor_img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor_img)
            probs = torch.softmax(outputs, dim=1)
            pred_class = int(torch.argmax(probs, dim=1).item())

        tier_label, condition_score = CONDITION_TIERS[pred_class]

        return {
            "predicted_tier": pred_class,
            "condition_label": tier_label,
            "visual_condition_score": float(condition_score),
            "confidence_probabilities": [round(float(p), 4) for p in probs[0].cpu().numpy()]
        }
