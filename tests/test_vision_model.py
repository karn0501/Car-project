"""
Unit tests for Phase 5: Deep Learning PyTorch Computer Vision Condition Scorer.
"""

import os
import sys
import torch
import pytest
from PIL import Image

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision_model import CarConditionCNN, CarConditionScorer, CONDITION_TIERS
from src.preprocessing import DataPreprocessor


def test_car_condition_cnn_forward():
    model = CarConditionCNN(num_classes=4, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 4)


def test_car_condition_scorer_inference(tmp_path):
    # Save untrained checkpoint for testing
    model = CarConditionCNN(num_classes=4, pretrained=False)
    checkpoint_path = str(tmp_path / "test_cnn.pt")
    torch.save(model.state_dict(), checkpoint_path)

    scorer = CarConditionScorer(model_path=checkpoint_path)
    test_img = Image.new("RGB", (224, 224), color=(150, 150, 150))

    result = scorer.predict_image_condition(test_img)

    assert "predicted_tier" in result
    assert "condition_label" in result
    assert "visual_condition_score" in result
    assert "confidence_probabilities" in result

    assert result["predicted_tier"] in [0, 1, 2, 3]
    assert 0.0 <= result["visual_condition_score"] <= 1.0
    assert len(result["confidence_probabilities"]) == 4


def test_preprocessing_visual_condition_score_fallback():
    import pandas as pd
    df_dummy = pd.DataFrame({
        "manufacture_year": [2020],
        "km_driven": [50000],
        "asking_price": [500000],
        "ex_showroom_price": [800000]
    })

    preprocessor = DataPreprocessor()
    df_proc = preprocessor.engineer_features(df_dummy)

    assert "visual_condition_score" in df_proc.columns
    assert df_proc["visual_condition_score"].iloc[0] == 1.0
