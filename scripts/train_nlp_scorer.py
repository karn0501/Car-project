"""
Phase 7 Training Script: NLP Description Quality Scorer.
Generates weak labels from keyword heuristics, optionally fine-tunes DistilBERT
with LoRA, and saves trained model checkpoint.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp_scorer import ListingDescriptionScorer


# Sample listing descriptions for training demonstration
SAMPLE_DESCRIPTIONS = [
    "Single owner, well maintained Maruti Swift VXi. Showroom condition, regularly serviced at authorized center. No scratches, no dents. Insurance valid till Dec 2027.",
    "First owner car in excellent condition. All documents clear. New tyres and battery replaced recently. Comprehensive insurance active.",
    "Good condition Hyundai i20. Second owner. Minor scratches on bumper. Service done at local garage. Insurance expired.",
    "Urgent sale! Third owner vehicle. AC not working properly. Some dents on sides. Repainted recently.",
    "Well-maintained Honda City. Company maintained with full service history. Accident free, zero scratch. Top variant with sunroof and alloy wheels.",
    "Taxi used vehicle, commercial registration. Engine problem reported. Multiple owners. As is condition.",
    "Mint condition Kia Seltos. Single owner, non accidental. All documents and NOC available. Fully loaded top model.",
    "Flood damaged car. Rust on underbody. Engine overheating issue. Needs major repair work.",
    "Like new Toyota Fortuner. First owner, regularly serviced. No accident history. Comprehensive insurance valid. All original parts.",
    "Second owner Tata Nexon. Good condition overall. New tyres. Insurance valid. RTO passing done.",
]


def train_nlp_scorer():
    print("=" * 80)
    print("PHASE 7: NLP DESCRIPTION QUALITY SCORER TRAINING")
    print("=" * 80)

    # 1. Initialize scorer
    print("\n[Step 1/3] Initializing keyword-based heuristic scorer...")
    scorer = ListingDescriptionScorer()
    print("  \\-- Keyword dictionaries loaded (positive + negative signals)")

    # 2. Generate weak labels and score sample descriptions
    print("\n[Step 2/3] Scoring sample listing descriptions...")
    scores = scorer.batch_score(SAMPLE_DESCRIPTIONS)

    for i, (desc, score) in enumerate(zip(SAMPLE_DESCRIPTIONS, scores)):
        quality = "HIGH" if score >= 0.7 else ("MEDIUM" if score >= 0.45 else "LOW")
        preview = desc[:60] + "..." if len(desc) > 60 else desc
        print(f"  Listing {i+1:2d}: Score={score:.4f} ({quality:6s}) | {preview}")

    # 3. Attempt DistilBERT training if transformers available
    print("\n[Step 3/3] Checking DistilBERT + LoRA availability...")
    try:
        import torch
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        print("  |-- transformers library detected")
        print("  |-- Initializing DistilBERT fine-tuning with weak labels...")

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=1
        )

        # Tokenize
        weak_labels = scorer.generate_weak_labels(SAMPLE_DESCRIPTIONS)
        encodings = tokenizer(
            SAMPLE_DESCRIPTIONS, return_tensors="pt",
            max_length=128, truncation=True, padding="max_length"
        )
        labels = torch.FloatTensor(weak_labels).unsqueeze(1)

        # Quick fine-tune (3 epochs on sample data)
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        criterion = torch.nn.MSELoss()
        model.train()

        for epoch in range(3):
            optimizer.zero_grad()
            outputs = model(**encodings)
            preds = torch.sigmoid(outputs.logits)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            print(f"  Epoch {epoch+1}/3 | Loss: {loss.item():.4f}")

        # Save checkpoint
        os.makedirs("models", exist_ok=True)
        model_path = os.path.abspath("models/nlp_description_scorer.pt")
        torch.save(model.state_dict(), model_path)
        print(f"  \\-- Saved NLP model checkpoint to: {model_path}")

    except ImportError:
        print("  |-- transformers not installed (pip install transformers)")
        print("  \\-- Using keyword heuristic scorer as production fallback (no training needed)")

    # Summary
    avg_score = sum(scores) / len(scores)
    high_quality = sum(1 for s in scores if s >= 0.7)
    low_quality = sum(1 for s in scores if s < 0.45)

    print("\n" + "=" * 80)
    print("NLP SCORER TRAINING COMPLETE")
    print("=" * 80)
    print(f"  |-- Average Quality Score : {avg_score:.4f}")
    print(f"  |-- High Quality Listings : {high_quality}/{len(scores)}")
    print(f"  |-- Low Quality Listings  : {low_quality}/{len(scores)}")
    print(f"  \\-- Scorer Mode           : Keyword Heuristic + Optional DistilBERT")
    print("=" * 80)


if __name__ == "__main__":
    train_nlp_scorer()
