"""
Phase 7 Module: NLP-Based Listing Description Quality Scorer.
Uses DistilBERT with LoRA fine-tuning (when available) or keyword-based heuristic
fallback to score used car listing descriptions on a 0.0–1.0 quality scale.

Quality signals include: ownership history, service records, condition keywords,
damage indicators, and overall listing professionalism.
"""

import re
import numpy as np

# Keyword dictionaries for heuristic scoring
POSITIVE_KEYWORDS = {
    # Ownership & History
    "single owner": 0.15,
    "first owner": 0.15,
    "1st owner": 0.15,
    "one owner": 0.15,
    "well maintained": 0.12,
    "well-maintained": 0.12,
    "regularly serviced": 0.10,
    "service history": 0.10,
    "full service": 0.10,
    "authorized service": 0.08,
    "company maintained": 0.10,

    # Condition
    "showroom condition": 0.15,
    "mint condition": 0.12,
    "excellent condition": 0.10,
    "good condition": 0.06,
    "like new": 0.10,
    "no scratches": 0.08,
    "scratch free": 0.08,
    "no dents": 0.08,
    "dent free": 0.08,
    "no accident": 0.12,
    "accident free": 0.12,
    "zero accident": 0.12,
    "non accidental": 0.10,

    # Documentation
    "insurance valid": 0.06,
    "comprehensive insurance": 0.08,
    "all documents": 0.05,
    "noc available": 0.04,
    "rto passing": 0.04,

    # Features
    "new tyres": 0.05,
    "new tires": 0.05,
    "new battery": 0.04,
    "alloy wheels": 0.03,
    "sunroof": 0.03,
    "automatic": 0.02,
    "top model": 0.04,
    "top variant": 0.04,
    "fully loaded": 0.05,
}

NEGATIVE_KEYWORDS = {
    # Damage indicators
    "accident": -0.15,
    "accidental": -0.15,
    "flood damaged": -0.20,
    "flood affected": -0.20,
    "major scratch": -0.10,
    "deep scratch": -0.10,
    "dented": -0.08,
    "major dent": -0.10,
    "rust": -0.10,
    "rusted": -0.10,
    "repainted": -0.08,
    "repaired": -0.06,

    # Mechanical issues
    "engine problem": -0.15,
    "engine issue": -0.15,
    "gear problem": -0.10,
    "ac not working": -0.08,
    "overheating": -0.10,
    "oil leak": -0.10,
    "smoke": -0.08,

    # Ownership issues
    "commercial": -0.06,
    "taxi": -0.10,
    "ola": -0.08,
    "uber": -0.08,
    "fleet": -0.08,
    "third owner": -0.06,
    "3rd owner": -0.06,
    "fourth owner": -0.08,
    "4th owner": -0.08,

    # Red flags
    "urgent sale": -0.04,
    "desperate": -0.05,
    "as is": -0.06,
    "no warranty": -0.04,
    "expired insurance": -0.04,
}


class ListingDescriptionScorer:
    """
    NLP-based Listing Description Quality Scorer.

    Uses a keyword-based heuristic engine with optional DistilBERT fine-tuned
    model for enhanced accuracy. The heuristic scorer serves as both a training
    label generator (weak supervision) and production fallback.
    """

    def __init__(self, model_path: str = None, use_transformer: bool = False):
        """
        Args:
            model_path: Path to fine-tuned DistilBERT checkpoint (.pt file)
            use_transformer: Whether to attempt loading the transformer model
        """
        self.model_path = model_path
        self.transformer_model = None
        self.tokenizer = None
        self.use_transformer = use_transformer

        if use_transformer and model_path:
            self._load_transformer(model_path)

    def _load_transformer(self, model_path: str):
        """Attempt to load DistilBERT fine-tuned model."""
        try:
            import torch
            from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            self.transformer_model = DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased", num_labels=1
            )

            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
            self.transformer_model.load_state_dict(checkpoint, strict=False)
            self.transformer_model.eval()
            print(f"  |-- Loaded DistilBERT NLP model from: {model_path}")
        except Exception as e:
            print(f"  |-- DistilBERT unavailable ({e}), using keyword heuristic fallback")
            self.transformer_model = None

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize description text."""
        if not text or not isinstance(text, str):
            return ""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\-]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    def score_description_heuristic(self, text: str) -> float:
        """
        Score a listing description using keyword-based heuristics.
        Returns a float between 0.0 (poor quality) and 1.0 (excellent quality).
        """
        if not text or not isinstance(text, str) or len(text.strip()) < 5:
            return 0.3  # Default neutral-low for empty/missing descriptions

        clean_text = self._preprocess_text(text)
        score = 0.5  # Start from neutral baseline

        # Apply positive keyword matches
        for keyword, weight in POSITIVE_KEYWORDS.items():
            if keyword in clean_text:
                score += weight

        # Apply negative keyword matches
        for keyword, weight in NEGATIVE_KEYWORDS.items():
            if keyword in clean_text:
                score += weight  # weight is already negative

        # Length bonus: longer, more detailed descriptions tend to be higher quality
        word_count = len(clean_text.split())
        if word_count >= 50:
            score += 0.05
        elif word_count >= 30:
            score += 0.03
        elif word_count < 10:
            score -= 0.05

        # Clamp to [0.0, 1.0]
        return round(max(0.0, min(1.0, score)), 4)

    def score_description_transformer(self, text: str) -> float:
        """Score using the fine-tuned DistilBERT model."""
        import torch

        clean_text = self._preprocess_text(text)
        if not clean_text:
            return 0.3

        inputs = self.tokenizer(
            clean_text, return_tensors="pt",
            max_length=128, truncation=True, padding="max_length"
        )

        with torch.no_grad():
            output = self.transformer_model(**inputs)
            score = torch.sigmoid(output.logits).item()

        return round(max(0.0, min(1.0, score)), 4)

    def score_description(self, text: str) -> float:
        """
        Score a listing description. Uses transformer if available, otherwise heuristic.

        Args:
            text: Raw listing description text

        Returns:
            Float in [0.0, 1.0] representing listing quality score
        """
        if self.transformer_model is not None:
            return self.score_description_transformer(text)
        return self.score_description_heuristic(text)

    def batch_score(self, texts: list) -> list:
        """
        Score multiple descriptions in batch.

        Args:
            texts: List of description strings

        Returns:
            List of float scores in [0.0, 1.0]
        """
        return [self.score_description(t) for t in texts]

    def generate_weak_labels(self, texts: list) -> list:
        """
        Generate weak supervision labels from keyword heuristics.
        Used to bootstrap training data for DistilBERT fine-tuning.

        Args:
            texts: List of listing descriptions

        Returns:
            List of float labels in [0.0, 1.0]
        """
        return [self.score_description_heuristic(t) for t in texts]

    def get_quality_breakdown(self, text: str) -> dict:
        """
        Get a detailed breakdown of quality signals found in the description.

        Args:
            text: Raw listing description text

        Returns:
            Dict with overall score, matched positive/negative keywords, and word count
        """
        clean_text = self._preprocess_text(text)
        matched_positive = {}
        matched_negative = {}

        for keyword, weight in POSITIVE_KEYWORDS.items():
            if keyword in clean_text:
                matched_positive[keyword] = weight

        for keyword, weight in NEGATIVE_KEYWORDS.items():
            if keyword in clean_text:
                matched_negative[keyword] = abs(weight)

        return {
            "overall_score": self.score_description(text),
            "word_count": len(clean_text.split()) if clean_text else 0,
            "positive_signals": matched_positive,
            "negative_signals": matched_negative,
            "positive_count": len(matched_positive),
            "negative_count": len(matched_negative),
        }
