"""
Unit tests for Phase 7: NLP-Based Listing Description Quality Scorer.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlp_scorer import ListingDescriptionScorer


def test_keyword_quality_scoring():
    """Test that keyword heuristic produces expected scores for clear cases."""
    scorer = ListingDescriptionScorer()

    # High quality listing
    high_quality = "Single owner, well maintained, showroom condition. Regularly serviced at authorized center. No scratches, no dents, accident free."
    score_high = scorer.score_description(high_quality)
    assert score_high >= 0.7, f"High quality listing should score >= 0.7, got {score_high}"

    # Low quality listing
    low_quality = "Taxi used, commercial registration. Engine problem, major dent, rust on body. Third owner, urgent sale."
    score_low = scorer.score_description(low_quality)
    assert score_low < 0.45, f"Low quality listing should score < 0.45, got {score_low}"

    # High should beat low
    assert score_high > score_low, "High quality listing should score higher than low quality"


def test_batch_scoring():
    """Test that batch scoring works correctly."""
    scorer = ListingDescriptionScorer()

    descriptions = [
        "Well maintained first owner car with full service history.",
        "Urgent sale, engine problem, accidental car.",
        "Good condition, second owner, insurance valid.",
    ]

    scores = scorer.batch_score(descriptions)
    assert len(scores) == 3, "Batch should return 3 scores"
    assert all(isinstance(s, float) for s in scores), "All scores should be floats"


def test_score_range():
    """Test that all scores fall within [0.0, 1.0] range."""
    scorer = ListingDescriptionScorer()

    test_cases = [
        "",
        None,
        "a",
        "Single owner well maintained showroom condition excellent no scratches no dents accident free regularly serviced",
        "Flood damaged rusted engine problem accidental taxi commercial urgent sale as is no warranty expired insurance",
        "Normal car in decent condition.",
    ]

    for text in test_cases:
        score = scorer.score_description(text)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range for text: {text!r}"


def test_quality_breakdown():
    """Test detailed quality breakdown output."""
    scorer = ListingDescriptionScorer()

    text = "First owner, well maintained. No accident history. Minor scratch on bumper."
    breakdown = scorer.get_quality_breakdown(text)

    assert "overall_score" in breakdown
    assert "positive_signals" in breakdown
    assert "negative_signals" in breakdown
    assert "word_count" in breakdown
    assert breakdown["positive_count"] >= 2  # "first owner" and "well maintained" at minimum
    assert isinstance(breakdown["overall_score"], float)


def test_empty_and_null_descriptions():
    """Test handling of empty, None, and very short descriptions."""
    scorer = ListingDescriptionScorer()

    assert scorer.score_description("") == 0.3
    assert scorer.score_description(None) == 0.3
    assert scorer.score_description("   ") == 0.3
    assert isinstance(scorer.score_description("ok"), float)
