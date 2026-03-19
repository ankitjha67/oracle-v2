"""Tests for the sentiment engine — blending and probability calculations."""
import pytest

from sentiment import blend_prediction


class TestBlendPrediction:
    """Prediction blending tests."""

    def test_basic_blend(self):
        result = blend_prediction(60, 55)
        assert "blended_prob_a" in result
        assert "blended_prob_b" in result
        assert abs(result["blended_prob_a"] + result["blended_prob_b"] - 100) < 0.2

    def test_model_and_market_agree(self):
        result = blend_prediction(70, 68)
        assert result["confidence"] == "HIGH"
        assert "AGREE" in result["signal"]

    def test_contrarian_signal(self):
        result = blend_prediction(65, 40)
        assert result["confidence"] == "CONTRARIAN"

    def test_uncertain_model_trusts_market_more(self):
        result = blend_prediction(51, 65)
        # When model is uncertain, market weight should be higher
        assert result["sentiment_weight"] > result["model_weight"]

    def test_confident_model_trusts_model_more(self):
        result = blend_prediction(80, 60)
        # When model is confident, model weight should be higher
        assert result["model_weight"] > result["sentiment_weight"]

    def test_clamped_to_valid_range(self):
        result = blend_prediction(95, 95)
        assert 5 <= result["blended_prob_a"] <= 95

    def test_moderate_confidence(self):
        result = blend_prediction(60, 55)
        # Same direction but different margins
        assert result["confidence"] in ("MODERATE", "HIGH")
