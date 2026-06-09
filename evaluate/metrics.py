"""Shared metrics utilities for training and evaluation."""
import numpy as np
from sklearn.metrics import roc_auc_score


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC-ROC. Returns 0.5 if only one class present (e.g., during early training)."""
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_fnr(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    """False Negative Rate (missed deepfakes) at given threshold."""
    preds = (scores >= threshold).astype(int)
    positives = labels == 1
    if positives.sum() == 0:
        return 0.0
    return float((preds[positives] == 0).sum() / positives.sum())


def compute_fpr(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    """False Positive Rate at given threshold."""
    preds = (scores >= threshold).astype(int)
    negatives = labels == 0
    if negatives.sum() == 0:
        return 0.0
    return float((preds[negatives] == 1).sum() / negatives.sum())


def compute_accuracy(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> float:
    preds = (scores >= threshold).astype(int)
    return float((preds == labels.astype(int)).mean())
