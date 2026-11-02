"""Optional class-weighted cross-entropy for the team's three-class NumPy NN.

This file is a separate copy/adaptation of capstone_nn/losses.py.
It never changes or imports the team's original loss functions.

Loss convention used here: mean(sample_class_weight * individual_loss).
This matches ordinary cross-entropy when class_weights=None or all ones.
Both the scalar loss and the gradient use the SAME normalization (batch size).
"""

import numpy as np


def softmax(logits):
    """Stable softmax: (batch, classes) logits -> probabilities."""
    logits = np.asarray(logits)
    if logits.ndim != 2 or logits.shape[0] == 0:
        raise ValueError("logits must have shape (nonempty_batch, num_classes)")
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exponents = np.exp(shifted_logits)
    return exponents / np.sum(exponents, axis=1, keepdims=True)


def calculate_class_weights(y_train, num_classes=3):
    """Inverse-frequency weights computed ONLY from ORIGINAL training labels.

    Formula: w[c] = n_training / (n_classes * n_training_in_class_c)
    Expected label IDs for this capstone: Angry=0, Happy=1, Sad=2.
    """
    labels = np.asarray(y_train)
    if labels.ndim != 1 or len(labels) == 0:
        raise ValueError("y_train must be a nonempty one-dimensional array")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("y_train must contain integer class labels")
    if np.any(labels < 0) or np.any(labels >= num_classes):
        raise ValueError(f"Expected class labels between 0 and {num_classes - 1}")
    counts = np.bincount(labels, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError("Every class must be present in y_train")
    return len(labels) / (num_classes * counts)


def _checked_weights(probabilities, labels, class_weights):
    """Validate dimensions/IDs and return per-image weights."""
    probs = np.asarray(probabilities)
    y = np.asarray(labels)
    if probs.ndim != 2 or probs.shape[0] == 0:
        raise ValueError("probabilities must have shape (nonempty_batch, num_classes)")
    if y.ndim != 1 or len(y) != len(probs) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels must be an integer array with one entry per image")
    if np.any(y < 0) or np.any(y >= probs.shape[1]):
        raise ValueError("Some labels are outside the model's class range")
    if class_weights is None:
        return np.ones(len(y), dtype=np.float64)
    weights = np.asarray(class_weights, dtype=np.float64)
    if weights.shape != (probs.shape[1],):
        raise ValueError("class_weights must have one value per model output class")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("class_weights must be positive and finite")
    return weights[y]


def cross_entropy(probabilities, labels, class_weights=None):
    """Mean of weighted negative log-likelihoods for a batch."""
    sample_weights = _checked_weights(probabilities, labels, class_weights)
    labels = np.asarray(labels)
    correct_probabilities = np.asarray(probabilities)[np.arange(len(labels)), labels]
    correct_probabilities = np.clip(correct_probabilities, 1e-12, 1.0)
    individual_losses = -np.log(correct_probabilities)
    return float(np.mean(individual_losses * sample_weights))


def softmax_cross_entropy_backward(probabilities, labels, class_weights=None):
    """d(weighted CE)/d(logits), matching cross_entropy() exactly."""
    sample_weights = _checked_weights(probabilities, labels, class_weights)
    labels = np.asarray(labels)
    gradient = np.asarray(probabilities).copy()
    gradient[np.arange(len(labels)), labels] -= 1.0
    gradient *= sample_weights[:, None]
    gradient /= len(labels)
    return gradient
