"""Independent training loop with optional class weights.

This file mirrors the interface of the team's capstone_nn/training.py, but
imports the local weighted-loss functions instead. No original code is edited.
"""

import numpy as np
from .losses import softmax, cross_entropy, softmax_cross_entropy_backward


def train_epoch(model, X, y, batch_size, learning_rate, rng, class_weights=None):
    """Run one epoch; pass class_weights=None for ordinary/unweighted training.

    Model must expose forward(), backward(), and update(), as the team's
    capstone_nn.model.Sequential already does.
    """
    if len(X) != len(y) or len(X) == 0:
        raise ValueError("X and y must have matching, nonempty lengths")
    if batch_size < 1 or learning_rate <= 0:
        raise ValueError("batch_size and learning_rate must be positive")

    indices = rng.permutation(len(X))
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for start in range(0, len(X), batch_size):
        batch_indices = indices[start:start + batch_size]
        X_batch, y_batch = X[batch_indices], y[batch_indices]

        logits = model.forward(X_batch)
        probabilities = softmax(logits)
        batch_loss = cross_entropy(probabilities, y_batch, class_weights)
        predictions = np.argmax(probabilities, axis=1)

        n = len(y_batch)
        total_loss += batch_loss * n
        total_correct += int(np.sum(predictions == y_batch))
        total_samples += n

        # The weighted gradient MUST match the weighted loss above.
        grad_logits = softmax_cross_entropy_backward(
            probabilities, y_batch, class_weights
        )
        model.backward(grad_logits)
        model.update(learning_rate)

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def evaluate(model, X, y, batch_size=256):
    """Measure UNWEIGHTED loss and accuracy for comparable validation results."""
    if len(X) != len(y) or len(X) == 0:
        raise ValueError("X and y must have matching, nonempty lengths")
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for start in range(0, len(X), batch_size):
        X_batch, y_batch = X[start:start + batch_size], y[start:start + batch_size]
        probabilities = softmax(model.forward(X_batch))
        n = len(y_batch)
        total_loss += cross_entropy(probabilities, y_batch) * n
        total_correct += int(np.sum(np.argmax(probabilities, axis=1) == y_batch))
        total_samples += n
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}
