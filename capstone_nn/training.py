import numpy as np

from capstone_nn.losses import (
    softmax,
    cross_entropy,
    softmax_cross_entropy_backward,
)


def train_epoch(
    model,
    X,
    y,
    batch_size,
    learning_rate,
    rng,
):
    """
    Train a model for one complete pass over the dataset.

    Args:
        model:
            Sequential neural network.

        X:
            Input samples.

        y:
            Integer class labels.

        batch_size:
            Number of samples processed per update.

        learning_rate:
            Gradient-descent step size.

        rng:
            NumPy random-number generator used for shuffling.

    Returns:
        Dictionary containing epoch loss and accuracy.
    """

    indices = rng.permutation(
        len(X)
    )

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for start in range(
        0,
        len(X),
        batch_size,
    ):
        end = start + batch_size

        batch_indices = indices[
            start:end
        ]

        X_batch = X[
            batch_indices
        ]

        y_batch = y[
            batch_indices
        ]

        logits = model.forward(
            X_batch
        )

        probabilities = softmax(
            logits
        )

        batch_loss = cross_entropy(
            probabilities,
            y_batch,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        current_batch_size = len(
            X_batch
        )

        total_loss += (
            batch_loss
            * current_batch_size
        )

        total_correct += np.sum(
            predictions == y_batch
        )

        total_samples += (
            current_batch_size
        )

        grad_logits = (
            softmax_cross_entropy_backward(
                probabilities,
                y_batch,
            )
        )

        model.backward(
            grad_logits
        )

        model.update(
            learning_rate
        )

    epoch_loss = (
        total_loss
        / total_samples
    )

    epoch_accuracy = (
        total_correct
        / total_samples
    )

    return {
        "loss": epoch_loss,
        "accuracy": epoch_accuracy,
    }


def evaluate(
    model,
    X,
    y,
    batch_size=256,
):
    """
    Evaluate a model without updating its parameters.

    Returns:
        Dictionary containing loss and accuracy.
    """

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for start in range(
        0,
        len(X),
        batch_size,
    ):
        end = start + batch_size

        X_batch = X[
            start:end
        ]

        y_batch = y[
            start:end
        ]

        logits = model.forward(
            X_batch
        )

        probabilities = softmax(
            logits
        )

        batch_loss = cross_entropy(
            probabilities,
            y_batch,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        current_batch_size = len(
            X_batch
        )

        total_loss += (
            batch_loss
            * current_batch_size
        )

        total_correct += np.sum(
            predictions == y_batch
        )

        total_samples += (
            current_batch_size
        )

    return {
        "loss": (
            total_loss
            / total_samples
        ),
        "accuracy": (
            total_correct
            / total_samples
        ),
    }
