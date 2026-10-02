import numpy as np

def softmax(logits):
    '''
    Converts logits into class probabilities

    Args:
        logits:
            Shape(batch_size, num_classes)

    Returns:
        probabilities:
            Same shape as logits
            Each row sums to 1
    '''

    shifted_logits = (
            logits
            - np.max(
                logits,
                axis=1,
                keepdims=True,
            )
        )

    exp_values = np.exp(
            shifted_logits
        )

    probabilities = (
            exp_values
            / np.sum(
                exp_values,
                axis=1,
                keepdims=True,
            )
        )

    return probabilities


def cross_entropy(
        probabilities,
        labels,
    ):

    '''
    Compute mean cross-entropy loss

    Args:
        probabilities:
            Shape(batch_size, num_classes)

        labels:
            Shape(batch_size,)
            Each value is the index of the correct class

    Returns:
        Scalar mean loss
    '''

    batch_size = probabilities.shape[0]
    
    correct_probabilities = probabilities[
            np.arange(batch_size),
            labels,
        ]

    correct_probabilities = np.clip(
            correct_probabilities,
            1e-12,
            1.0,
        )

    losses = -np.log(
            correct_probabilities
        )

    return np.mean(losses)



