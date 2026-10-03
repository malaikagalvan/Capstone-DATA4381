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


def softmax_cross_entropy_backward(
        probabilities,
        labels,
    ):

        '''
        comput the gradient of mean cross-entropy loss
        with respect to the logits

        Args:
            probabilities:
                Softmax probabilities with shape
                (batch_size, num_classes)

            labels:
                Integer class labels with shape
                (batch_size)

            Returns:
                Gradient with respect to logits
                Shape is the same as probabilities
        '''

        batch_size = probabilities.shape[0]

        grad_logits = probabilities.copy()

        grad_logits[
                np.arange(batch_size),
                labels,
            ] -= 1

        grad_logits /= batch_size

        return grad_logits


         


