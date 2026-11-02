"""Quick checks of loss, class-weight math, and gradient; no dataset needed.

From repository root: python -m unittest capstone_nn.franklin_class_weighting.test_weighting -v
"""
import unittest
import numpy as np
from .losses import calculate_class_weights, cross_entropy, softmax, softmax_cross_entropy_backward


class WeightingTests(unittest.TestCase):
    def test_class_weights(self):
        y = np.array([0, 0, 0, 0, 1, 1, 2])
        w = calculate_class_weights(y)
        np.testing.assert_allclose(w, [7 / 12, 7 / 6, 7 / 3])

    def test_all_ones_is_standard(self):
        p = softmax(np.array([[0.2, 0.5, -0.1], [1.0, -1.0, 0.5]]))
        y = np.array([1, 2])
        self.assertAlmostEqual(cross_entropy(p, y), cross_entropy(p, y, np.ones(3)))
        np.testing.assert_allclose(softmax_cross_entropy_backward(p, y),
                                   softmax_cross_entropy_backward(p, y, np.ones(3)))

    def test_weighted_gradient_finite_difference(self):
        logits = np.array([[0.1, 1.7, -0.6], [0.5, -0.9, 0.2]], dtype=float)
        y = np.array([1, 2])
        weights = np.array([0.7, 1.5, 3.1])
        analytic = softmax_cross_entropy_backward(softmax(logits), y, weights)
        numerical = np.zeros_like(logits)
        h = 1e-6
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                plus, minus = logits.copy(), logits.copy()
                plus[i, j] += h
                minus[i, j] -= h
                numerical[i, j] = (
                    cross_entropy(softmax(plus), y, weights) -
                    cross_entropy(softmax(minus), y, weights)
                ) / (2 * h)
        np.testing.assert_allclose(analytic, numerical, atol=1e-7, rtol=1e-6)

    def test_missing_class_not_allowed(self):
        with self.assertRaises(ValueError):
            calculate_class_weights(np.array([0, 0, 1]))


if __name__ == "__main__":
    unittest.main()
