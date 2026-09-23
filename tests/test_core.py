"""Tests for the streaming quantile estimator.

These tests verify the core behavior of the P² algorithm implementation,
including edge cases and error conditions.
"""

import math
import unittest

from streaming_quantile import QuantileEstimator


class TestQuantileEstimator(unittest.TestCase):
    def test_rejects_invalid_quantile(self):
        with self.assertRaises(ValueError):
            QuantileEstimator(0.0)
        with self.assertRaises(ValueError):
            QuantileEstimator(1.0)
        with self.assertRaises(ValueError):
            QuantileEstimator(-0.5)
        with self.assertRaises(ValueError):
            QuantileEstimator(1.5)

    def test_estimate_before_five_observations_is_nan(self):
        est = QuantileEstimator(0.5)
        self.assertTrue(math.isnan(est.estimate()))
        est.observe(10)
        self.assertTrue(math.isnan(est.estimate()))
        est.observe(20)
        est.observe(30)
        est.observe(40)
        self.assertTrue(math.isnan(est.estimate()))

    def test_median_of_five_sorted_values(self):
        est = QuantileEstimator(0.5)
        for v in [1, 2, 3, 4, 5]:
            est.observe(v)
        self.assertEqual(est.estimate(), 3.0)

    def test_median_of_five_unsorted_values(self):
        est = QuantileEstimator(0.5)
        for v in [5, 1, 4, 2, 3]:
            est.observe(v)
        self.assertEqual(est.estimate(), 3.0)

    def test_extreme_quantiles_after_five(self):
        est = QuantileEstimator(0.0 + 1e-9)  # very close to 0, but valid
        for v in [10, 20, 30, 40, 50]:
            est.observe(v)
        # Estimate should be near the minimum, but not exactly due to
        # marker initialization; we only check it is within the data range.
        self.assertTrue(10 <= est.estimate() <= 50)

        est = QuantileEstimator(0.999999999)
        for v in [10, 20, 30, 40, 50]:
            est.observe(v)
        self.assertTrue(10 <= est.estimate() <= 50)

    def test_repeated_values(self):
        est = QuantileEstimator(0.5)
        for v in [7, 7, 7, 7, 7, 7, 7]:
            est.observe(v)
        self.assertEqual(est.estimate(), 7.0)

    def test_stream_of_ascending_integers(self):
        est = QuantileEstimator(0.5)
        for v in range(100):
            est.observe(v)
        # For a uniform distribution, the median estimate should be near 50.
        self.assertAlmostEqual(est.estimate(), 50, delta=10)

    def test_stream_of_descending_integers(self):
        est = QuantileEstimator(0.5)
        for v in range(100, 0, -1):
            est.observe(v)
        self.assertAlmostEqual(est.estimate(), 50, delta=10)

    def test_observe_many(self):
        est = QuantileEstimator(0.5)
        est.observe_many([1, 2, 3, 4, 5])
        self.assertEqual(est.estimate(), 3.0)

    def test_observe_rejects_non_numeric(self):
        est = QuantileEstimator(0.5)
        with self.assertRaises(TypeError):
            est.observe("a")
        with self.assertRaises(TypeError):
            est.observe(None)

    def test_float_and_int_mix(self):
        est = QuantileEstimator(0.5)
        est.observe(1)
        est.observe(2.0)
        est.observe(3)
        est.observe(4.0)
        est.observe(5)
        self.assertEqual(est.estimate(), 3.0)

    def test_large_stream_constant_memory_property(self):
        # We cannot directly test memory, but we can process many values
        # and ensure the estimator still returns a finite number within
        # the expected range.
        est = QuantileEstimator(0.9)
        for v in range(1000):
            est.observe(v)
        self.assertTrue(0 <= est.estimate() <= 999)

    def test_quantile_estimate_monotonic_with_data(self):
        # For a simple monotonic stream, the estimate should stay within
        # the observed min/max.
        est = QuantileEstimator(0.5)
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for v in data:
            est.observe(v)
        self.assertTrue(1 <= est.estimate() <= 10)


if __name__ == "__main__":
    unittest.main()
