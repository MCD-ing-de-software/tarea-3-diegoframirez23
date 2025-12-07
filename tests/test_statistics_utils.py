import numpy as np
import numpy.testing as npt
import unittest

from src.statistics_utils import StatisticsUtils


class TestStatisticsUtils(unittest.TestCase):
    """Test suite for StatisticsUtils class."""

    def test_example_moving_average_with_numpy_testing(self):
        utils = StatisticsUtils()
        arr = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = utils.moving_average(arr, window=3)
        expected = np.array([2.0, 3.0, 4.0])
        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_example_min_max_scale_with_numpy_testing(self):
        utils = StatisticsUtils()
        arr = [10.0, 20.0, 30.0, 40.0]
        result = utils.min_max_scale(arr)
        expected = np.array([0.0, 1/3, 2/3, 1.0])
        npt.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)

    def test_moving_average_basic_case(self):
        utils = StatisticsUtils()
        arr = [1, 2, 3, 4]
        result = utils.moving_average(arr, window=2)
        expected = np.array([1.5, 2.5, 3.5])

        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)
        self.assertEqual(result.shape, expected.shape)

    def test_moving_average_raises_for_invalid_window(self):
        utils = StatisticsUtils()
        arr = [1, 2, 3]

        with self.assertRaises(ValueError):
            utils.moving_average(arr, window=0)

        with self.assertRaises(ValueError):
            utils.moving_average(arr, window=5)

    def test_moving_average_only_accepts_1d_sequences(self):
        utils = StatisticsUtils()
        arr_2d = [[1, 2], [3, 4]]

        with self.assertRaises(ValueError):
            utils.moving_average(arr_2d, window=2)

    def test_zscore_has_mean_zero_and_unit_std(self):
        utils = StatisticsUtils()
        arr = [10, 20, 30, 40]

        result = utils.zscore(arr)

        mean = np.mean(result)
        std = np.std(result)

        self.assertAlmostEqual(mean, 0.0, places=7)
        self.assertAlmostEqual(std, 1.0, places=7)

    def test_zscore_raises_for_zero_std(self):
        utils = StatisticsUtils()
        arr = [5, 5, 5]

        with self.assertRaises(ValueError):
            utils.zscore(arr)

    def test_min_max_scale_maps_to_zero_one_range(self):
        utils = StatisticsUtils()
        arr = [2, 4, 6]

        result = utils.min_max_scale(arr)
        expected = np.array([0.0, 0.5, 1.0])

        self.assertAlmostEqual(result.min(), 0.0, places=7)
        self.assertAlmostEqual(result.max(), 1.0, places=7)

        npt.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)

    def test_min_max_scale_raises_for_constant_values(self):
        utils = StatisticsUtils()
        arr = [3, 3, 3]

        with self.assertRaises(ValueError):
            utils.min_max_scale(arr)


if __name__ == "__main__":
    unittest.main()