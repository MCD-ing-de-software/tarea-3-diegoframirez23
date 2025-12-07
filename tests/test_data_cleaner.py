import pandas as pd
import pandas.testing as pdt
import unittest

from src.data_cleaner import DataCleaner


def make_sample_df() -> pd.DataFrame:
    """Create a small DataFrame for testing."""
    return pd.DataFrame(
        {
            "name": [" Alice ", "Bob", None, " Carol  "],
            "age": [25, None, 35, 120],  # 120 is a likely outlier
            "city": ["SCL", "LPZ", "SCL", "LPZ"],
        }
    )


class TestDataCleaner(unittest.TestCase):
    """Test suite for DataCleaner class."""

    def test_example_trim_strings_with_pandas_testing(self):
        df = pd.DataFrame({
            "name": ["  Alice  ", "  Bob  ", "Carol"],
            "age": [25, 30, 35]
        })
        cleaner = DataCleaner()

        result = cleaner.trim_strings(df, ["name"])

        expected = pd.DataFrame({
            "name": ["Alice", "Bob", "Carol"],
            "age": [25, 30, 35]
        })

        pdt.assert_frame_equal(result, expected)

    def test_example_drop_invalid_rows_with_pandas_testing(self):
        df = pd.DataFrame({
            "name": ["Alice", None, "Bob"],
            "age": [25, 30, None],
            "city": ["SCL", "LPZ", "SCL"]
        })
        cleaner = DataCleaner()

        result = cleaner.drop_invalid_rows(df, ["name"])

        expected_name_series = pd.Series(
            ["Alice", "Bob"], index=[0, 2], name="name"
        )

        pdt.assert_series_equal(
            result["name"], expected_name_series, check_names=True
        )

    def test_drop_invalid_rows_removes_rows_with_missing_values(self):
        df = make_sample_df()
        cleaner = DataCleaner()

        result = cleaner.drop_invalid_rows(df, ["name", "age"])

        self.assertEqual(result["name"].isna().sum(), 0)
        self.assertEqual(result["age"].isna().sum(), 0)
        self.assertLess(len(result), len(df))

    def test_drop_invalid_rows_raises_keyerror_for_unknown_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()

        with self.assertRaises(KeyError):
            cleaner.drop_invalid_rows(df, ["does_not_exist"])

    def test_trim_strings_strips_whitespace_without_changing_other_columns(self):
        df = make_sample_df()
        df["name"] = df["name"].astype("string")  # asegurar dtype string
        cleaner = DataCleaner()

        original_name_0 = df.loc[0, "name"]
        original_city_0 = df.loc[0, "city"]

        result = cleaner.trim_strings(df, ["name"])

        # verificar que el DataFrame original NO fue modificado
        self.assertEqual(df.loc[0, "name"], original_name_0)

        # verificar trimming correcto
        self.assertEqual(result.loc[0, "name"], "Alice")
        self.assertEqual(result.loc[3, "name"], "Carol")

        # columnas no especificadas no cambian
        pdt.assert_series_equal(result["city"], df["city"])

    def test_trim_strings_raises_typeerror_for_non_string_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()

        with self.assertRaises(TypeError):
            cleaner.trim_strings(df, ["age"])

    def test_remove_outliers_iqr_removes_extreme_values(self):
        df = make_sample_df()
        cleaner = DataCleaner()

        # usar un factor más estricto para forzar la eliminación del outlier
        result = cleaner.remove_outliers_iqr(df, "age", factor=0.5)

        self.assertNotIn(120, result["age"].values)
        self.assertTrue(
            (25 in result["age"].values) or (35 in result["age"].values)
        )

    def test_remove_outliers_iqr_raises_keyerror_for_missing_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()

        with self.assertRaises(KeyError):
            cleaner.remove_outliers_iqr(df, "salary", factor=1.5)

    def test_remove_outliers_iqr_raises_typeerror_for_non_numeric_column(self):
        df = make_sample_df()
        cleaner = DataCleaner()

        with self.assertRaises(TypeError):
            cleaner.remove_outliers_iqr(df, "city", factor=1.5)


if __name__ == "__main__":
    unittest.main()