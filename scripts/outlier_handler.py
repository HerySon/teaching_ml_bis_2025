"""
Module for detecting and handling outliers in datasets.
"""

try:
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
except ImportError as exc:
    raise ImportError('numpy, pandas, and scikit-learn are required for this module') from exc


def detect_outliers_tukey(data: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Detect outliers using Tukey's method (IQR).

    Args:
        data: Data series to analyze
        k: Multiplier for IQR (default 1.5)

    Returns:
        Boolean series indicating outliers
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr

    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detect outliers using Z-score method.

    Args:
        data: Data series to analyze
        threshold: Z-score threshold (default 3.0)

    Returns:
        Boolean series indicating outliers
    """
    z_scores = (data - data.mean()) / data.std()
    return abs(z_scores) > threshold


def detect_outliers_isolation_forest(data: pd.DataFrame, contamination: float = 0.1) -> pd.Series:
    """
    Detect outliers using Isolation Forest algorithm.

    Args:
        data: DataFrame with features
        contamination: Expected proportion of outliers (default 0.1)

    Returns:
        Boolean series indicating outliers
    """
    model = IsolationForest(contamination=contamination, random_state=42)
    return pd.Series(model.fit_predict(data) == -1, index=data.index)


def detect_outliers_lof(data: pd.DataFrame, n_neighbors: int = 20, contamination: float = 0.1) -> pd.Series:
    """
    Detect outliers using Local Outlier Factor algorithm.

    Args:
        data: DataFrame with features
        n_neighbors: Number of neighbors to consider (default 20)
        contamination: Expected proportion of outliers (default 0.1)

    Returns:
        Boolean series indicating outliers
    """
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    return pd.Series(model.fit_predict(data) == -1, index=data.index)


def get_outlier_strategies():
    """
    Return available outlier handling strategies.

    Returns:
        dict: Available strategies
    """
    return {
        "keep": "Keep outliers in the dataset",
        "remove": "Remove outliers from the dataset",
        "impute_mean": "Replace outliers with mean values",
        "impute_median": "Replace outliers with median values",
        "impute_mode": "Replace outliers with mode values",
        "impute_boundary": "Replace outliers with boundary values (Winsorizing)"
    }


def handle_outliers(
    data: pd.DataFrame,
    outlier_mask: pd.Series,
    strategy: str,
    columns: list[str] | None = None
) -> pd.DataFrame:
    """
    Handle outliers according to the selected strategy.

    Args:
        data: DataFrame containing data
        outlier_mask: Boolean series indicating which rows contain outliers
        strategy: Strategy to handle outliers from get_outlier_strategies()
        columns: List of column names to apply strategy (default: all numeric columns)

    Returns:
        DataFrame with handled outliers
    """
    if columns is None:
        columns = data.select_dtypes(include=np.number).columns.tolist()

    result = data.copy()

    if strategy == "keep":
        # Do nothing
        return result

    elif strategy == "remove":
        return result[~outlier_mask]

    elif strategy.startswith("impute_"):
        for col in columns:
            mask = outlier_mask & ~result[col].isna()

            if strategy == "impute_mean":
                value = result.loc[~outlier_mask, col].mean()
            elif strategy == "impute_median":
                value = result.loc[~outlier_mask, col].median()
            elif strategy == "impute_mode":
                value = result.loc[~outlier_mask, col].mode()[0]
            elif strategy == "impute_boundary":
                series = result[col]
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                # Apply boundary values only to numeric values
                for idx in result[mask].index:
                    val = result.loc[idx, col]
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        if val < lower_bound:
                            result.loc[idx, col] = lower_bound
                        elif val > upper_bound:
                            result.loc[idx, col] = upper_bound
                continue

            result.loc[mask, col] = value

    return result


class OutlierDetector:
    """Class to detect and handle outliers in datasets."""

    def __init__(self, data: pd.DataFrame):
        """
        Initialize with a dataframe.

        Args:
            data: DataFrame to analyze
        """
        self.data = data
        self.numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
        self.outlier_masks = {}

    def detect_outliers(self, method: str = "tukey", **kwargs) -> pd.Series:
        """
        Detect outliers using the specified method.

        Args:
            method: Method to use ('tukey', 'zscore', 'isolation_forest', 'lof')
            **kwargs: Additional parameters for the detection method

        Returns:
            Boolean series indicating outliers
        """
        if method == "tukey":
            results = {}
            for col in self.numeric_columns:
                results[col] = detect_outliers_tukey(self.data[col], **kwargs)

            # A row is an outlier if any of its values is an outlier
            outlier_mask = pd.DataFrame(results).any(axis=1)

        elif method == "zscore":
            results = {}
            for col in self.numeric_columns:
                results[col] = detect_outliers_zscore(self.data[col], **kwargs)

            outlier_mask = pd.DataFrame(results).any(axis=1)

        elif method == "isolation_forest":
            numeric_data = self.data[self.numeric_columns].copy()
            outlier_mask = detect_outliers_isolation_forest(numeric_data, **kwargs)

        elif method == "lof":
            numeric_data = self.data[self.numeric_columns].copy()
            outlier_mask = detect_outliers_lof(numeric_data, **kwargs)

        else:
            raise ValueError(f"Unknown method: {method}")

        self.outlier_masks[method] = outlier_mask
        return outlier_mask

    def get_outlier_summary(self) -> dict:
        """
        Get summary of detected outliers.

        Returns:
            Dictionary with outlier counts and percentages
        """
        summary = {}
        for method, mask in self.outlier_masks.items():
            outlier_count = mask.sum()
            total_count = len(mask)
            summary[method] = {
                "count": outlier_count,
                "percentage": outlier_count / total_count * 100
            }
        return summary

    def handle_outliers(self, method: str, strategy: str, columns: list[str] | None = None) -> pd.DataFrame:
        """
        Handle outliers using the specified strategy.

        Args:
            method: Outlier detection method used
            strategy: Strategy to handle outliers
            columns: List of column names to apply strategy (default: all numeric columns)

        Returns:
            DataFrame with handled outliers
        """
        if method not in self.outlier_masks:
            raise ValueError(f"Method {method} not used for detection yet. Call detect_outliers first.")

        return handle_outliers(
            self.data,
            self.outlier_masks[method],
            strategy,
            columns=columns
        )
