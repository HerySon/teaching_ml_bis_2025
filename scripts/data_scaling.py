"""
Data scaling module for OpenFoodFacts dataset.
This module provides functions to scale numerical features in the dataset
using different scaling methods.
"""

try:
    import pandas as pd
    from typing import Callable, Literal, Optional
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.preprocessing import PowerTransformer, QuantileTransformer
except ImportError as exc:
    raise ImportError('pandas, numpy, and scikit-learn are required for this module') from exc


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """
    Get all numeric columns from the dataframe.

    Args:
        df: Input dataframe

    Returns:
        list of numeric column names
    """
    return df.select_dtypes(include=['number']).columns.tolist()


def standard_scaler(df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Apply StandardScaler (Z-score normalization) to specified columns.

    Args:
        df: Input dataframe
        columns: List of columns to scale, if None uses all numeric columns

    Returns:
        DataFrame with scaled columns
    """
    if columns is None:
        columns = get_numeric_columns(df)

    scaled_df = df.copy()
    scaler = StandardScaler()

    if columns:
        scaled_df[columns] = scaler.fit_transform(scaled_df[columns].fillna(0))

    return scaled_df


def minmax_scaler(df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Apply MinMaxScaler (scales to range [0, 1]) to specified columns.

    Args:
        df: Input dataframe
        columns: List of columns to scale, if None uses all numeric columns

    Returns:
        DataFrame with scaled columns
    """
    if columns is None:
        columns = get_numeric_columns(df)

    scaled_df = df.copy()
    scaler = MinMaxScaler()

    if columns:
        scaled_df[columns] = scaler.fit_transform(scaled_df[columns].fillna(0))

    return scaled_df


def robust_scaler(df: pd.DataFrame, columns: Optional[list[str]] = None) -> pd.DataFrame:
    """
    Apply RobustScaler (scales based on quantiles, less affected by outliers) to specified columns.

    Args:
        df: Input dataframe
        columns: List of columns to scale, if None uses all numeric columns

    Returns:
        DataFrame with scaled columns
    """
    if columns is None:
        columns = get_numeric_columns(df)

    scaled_df = df.copy()
    scaler = RobustScaler()

    if columns:
        scaled_df[columns] = scaler.fit_transform(scaled_df[columns].fillna(0))

    return scaled_df


def power_transformer(
    df: pd.DataFrame,
    method: Literal['yeo-johnson', 'box-cox'] = 'yeo-johnson',
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Apply PowerTransformer to make data more Gaussian-like.

    Args:
        df: Input dataframe
        method: 'yeo-johnson' or 'box-cox'. Yeo-Johnson works with negative values.
        columns: List of columns to transform, if None uses all numeric columns

    Returns:
        DataFrame with transformed columns
    """
    if columns is None:
        columns = get_numeric_columns(df)

    scaled_df = df.copy()
    pt = PowerTransformer(method=method)

    if columns:
        scaled_df[columns] = pt.fit_transform(scaled_df[columns].fillna(0))

    return scaled_df


def quantile_transformer(
    df: pd.DataFrame,
    n_quantiles: int = 1000,
    distribution: Literal['uniform', 'normal'] = 'uniform',
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Apply QuantileTransformer to transform to uniform or normal distribution.

    Args:
        df: Input dataframe
        n_quantiles: Number of quantiles
        distribution: Either 'uniform' or 'normal'
        columns: List of columns to transform, if None uses all numeric columns

    Returns:
        DataFrame with transformed columns
    """
    if columns is None:
        columns = get_numeric_columns(df)

    scaled_df = df.copy()
    qt = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=distribution)

    if columns:
        scaled_df[columns] = qt.fit_transform(scaled_df[columns].fillna(0))

    return scaled_df


def select_scaling_method(method: str) -> Callable:
    """
    Select a scaling method based on the method name.

    Args:
        method: Name of the scaling method

    Returns:
        Scaling function

    Raises:
        ValueError: If method is not recognized
    """
    method_map = {
        'standard': standard_scaler,
        'minmax': minmax_scaler,
        'robust': robust_scaler,
        'power': power_transformer,
        'quantile': quantile_transformer,
    }

    if method not in method_map:
        raise ValueError(
            f"Unknown scaling method: {method}."
            f"Available methods: {list(method_map.keys())}",
        )

    return method_map[method]


def scale_dataframe(
    df: pd.DataFrame, method: str = 'standard',
    columns: Optional[list[str]] = None, **kwargs,
) -> pd.DataFrame:
    """
    Scale a dataframe using the specified method.

    Args:
        df: Input dataframe
        method: Scaling method ('standard', 'minmax', 'robust', 'power', 'quantile')
        columns: Columns to scale, if None uses all numeric columns
        **kwargs: Additional arguments for the specific scaling method

    Returns:
        DataFrame with scaled columns
    """
    scaling_function = select_scaling_method(method)

    if method in ['power', 'quantile']:
        return scaling_function(df, columns=columns, **kwargs)
    else:
        return scaling_function(df, columns=columns)
