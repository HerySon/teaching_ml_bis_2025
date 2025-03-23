"""
DataFrame filtering utilities for the OpenFoodFacts dataset.

This module provides utilities for filtering and selecting relevant columns from DataFrames,
particularly optimized for the OpenFoodFacts dataset.
"""

import pandas as pd
from typing import Any, Optional


def identify_column_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Identify column types in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    dict
        Dictionary with column types as keys and column names as values
    """
    # Initialize column type dictionaries
    column_types: dict[str, list[str]] = {
        "numeric": [], "categorical_ordinal": [], "categorical_non_ordinal": [], "date": [], "text": [], "other": [],
    }

    # Known ordinal columns in OpenFoodFacts
    known_ordinal_columns = ["nutriscore_grade", "nova_group", "environmental_score_grade"]

    for col in df.columns:
        # Check for numeric columns
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            column_types["numeric"].append(col)

        # Check for datetime columns
        elif pd.api.types.is_datetime64_dtype(df[col].dtype) or (
            df[col].dtype == object and ("datetime" in col.lower() or "date" in col.lower() or "time" in col.lower())):
            column_types["date"].append(col)

        # Check for known ordinal categorical columns
        elif col in known_ordinal_columns:
            column_types["categorical_ordinal"].append(col)

        # Check for categorical columns, function is_categorical_dtype isn't present in 2.2.3 so we use isinstance
        elif isinstance(df[col].dtype, pd.CategoricalDtype) or (df[col].nunique() < 0.1 * len(df) and df[col].nunique() <= 100):
            column_types["categorical_non_ordinal"].append(col)

        # Check for text columns (likely non-ordinal categorical with many values)
        elif df[col].dtype == object and df[col].nunique() > 100:
            column_types["text"].append(col)

        # Other columns
        else:
            column_types["other"].append(col)

    return column_types


def filter_categorical_by_cardinality(
    df: pd.DataFrame, columns: list[str], max_categories: int = 50, min_categories: int = 2,
) -> list[str]:
    """
    Filter categorical columns based on their cardinality.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    columns : list
        List of categorical column names to filter
    max_categories : int, default=50
        Maximum number of unique categories allowed
    min_categories : int, default=2
        Minimum number of unique categories required

    Returns:
    --------
    list
        Filtered list of categorical column names
    """
    filtered_columns: list[str] = []

    for col in columns:
        nunique = df[col].nunique()
        if min_categories <= nunique <= max_categories:
            filtered_columns.append(col)

    return filtered_columns


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast numeric columns to reduce memory usage.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame with downcasted columns
    """
    df_copy = df.copy()

    # Downcast float columns
    float_cols = df_copy.select_dtypes(include=['float']).columns
    for col in float_cols:
        df_copy[col] = pd.to_numeric(df_copy[col], downcast='float')

    # Downcast integer columns
    int_cols = df_copy.select_dtypes(include=['int']).columns
    for col in int_cols:
        df_copy[col] = pd.to_numeric(df_copy[col], downcast='integer')

    # Convert object columns with low cardinality to categorical
    obj_cols = df_copy.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if df_copy[col].nunique() < 0.5 * len(df_copy):
            df_copy[col] = df_copy[col].astype('category')

    return df_copy


def filter_by_missing_values(
    df: pd.DataFrame, threshold: float = 0.5,
) -> list[str]:
    """
    Filter columns based on the percentage of missing values.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    threshold : float, default=0.5
        Maximum percentage of missing values allowed

    Returns:
    --------
    list
        List of column names with missing values below the threshold
    """
    missing_values = df.isnull().mean()
    columns_to_keep = missing_values[missing_values <= threshold].index.tolist()
    return columns_to_keep


def select_relevant_columns(
    df: pd.DataFrame, missing_threshold: float = 0.5, max_categories: int = 50, min_categories: int = 2,
) -> dict[str, list[str]]:
    """
    Select relevant columns from the DataFrame based on various criteria.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    missing_threshold : float, default=0.5
        Maximum percentage of missing values allowed
    max_categories : int, default=50
        Maximum number of unique categories allowed for categorical columns
    min_categories : int, default=2
        Minimum number of unique categories required for categorical columns

    Returns:
    --------
    dict
        Dictionary with filtered column types
    """
    # Get columns with acceptable missing values
    valid_columns = filter_by_missing_values(df, threshold=missing_threshold)
    df_filtered = df[valid_columns]

    # Identify column types
    column_types = identify_column_types(df_filtered)

    # Filter categorical columns by cardinality
    column_types["categorical_non_ordinal"] = filter_categorical_by_cardinality(
        df_filtered, column_types["categorical_non_ordinal"], max_categories=max_categories, min_categories=min_categories,
    )

    return column_types


def filter_dataframe(
    df: pd.DataFrame,
    column_types_to_keep: list[str] | None = None,
    missing_threshold: float = 0.5,
    max_categories: int = 50,
    min_categories: int = 2,
    downcast: bool = True,
) -> pd.DataFrame:
    """
    Filter the DataFrame based on column types and other criteria.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    column_types_to_keep : list, default=None
        List of column types to keep. If None, uses ["numeric", "categorical_ordinal", "categorical_non_ordinal"]
    missing_threshold : float, default=0.5
        Maximum percentage of missing values allowed
    max_categories : int, default=50
        Maximum number of unique categories allowed for categorical columns
    min_categories : int, default=2
        Minimum number of unique categories required for categorical columns
    downcast : bool, default=True
        Whether to downcast the DataFrame to reduce memory usage

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame
    """
    if column_types_to_keep is None:
        column_types_to_keep = ["numeric", "categorical_ordinal", "categorical_non_ordinal"]

    # Select relevant columns
    column_types = select_relevant_columns(
        df, missing_threshold=missing_threshold, max_categories=max_categories, min_categories=min_categories,
    )

    # Collect columns to keep
    columns_to_keep: list[str] = []
    for col_type in column_types_to_keep:
        columns_to_keep.extend(column_types[col_type])

    # Filter DataFrame
    df_filtered = df[columns_to_keep]

    # Downcast if requested
    if downcast:
        df_filtered = downcast_dataframe(df_filtered)

    return df_filtered


def get_column_info(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get detailed information about each column in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame

    Returns:
    --------
    pd.DataFrame
        DataFrame with column information
    """
    # Initialize data
    data: dict[str, list[Any]] = {
        "column_name": [], "dtype": [], "unique_values": [], "missing_percentage": [], "memory_usage_mb": [],
    }

    # Collect information
    for col in df.columns:
        data["column_name"].append(col)
        data["dtype"].append(str(df[col].dtype))
        data["unique_values"].append(df[col].nunique())
        data["missing_percentage"].append(df[col].isnull().mean() * 100)
        data["memory_usage_mb"].append(df[col].memory_usage(deep=True) / (1024 * 1024))

    # Create DataFrame
    column_info = pd.DataFrame(data)
    column_info = column_info.sort_values(by="memory_usage_mb", ascending=False)

    return column_info
