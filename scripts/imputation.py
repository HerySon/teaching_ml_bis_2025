"""
Utility functions for imputing missing values in the OpenFoodFacts dataset.
"""

try:
    import pandas as pd
except ImportError:
    raise ImportError("pandas is required for this module")


def identify_imputable_columns(df: pd.DataFrame, threshold: float = 0.5) -> dict[str, str]:
    """
    Identify columns that can be imputed with difference strategies.

    Args:
        df: Input DataFrame
        threshold: Maximum missing value percentage to consider a column imputable

    Returns:
        Dictionary with column names as keys and imputation strategy as values
    """
    missing_percentage = df.isnull().mean()
    imputable_cols = missing_percentage[
        (missing_percentage > 0) & (missing_percentage < threshold)
    ].index.tolist()

    # Determine imputation strategy based on data type
    imputation_strategy = {}
    for col in imputable_cols:
        if col not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            # For numeric columns, use median
            imputation_strategy[col] = 'median'
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # For categorical columns, use most frequent value
            imputation_strategy[col] = 'most_frequent'

    print(f"Identified {len(imputation_strategy)} imputable columns")
    return imputation_strategy


def impute_numeric(
    df: pd.DataFrame,
    column: str,
    strategy: str = 'median',
) -> pd.DataFrame:
    """
    Impute missing values in a numeric column.

    Args:
        df: Input DataFrame
        column: Column name to impute
        strategy: Imputation strategy ('median', 'mean', 'zero')

    Returns:
        DataFrame with imputed values
    """
    if column not in df.columns:
        print(f"Column {column} not found in DataFrame")
        return df

    df_imputed = df.copy()

    if strategy == 'median':
        fill_value = df[column].median()
    elif strategy == 'mean':
        fill_value = df[column].mean()
    elif strategy == 'zero':
        fill_value = 0
    else:
        print(f"Unknown imputation strategy: {strategy}")
        return df

    df_imputed[column] = df_imputed[column].fillna(fill_value)
    return df_imputed


def impute_categorical(
    df: pd.DataFrame,
    column: str,
    strategy: str = 'most_frequent',
) -> pd.DataFrame:
    """
    Impute missing values in a categorical column.

    Args:
        df: Input DataFrame
        column: Column name to impute
        strategy: Imputation strategy ('most_frequent', 'missing_category')

    Returns:
        DataFrame with imputed values
    """
    if column not in df.columns:
        print(f"Column {column} not found in DataFrame")
        return df

    df_imputed = df.copy()

    if strategy == 'most_frequent':
        # Get most frequent value - handle empty case
        if df[column].value_counts().empty:
            fill_value = "Unknown"
        else:
            fill_value = str(df[column].value_counts().index[0])
        df_imputed[column] = df_imputed[column].fillna(fill_value)
    elif strategy == 'missing_category':
        # Create a separate category for missing values
        df_imputed[column] = df_imputed[column].fillna('Missing')
    else:
        print(f"Unknown imputation strategy: {strategy}")
        return df

    return df_imputed


def impute_columns(
    df: pd.DataFrame,
    strategies: dict[str, str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Impute missing values in multiple columns based on strategies.

    Args:
        df: Input DataFrame
        strategies: Dictionary mapping column names to imputation strategies
                   If None, will determine strategies automatically
        threshold: Maximum missing value percentage to consider for automatic imputation

    Returns:
        DataFrame with imputed values
    """
    if strategies is None:
        strategies = identify_imputable_columns(df, threshold)

    df_imputed = df.copy()

    for column, strategy in strategies.items():
        if column not in df.columns:
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            df_imputed = impute_numeric(df_imputed, column, strategy)
        else:
            df_imputed = impute_categorical(df_imputed, column, strategy)

        missing_before = df[column].isnull().sum()
        missing_after = df_imputed[column].isnull().sum()
        print(
            f"Imputed {missing_before - missing_after}"
            f"missing values in {column} using {strategy}",
        )

    return df_imputed


def impute_by_group(
    df: pd.DataFrame,
    column: str,
    group_by: str,
    strategy: str = 'median',
) -> pd.DataFrame:
    """
    Impute missing values in a column based on group statistics.

    Args:
        df: Input DataFrame
        column: Column name to impute
        group_by: Column name to group by
        strategy: Imputation strategy ('median', 'mean')

    Returns:
        DataFrame with imputed values
    """
    if column not in df.columns or group_by not in df.columns:
        print(f"Column {column} or {group_by} not found in DataFrame")
        return df

    df_imputed = df.copy()

    # Calculate the aggregation value per group
    if strategy == 'median':
        group_values = df.groupby(group_by)[column].transform('median')
    elif strategy == 'mean':
        group_values = df.groupby(group_by)[column].transform('mean')
    else:
        print(f"Unknown imputation strategy: {strategy}")
        return df

    # Fill NaN values with the group value, or with the overall column value if group is also NaN
    mask = df_imputed[column].isna()
    df_imputed.loc[mask, column] = group_values.loc[mask]

    # Use global imputation for any values still missing (group was also NaN)
    if df_imputed[column].isna().any():
        if strategy == 'median':
            global_value = df[column].median()
        else:
            global_value = df[column].mean()

        df_imputed[column] = df_imputed[column].fillna(global_value)

    missing_before = df[column].isnull().sum()
    missing_after = df_imputed[column].isnull().sum()
    print(
        f"Imputed {missing_before - missing_after} missing values in {column} using {strategy} by {group_by}",
    )

    return df_imputed
