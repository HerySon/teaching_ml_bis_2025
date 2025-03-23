"""
Script to test different scaling methods on the OpenFoodFacts dataset.
This script loads a subset of the OpenFoodFacts data and tests different scaling
methods to determine the most appropriate one for each type of variable.
"""

try:
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from typing import Optional, Tuple, Literal, List, cast
except ImportError as exc:
    raise ImportError('pandas, numpy, matplotlib, seaborn, and scikit-learn are required') from exc

from data_scaling import (
    standard_scaler,
    minmax_scaler,
    robust_scaler,
    power_transformer,
    quantile_transformer,
    get_numeric_columns
)


def load_data(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Load the OpenFoodFacts dataset.

    Args:
        path: Path to the dataset (can be URL or local file)
        nrows: Number of rows to load (None for all)

    Returns:
        DataFrame containing the dataset
    """
    print(f"Loading data from {path}...")
    return pd.read_csv(path, nrows=nrows, sep='\t', encoding="utf-8")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the dataset for scaling:
    - Select only relevant columns
    - Handle missing values
    - Convert data types if needed

    Args:
        df: Input dataframe

    Returns:
        Preprocessed dataframe
    """
    print("Preprocessing data...")

    # Select only numeric columns that might be relevant for clustering
    # Focus on nutritional values and product characteristics
    nutrient_cols = [col for col in df.columns if
                     any(x in col for x in ['100g', 'serving', 'energy', 'fat',
                                          'carbohydrates', 'sugars', 'proteins',
                                          'salt', 'sodium', 'fiber'])]

    # Add some metadata columns
    meta_cols = ['code', 'created_t', 'last_modified_t']

    # Select columns and drop rows with all missing nutritional values
    selected_cols = nutrient_cols + meta_cols
    subset_df = df[selected_cols]

    # Fill NaN values with 0 for nutritional values (assuming missing = not present)
    subset_df[nutrient_cols] = subset_df[nutrient_cols].fillna(0)

    # Remove any remaining rows with NaN
    subset_df = subset_df.dropna()

    print(f"Preprocessed data shape: {subset_df.shape}")
    return subset_df


def analyze_distributions(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    save_dir: str = 'results/distributions'
) -> None:
    """
    Analyze the distributions of numeric columns before scaling.

    Args:
        df: Input dataframe
        columns: List of columns to analyze (if None, uses all numeric columns)
        save_dir: Directory to save the distribution plots
    """
    if columns is None:
        columns = get_numeric_columns(df)

    # Ensure columns is a list
    columns_list = list(columns)

    # Create sample of columns if too many
    if len(columns_list) > 10:
        column_sample = cast(
            list[str],
            np.random.choice(columns_list, 10, replace=False).tolist()
        )
    else:
        column_sample = columns_list

    print(f"Analyzing distributions of {len(column_sample)} columns...")

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Create distribution plots
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(column_sample, 1):
        plt.subplot(4, 3, i)
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.tight_layout()

    plt.savefig(f"{save_dir}/original_distributions.png")
    plt.close()


def compare_scaling_methods(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    save_dir: str = 'results/scaling_comparison'
) -> None:
    """
    Compare different scaling methods on selected columns.

    Args:
        df: Input dataframe
        columns: List of columns to analyze (if None, uses sample of numeric columns)
        save_dir: Directory to save the comparison plots
    """
    if columns is None:
        all_numeric = get_numeric_columns(df)
        # Select a sample of numeric columns
        columns = cast(
            list[str],
            np.random.choice(all_numeric, min(5, len(all_numeric)), replace=False).tolist()
        )

    print(f"Comparing scaling methods on columns: {columns}")

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Apply different scaling methods
    scaled_dfs = {
        'Original': df,
        'StandardScaler': standard_scaler(df, columns),
        'MinMaxScaler': minmax_scaler(df, columns),
        'RobustScaler': robust_scaler(df, columns),
        'PowerTransformer': power_transformer(df, columns=columns),
        'QuantileTransformer (uniform)': quantile_transformer(df, distribution='uniform', columns=columns),
        'QuantileTransformer (normal)': quantile_transformer(df, distribution='normal', columns=columns)
    }

    # Create comparison plots for each column
    for col in columns:
        plt.figure(figsize=(15, 10))

        for i, (name, scaled_df) in enumerate(scaled_dfs.items(), 1):
            plt.subplot(3, 3, i)
            sns.histplot(data=scaled_df, x=col, kde=True)
            plt.title(f'{name} - {col}')

        plt.tight_layout()
        plt.savefig(f"{save_dir}/{col}_comparison.png")
        plt.close()


def evaluate_scaling_method(
    df: pd.DataFrame,
    method: Literal['standard', 'minmax', 'robust', 'power', 'quantile'],
    columns: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a scaling method by comparing statistics before and after scaling.

    Args:
        df: Input dataframe
        method: Scaling method to evaluate
        columns: Columns to scale (if None, uses all numeric columns)

    Returns:
        Tuple of (scaled dataframe, statistics comparison dataframe)
    """
    if columns is None:
        columns = get_numeric_columns(df)

    print(f"Evaluating {method} scaling method...")

    # Get statistics before scaling
    before_stats = df[columns].describe().T
    before_stats.columns = [f'before_{col}' for col in before_stats.columns]

    # Apply scaling
    if method == 'standard':
        scaled_df = standard_scaler(df, columns)
    elif method == 'minmax':
        scaled_df = minmax_scaler(df, columns)
    elif method == 'robust':
        scaled_df = robust_scaler(df, columns)
    elif method == 'power':
        scaled_df = power_transformer(df, columns=columns)
    elif method == 'quantile':
        scaled_df = quantile_transformer(df, columns=columns)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Get statistics after scaling
    after_stats = scaled_df[columns].describe().T
    after_stats.columns = [f'after_{col}' for col in after_stats.columns]

    # Combine statistics
    stats_comparison = pd.concat([before_stats, after_stats], axis=1)

    return scaled_df, stats_comparison


def recommend_scaling_methods(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze columns and recommend appropriate scaling methods for each.

    Args:
        df: Input dataframe with numeric columns

    Returns:
        DataFrame with columns and recommended scaling methods
    """
    numeric_cols = get_numeric_columns(df)
    recommendations = []

    print("Analyzing columns to recommend scaling methods...")

    for col in numeric_cols:
        col_data = df[col].dropna()

        # Skip if column has no data
        if len(col_data) == 0:
            continue

        # Calculate statistics
        skewness = col_data.skew()
        has_outliers = np.abs(col_data - col_data.mean()) > (3 * col_data.std())
        pct_outliers = has_outliers.mean() * 100

        # Determine recommended method based on skewness and outliers
        if pct_outliers > 5:
            if isinstance(
                skewness,
                (int, float)) and (skewness > 1 or skewness < -1
            ):
                recommendation = "power"  # For skewed data with outliers
            else:
                recommendation = "robust"  # For data with outliers
        else:
            if isinstance(
                skewness,
                (int, float)) and (skewness > 1 or skewness < -1
            ):
                recommendation = "quantile"  # For skewed data
            else:
                recommendation = "standard"  # For normally distributed data

        # Add to recommendations
        recommendations.append({
            'column': col,
            'skewness': skewness,
            'pct_outliers': pct_outliers,
            'recommended_scaling': recommendation
        })

    return pd.DataFrame(recommendations)


def main():
    """Main function to test scaling methods."""

    # URL for the dataset
    data_url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"

    # Load a subset of the data for testing
    print("Starting data scaling tests...")
    df = load_data(data_url, nrows=1000)

    # Preprocess the data
    processed_df = preprocess_data(df)

    # Analyze distributions before scaling
    analyze_distributions(processed_df)

    # Compare different scaling methods
    compare_scaling_methods(processed_df)

    # Recommend scaling methods for each column
    recommendations = recommend_scaling_methods(processed_df)

    # Save recommendations
    os.makedirs('results', exist_ok=True)
    recommendations.to_csv('results/scaling_recommendations.csv', index=False)

    print("Scaling recommendations saved to results/scaling_recommendations.csv")

    # Group columns by recommended scaling method
    grouped_cols = recommendations.groupby('recommended_scaling')['column'].apply(list).to_dict()

    # Apply recommended scaling methods to each group
    final_scaled_df = processed_df.copy()

    for method, cols in grouped_cols.items():
        if method == 'standard':
            final_scaled_df[cols] = standard_scaler(processed_df, cols)[cols]
        elif method == 'minmax':
            final_scaled_df[cols] = minmax_scaler(processed_df, cols)[cols]
        elif method == 'robust':
            final_scaled_df[cols] = robust_scaler(processed_df, cols)[cols]
        elif method == 'power':
            final_scaled_df[cols] = power_transformer(processed_df, columns=cols)[cols]
        elif method == 'quantile':
            final_scaled_df[cols] = quantile_transformer(processed_df, columns=cols)[cols]

    # Save the final scaled dataframe
    final_scaled_df.to_csv('results/scaled_data.csv', index=False)
    print("Final scaled data saved to results/scaled_data.csv")

    print("Data scaling tests completed successfully!")


if __name__ == "__main__":
    main()
