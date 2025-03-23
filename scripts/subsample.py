#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides functions to subsample the Open Food Facts dataset
to create a representative subset while maintaining the distribution of key variables.
"""

import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_dataset(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Load the Open Food Facts dataset from a file.

    Args:
        filepath: Path to the dataset file (CSV format expected)

    Returns:
        DataFrame containing the dataset
    """
    logger.info(f"Loading dataset from {filepath}")

    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise


def analyze_categorical_columns(df: pd.DataFrame,
                               max_categories: int = 30,
                               figsize: tuple = (15, 10)) -> dict:
    """
    Analyze categorical columns in the dataset and visualize their distributions.

    Args:
        df: Input dataframe
        max_categories: Maximum number of categories to consider for stratification
        figsize: Size of the figure for plots

    Returns:
        Dictionary with categorical columns and their cardinality
    """
    logger.info("Analyzing categorical columns")

    # Find categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Calculate cardinality of each categorical column
    cardinality = {}
    for col in categorical_cols:
        n_unique = df[col].nunique()
        cardinality[col] = n_unique

    # Sort by cardinality
    sorted_cardinality = dict(sorted(cardinality.items(), key=lambda x: x[1]))

    # Filter columns with reasonable cardinality for stratification
    stratification_candidates = {k: v for k, v in sorted_cardinality.items()
                               if v >= 2 and v <= max_categories}

    # Visualize cardinality
    plt.figure(figsize=figsize)

    # Create a bar plot for columns with reasonable cardinality
    cols = list(stratification_candidates.keys())
    counts = list(stratification_candidates.values())

    if cols:
        bars = plt.bar(cols, counts)
        plt.xticks(rotation=90)
        plt.title('Number of Categories by Column')
        plt.xlabel('Column')
        plt.ylabel('Number of Categories')
        plt.tight_layout()

        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom')

        plt.savefig('results/categorical_columns_analysis.png')
        plt.close()

    logger.info(f"Found {len(stratification_candidates)} categorical columns suitable for stratification")

    return stratification_candidates


def visualize_categorical_distribution(df: pd.DataFrame,
                                      column: str,
                                      top_n: int = 10,
                                      figsize: tuple = (12, 8)) -> None:
    """
    Visualize the distribution of a categorical column.

    Args:
        df: Input dataframe
        column: Column name to visualize
        top_n: Number of top categories to show
        figsize: Size of the figure
    """
    if column not in df.columns:
        logger.warning(f"Column {column} not found in dataframe")
        return

    value_counts = df[column].value_counts()

    if len(value_counts) > top_n:
        # Keep top N categories and group others into "Other"
        top_values = value_counts.nlargest(top_n)
        others_count = value_counts[top_n:].sum()

        # Create a new series with top categories and "Other"
        plot_data = pd.Series({**top_values.to_dict(), "Other": others_count})
    else:
        plot_data = value_counts

    plt.figure(figsize=figsize)
    ax = sns.barplot(x=plot_data.index, y=plot_data.values)
    plt.title(f'Distribution of {column}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Add count labels
    for i, (idx, val) in enumerate(plot_data.items()):
        percentage = 100 * val / value_counts.sum()
        ax.annotate(f"{val:,}\n({percentage:.1f}%)",
                   (i, val),
                   ha='center', va='bottom')

    plt.savefig(f'results/{column}_distribution.png')
    plt.close()


def subsample_dataset(df: pd.DataFrame,
                     stratify_by: str,
                     sample_size: int | None = None,
                     sample_fraction: float = 0.1,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Create a stratified subsample of the dataset to maintain the distribution of a key variable.

    Args:
        df: Input dataframe
        stratify_by: Column name to stratify by
        sample_size: Specific sample size (if None, sample_fraction is used)
        sample_fraction: Fraction of original data to sample (if sample_size is None)
        random_state: Random seed for reproducibility

    Returns:
        Subsampled dataframe
    """
    if stratify_by not in df.columns:
        logger.error(f"Stratification column '{stratify_by}' not found in dataset")
        raise ValueError(f"Column '{stratify_by}' not found in dataset")

    # Handle missing values in stratification column
    if df[stratify_by].isna().any():
        logger.warning(f"Found {df[stratify_by].isna().sum()} missing values in '{stratify_by}'. "
                      f"Filling with 'Unknown'")
        df[stratify_by] = df[stratify_by].fillna('Unknown')

    # Determine sample size
    if sample_size is None:
        sample_size = int(len(df) * sample_fraction)
        logger.info(f"Using sample_fraction: {sample_fraction} to get sample_size: {sample_size}")

    logger.info(f"Creating stratified subsample of size {sample_size} stratified by '{stratify_by}'")

    # Check if we have very rare categories
    value_counts = df[stratify_by].value_counts()
    rare_categories = value_counts[value_counts < 5].index.tolist()

    if rare_categories:
        logger.warning(f"Found {len(rare_categories)} categories with fewer than 5 samples. "
                      f"These will be grouped as 'Other'")
        # Create a temporary stratification column
        strat_col = df[stratify_by].copy()
        strat_col = strat_col.apply(lambda x: 'Other' if x in rare_categories else x)
    else:
        strat_col = df[stratify_by]

    # Use stratified sampling to maintain distribution
    subsample = df.groupby(strat_col, group_keys=False).apply(
        lambda x: x.sample(
            n=max(1, int(sample_size * len(x) / len(df))),
            random_state=random_state
        )
    )

    # If we got more samples than requested (due to rounding), trim randomly
    if len(subsample) > sample_size:
        subsample = subsample.sample(sample_size, random_state=random_state)

    logger.info(f"Created subsample with {len(subsample)} rows")
    return subsample


def balanced_subsample_multiple_columns(df: pd.DataFrame,
                                       columns: list,
                                       sample_size: int | None = None,
                                       sample_fraction: float = 0.1,
                                       random_state: int = 42) -> pd.DataFrame:
    """
    Create a balanced subsample considering multiple columns using StratifiedKFold.

    Args:
        df: Input dataframe
        columns: List of column names to consider for balancing
        sample_size: Specific sample size (if None, sample_fraction is used)
        sample_fraction: Fraction of original data to sample (if sample_size is None)
        random_state: Random seed for reproducibility

    Returns:
        Balanced subsampled dataframe
    """
    # Validate columns
    for col in columns:
        if col not in df.columns:
            logger.error(f"Column '{col}' not found in dataset")
            raise ValueError(f"Column '{col}' not found in dataset")

    # Handle missing values in specified columns
    for col in columns:
        if df[col].isna().any():
            logger.warning(f"Filling {df[col].isna().sum()} missing values in '{col}' with 'Unknown'")
            df[col] = df[col].fillna('Unknown')

    # Create a composite key for stratification
    df['_strat_key'] = df[columns].astype(str).agg('_'.join, axis=1)

    # Determine sample size
    if sample_size is None:
        sample_size = int(len(df) * sample_fraction)

    logger.info(f"Creating balanced subsample of size {sample_size} considering columns: {', '.join(columns)}")

    # Use StratifiedKFold for balanced sampling
    skf = StratifiedKFold(n_splits=int(1/sample_fraction), random_state=random_state, shuffle=True)

    # Get indices of the first fold
    train_idx, _ = next(skf.split(df, df['_strat_key']))

    # Create subsample
    subsample = df.iloc[train_idx].copy()

    # If we have more samples than requested, trim randomly
    if len(subsample) > sample_size:
        subsample = subsample.sample(sample_size, random_state=random_state)
    elif len(subsample) < sample_size:
        logger.warning(f"Requested sample_size {sample_size} is larger than what StratifiedKFold "
                      f"provided ({len(subsample)}). Using all available samples.")

    # Remove temporary stratification key
    subsample = subsample.drop('_strat_key', axis=1)

    logger.info(f"Created subsample with {len(subsample)} rows")
    return subsample


def evaluate_subsample_quality(original_df: pd.DataFrame,
                              subsample_df: pd.DataFrame,
                              columns_to_check: list,
                              figsize: tuple = (15, 10)) -> None:
    """
    Evaluate how well the subsample represents the original dataset.

    Args:
        original_df: Original dataframe
        subsample_df: Subsampled dataframe
        columns_to_check: Columns to check for distribution similarity
        figsize: Figure size for plots
    """
    logger.info("Evaluating subsample quality")

    # Create directory for quality evaluation plots
    Path('results/subsample_quality').mkdir(exist_ok=True, parents=True)

    for col in columns_to_check:
        if col not in original_df.columns or col not in subsample_df.columns:
            logger.warning(f"Column {col} not found in one of the dataframes")
            continue

        # Get value counts and convert to percentages
        orig_counts = original_df[col].value_counts(normalize=True).sort_index()
        sub_counts = subsample_df[col].value_counts(normalize=True).sort_index()

        # Get union of all categories
        all_categories = sorted(set(orig_counts.index) | set(sub_counts.index))

        # Reindex to ensure both have the same categories
        orig_counts = orig_counts.reindex(all_categories, fill_value=0)
        sub_counts = sub_counts.reindex(all_categories, fill_value=0)

        # Top N categories in original dataset
        top_n = 15
        top_categories = original_df[col].value_counts().nlargest(top_n).index

        # Filter to top categories
        orig_counts_top = orig_counts.loc[top_categories]
        sub_counts_top = sub_counts.loc[top_categories]

        # Create comparison plot
        plt.figure(figsize=figsize)

        # Create a DataFrame for plotting
        plot_df = pd.DataFrame({
            'Original': orig_counts_top * 100,
            'Subsample': sub_counts_top * 100
        })

        # Plot
        plot_df.plot(kind='bar', figsize=figsize)
        plt.title(f'Distribution Comparison for {col} (Top {top_n} categories)')
        plt.ylabel('Percentage (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'results/subsample_quality/{col}_distribution_comparison.png')
        plt.close()

        # Calculate and log distribution similarity metrics
        jsd = jensen_shannon_distance(orig_counts, sub_counts)
        logger.info(f"Jensen-Shannon distance for {col}: {jsd:.4f} (lower is better, 0 is identical)")


def jensen_shannon_distance(p: pd.Series, q: pd.Series) -> float:
    """
    Calculate Jensen-Shannon distance between two probability distributions.

    Args:
        p: First probability distribution
        q: Second probability distribution

    Returns:
        Jensen-Shannon distance (between 0 and 1)
    """
    # Convert to numpy arrays
    p_values = p.to_numpy(dtype=np.float64)
    q_values = q.to_numpy(dtype=np.float64)

    # Calculate midpoint distribution
    m = 0.5 * (p_values + q_values)

    # Replace zeros to avoid log(0)
    p_values = np.maximum(p_values, 1e-10)
    q_values = np.maximum(q_values, 1e-10)
    m = np.maximum(m, 1e-10)

    # Calculate KL divergences
    kl_p_m = np.sum(p_values * np.log2(p_values / m))
    kl_q_m = np.sum(q_values * np.log2(q_values / m))

    # Calculate JS divergence and distance
    js_divergence = 0.5 * (kl_p_m + kl_q_m)
    js_distance = np.sqrt(js_divergence)

    return js_distance


if __name__ == "__main__":
    # Example usage - this would be run from a notebook or script
    # The module provides the functionality for subsampling
    print("This module provides functions for subsampling the Open Food Facts dataset.")
    print("Import and use the functions in your code or notebooks.")
