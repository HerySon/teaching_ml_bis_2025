#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module provides functions to subsample the Open Food Facts dataset
to create a representative subset while maintaining the distribution of key variables.
"""

import logging
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from sklearn.model_selection import StratifiedKFold
except ImportError as e:
    logging.error("Required dependency not found: %s", e)
    raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_dataset(filepath: str | Path) -> pd.DataFrame:
    """
    Load the Open Food Facts dataset from a file.

    Args:
        filepath: Path to the dataset file (CSV format expected)

    Returns:
        DataFrame containing the dataset
    """
    logger.info("Loading dataset from %s", filepath)

    try:
        df = pd.read_csv(filepath)
        logger.info("Dataset loaded successfully. Shape: %s", df.shape)
        return df
    except Exception as e:
        logger.error("Error loading dataset: %s", e)
        raise


def analyze_categorical_columns(
    df: pd.DataFrame,
    max_categories: int = 30,
    figsize: tuple[int, int] = (15, 10),
) -> dict[str, int]:
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
    stratification_candidates = {
        k: v for k, v in sorted_cardinality.items()
        if 2 <= v <= max_categories
    }

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
        for plot_bar in bars:
            height = plot_bar.get_height()
            plt.text(
                plot_bar.get_x() + plot_bar.get_width() / 2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom',
            )

        plt.savefig('results/categorical_columns_analysis.png')
        plt.close()

    logger.info(
        "Found %s categorical columns suitable for stratification",
        len(stratification_candidates),
    )

    return stratification_candidates


def visualize_categorical_distribution(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """
    Visualize the distribution of a categorical column.

    Args:
        df: Input dataframe
        column: Column name to visualize
        top_n: Number of top categories to show
        figsize: Size of the figure
    """
    if column not in df.columns:
        logger.warning("Column %s not found in dataframe", column)
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
    for i, (_, val) in enumerate(plot_data.items()):
        percentage = 100 * val / value_counts.sum()
        ax.annotate(
            f"{val:,}\n({percentage:.1f}%)",
            (i, val),
            ha='center', va='bottom',
        )

    plt.savefig(f'results/{column}_distribution.png')
    plt.close()


def subsample_dataset(
    df: pd.DataFrame,
    stratify_by: str,
    sample_size: int | None = None,
    sample_fraction: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
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
        logger.error("Stratification column '%s' not found in dataset", stratify_by)
        raise ValueError(f"Column '{stratify_by}' not found in dataset")

    # Handle missing values in stratification column
    missing_count = df[stratify_by].isna().sum()
    if missing_count > 0:
        logger.warning(
            "Found %s missing values in '%s'. Filling with 'Unknown'",
            missing_count, stratify_by,
        )
        df[stratify_by] = df[stratify_by].fillna('Unknown')

    # Determine sample size
    if sample_size is None:
        sample_size = int(len(df) * sample_fraction)
        logger.info(
            "Using sample_fraction: %s to get sample_size: %s",
            sample_fraction, sample_size,
        )

    logger.info(
        "Creating stratified subsample of size %s stratified by '%s'",
        sample_size, stratify_by,
    )

    # Check if we have very rare categories
    value_counts = df[stratify_by].value_counts()
    rare_categories = value_counts[value_counts < 5].index.tolist()

    if rare_categories:
        logger.warning(
            "Found %s categories with fewer than 5 samples. These will be grouped as 'Other'",
            len(rare_categories),
        )
        # Create a temporary stratification column
        strat_col = df[stratify_by].copy()
        strat_col = strat_col.apply(lambda x: 'Other' if x in rare_categories else x)
    else:
        strat_col = df[stratify_by]

    # Use stratified sampling to maintain distribution
    subsample = df.groupby(strat_col, group_keys=False).apply(
        lambda x: x.sample(
            n=max(1, int(sample_size * len(x) / len(df))),
            random_state=random_state,
        ),
    )

    # If we got more samples than requested (due to rounding), trim randomly
    if len(subsample) > sample_size:
        subsample = subsample.sample(sample_size, random_state=random_state)

    logger.info("Created subsample with %s rows", len(subsample))
    return subsample


def balanced_subsample_multiple_columns(
    df: pd.DataFrame,
    columns: list[str],
    sample_size: int | None = None,
    sample_fraction: float = 0.1,
    random_state: int = 42,
) -> pd.DataFrame:
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
            logger.error("Column '%s' not found in dataset", col)
            raise ValueError(f"Column '{col}' not found in dataset")

    # Handle missing values in specified columns
    for col in columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            logger.warning(
                "Filling %s missing values in '%s' with 'Unknown'",
                missing_count, col,
            )
            df[col] = df[col].fillna('Unknown')

    # Create a composite key for stratification
    df['_strat_key'] = df[columns].astype(str).agg('_'.join, axis=1)

    # Determine sample size
    if sample_size is None:
        sample_size = int(len(df) * sample_fraction)

    logger.info(
        "Creating balanced subsample of size %s considering columns: %s",
        sample_size, ', '.join(columns),
    )

    # Use StratifiedKFold for balanced sampling
    skf = StratifiedKFold(
        n_splits=int(1 / sample_fraction),
        random_state=random_state,
        shuffle=True,
    )

    # Get indices of the first fold
    train_idx, _ = next(skf.split(df, df['_strat_key']))

    # Create subsample
    subsample = df.iloc[train_idx].copy()

    # If we have more samples than requested, trim randomly
    if len(subsample) > sample_size:
        subsample = subsample.sample(sample_size, random_state=random_state)
    elif len(subsample) < sample_size:
        logger.warning(
            "Requested sample_size %s is larger than what StratifiedKFold provided (%s). "
            "Using all available samples.",
            sample_size, len(subsample),
        )

    # Remove temporary stratification key
    subsample = subsample.drop('_strat_key', axis=1)

    logger.info("Created subsample with %s rows", len(subsample))
    return subsample


def evaluate_subsample_quality(
    original_df: pd.DataFrame,
    subsample_df: pd.DataFrame,
    columns_to_check: list[str],
    figsize: tuple[int, int] = (15, 10),
) -> None:
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
            logger.warning("Column %s not found in one of the dataframes", col)
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
        plot_df = pd.DataFrame(
            {
                'Original': orig_counts_top * 100,
                'Subsample': sub_counts_top * 100,
            },
        )

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
        logger.info(
            "Jensen-Shannon distance for %s: %.4f (lower is better, 0 is identical)",
            col, jsd,
        )


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
