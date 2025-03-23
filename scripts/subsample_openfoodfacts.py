#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a representative subsample of the Open Food Facts dataset.
This script implements task 0.1 of the clustering project.
"""

import argparse
import logging
from pathlib import Path

from scripts.subsample import (
    analyze_categorical_columns,
    balanced_subsample_multiple_columns,
    evaluate_subsample_quality,
    load_dataset,
    subsample_dataset,
    visualize_categorical_distribution
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_subsample(
    input_file: str,
    output_file: str,
    method: str = "single",
    stratify_column: str | None = None,
    stratify_columns: list[str] | None = None,
    sample_size: int | None = None,
    sample_fraction: float = 0.1,
    random_state: int = 42
) -> None:
    """
    Create a representative subsample of the Open Food Facts dataset.

    Args:
        input_file: Path to the input CSV file
        output_file: Path to save the resulting subsample
        method: Subsampling method ('single' for single column, 'multiple' for multiple columns)
        stratify_column: Column to stratify by when using 'single' method
        stratify_columns: List of columns to stratify by when using 'multiple' method
        sample_size: Number of samples to select
        sample_fraction: Fraction of dataset to sample (used if sample_size is None)
        random_state: Random seed for reproducibility
    """
    # Create results directory if it doesn't exist
    Path('results').mkdir(exist_ok=True)

    # Load dataset
    df = load_dataset(input_file)
    logger.info(f"Original dataset has {len(df)} rows and {len(df.columns)} columns")

    # Analyze categorical columns to identify good candidates for stratification
    strat_candidates = analyze_categorical_columns(df)

    # If stratification column(s) not provided, suggest based on analysis
    if method == "single" and stratify_column is None:
        # Find columns with moderate cardinality (not too many or too few categories)
        moderate_candidates = {k: v for k, v in strat_candidates.items() if 5 <= v <= 20}
        if moderate_candidates:
            stratify_column = list(moderate_candidates.keys())[0]
            logger.info(f"No stratification column specified. Using '{stratify_column}' based on analysis.")
        else:
            # If no suitable column found, use the one with most reasonable number of categories
            stratify_column = list(strat_candidates.keys())[0]
            logger.info(f"No stratification column specified. Using '{stratify_column}' as fallback.")

    elif method == "multiple" and stratify_columns is None:
        # Select top 2-3 columns with reasonable cardinality
        stratify_columns = list(strat_candidates.keys())[:min(3, len(strat_candidates))]
        logger.info(f"No stratification columns specified. Using {stratify_columns} based on analysis.")

    # Visualize distribution of stratification column(s)
    if method == "single" and stratify_column is not None:
        visualize_categorical_distribution(df, stratify_column)
    elif stratify_columns is not None:
        for col in stratify_columns:
            visualize_categorical_distribution(df, col)

    # Create subsample based on chosen method
    if method == "single":
        logger.info(f"Creating stratified subsample using column: {stratify_column}")
        subsample = subsample_dataset(
            df=df,
            stratify_by=stratify_column,
            sample_size=sample_size,
            sample_fraction=sample_fraction,
            random_state=random_state
        )
    else:
        logger.info(f"Creating stratified subsample using columns: {stratify_columns}")
        subsample = balanced_subsample_multiple_columns(
            df=df,
            columns=stratify_columns,
            sample_size=sample_size,
            sample_fraction=sample_fraction,
            random_state=random_state
        )

    # Save subsample
    subsample.to_csv(output_file, index=False)
    logger.info(f"Saved subsample with {len(subsample)} rows to {output_file}")

    # Evaluate subsample quality
    columns_to_check = stratify_columns if method == "multiple" else [stratify_column]

    # Add a few more columns to check for representativeness
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns[:3].tolist()
    categorical_cols = list(strat_candidates.keys())

    # Ensure we're not checking too many columns
    all_check_cols = list(set(columns_to_check + categorical_cols[:5] + numerical_cols))

    evaluate_subsample_quality(df, subsample, all_check_cols)

    logger.info("Subsampling completed successfully")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Create a representative subsample of the Open Food Facts dataset")

    parser.add_argument("--input", "-i", required=True,
                       help="Path to the input CSV file containing the Open Food Facts dataset")

    parser.add_argument("--output", "-o", required=True,
                       help="Path to save the resulting subsample CSV file")

    parser.add_argument("--method", "-m", choices=["single", "multiple"], default="multiple",
                       help="Stratification method: single column or multiple columns")

    parser.add_argument("--stratify-column", "-s",
                       help="Column to stratify by when using 'single' method")

    parser.add_argument("--stratify-columns", "-c", nargs="+",
                       help="List of columns to stratify by when using 'multiple' method")

    parser.add_argument("--sample-size", "-n", type=int,
                       help="Number of samples to select (overrides sample-fraction)")

    parser.add_argument("--sample-fraction", "-f", type=float, default=0.1,
                       help="Fraction of dataset to sample (default: 0.1)")

    parser.add_argument("--random-state", "-r", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Convert stratify-columns from list of strings to actual list if provided
    stratify_cols = args.stratify_columns

    # Create subsample
    create_subsample(
        input_file=args.input,
        output_file=args.output,
        method=args.method,
        stratify_column=args.stratify_column,
        stratify_columns=stratify_cols,
        sample_size=args.sample_size,
        sample_fraction=args.sample_fraction,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()
