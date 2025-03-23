"""
Main script to clean OpenFoodFacts dataset by integrating all cleaning functionalities.
"""

import argparse
import pandas as pd
from pathlib import Path

from data_cleaning import (
    load_data,
    identify_irrelevant_columns,
    remove_irrelevant_columns,
    fix_data_errors
)
from imputation import (
    identify_imputable_columns,
    impute_columns
)
from extract_patterns import (
    extract_patterns
)


def clean_openfoodfacts_dataset(
    input_path: str,
    output_path: str | None = None,
    nrows: int | None = None,
    threshold_irrelevant: float = 0.95,
    threshold_imputable: float = 0.5
) -> pd.DataFrame:
    """
    Main function to clean the OpenFoodFacts dataset by integrating all cleaning steps.

    Args:
        input_path: Path to the dataset
        output_path: Path to save the cleaned dataset (if None, won't save)
        nrows: Number of rows to load (None for all)
        threshold_irrelevant: Missing value threshold to consider a column irrelevant
        threshold_imputable: Maximum missing value percentage to consider a column imputable

    Returns:
        Cleaned DataFrame
    """
    print(f"\n{'='*50}")
    print("CLEANING OPENFOODFACTS DATASET")
    print(f"{'='*50}\n")

    # Step 1: Load data
    print("\n[Step 1] Loading dataset...")
    df = load_data(input_path, nrows)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    # Step 2: Remove irrelevant columns
    print("\n[Step 2] Removing irrelevant columns...")
    irrelevant_cols = identify_irrelevant_columns(df, threshold_irrelevant)
    df_clean = remove_irrelevant_columns(df, irrelevant_cols)

    # Step 3: Extract patterns from text fields
    print("\n[Step 3] Extracting patterns from text fields...")
    df_clean = extract_patterns(
        df_clean,
        process_serving_size=True,
        process_quantity=True,
        process_brands=True
    )

    # Step 4: Fix data errors
    print("\n[Step 4] Fixing data errors...")
    df_clean = fix_data_errors(df_clean)

    # Step 5: Identify and impute missing values
    print("\n[Step 5] Imputing missing values...")
    imputation_strategies = identify_imputable_columns(df_clean, threshold_imputable)
    df_clean = impute_columns(
        df_clean,
        imputation_strategies,
        threshold_imputable
    )

    print("\n[Summary] Data cleaning completed:")
    print(f"  - Initial dataset shape: {df.shape[0]} rows and {df.shape[1]} columns")
    print(f"  - Cleaned dataset shape: {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")
    print(f"  - Removed {df.shape[1] - df_clean.shape[1]} irrelevant columns")
    print(f"  - Added {len([col for col in df_clean.columns if col not in df.columns])}"
          f"new features")

    # Calculate percentage of missing values before and after cleaning
    missing_before = df.isnull().mean().mean() * 100
    missing_after = df_clean.isnull().mean().mean() * 100
    print(f"  - Missing values: {missing_before:.2f}% → {missing_after:.2f}%")

    # Save cleaned data if output path is provided
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"\nSaved cleaned dataset to {output_path}")

    return df_clean


def main():
    """Parse command line arguments and run the cleaning process."""
    parser = argparse.ArgumentParser(description="Clean OpenFoodFacts dataset")
    parser.add_argument(
        "--input", "-i",
        default="https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz",
        help="Path or URL to the OpenFoodFacts dataset"
    )
    parser.add_argument(
        "--output", "-o",
        default="data/openfoodfacts_clean.csv",
        help="Path to save the cleaned dataset"
    )
    parser.add_argument(
        "--nrows", "-n",
        type=int,
        default=None,
        help="Number of rows to load (None for all)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use a random sample of the dataset"
    )
    parser.add_argument(
        "--threshold-irrelevant",
        type=float,
        default=0.95,
        help="Missing value threshold to consider a column irrelevant"
    )
    parser.add_argument(
        "--threshold-imputable",
        type=float,
        default=0.5,
        help="Maximum missing value percentage to consider a column imputable"
    )
    args = parser.parse_args()

    # For testing purposes, you can use a smaller random sample
    nrows = args.sample if args.sample else args.nrows

    # Run the cleaning process
    clean_openfoodfacts_dataset(
        input_path=args.input,
        output_path=args.output,
        nrows=nrows,
        threshold_irrelevant=args.threshold_irrelevant,
        threshold_imputable=args.threshold_imputable
    )


if __name__ == "__main__":
    main()
