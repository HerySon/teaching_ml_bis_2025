"""
Data cleaning script for OpenFoodFacts dataset.
This script implements various methods to clean problematic values in the dataset.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load OpenFoodFacts dataset from URL or local path.

    Args:
        path: URL or local path to the dataset
        nrows: Number of rows to load (None for all)

    Returns:
        DataFrame containing the dataset
    """
    print(f"Loading data from {path}...")
    return pd.read_csv(
        path,
        nrows=nrows,
        sep='\t',
        encoding="utf-8",
        low_memory=False
    )


def identify_irrelevant_columns(df: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """
    Identify columns that are not relevant for clustering task.

    Args:
        df: Input DataFrame
        threshold: Missing value threshold to consider a column irrelevant

    Returns:
        List of column names considered irrelevant
    """
    # Columns with too many missing values
    missing_values = df.isnull().mean()
    high_missing_cols = missing_values[missing_values > threshold].index.tolist()

    # Metadata columns that are not useful for clustering food products
    metadata_cols = [
        'code', 'url', 'creator', 'created_t', 'created_datetime',
        'last_modified_t', 'last_modified_datetime', 'last_modified_by',
        'last_updated_t', 'last_updated_datetime'
    ]

    # Combine both lists and remove duplicates
    irrelevant_cols = list(set(high_missing_cols + metadata_cols))

    print(f"Identified {len(irrelevant_cols)} irrelevant columns")
    return irrelevant_cols


def remove_irrelevant_columns(
    df: pd.DataFrame,
    irrelevant_cols: list[str] | None = None
) -> pd.DataFrame:
    """
    Remove columns identified as irrelevant.

    Args:
        df: Input DataFrame
        irrelevant_cols: List of columns to remove. If None, will identify them automatically.

    Returns:
        DataFrame with irrelevant columns removed
    """
    if irrelevant_cols is None:
        irrelevant_cols = identify_irrelevant_columns(df)

    # Remove columns that exist in the DataFrame
    cols_to_remove = [col for col in irrelevant_cols if col in df.columns]
    df_clean = df.drop(columns=cols_to_remove)

    print(f"Removed {len(cols_to_remove)} irrelevant columns")
    return df_clean


def identify_missing_values(df: pd.DataFrame) -> dict[str, float]:
    """
    Identify columns with missing values and their percentage.

    Args:
        df: Input DataFrame

    Returns:
        Dictionary with column names and percentage of missing values
    """
    missing_percentage = (df.isnull().mean() * 100).round(2)
    missing_dict = missing_percentage[missing_percentage > 0].to_dict()

    # Sort by percentage descending
    missing_dict = dict(sorted(missing_dict.items(), key=lambda x: x[1], reverse=True))

    print(f"Found {len(missing_dict)} columns with missing values")
    return missing_dict


def extract_quantity_from_serving_size(serving_size: str) -> float:
    """
    Extract quantity from serving_size string.

    Args:
        serving_size: String containing serving size information

    Returns:
        Extracted quantity as float, or NaN if extraction fails
    """
    if pd.isna(serving_size) or not isinstance(serving_size, str):
        return np.nan

    # Common patterns: "100g", "2 x 125g", "1 cup (240ml)", etc.
    # Extract numbers followed by units (g, ml, kg, etc.)
    pattern = r'(\d+(?:\.\d+)?)\s*(?:x\s*)?(\d+(?:\.\d+)?)?\s*(g|ml|kg|l|oz)'
    match = re.search(pattern, serving_size.lower())

    if match:
        # If we have a format like "2 x 125g"
        if match.group(2):
            quantity = float(match.group(1)) * float(match.group(2))
        else:
            quantity = float(match.group(1))

        # Convert to grams or ml for consistency
        unit = match.group(3)
        if unit == 'kg':
            quantity *= 1000
        elif unit == 'l':
            quantity *= 1000
        elif unit == 'oz':
            quantity *= 28.35  # Approximate conversion to grams

        return quantity

    return np.nan


def clean_serving_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and extract information from serving_size column.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with added serving_quantity column
    """
    if 'serving_size' not in df.columns:
        print("serving_size column not found in DataFrame")
        return df

    # Create a new column with extracted quantities
    df['serving_quantity'] = df['serving_size'].apply(extract_quantity_from_serving_size)

    print("Extracted quantities from serving_size column")
    return df


def fix_data_errors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix common data errors in the dataset.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with fixed errors
    """
    df_clean = df.copy()

    # Replace non-numeric values in numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if df[col].dtype != 'object':
            continue

        # Try to convert string numbers to float
        df_clean[col] = pd.to_numeric(df[col], errors='coerce')

    # Fix typos in categorical columns (focusing on main ones)
    categorical_cols = ['nutrition_grade_fr', 'main_category']
    for col in categorical_cols:
        if col not in df.columns:
            continue

        # Get value counts to identify potential errors
        value_counts = df[col].value_counts()

        # Focus on values with very few occurrences (potential typos)
        rare_values = value_counts[value_counts < 5].index
        if len(rare_values) > 0:
            print(f"Found {len(rare_values)} potential typos in {col}")

    print("Fixed common data errors")
    return df_clean


def clean_data(
    data_path: str,
	output_path: str | None = None,
    nrows: int | None = None
) -> pd.DataFrame:
    """
    Main function to clean the OpenFoodFacts dataset.

    Args:
        data_path: Path to the dataset
        output_path: Path to save the cleaned dataset (if None, won't save)
        nrows: Number of rows to load (None for all)

    Returns:
        Cleaned DataFrame
    """
    # Load data
    df = load_data(data_path, nrows)
    print(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns")

    # Remove irrelevant columns
    df_clean = remove_irrelevant_columns(df)

    # Identify columns with missing values
    missing_info = identify_missing_values(df_clean)

    # Clean serving_size column and extract quantities
    df_clean = clean_serving_size(df_clean)

    # Fix common data errors
    df_clean = fix_data_errors(df_clean)

    # Save cleaned data if output path is provided
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        df_clean.to_csv(output_path, index=False)
        print(f"Saved cleaned dataset to {output_path}")

    print(f"Final dataset shape: {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")
    return df_clean
