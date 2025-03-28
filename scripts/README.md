# OpenFoodFacts Data Cleaning

This directory contains scripts to clean the OpenFoodFacts dataset for clustering tasks.

## Overview

The scripts handle several data cleaning tasks:

1. Removing irrelevant columns (metadata, high missing values)
2. Identifying and imputing missing values
3. Extracting patterns from text fields (serving_size, quantity)
4. Cleaning and fixing data errors

## Files

- `data_cleaning.py`: Base data cleaning functions
- `imputation.py`: Strategies for handling missing values
- `extract_patterns.py`: Functions to extract structured information from text fields
- `clean_openfoodfacts.py`: Main script integrating all cleaning steps

## Usage

You can run the main script directly:

```bash
python scripts/clean_openfoodfacts.py --nrows 1000 --output data/openfoodfacts_clean.csv
```

### Command Line Arguments

- `--input`, `-i`: Path or URL to the dataset (default: OpenFoodFacts URL)
- `--output`, `-o`: Path to save the cleaned dataset (default: data/openfoodfacts_clean.csv)
- `--nrows`, `-n`: Number of rows to load (default: None = all rows)
- `--sample`: Use a random sample of the dataset
- `--threshold-irrelevant`: Missing value threshold to consider a column irrelevant (default: 0.95)
- `--threshold-imputable`: Maximum missing value percentage to impute (default: 0.5)

## Features

### Irrelevant Columns

The scripts identify and remove columns that:
- Have too many missing values (above threshold)
- Contain metadata not useful for clustering (code, URL, creation date, etc.)

### Missing Value Imputation

Two imputation strategies are used depending on the data type:
- Numeric columns: Median imputation
- Categorical columns: Most frequent value imputation

### Pattern Extraction

The scripts extract structured information from text fields:
- `serving_size`: Extract quantity and unit (e.g., "100g" → 100, "g")
- `quantity`: Extract value and unit
- `brands`: Extract individual brands from comma-separated lists

### Data Error Fixing

Several error-fixing methods are implemented:
- Converting string numbers to numeric types
- Identifying potential typos in categorical columns

## Python Usage

You can also import the functions and use them in your own code:

```python
from scripts.clean_openfoodfacts import clean_openfoodfacts_dataset

# Clean dataset with custom parameters
df_clean = clean_openfoodfacts_dataset(
    input_path="path/to/dataset.csv",
    output_path="path/to/output.csv",
    nrows=1000,
    threshold_irrelevant=0.95,
    threshold_imputable=0.5
)
