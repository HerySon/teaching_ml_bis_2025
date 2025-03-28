# Open Food Facts Dataset Subsampling

This directory contains scripts for subsampling the Open Food Facts dataset to create a representative subset while maintaining the distribution of key variables.

## Scripts

- `subsample.py`: Core module with functions for dataset subsampling and analysis
- `subsample_openfoodfacts.py`: Command-line script for creating representative subsamples

## Usage

### Command Line

You can use the `subsample_openfoodfacts.py` script from the command line to create a subsample:

```bash
# Basic usage with default parameters (10% sample, stratified by multiple columns)
python -m scripts.subsample_openfoodfacts --input data/openfoodfacts.csv --output data/openfoodfacts_subsample.csv

# Specify stratification method (single column)
python -m scripts.subsample_openfoodfacts --input data/openfoodfacts.csv --output data/openfoodfacts_subsample.csv --method single --stratify-column categories

# Use multiple specific columns for stratification
python -m scripts.subsample_openfoodfacts --input data/openfoodfacts.csv --output data/openfoodfacts_subsample.csv --method multiple --stratify-columns brands categories countries

# Specify sample size (instead of sample fraction)
python -m scripts.subsample_openfoodfacts --input data/openfoodfacts.csv --output data/openfoodfacts_subsample.csv --sample-size 10000
```

### API Usage

You can also use the functions from the `subsample.py` module in your Python code:

```python
from scripts.subsample import load_dataset, analyze_categorical_columns, balanced_subsample_multiple_columns

# Load the dataset
df = load_dataset('data/openfoodfacts.csv')

# Analyze categorical columns to find good candidates for stratification
strat_candidates = analyze_categorical_columns(df)

# Create a stratified subsample based on multiple columns
stratify_columns = ['brands', 'categories', 'countries']
subsample = balanced_subsample_multiple_columns(
    df=df,
    columns=stratify_columns,
    sample_size=10000,
    random_state=42
)

# Save the subsample
subsample.to_csv('data/openfoodfacts_subsample.csv', index=False)
```

## Features

- **Stratified Sampling**: Maintains the distribution of key categorical variables
- **Multiple Column Stratification**: Balances across multiple categorical variables simultaneously
- **Automatic Analysis**: Identifies good categorical variables for stratification
- **Quality Evaluation**: Compares the distribution of variables in the original dataset and the subsample
- **Visualization**: Creates plots to understand the dataset and validate the subsampling

## Output

The scripts generate the following outputs in the `results/` directory:

- Analysis of categorical columns (`categorical_columns_analysis.png`)
- Distribution plots for key categorical variables
- Subsample quality evaluation plots showing distribution comparisons
