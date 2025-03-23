# OpenFoodFacts DataFrame Filtering

This module provides utilities for filtering and selecting relevant columns from the OpenFoodFacts dataset.

## Features

- Automatically detect column types (numeric, categorical, ordinal, etc.)
- Filter columns based on missing value percentage
- Filter categorical columns based on cardinality (number of unique values)
- Downcast DataFrame columns to reduce memory usage
- Get detailed information about columns

## Usage

### Basic Usage

```python
import pandas as pd
from scripts.dataframe_filtering import filter_dataframe

# Load OpenFoodFacts data
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=1000, sep='\t', encoding="utf-8", low_memory=False)

# Filter the DataFrame
filtered_df = filter_dataframe(
    df,
    missing_threshold=0.5,  # Keep columns with less than 50% missing values
    max_categories=50,      # For categorical columns, keep those with at most 50 categories
    min_categories=2,       # For categorical columns, keep those with at least 2 categories
    downcast=True           # Reduce memory usage by downcasting numeric types
)
```

## Functions

### `identify_column_types(df)`

Automatically identifies the types of columns in the DataFrame:
- Numeric columns
- Categorical ordinal columns
- Categorical non-ordinal columns
- Date columns
- Text columns

### `filter_by_missing_values(df, threshold=0.5)`

Filters columns based on the percentage of missing values.

### `filter_categorical_by_cardinality(df, columns, max_categories=50, min_categories=2)`

Filters categorical columns based on their number of unique values.

### `downcast_dataframe(df)`

Reduces memory usage by downcasting numeric columns and converting object columns with low cardinality to categorical.

### `filter_dataframe(df, ...)`

Main function that combines all filtering operations.

### `get_column_info(df)`

Provides detailed information about each column in the DataFrame.
