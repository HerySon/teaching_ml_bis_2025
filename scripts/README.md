# Data Scaling for OpenFoodFacts Dataset

This module provides functionality for scaling numerical features in the OpenFoodFacts dataset. Multiple scaling methods are implemented and can be compared to determine the most appropriate technique for each variable.

## Features

- **Data Loading**: Load data from OpenFoodFacts CSV file
- **Data Preprocessing**: Clean and prepare data for scaling
- **Multiple Scaling Methods**:
  - Standard Scaling (Z-score normalization)
  - Min-Max Scaling (0-1 normalization)
  - Robust Scaling (using quantiles)
  - Power Transformation (Yeo-Johnson)
  - Quantile Transformation (uniform or normal distribution)
- **Automatic Recommendations**: Automatically determine the best scaling method for each column based on its distribution and presence of outliers
- **Visualization**: Generate plots to visualize data distributions before and after scaling

## Files

- `data_scaling.py`: Core scaling functionality
- `test_scaling.py`: Testing and evaluation of different scaling methods
- `main.py`: Command-line interface to run the scaling operations

## Usage

### Running from Command Line

```bash
python main.py [options]
```

#### Command Line Options

- `--data-url`: URL or file path of the dataset (default: OpenFoodFacts URL)
- `--nrows`: Number of rows to load from the dataset (default: 1000)
- `--output-dir`: Directory to save results (default: 'results')
- `--operation`: Operation to perform (choices: analyze, compare, recommend, scale, all)
- `--method`: Scaling method to use (choices: standard, minmax, robust, power, quantile, recommend)

### Examples

#### Analyze Data Distributions

```bash
python main.py --operation analyze
```

#### Compare Different Scaling Methods

```bash
python main.py --operation compare
```

#### Get Scaling Method Recommendations

```bash
python main.py --operation recommend
```

#### Scale Data with Specific Method

```bash
python main.py --operation scale --method robust
```

#### Run All Operations

```bash
python main.py --operation all
```

### Using the API in Your Code

```python
from data_scaling import scale_dataframe
import pandas as pd

# Load your data
df = pd.read_csv('your_data.csv')

# Scale using a specific method
scaled_df = scale_dataframe(df, method='standard')

# Or use a combination of methods for different columns
from data_scaling import standard_scaler, robust_scaler

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
cols_with_outliers = ['col1', 'col2']
normal_cols = [col for col in numeric_cols if col not in cols_with_outliers]

# Apply robust scaling to columns with outliers
df = robust_scaler(df, cols_with_outliers)

# Apply standard scaling to remaining numeric columns
df = standard_scaler(df, normal_cols)
```

## How Scaling Methods are Recommended

The script analyzes each numeric column and recommends a scaling method based on:

1. **Presence of Outliers**: Checks if the column has significant outliers (> 3 standard deviations from the mean)
2. **Skewness**: Measures how symmetrical the distribution is

The recommendations follow these rules:

- **Standard Scaling**: For normally distributed data without outliers
- **MinMax Scaling**: When range is important and there are no outliers
- **Robust Scaling**: For data with outliers but not heavily skewed
- **Power Transformation**: For skewed data with outliers
- **Quantile Transformation**: For heavily skewed data without significant outliers

## Output Files

The script generates the following output files in the specified output directory:

- `preprocessed_data.csv`: Data after cleaning and preprocessing
- `scaling_recommendations.csv`: Recommended scaling method for each column
- `scaled_data_*.csv`: Data after applying scaling
- Distribution plots in the `distributions` subdirectory
- Comparison plots in the `scaling_comparison` subdirectory
