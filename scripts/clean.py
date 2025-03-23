"""Module providing a function clean a DataFrame from its missing and uninteresting values"""

import pandas as pd


def clean(df: pd.DataFrame, irrelevant_columns: [str] = [], missing_threshold: float = 0.5) \
        -> pd.DataFrame:
    """
    Clean a DataFrame from its missing and uninteresting values.
    Drop the columns with a proportion of missing values above the threshold.
    Drop the columns containing only one unique value.
    Drop the columns listed as irrelevant.

    Parameters :
    df : pd.DataFrame -> Entry DataFrame
    irrelevant_columns : [str] -> List of columns to drop
    missing_threshold : float -> Proportion of missing values above which a column is dropped

    Return :
    pd.DataFrame -> Cleaned DataFrame
    """

    # Drop columns with a proportion of missing values above the threshold
    missing_values = df.isnull().mean()
    df = df.drop(columns=missing_values[missing_values > missing_threshold].index)

    # Drop columns containing only one unique value
    unique_values = df.nunique()
    df = df.drop(columns=unique_values[unique_values == 1].index)

    # Drop irrelevant columns
    df = df.drop(columns=irrelevant_columns)

    return df
