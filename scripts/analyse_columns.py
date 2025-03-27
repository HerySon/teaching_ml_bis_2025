"""Module providing a function to analyse and classify the columns of a DataFrame"""

import pandas as pd
import numpy as np


def analyse_columns(df: pd.DataFrame, ordinal_categories: []) -> dict:
    """
    Analyse a DataFrame and classify its columns into three categories:
    - Numeric
    - Ordinal categorical
    - Non-ordinal categorical
    Also apply a downcast to numeric columns.

    Parameters :
    df : pd.DataFrame -> Entry DataFrame
    category_threshold : int -> Maximum number of categories for a column to be considered ordinal

    Return :
    dict containing the three categories of columns
    """

    # Dictionnaire to store the columns
    filtered_columns = {
        "numerical": {},
        "ordinal_categorical": {},
        "non_ordinal_categorical": {}
    }

    for numerical_column in df.select_dtypes(include=["int64", "float64"]).columns:
        filtered_columns["numerical"][numerical_column] = df[numerical_column].nunique()
        col_type = df[numerical_column].dtype

        if np.issubdtype(col_type, np.integer):
            for dtype in ["int8", "int16", "int32"]:
                if df[numerical_column].min() >= np.iinfo(dtype).min:
                    if df[numerical_column].max() <= np.iinfo(dtype).max:
                        df.loc[:, numerical_column] = df[numerical_column].astype(dtype)

        elif np.issubdtype(col_type, np.floating):
            if np.allclose(df[numerical_column], df[numerical_column].astype(np.float32)):
                df.loc[:, numerical_column] = df[numerical_column].astype(np.float32)

    for columns in df.select_dtypes(include=["object"]).columns:
        if columns in ordinal_categories:
            filtered_columns["ordinal_categorical"][columns] = df[columns].nunique()
        else:
            filtered_columns["non_ordinal_categorical"][columns] = df[columns].nunique()

    return filtered_columns
