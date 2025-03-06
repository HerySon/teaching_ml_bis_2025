"""Module providing a function to analyse the columns of a DataFrame and classify them into three categories"""

import pandas as pd


def analyse_columns(df: pd.DataFrame, category_threshold: int = 10) \
        -> [dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
        "numeric": [],
        "ordinal_categorical": [],
        "non_ordinal_categorical": []
    }

    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            filtered_columns["numeric"].append(col)
        elif df[col].dtype == "object" or df[col].dtype.name == "category":
            nb_categories = df[col].nunique()
            if nb_categories <= category_threshold:
                filtered_columns["ordinal_categorical"].append(col)
            else:
                filtered_columns["non_ordinal_categorical"].append(col)

        # Downcasting des variables numériques
        df.loc[:, filtered_columns["numeric"]] = df[filtered_columns["numeric"]].apply(
            pd.to_numeric, downcast="float")

    return (filtered_columns, df[filtered_columns["numeric"]], df[filtered_columns["ordinal_categorical"]],
            df[filtered_columns["non_ordinal_categorical"]])
