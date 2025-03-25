"""Module providing a function to subsample and the columns of a DataFrame"""
from sklearn.model_selection import StratifiedKFold
import pandas as pd


def subsample(df: pd.DataFrame, target: str) -> list:
    """
    Subsample a DataFrame into a certain number of folds using Stratified KFold.:
    :param df: pd.DataFrame -> Entry DataFrame
    :param n_sample: int -> Number of folds
    :param target: str -> Target column
    :return: List of DataFrames containing the folds
    """

    # Split the data into features and target
    x = df.drop(columns=[target])  # Features
    y = df[target]

    # Get the minimum class count to determine the number of splits
    n_sample = min(2, y.value_counts().min())

    # Use stratified KFold to subsample the data
    skf = StratifiedKFold(n_splits=n_sample, shuffle=True, random_state=42)

    # List to store the folds
    folds = []

    # Create the folds and store them
    for train_index, test_index in skf.split(x, y):
        folds.append({
            'X_train': x.iloc[train_index],
            'X_test': x.iloc[test_index],
            'y_train': y.iloc[train_index],
            'y_test': y.iloc[test_index]
        })
    return folds
