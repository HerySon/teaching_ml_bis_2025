# Encodage des features catégorielles

# Concernant l’encodage des features catégorielles, si vous êtes limité par une approche classique utilisant le OneHotEncoder voici des solutions possibles:

# Optimiser votre utilisation de OHE:

# en encodant les features candidates de manière incrémentale : encoder et sauvegarder (sur disque dur) les features une par une

# utiliser les sparses matrix pour stocker vos features ré-encodées (même si vous devez les stocker dans un fichier à part)

# Tenter de gérer vos catégories au préalable

# supprimer les catégories qui sont très rares (par exemple en fixant un seuil pour supprimer les 1% les plus rares) mais au risque de perdre des informations

# fusionner les catégories au sein des features : par la connaissance métier (par exemple en groupant des catégories similaires) par des méthodes de clustering (K-Means, Word embedding)

# Utiliser des méthodes alternatives (voir le module Category Encoding)

# le HashingEncoder

# utiliser une méthode simple basée sur la fréquence d'apparition de la catégorie CountEncoder

import pandas as pd
from scipy import sparse
from category_encoders import HashingEncoder
from category_encoders import CountEncoder

class DataEncoder(object):
    """
    A class used to encode data using various encoding techniques.
    Attributes
    ----------
    df : pandas.DataFrame
        The dataframe containing the dataset.
    limit : int
        The number of rows to read from the dataset.
    Methods
    -------
    one_hot_encode(column_name)
        Applies one-hot encoding to the specified column.
    incremental_one_hot_encode(column_name, save_path)
        Applies one-hot encoding to the specified column and saves the result to a CSV file.
    sparse_one_hot_encode(column_name)
        Applies one-hot encoding to the specified column and returns a sparse matrix.
    remove_rare_categories(column_name, threshold=0.01)
        Replaces rare categories in the specified column with 'Other' based on a threshold.
    merge_categories(column_name, mapping)
        Merges categories in the specified column according to a given mapping.
    hash_encode(column_name, n_features=10)
        Applies hash encoding to the specified column with a given number of features.
    count_encode(column_name)
        Applies count encoding to the specified column.
    """

    def __init__(self, url_dataset, limit=100, sep='\t') -> None:
        """
        Initializes the DataEncoder with a dataset URL, row limit, and separator.
        Parameters
        ----------
        url_dataset : str
            The URL or file path of the dataset.
        limit : int, optional
            The number of rows to read from the dataset (default is 100).
        sep : str, optional
            The separator used in the dataset (default is '\t').
        """
        self.df = pd.read_csv(url_dataset, nrows=limit, sep=sep, encoding="utf-8")
        self.limit = limit

    def one_hot_encode(self, column_name):
        """
        Applies one-hot encoding to the specified column.
        Parameters
        ----------
        column_name : str
            The name of the column to encode.
        Returns
        -------
        pandas.DataFrame
            The dataframe with one-hot encoded column.
        """
        encoded_df = pd.get_dummies(self.df[column_name], prefix=column_name)
        print(f"One-hot encoding applied to {column_name}.")
        return encoded_df

    def incremental_one_hot_encode(self, column_name, save_path):
        """
        Applies one-hot encoding to the specified column and saves the result to a CSV file.
        Parameters
        ----------
        column_name : str
            The name of the column to encode.
        save_path : str
            The file path to save the encoded dataframe.
        Returns
        -------
        pandas.DataFrame
            The dataframe with one-hot encoded column.
        """
        encoded_df = pd.get_dummies(self.df[column_name], prefix=column_name)
        encoded_df.to_csv(save_path, index=False)
        print(f"Incremental one-hot encoding applied to {column_name} and saved to {save_path}.")
        return encoded_df

    def sparse_one_hot_encode(self, column_name):
        """
        Applies one-hot encoding to the specified column and returns a sparse matrix.
        Parameters
        ----------
        column_name : str
            The name of the column to encode.
        Returns
        -------
        scipy.sparse.csr_matrix
            The sparse matrix with one-hot encoded column.
        """
        encoded_df = pd.get_dummies(self.df[column_name], prefix=column_name)
        print(f"Sparse one-hot encoding applied to {column_name}.")
        return sparse.csr_matrix(encoded_df.values)

    def remove_rare_categories(self, column_name, threshold=0.01):
        """
        Replaces rare categories in the specified column with 'Other' based on a threshold.
        Parameters
        ----------
        column_name : str
            The name of the column to process.
        threshold : float, optional
            The threshold below which categories are considered rare (default is 0.01).
        Returns
        -------
        pandas.DataFrame
            The dataframe with rare categories replaced.
        """
        value_counts = self.df[column_name].value_counts(normalize=True)
        to_remove = value_counts[value_counts < threshold].index
        self.df[column_name] = self.df[column_name].apply(lambda x: 'Other' if x in to_remove else x)
        print(f"Rare categories in {column_name} removed with threshold {threshold}.")
        return self.df

    def merge_categories(self, column_name, mapping):
        """
        Merges categories in the specified column according to a given mapping.
        Parameters
        ----------
        column_name : str
            The name of the column to process.
        mapping : dict
            A dictionary defining the mapping of old categories to new categories.
        Returns
        -------
        pandas.DataFrame
            The dataframe with merged categories.
        """
        self.df[column_name] = self.df[column_name].map(mapping).fillna(self.df[column_name])
        print(f"Categories in {column_name} merged according to mapping.")
        return self.df

    def hash_encode(self, column_name, n_features=10):
        """
        Applies hash encoding to the specified column with a given number of features.
        Parameters
        ----------
        column_name : str
            The name of the column to encode.
        n_features : int, optional
            The number of features for the hash encoding (default is 10).
        Returns
        -------
        pandas.DataFrame
            The dataframe with hash encoded column.
        """
        encoder = HashingEncoder(cols=[column_name], n_components=n_features)
        encoded_df = encoder.fit_transform(self.df)
        print(f"Hash encoding applied to {column_name} with {n_features} features.")
        return encoded_df

    def count_encode(self, column_name):
        """
        Applies count encoding to the specified column.
        Parameters
        ----------
        column_name : str
            The name of the column to encode.
        Returns
        -------
        pandas.DataFrame
            The dataframe with count encoded column.
        """
        encoder = CountEncoder(cols=[column_name])
        encoded_df = encoder.fit_transform(self.df)
        print(f"Count encoding applied to {column_name}.")
        return encoded_df


# Example usage
data_encoding_instance = DataEncoder("C:/Users/valen/Desktop/Machine Learning/teaching_ml_bis_2025/data/en.openfoodfacts.org.products.csv")

data_encoding_instance.one_hot_encode("product_name")

data_encoding_instance.incremental_one_hot_encode("product_name", "product_name_encoded.csv")

data_encoding_instance.sparse_one_hot_encode("product_name")

data_encoding_instance.remove_rare_categories("product_name")

mapping = {"cat1": "group1", "cat2": "group2", "cat3": "group3"}
data_encoding_instance.merge_categories("product_name", mapping)

data_encoding_instance.hash_encode("product_name", 5)

data_encoding_instance.count_encode("product_name")