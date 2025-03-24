# Vous proposerez des méthodes pour repérer et nettoyer le data set des données problématiques, en traitant, par exemple:

# les variables que vous jugez non pertinentes pour la tâche

# les variables ayant trop de valeurs manquantes et celle pouvant être imputées

# les variables pour lesquelles on a besoin d’extraire des motifs particuliers (comme serving_size)

# les variables présentant des erreurs

# …

# Remarque : il n’est pas nécessaire de réaliser l’encodage ou la détection des outliers car elle seront traitées dans d’autres tâches

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator
import re


class DataClean(object):
    def __init__(self, url_dataset, limit=100, sep='\t') -> None:
        self.df = pd.read_csv(url_dataset, nrows=limit, sep=sep, encoding="utf-8")
        self.limit = limit

    def __call_of_method__(self, method_name: str, *args, **kwargs) -> any:
        """
        Calls a method by its name with the provided arguments.
        
        Parameters:
        method_name (str): The name of the method to call.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        
        Returns:
        any: The return value of the called method.
        
        Raises:
        AttributeError: If the method does not exist.
        """
        if not hasattr(self, method_name):
            raise AttributeError(f"{self.__class__.__name__} does not have method {method_name}")
        
        method = getattr(self, method_name)
        return method(*args, **kwargs)
    
    def _imputer(self, df, imputer_class, **kwargs) -> pd.DataFrame:
        """
        Imputes missing values in the dataframe using the specified imputer class.
        
        Parameters:
        df (pd.DataFrame): The dataframe to impute.
        imputer_class: The imputer class to use (e.g., KNNImputer, IterativeImputer).
        **kwargs: Additional keyword arguments for the imputer class.
        
        Returns:
        pd.DataFrame: The dataframe with imputed values.
        """
        numeric_cols = df.select_dtypes(include=['number']).columns
        imputer = imputer_class(**kwargs)

        imputed_data = imputer.fit_transform(df[numeric_cols])

        imputed_df = pd.DataFrame(imputed_data, columns=numeric_cols[:imputed_data.shape[1]], index=df.index)
        df.loc[:, numeric_cols] = imputed_df

        return df
    
    def _moyenne(self, df) -> pd.DataFrame:
        """
        Imputes missing values with the mean of each column.
        
        Parameters:
        df (pd.DataFrame): The dataframe to impute.
        
        Returns:
        pd.DataFrame: The dataframe with imputed values.
        """
        return df.fillna(df.mean(numeric_only=True))
    
    def _median(self, df) -> pd.DataFrame:
        """
        Imputes missing values with the median of each column.
        
        Parameters:
        df (pd.DataFrame): The dataframe to impute.
        
        Returns:
        pd.DataFrame: The dataframe with imputed values.
        """
        return df.fillna(df.median(numeric_only=True))
    
    def _delete(self, df) -> pd.DataFrame:
        """
        Deletes rows with any missing values.
        
        Parameters:
        df (pd.DataFrame): The dataframe to clean.
        
        Returns:
        pd.DataFrame: The dataframe with rows containing missing values removed.
        """
        return df.dropna()
    
    def _simple(self, df) -> pd.DataFrame:
        """
        Imputes missing values with the most frequent value in each column.
        
        Parameters:
        df (pd.DataFrame): The dataframe to impute.
        
        Returns:
        pd.DataFrame: The dataframe with imputed values.
        """
        imputer = SimpleImputer(strategy='most_frequent')
        return imputer.fit_transform(df)
    
    def clean_data_replace_value(self, methode: str) -> None:
        """
        Cleans the dataframe by replacing missing values using the specified method.
        
        Parameters:
        methode (str): The imputation method to use ('moyenne', 'median', 'delete', 'simple', 'knn', 'iterative').
        
        Raises:
        ValueError: If the specified method is not available.
        """
        missing_values = self.df.isna().sum().sum()
        threshold = (self.limit * len(self.df.columns)) / 10

        methodes_of_imputation = {
            'moyenne': self._moyenne,
            'median': self._median,
            'delete': self._delete,
            'simple': self._simple,
            'knn': self.__call_of_method__('_imputer', self.df, KNNImputer),
            'iterative': self.__call_of_method__('_imputer', self.df, IterativeImputer)
        }

        if methode.lower() not in methodes_of_imputation.keys():
            raise ValueError(f"La méthode {methode} n'est pas disponible. Les méthodes disponibles sont : {', '.join(methodes_of_imputation.keys())}")
        
        methodes_of_imputation[methode.lower()] if missing_values >= threshold else print("Le nombre de données manquantes est inférieur à 10% et ne sera pas remplacé.")

    def clean_data_delete_replica(self):
        """
        Removes duplicate rows from the dataframe.
        """
        self.df.drop_duplicates(inplace=True)
        print("Duplicats supprimés.")
    
    def remove_irrelevant_columns(self, columns_to_remove: list):
        """
        Removes specified columns from the dataframe.
        
        Parameters:
        columns_to_remove (list): List of column names to remove.
        """
        self.df.drop(columns=columns_to_remove, inplace=True, errors='ignore')
        print(f"Colonnes supprimées : {columns_to_remove}")

    def remove_high_missing_columns(self, threshold: float = 0.5):
        """
        Removes columns with a high percentage of missing values.
        
        Parameters:
        threshold (float): The threshold for missing values (default is 0.5).
        """
        missing_ratio = self.df.isnull().mean()
        columns_to_remove = missing_ratio[missing_ratio > threshold].index.tolist()
        self.df.drop(columns=columns_to_remove, inplace=True)
        print(f"Colonnes avec plus de {threshold*100}% de valeurs manquantes supprimées : {columns_to_remove}")

    def extract_patterns(self, column_name: str, pattern: str):
        """
        Extracts patterns from a specified column using a regular expression.
        
        Parameters:
        column_name (str): The name of the column to extract patterns from.
        pattern (str): The regular expression pattern to use.
        """
        if (column_name in self.df.columns):
            self.df[column_name + '_extracted'] = self.df[column_name].astype(str).apply(lambda x: re.findall(pattern, x))
            print(f"Motifs extraits pour {column_name}")
        else:
            print(f"La colonne {column_name} n'existe pas.")
    
    def correct_errors(self, column_name: str, correction_dict: dict):
        """
        Corrects errors in a specified column using a dictionary of corrections.
        
        Parameters:
        column_name (str): The name of the column to correct.
        correction_dict (dict): A dictionary where keys are incorrect values and values are the corrections.
        """
        if column_name in self.df.columns:
            self.df[column_name] = self.df[column_name].replace(correction_dict)
            print(f"Erreurs corrigées dans {column_name}")
        else:
            print(f"La colonne {column_name} n'existe pas.")
    
    def get_cleaned_data(self):
        """
        Returns the cleaned dataframe.
        
        Returns:
        pd.DataFrame: The cleaned dataframe.
        """
        return self.df

""""
méthode:"
    - moyenne
    - median
    - delete
    - simple"
    -knn
    -iterative"
"""""
data_clean_instance = DataClean("C:/Users/valen/Desktop/Machine Learning/teaching_ml_bis_2025/data/en.openfoodfacts.org.products.csv")
data_clean_instance.clean_data_replace_value(methode='moyenne')
#data_clean_instance.clean_data_delete_replica()
#data_clean_instance.remove_irrelevant_columns(['column_to_remove1', 'column_to_remove2'])
#data_clean_instance.remove_high_missing_columns(threshold=0.6)
#data_clean_instance.extract_patterns('serving_size', r'\d+')
#data_clean_instance.correct_errors('product_name', {'erreur1': 'correction1', 'erreur2': 'correction2'})
#cleaned_df = data_clean_instance.get_cleaned_data()v