# Vous proposerez du code permettant à minima, de:

#     détecter et tagger les valeurs abbérantes (en utilisant des critères métiers comme des méthodes algorithmiques)
#     proposer à l’utilisateur le choix entre plusieurs stratégies (conserver les outliers, les imputer, …)

# Ressources

#     le critère de Tukey
#     le z-score
#     les méthodes de scikit-learn


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import (
    MaxAbsScaler, MinMaxScaler, PowerTransformer,
    QuantileTransformer, RobustScaler, StandardScaler
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

class DataOutliers:
    def __init__(self, url_dataset, limit=100):
        self.df = pd.read_csv(url_dataset, nrows=limit, sep='\t', encoding="utf-8")
        print("Colonnes du fichier chargé :", self.df.columns.tolist())
        self.limit = limit

    def __call_of_method__(self, method_name: str, *args, **kwargs) -> any:
        if not hasattr(self, method_name):
            raise AttributeError(f"{self.__class__.__name__} does not have method {method_name}")
        
        method = getattr(self, method_name)
        return method(*args, **kwargs)
    
    def __numeric_columns__(self) -> list:
        """
        Returns a list of numeric columns in the dataframe.
        """
        return [col for col in self.df.select_dtypes(include=[np.number]).columns if col in self.df.columns]
    
    def _zscore(self, col, seuil=3) -> np.ndarray:
        """
        Detects outliers in a column using the z-score method.
        
        Parameters:
        col (str): The column name.
        seuil (int): The z-score threshold to identify outliers.
        
        Returns:
        np.ndarray: A boolean array indicating outliers.
        """
        z_scores = zscore(self.df[col], nan_policy='omit')
        outliers = np.abs(z_scores) > seuil
        return outliers

    def _tukey(self, col, k=1.5) -> np.ndarray:
        """
        Detects outliers in a column using Tukey's method.
        
        Parameters:
        col (str): The column name.
        k (float): The multiplier for the interquartile range.
        
        Returns:
        np.ndarray: A boolean array indicating outliers.
        """
        q1 = self.df[col].quantile(0.25)
        q3 = self.df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        outliers = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
        return outliers

    def _isolation_forest(self, col) -> np.ndarray:
        """
        Detects outliers in a column using the Isolation Forest method.
        
        Parameters:
        col (str): The column name.
        
        Returns:
        np.ndarray: A boolean array indicating outliers.
        """
        iso_forest = IsolationForest(contamination=0.1)
        iso_forest.fit(self.df[[col]])
        return iso_forest.predict(self.df[[col]]) == -1

    def _local_outlier_factor(self, col, n_neighbors=20) -> np.ndarray:
        """
        Detects outliers in a column using the Local Outlier Factor method.
        
        Parameters:
        col (str): The column name.
        n_neighbors (int): The number of neighbors to use for LOF.
        
        Returns:
        np.ndarray: A boolean array indicating outliers.
        """
        self.df = self.df.dropna(subset=[col])
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        outliers = lof.fit_predict(self.df[[col]])
        return outliers == -1

    def _elliptic_envelope(self, col, contamination=0.1) -> np.ndarray:
        """
        Detects outliers in a column using the Elliptic Envelope method.
        
        Parameters:
        col (str): The column name.
        contamination (float): The proportion of outliers in the data set.
        
        Returns:
        np.ndarray: A boolean array indicating outliers.
        """
        self.df = self.df.dropna(subset=[col])
        envelope = EllipticEnvelope(contamination=contamination)
        outliers = envelope.fit_predict(self.df[[col]])
        return outliers == -1

    def _plot_outliers(self, col) -> None:
        """
        Plots the outliers in a column.
        
        Parameters:
        col (str): The column name.
        """
        outliers = self._zscore(col)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.df.index, self.df[col], c=outliers, cmap='coolwarm', label='Outliers')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.title(f'Outliers in {col}')
        plt.legend()
        plt.show()

    def _delete(self, col, outliers) -> None:
        """
        Deletes the outliers in a column.
        
        Parameters:
        col (str): The column name.
        outliers (np.ndarray): A boolean array indicating outliers.
        """
        before_rows = len(self.df)
        self.df = self.df[~outliers]
        after_rows = len(self.df)
        print(f"{before_rows - after_rows} valeurs aberrantes supprimées dans {col}.")

    def _moyenne(self, col, outliers) -> None:
        """
        Replaces outliers in a column with the mean value.
        
        Parameters:
        col (str): The column name.
        outliers (np.ndarray): A boolean array indicating outliers.
        """
        mean_value = self.df[col][~outliers].mean()
        self.df[col] = np.where(outliers, mean_value, self.df[col])
        print(f"Valeurs aberrantes remplacées par la moyenne dans {col}.")

    def _median(self, col, outliers) -> None:
        """
        Replaces outliers in a column with the median value.
        
        Parameters:
        col (str): The column name.
        outliers (np.ndarray): A boolean array indicating outliers.
        """
        median_value = self.df[col][~outliers].median()
        self.df[col] = np.where(outliers, median_value, self.df[col])
        print(f"Valeurs aberrantes remplacées par la médiane dans {col}.")
    
    def _methode(self, methode) -> None:
        """
        Applies a specified transformation method to numeric columns.
        
        Parameters:
        methode (str): The transformation method to apply.
        """
        numeric_cols = self.__numeric_columns__()

        transformer_dict = {
            'minmax': MinMaxScaler(),
            'maxabs': MaxAbsScaler(),
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'quantile': QuantileTransformer(),
            'power': PowerTransformer(),
        }
        
        methode_lower = methode.lower()
        if methode_lower not in transformer_dict:
            raise ValueError(f"La méthode {methode} n'est pas valide. Les méthodes disponibles sont : {', '.join(transformer_dict.keys())}")
        
        transformer = transformer_dict[methode_lower]

        pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('transformer', transformer)
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', pipeline, numeric_cols)
            ],
            remainder='passthrough'  
        )

        transformed_data = preprocessor.fit_transform(self.df)

        self.df.loc[:, numeric_cols] = transformed_data if transformed_data.shape[1] == len(numeric_cols) else None

        print("⚠️ La longueur des données transformées ne correspond pas à la longueur des colonnes numériques.")

    def clean_remove_outliers(self, methode: str, search: str, seuil=3, strategy="imputer", **kwargs) -> None:
        """
        Cleans and removes outliers from the dataframe using specified methods and strategies.
        
        Parameters:
        methode (str): The transformation method to apply.
        search (str): The method to use for detecting outliers.
        seuil (int): The threshold for detecting outliers.
        strategy (str): The strategy to apply to outliers.
        """
        for col in self.__numeric_columns__():
            if self.df[col].isna().all():  
                print(f"⚠️ Colonne {col} vide, ignorée.")
                continue

            self.df[col] = self.df[col].fillna(self.df[col].median())

            search_dict = {
                'zscore': lambda col: self._zscore(col, seuil),
                'tukey': lambda col: self._tukey(col),
                'isolation_forest': lambda col: self._isolation_forest(col),
                'local_outlier_factor': lambda col: self._local_outlier_factor(col),
                'elliptic_envelope': lambda col: self._elliptic_envelope(col),
                'plot_outliers': lambda col: self._plot_outliers(col)
            }

            if search.lower() not in search_dict.keys():
                raise ValueError(f"La méthode recherche {search} n'est pas disponible. Les méthodes de recherches disponibles sont : {', '.join(search_dict.keys())}")

            search_dict[search.lower()](col)

            strategy_dict = {
                'delete': self._delete,
                'moyenne': self._moyenne,
                'median': self._median,
                'minmax': self._methode,
                'maxabs': self._methode,
                'robust': self._methode,
                'standard': self._methode,
                'quantile': self._methode,
                'power': self._methode,
                'imputer': self._imputer
            }

            if strategy not in strategy_dict:
                raise ValueError(f"La stratégie {strategy} n'est pas valide. Les stratégies disponibles sont : {', '.join(strategy_dict.keys())}")

            outliers = search_dict[search.lower()](col)
            strategy_dict[strategy](col, outliers)

    def _imputer(self, col, outliers) -> None:
        """
        Imputes outliers in a column with the median value.
        
        Parameters:
        col (str): The column name.
        outliers (np.ndarray): A boolean array indicating outliers.
        """
        self.df[col] = np.where(outliers, np.nan, self.df[col])
        self.df[col] = self.df[col].fillna(self.df[col].median())
        print(f"Valeurs aberrantes imputées dans {col}.")

    def display_outlier_columns(self):
        """
        Displays columns with detected outliers.
        """
        outlier_columns = [col for col in self.df.columns if col.endswith('_outlier')]
        if outlier_columns:
            print("Columns with detected outliers:", outlier_columns)
        else:
            print("No outliers detected.")
    
    def detect_and_tag_outliers(self, seuil_zscore=3) :
        """
        Detects and tags outliers in numeric columns using the z-score method.
        
        Parameters:
        seuil_zscore (int): The z-score threshold to identify outliers.
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].dropna().empty:
                print(f"⚠️ La colonne '{col}' ne contient pas assez de données pour détecter les outliers.")
                continue

            z_scores = zscore(self.df[col], nan_policy='omit')
            outliers_zscore = np.abs(z_scores) > seuil_zscore

            self.df[f'{col}_outlier'] = outliers_zscore

        print("Outliers détectés et marqués dans de nouvelles colonnes.")
        print("Outliers : " + str(self.df.columns.tolist()))

# Exemple d'utilisation
data_outliers_instance = DataOutliers("C:/Users/valen/Desktop/Machine Learning/teaching_ml_bis_2025/data/en.openfoodfacts.org.products.csv")
data_outliers_instance.clean_remove_outliers(search="local_outlier_factor", methode='quantile', seuil=3) 
data_outliers_instance.detect_and_tag_outliers()
data_outliers_instance.display_outlier_columns()
