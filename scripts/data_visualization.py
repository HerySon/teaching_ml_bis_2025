# Proposer des méthodes de visualisations pouvant être utilisées de manière mutli-variées, permettant, a minima de :

#     d'explorer certaines variables de départ dans votre data set
#     d’explorer les resultats de vos traitements : clusters formés

# Voici quelques exemples, non exhaustif, de représentations graphiques que vous pouvez utiliser:

#     Matrice de corrélation et heatmap
#     Pair-wise Scatter Plot (toutes les paires de features sous forme de nuages de points)
#     Les partial dependency plot
#     …

import pandas as pd
import seaborn as sns
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

class DataVisualization(object):
    def __init__(self, url_dataset, limit=10, sep='\t') -> None:
        """
        Initializes the DataVisualization object by loading the dataset.
        Parameters
        ----------
        url_dataset : str
            The URL or file path to the dataset.
        limit : int, optional
            The maximum number of rows to read from the dataset (default is 10).
        sep : str, optional
            The delimiter to use for parsing the dataset (default is '\t').
        """
        self.df = pd.read_csv(url_dataset, nrows=limit, sep=sep, encoding="utf-8")
        self.limit = limit

    def plot_correlation_matrix(self):
        """
        Plots the correlation matrix of the numeric features in the dataset.
        This method selects the numeric columns from the dataframe, computes the correlation matrix,
        and then uses seaborn's heatmap to visualize the correlations. The plot is displayed with
        annotations and a color map to indicate the strength of correlations.
        Methods used:
        - pandas.DataFrame.select_dtypes: To select numeric columns.
        - pandas.DataFrame.corr: To compute the correlation matrix.
        - seaborn.heatmap: To plot the heatmap of the correlation matrix.
        - matplotlib.pyplot: To configure and display the plot.
        """
        plt.figure(figsize=(12, 10))
        
        numeric_df = self.df.select_dtypes(include=['number'])
        correlation_matrix = numeric_df.corr()
        
        sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            linewidths=0.5, 
            fmt=".2f", 
            annot_kws={"size": 10}
        )
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.title('Correlation Matrix', fontsize=14)
        plt.show()

    def plot_pairwise_scatter(self, max_vars=5):
        """
        Plots pairwise scatter plots for the numeric features in the dataset.
        This method selects the first few numeric columns (up to max_vars) and uses seaborn's pairplot
        to create scatter plots for each pair of selected features. The plot is displayed to show the
        relationships between the features.
        Parameters
        ----------
        max_vars : int, optional
            The maximum number of numeric features to include in the pairwise scatter plots (default is 5).
        Methods used:
        - pandas.DataFrame.select_dtypes: To select numeric columns.
        - seaborn.pairplot: To create pairwise scatter plots.
        - matplotlib.pyplot: To display the plot.
        """
        numeric_df = self.df.select_dtypes(include=['number'])
        selected_columns = numeric_df.columns[:max_vars]  
        sns.pairplot(self.df[selected_columns])
        plt.show()

    def plot_partial_dependency(self, feature):
        """
        Plots the partial dependency of a specified numeric feature.
        This method checks if the specified feature exists and is numeric. It then removes rows with NaN values,
        trains a RandomForestRegressor model using the remaining numeric features, and plots the partial dependency
        of the specified feature using sklearn's PartialDependenceDisplay.
        Parameters
        ----------
        feature : str
            The name of the numeric feature for which to plot the partial dependency.
        Raises
        ------
        ValueError
            If the feature is not found in the dataset columns or is not numeric.
            If no valid data is left after removing NaN values.
        Methods used:
        - pandas.DataFrame.select_dtypes: To select numeric columns.
        - pandas.DataFrame.drop: To drop the specified feature column.
        - pandas.DataFrame.isna: To identify NaN values.
        - pandas.DataFrame.loc: To filter valid rows.
        - sklearn.ensemble.RandomForestRegressor: To train the model.
        - sklearn.inspection.PartialDependenceDisplay: To plot the partial dependency.
        - matplotlib.pyplot: To configure and display the plot.
        """
        if feature not in self.df.columns:
            raise ValueError(f"Feature '{feature}' not found in dataset columns.")

        if feature not in self.df.select_dtypes(include=['number']).columns:
            raise ValueError(f"Feature '{feature}' is not numeric and cannot be used.")

        X = self.df.select_dtypes(include=['number']).drop(columns=[feature])
        y = self.df[feature]

        # Supprimer les lignes contenant des NaN
        valid_indices = ~X.isna().any(axis=1) & ~y.isna()
        X = X.loc[valid_indices]
        y = y.loc[valid_indices]

        if X.empty or y.empty:
            raise ValueError("No valid data left after removing NaN values.")

        print("Features used for training:", X.columns.tolist())

        model = RandomForestRegressor()
        model.fit(X, y)

        _, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(model, X, [feature], ax=ax)
        plt.title(f'Partial Dependency Plot for {feature}')
        plt.show()

    

data_visualization_instance = DataVisualization("C:/Users/valen/Desktop/Machine Learning/teaching_ml_bis_2025/data/en.openfoodfacts.org.products.csv")
data_visualization_instance.plot_correlation_matrix()
data_visualization_instance.plot_pairwise_scatter()
data_visualization_instance.plot_partial_dependency('fat_100g')

