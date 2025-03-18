import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import seaborn as sns

class OutlierDetector:
    """
    Classe pour détecter les outliers dans un dataset en utilisant différentes méthodes.

    Méthodes disponibles :
    - tukey_method : Détecte les outliers en utilisant la méthode de Tukey (IQR).
    - z_score_method : Détecte les outliers en utilisant la méthode du Z-score.
    - isolation_forest : Détecte les outliers en utilisant la méthode Isolation Forest.
    - local_outlier_factor : Détecte les outliers en utilisant la méthode Local Outlier Factor (LOF).
    - elliptic_envelope : Détecte les outliers en utilisant la méthode Elliptic Envelope.
    - plot_outliers : Visualise les outliers détectés dans une colonne en utilisant une méthode spécifiée.
    """

    def __init__(self, data):
        """
        Initialise la classe avec les données.

        Arguments :
            data (pd.DataFrame) : Les données à analyser.
        """
        if data.empty:
            raise ValueError("Data is empty. Please provide a valid DataFrame.")
        self.data = data

    def tukey_method(self, column):
        """
        Détecte les outliers dans une colonne en utilisant la méthode de Tukey (IQR).

        Arguments :
            column (str) : Nom de la colonne à analyser.

        Retour :
            pd.Series : Un booléen indiquant si chaque point est un outlier.
        """
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (self.data[column] < lower_bound) | (self.data[column] > upper_bound)

    def z_score_method(self, column, threshold=3):
        """
        Détecte les outliers dans une colonne en utilisant la méthode du Z-score.

        Arguments :
            column (str) : Nom de la colonne à analyser.
            threshold (float) : Seuil de Z-score pour considérer un point comme un outlier (par défaut 3).

        Retour :
            pd.Series : Un booléen indiquant si chaque point est un outlier.
        """
        z_scores = np.abs(stats.zscore(self.data[column]))
        return z_scores > threshold

    def isolation_forest(self, contamination=0.05):
        """
        Détecte les outliers dans les données en utilisant la méthode Isolation Forest.

        Arguments :
            contamination (float) : Proportion de points à considérer comme des outliers (par défaut 0.05).

        Retour :
            pd.Series : Un booléen indiquant si chaque point est un outlier.
        """
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outliers = iso_forest.fit_predict(self.data)
        return outliers == -1

    def local_outlier_factor(self, n_neighbors=20, contamination=0.05):
        """
        Détecte les outliers dans les données en utilisant la méthode Local Outlier Factor (LOF).

        Arguments :
            n_neighbors (int) : Nombre de voisins à utiliser pour calculer le LOF (par défaut 20).
            contamination (float) : Proportion de points à considérer comme des outliers (par défaut 0.05).

        Retour :
            pd.Series : Un booléen indiquant si chaque point est un outlier.
        """
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outliers = lof.fit_predict(self.data)
        return outliers == -1

    def elliptic_envelope(self, contamination=0.05):
        """
        Détecte les outliers dans les données en utilisant la méthode Elliptic Envelope.

        Arguments :
            contamination (float) : Proportion de points à considérer comme des outliers (par défaut 0.05).

        Retour :
            pd.Series : Un booléen indiquant si chaque point est un outlier.
        """
        envelope = EllipticEnvelope(contamination=contamination, random_state=42)
        outliers = envelope.fit_predict(self.data)
        return outliers == -1

    def plot_outliers(self, column, method):
        """
        Visualise les outliers détectés dans une colonne en utilisant une méthode spécifiée.

        Arguments :
            column (str) : Nom de la colonne à analyser.
            method (str) : Méthode à utiliser pour détecter les outliers ('tukey', 'z_score', 'isolation_forest', 'lof', 'elliptic_envelope').
        """
        if method == 'tukey':
            outliers = self.tukey_method(column)
        elif method == 'z_score':
            outliers = self.z_score_method(column)
        elif method == 'isolation_forest':
            outliers = self.isolation_forest()
        elif method == 'lof':
            outliers = self.local_outlier_factor()
        elif method == 'elliptic_envelope':
            outliers = self.elliptic_envelope()
        else:
            raise ValueError("Invalid method. Choose from 'tukey', 'z_score', 'isolation_forest', 'lof', 'elliptic_envelope'.")

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data.index, y=self.data[column], hue=outliers, palette={True: 'red', False: 'blue'}, alpha=0.5)
        plt.title(f"Outliers detected using {method} method")
        plt.xlabel("Index")
        plt.ylabel(column)
        plt.legend(title="Outlier", loc='upper right', labels=['No', 'Yes'])
        plt.show()

# Example usage of the OutlierDetector class

# Load sample data
data = pd.DataFrame({
    'feature1': np.random.normal(0, 1, 1000),
    'feature2': np.random.normal(0, 1, 1000)
})

# Introduce some outliers
data.loc[::50, 'feature1'] = np.random.normal(10, 1, 20)
data.loc[::50, 'feature2'] = np.random.normal(10, 1, 20)

# Initialize the OutlierDetector class
outlier_detector = OutlierDetector(data)

# Detect outliers using Tukey's method
print("\nDetecting outliers using Tukey's method...")
tukey_outliers = outlier_detector.tukey_method('feature1')
print(tukey_outliers)

# Detect outliers using Z-score method
print("\nDetecting outliers using Z-score method...")
z_score_outliers = outlier_detector.z_score_method('feature1')
print(z_score_outliers)

# Detect outliers using Isolation Forest
print("\nDetecting outliers using Isolation Forest...")
iso_forest_outliers = outlier_detector.isolation_forest()
print(iso_forest_outliers)

# Detect outliers using Local Outlier Factor (LOF)
print("\nDetecting outliers using Local Outlier Factor (LOF)...")
lof_outliers = outlier_detector.local_outlier_factor()
print(lof_outliers)

# Detect outliers using Elliptic Envelope
print("\nDetecting outliers using Elliptic Envelope...")
elliptic_envelope_outliers = outlier_detector.elliptic_envelope()
print(elliptic_envelope_outliers)

# Plot outliers detected using Tukey's method
print("\nPlotting outliers detected using Tukey's method...")
outlier_detector.plot_outliers('feature1', method='tukey')

# Plot outliers detected using Z-score method
print("\nPlotting outliers detected using Z-score method...")
outlier_detector.plot_outliers('feature1', method='z_score')

# Plot outliers detected using Isolation Forest
print("\nPlotting outliers detected using Isolation Forest...")
outlier_detector.plot_outliers('feature1', method='isolation_forest')

# Plot outliers detected using Local Outlier Factor (LOF)
print("\nPlotting outliers detected using Local Outlier Factor (LOF)...")
outlier_detector.plot_outliers('feature1', method='lof')

# Plot outliers detected using Elliptic Envelope
print("\nPlotting outliers detected using Elliptic Envelope...")
outlier_detector.plot_outliers('feature1', method='elliptic_envelope')
