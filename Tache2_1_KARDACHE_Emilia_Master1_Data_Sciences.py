import pandas as pd
import numpy as np
from Tache1_0_KARDACHE_Emilia_Master1_Data_Sciences import clean_dataset


def highly_correlated(path: str, threshold=0.9) -> list:
    """
    Fonction pour identifier les colonnes fortement corrélées après nettoyage.

    Paramètres :
    - path : str
        Le chemin vers le dataset CSV.
    - threshold : float
        Seuil de corrélation au-delà duquel les colonnes sont considérées comme fortement corrélées.

    Retourne :
    - Liste des colonnes fortement corrélées.
    """
    # Nettoyage du dataset
    df_cleaned = clean_dataset(path)

    # Calcul de la matrice de corrélation absolue
    corr_matrix = df_cleaned.corr().abs()

    # Récupérer uniquement le triangle supérieur pour éviter les doublons
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Extraire les colonnes avec une corrélation > threshold
    highly_correlated = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    return highly_correlated

# Exemple d'utilisation
path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
correlated_columns = highly_correlated(path, threshold=0.9)
print(f"Colonnes fortement corrélées : {correlated_columns}")  
