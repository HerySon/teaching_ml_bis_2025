import pandas as pd
import numpy as np

def clean_dataset_local(path: str) -> pd.DataFrame:
    """
    Fonction de nettoyage du dataset :
    - Charge les données
    - Supprime les colonnes vides
    - Supprime les doublons
    - Remplace les valeurs infinies et les NaN

    Paramètre :
    - path : str
        Le chemin vers le dataset CSV.

    Retourne :
    - pd.DataFrame : dataset nettoyé.
    """
    df = pd.read_csv(path, sep='\t', encoding="utf-8", low_memory=False, na_filter=True, nrows=10000)

    # Suppression des colonnes vides
    df.dropna(axis=1, how='all', inplace=True)

    # Suppression des doublons
    df.drop_duplicates(inplace=True)

    # Remplacement des valeurs infinies et NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

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
    df_cleaned = clean_dataset_local(path)

    # Sélectionner uniquement les colonnes numériques
    numeric_columns = df_cleaned.select_dtypes(include=['number'])
    
    if numeric_columns.empty:
        print("Aucune colonne numérique disponible pour la corrélation.")
        return []

    # Calcul de la matrice de corrélation absolue
    corr_matrix = numeric_columns.corr().abs()

    # Récupérer uniquement le triangle supérieur pour éviter les doublons
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Extraire les colonnes avec une corrélation > threshold
    highly_correlated = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    return highly_correlated

# Exemple d'utilisation
path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
correlated_columns = highly_correlated(path, threshold=0.9)
print(f"Colonnes fortement corrélées : {correlated_columns}")
