import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from Tache0_KARDACHE_Emilia_Master1_Data_Sciences import detect_variables
from Tache1_0_KARDACHE_Emilia_Master1_Data_Sciences import clean_dataset  

def standard_scaling(df, numeric_columns):
    """
    (Données normalement distribuées)
    Applique le StandardScaler sur les colonnes numériques pour les normaliser.
    Le StandardScaler transforme les données de sorte qu'elles aient une moyenne de 0 
    et un écart-type de 1.
    
    Paramètres :
    - df : DataFrame contenant les données à scaler.
    - numeric_columns : Liste des colonnes numériques à transformer.
    
    Retourne :
    - DataFrame avec les colonnes numériques transformées.
    """
    scaler = StandardScaler()
    scaled_df = df.copy()  # Création d'une copie pour éviter de modifier l'original
    scaled_df[numeric_columns] = scaler.fit_transform(df[numeric_columns])  # Applique le scaler
    return scaled_df


def minmax_scaling(df, numeric_columns):
    """
    (Meme échelle)
    Applique le MinMaxScaler sur les colonnes numériques pour les ramener dans une plage de [0, 1]. 
    
    Paramètres :
    - df : DataFrame contenant les données à scaler.
    - numeric_columns : Liste des colonnes numériques à transformer.
    
    Retourne :
    - DataFrame avec les colonnes numériques transformées.
    """
    scaler = MinMaxScaler()
    scaled_df = df.copy()  # Création d'une copie pour éviter de modifier l'original
    scaled_df[numeric_columns] = scaler.fit_transform(df[numeric_columns])  # Applique le scaler
    return scaled_df


def robust_scaling(df, numeric_columns):
    """
    (Outliers)
    Applique le RobustScaler sur les colonnes numériques (en utilisant la médiane et l'écart interquartile) pour les rendre plus robustes aux valeurs extrêmes.
    
    Paramètres :
    - df : DataFrame contenant les données à scaler.
    - numeric_columns : Liste des colonnes numériques à transformer.
    
    Retourne :
    - DataFrame avec les colonnes numériques transformées.
    """
    scaler = RobustScaler()
    scaled_df = df.copy()  # Création d'une copie pour éviter de modifier l'original
    scaled_df[numeric_columns] = scaler.fit_transform(df[numeric_columns])  # Applique le scaler
    return scaled_df


def apply_scaling_methods(df, numeric_columns):
    """
    Applique différentes méthodes de scaling (StandardScaler, MinMaxScaler, RobustScaler) 
    sur les données numériques et retourne les résultats dans un dictionnaire.
    
    Paramètres :
    - df : DataFrame contenant les données à scaler.
    - numeric_columns : Liste des colonnes numériques à transformer.
    
    Retourne :
    - Dictionnaire avec les DataFrames transformés pour chaque méthode de scaling.
    """
    scaled_results = {}

    # Appliquer StandardScaler
    scaled_results["StandardScaler"] = standard_scaling(df, numeric_columns)

    # Appliquer MinMaxScaler
    scaled_results["MinMaxScaler"] = minmax_scaling(df, numeric_columns)

    # Appliquer RobustScaler
    scaled_results["RobustScaler"] = robust_scaling(df, numeric_columns)

    return scaled_results


# Exemple d'utilisation :

# Charger un échantillon de données (par exemple Open Food Facts)
path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(path, nrows=10000, sep='\t', encoding="utf-8", low_memory=False, na_filter=True)

# Nettoyer le dataset en supprimant les colonnes non pertinentes et en imputant les valeurs manquantes
df_cleaned = clean_dataset(path)

# Appliquer la détection des variables
result = detect_variables(df_cleaned, sample_size=1000)

# Obtenir les colonnes numériques à partir du résultat de la détection
numeric_columns = result["Types de variables"].loc[result["Types de variables"]["Type de variable"] == "Numérique", "Nom de la colonne"].tolist()

# Appliquer les méthodes de scaling et obtenir les résultats
scaled_results = apply_scaling_methods(df_cleaned, numeric_columns)

# Afficher les résultats pour chaque méthode de scaling
for scaler_name, scaled_df in scaled_results.items():
    print(f"Résultats après application de {scaler_name} :")
    print(scaled_df[numeric_columns].head())  # Afficher les premières lignes des résultats
    print()

# TODO: Ajouter les colonnes catégorielles 
    # Encoder les features non numériques (Tache1.1).
    # Appliquer de la meme manière le scaling sur ces dernières.