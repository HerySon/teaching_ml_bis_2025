from sklearn.impute import KNNImputer
import pandas as pd

#NON CATEGORIELLE IMPUTATION la modalité la plus fréquente.

def clean_dataset(path: str, columns_to_drop=None, missing_threshold=90, n_neighbors=5) -> pd.DataFrame:
    """
    Fonction pour nettoyer un dataset en plusieurs étapes :
    - Suppression des colonnes non pertinentes
    - Calcul du pourcentage de valeurs manquantes
    - Suppression des colonnes avec trop de valeurs manquantes
    - Imputation des valeurs manquantes en utilisant KNN (et Médiane commentée ici)

    Paramètres :
    - path : str
        Chemin du fichier CSV.
    - columns_to_drop : list (optionnel)
        Liste des colonnes à supprimer. Si None, une liste par défaut est utilisée.
    - missing_threshold : int
        Seuil en pourcentage de valeurs manquantes au-dessus duquel une colonne est supprimée (par défaut 90%).
    - n_neighbors : int
        Nombre de voisins à utiliser pour l'imputation KNN.

    Retourne :
    - DataFrame nettoyé
    """
    # Chargement du dataset
    df = pd.read_csv(path, nrows=10000, sep='\t', encoding="utf-8", low_memory=False, na_filter=True)

    # Affichage des informations initiales sur le dataset
    print("Informations sur le dataset initial :")
    df.info()
    print("\nStatistiques descriptives :")
    print(df.describe())

    # Suppression des colonnes non pertinentes (passées en paramètre)
    if columns_to_drop is None:
        columns_to_drop = [
            "code", "url", "creator", "created_t", "created_datetime",
            "last_modified_t", "last_modified_datetime", "packaging", "packaging_tags",
            "brands_tags", "categories_tags", "categories_fr",
            "origins_tags", "manufacturing_places", "manufacturing_places_tags",
            "labels_tags", "labels_fr", "emb_codes", "emb_codes_tags",
            "first_packaging_code_geo", "cities", "cities_tags", "purchase_places",
            "countries_tags", "countries_fr", "image_ingredients_url",
            "image_ingredients_small_url", "image_nutrition_url", "image_nutrition_small_url",
            "image_small_url", "image_url", "last_updated_t", "last_updated_datetime", "last_modified_by"
        ]
    df.drop(columns=columns_to_drop, errors='ignore', inplace=True)

    # Calcul du pourcentage de valeurs manquantes par colonne
    missing_percentage = df.isnull().mean() * 100

    # Suppression des colonnes avec trop de valeurs manquantes
    df.drop(columns=missing_percentage[missing_percentage > missing_threshold].index, inplace=True)

    # Affichage du DataFrame après suppression des colonnes avec trop de valeurs manquantes
    print(f"\nDataFrame après suppression des colonnes avec plus de {missing_threshold}% de valeurs manquantes :")
    print(df.head())

    # Sélectionner uniquement les colonnes numériques
    df_numeric = df.select_dtypes(include=['float64', 'int64'])

    # Application de l'imputation KNN sur les colonnes numériques
    knn_imputer = KNNImputer(n_neighbors=n_neighbors)
    df_knn_imputed = knn_imputer.fit_transform(df_numeric)
    df_knn_imputed = pd.DataFrame(df_knn_imputed, columns=df_numeric.columns)

    # Optionnel : imputation par médiane (commentée ici)
    # df_median_imputed = df_numeric.copy()
    # df_median_imputed.fillna(df_median_imputed.median(), inplace=True)

    # Affichage du DataFrame après l'imputation KNN
    print("\nDataFrame après l'imputation KNN des valeurs manquantes :")
    print(df_knn_imputed.head())

    return df_knn_imputed

# Exemple d'utilisation
path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df_knn_cleaned = clean_dataset(path)

