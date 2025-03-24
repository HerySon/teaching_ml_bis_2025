import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.stats import zscore

# Chargement du fichier CSV (assurez-vous que le chemin du fichier est correct)
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=1000, sep='\t', encoding="utf-8")

# Prétraitement des données : supprimer les NaN, les doublons, et gérer les valeurs manquantes

def preprocess_data(df, threshold=0.9):
    # Suppression des doublons
    df = df.drop_duplicates()

    # Suppression des colonnes où plus de 90% des valeurs sont manquantes
    missing_percentage = df.isnull().mean()
    cols_to_drop = missing_percentage[missing_percentage > threshold].index
    df = df.drop(cols_to_drop, axis=1)

    # Suppression des lignes avec des valeurs NaN restantes
    df = df.dropna()

    return df

# Appliquer le prétraitement
df = preprocess_data(df)

# Sélectionner les colonnes numériques pour l'analyse des outliers
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# Fonction 1 : Détection des outliers avec le critère de Tukey
def tukey_outliers(df, columns):
    outliers = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
    return outliers

# Fonction 2 : Détection des outliers avec le Z-score
def zscore_outliers(df, columns, threshold=3):
    outliers = {}
    for col in columns:
        # Enlever les NaN avant de calculer les Z-scores
        df_cleaned = df[col].dropna()
        z_scores = zscore(df_cleaned) 
        outliers[col] = np.where(np.abs(z_scores) > threshold)[0].tolist()
    return outliers

# Fonction 3 : Détection des outliers avec Isolation Forest
def isolation_forest_outliers(df, columns):
    outliers = {}
    model = IsolationForest(contamination=0.05)  # Assume 5% des données sont des outliers
    for col in columns:
        # Enlever les NaN avant de faire la détection avec Isolation Forest
        df_cleaned = df[[col]].dropna()

        # Vérifier si la colonne contient des données après suppression des NaN
        if df_cleaned.empty:
            print(f"Aucune donnée dans la colonne '{col}' après suppression des NaN.")
            continue

        # Appliquer le modèle Isolation Forest
        outliers[col] = df_cleaned[model.fit_predict(df_cleaned) == -1].index.tolist()
    return outliers



# Fonction 4 : Détection des outliers avec One-Class SVM (scikit-learn)
def one_class_svm_outliers(df, columns):
    outliers = {}
    model = OneClassSVM(nu=0.05)  # Assume 5% des données sont des outliers
    for col in columns:
        # Enlever les NaN avant de faire la détection avec One-Class SVM
        df_cleaned = df[[col]].dropna()
        
        # Vérifier si la colonne est vide après suppression des NaN
        if df_cleaned.shape[0] == 0:
            print(f"Aucune donnée disponible pour {col} après suppression des NaN.")
            continue
        
        outliers[col] = df_cleaned[model.fit_predict(df_cleaned) == -1].index.tolist()
    return outliers

one_class_svm_outliers_result = one_class_svm_outliers(df, numeric_columns)

# Fonction pour gérer les outliers en fonction de la stratégie choisie
def handle_outliers(df, outlier_indices, strategy="remove"):
    """
    Stratégie pour traiter les outliers :
    - "keep" : Conserver les outliers
    - "impute" : Remplacer les outliers par la médiane
    - "remove" : Supprimer les outliers
    """
    if strategy == "remove":
        return df.drop(index=outlier_indices)
    
    elif strategy == "impute":
        # Remplacer les outliers par la médiane de chaque colonne
        for col in df.columns:
            median_value = df[col].median()
            df.loc[outlier_indices, col] = median_value
        return df
    
    elif strategy == "keep":
        # Ne rien changer, conserver les outliers
        return df
    
    else:
        print("Stratégie inconnue. La stratégie 'remove' sera utilisée par défaut.")
        return df.drop(index=outlier_indices)

# Détection des outliers avec les différentes méthodes
tukey_outliers_result = tukey_outliers(df, numeric_columns)
zscore_outliers_result = zscore_outliers(df, numeric_columns)
isolation_forest_outliers_result = isolation_forest_outliers(df, numeric_columns)
one_class_svm_outliers_result = one_class_svm_outliers(df, numeric_columns)

# Affichage des résultats de la détection
print("\nOutliers détectés avec le critère de Tukey :")
print(tukey_outliers_result)

print("\nOutliers détectés avec le Z-score :")
print(zscore_outliers_result)

print("\nOutliers détectés avec Isolation Forest :")
print(isolation_forest_outliers_result)

print("\nOutliers détectés avec One-Class SVM :")
print(one_class_svm_outliers_result)

# Demander à l'utilisateur quelle stratégie appliquer pour gérer les outliers
print("\nChoisissez une stratégie pour gérer les outliers :")
print("keep : Conserver les outliers")
print("impute : Remplacer les outliers par la médiane")
print("remove : Supprimer les outliers")

# Option utilisateur
choice = input("Entrez votre choix (keep, impute, ou remove) : ").lower()

# Appliquer la stratégie choisie à la détection des outliers
all_outliers = set(sum(tukey_outliers_result.values(), [])) | \
               set(sum(zscore_outliers_result.values(), [])) | \
               set(sum(isolation_forest_outliers_result.values(), [])) | \
               set(sum(one_class_svm_outliers_result.values(), []))

# Appliquer la stratégie choisie sur les données
df_processed = handle_outliers(df, all_outliers, strategy=choice)

# Affichage des résultats après gestion des outliers
print("\nDonnées après traitement des outliers :") 
print(df_processed.head())  


