import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from scipy.stats import zscore

# Chargement du fichier CSV (assurez-vous que le chemin du fichier est correct)
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=1000, sep='\t', encoding="utf-8")

# Sélectionner les colonnes numériques pour l'analyse des outliers
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# 1. Détection des outliers avec le critère de Tukey
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

tukey_outliers_result = tukey_outliers(df, numeric_columns)

# 2. Détection des outliers avec le Z-score
def zscore_outliers(df, columns, threshold=3):
    outliers = {}
    for col in columns:
        z_scores = zscore(df[col].dropna())  # On enlève les NaN avant de calculer les Z-scores
        outliers[col] = np.where(np.abs(z_scores) > threshold)[0].tolist()
    return outliers

zscore_outliers_result = zscore_outliers(df, numeric_columns)

# 3. Détection des outliers avec Isolation Forest (scikit-learn)
def isolation_forest_outliers(df, columns):
    outliers = {}
    model = IsolationForest(contamination=0.05)  # Assume 5% des données sont des outliers
    for col in columns:
        outliers[col] = df[model.fit_predict(df[[col]]) == -1].index.tolist()
    return outliers

isolation_forest_outliers_result = isolation_forest_outliers(df, numeric_columns)

# 4. Détection des outliers avec One-Class SVM (scikit-learn)
def one_class_svm_outliers(df, columns):
    outliers = {}
    model = OneClassSVM(nu=0.05)  # Assume 5% des données sont des outliers
    for col in columns:
        outliers[col] = df[model.fit_predict(df[[col]]) == -1].index.tolist()
    return outliers

one_class_svm_outliers_result = one_class_svm_outliers(df, numeric_columns)

# Fonction pour gérer les outliers
def handle_outliers(df, outlier_indices, strategy="remove"):
    """
    Stratégie pour traiter les outliers :
    - "remove" : Supprimer les outliers
    - "impute" : Remplacer les outliers par la médiane
    - "keep" : Conserver les outliers
    """
    if strategy == "remove":
        return df.drop(index=outlier_indices)
    elif strategy == "impute":
        # Remplacer les outliers par la médiane
        for col in df.columns:
            median_value = df[col].median()
            df.loc[outlier_indices, col] = median_value
        return df
    elif strategy == "keep":
        # Conserver les outliers sans modification
        return df

# Affichage des résultats
print("\nOutliers détectés avec le critère de Tukey :")
print(tukey_outliers_result)

print("\nOutliers détectés avec le Z-score :")
print(zscore_outliers_result)

print("\nOutliers détectés avec Isolation Forest :")
print(isolation_forest_outliers_result)

print("\nOutliers détectés avec One-Class SVM :")
print(one_class_svm_outliers_result)

# Demander à l'utilisateur quelle stratégie appliquer
print("\nChoisissez une stratégie pour gérer les outliers :")
print("1: Conserver les outliers")
print("2: Imputer les outliers (remplacer par la médiane)")
print("3: Supprimer les outliers")

# Option utilisateur
choice = input("Entrez le numéro de la stratégie choisie (1, 2, ou 3) : ")

# Sélectionner la stratégie
if choice == "1":
    strategy = "keep"
elif choice == "2":
    strategy = "impute"
elif choice == "3":
    strategy = "remove"
else:
    print("Choix invalide. La stratégie par défaut 'remove' sera appliquée.")
    strategy = "remove"

# Appliquer la stratégie choisie à la détection des outliers
all_outliers = set(sum(tukey_outliers_result.values(), [])) | \
               set(sum(zscore_outliers_result.values(), [])) | \
               set(sum(isolation_forest_outliers_result.values(), [])) | \
               set(sum(one_class_svm_outliers_result.values(), []))

# Appliquer la stratégie
df_processed = handle_outliers(df, all_outliers, strategy)

# Afficher les résultats après gestion des outliers
print("\nDonnées après traitement des outliers :")
print(df_processed.head())

# Sauvegarde des résultats après traitement des outliers
df_processed.to_csv("processed_data_with_outliers_handled.csv", index=False)
