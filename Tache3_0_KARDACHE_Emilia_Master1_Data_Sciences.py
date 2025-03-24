import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


"""
Ce script détecte et traite les outliers dans un DataFrame en utilisant trois méthodes :
- IQR (Interquartile Range)
- Z-score
- Isolation Forest

L'utilisateur peut choisir une stratégie pour gérer les outliers : "remove" (supprimer), "impute" (imputer avec la médiane), ou "keep" (conserver).

### Fonctions :
1. **detect_outliers_iqr(df, cols, threshold=1.5)** : Détecte les outliers avec la méthode IQR.
2. **detect_outliers_zscore(df, cols, threshold=3)** : Détecte les outliers avec le Z-score.
3. **detect_outliers_isolation_forest(df, cols)** : Détecte les outliers avec Isolation Forest.
4. **handle_outliers(df, outliers_df, strategy="remove")** : Applique une stratégie pour gérer les outliers.

### Utilisation :
- Le script charge un jeu de données (Open Food Facts), sélectionne les colonnes numériques, et détecte les outliers.
- L'utilisateur choisit la stratégie de traitement des outliers via un prompt.
- Les distributions des données après traitement sont affichées sous forme de boxplots.
"""

def detect_outliers_iqr(df, cols, threshold=1.5):
    """Détecte les outliers avec la méthode IQR pour les colonnes sélectionnées."""
    df_outliers = df.copy()
    for col in cols:
        #après suppression des NaN
        if df[col].dropna().empty:  
            print(f"Colonne {col} vide ou trop de NaN, saut de la détection d'outliers.")
            continue  
        Q1, Q3 = np.percentile(df[col].dropna(), [25, 75]) 
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_outliers[f'outlier_iqr_{col}'] = (df[col] < lower_bound) | (df[col] > upper_bound)
    return df_outliers

def detect_outliers_zscore(df, cols, threshold=3):
    """Détecte les outliers avec le Z-score pour les colonnes sélectionnées."""
    df_outliers = df.copy()
    for col in cols:
        #après suppression des NaN
        if df[col].dropna().empty: 
            print(f"Colonne {col} vide ou trop de NaN, saut de la détection d'outliers.")
            continue  
        df_outliers[f'outlier_zscore_{col}'] = np.abs(zscore(df[col].dropna())) > threshold
    return df_outliers

def detect_outliers_isolation_forest(df, cols):
    """Détecte les outliers avec Isolation Forest pour les colonnes sélectionnées."""
    df_outliers = df.copy()
    for col in cols:
        #Après suppression des NaN
        if df[col].dropna().empty:  
            print(f"Colonne {col} vide ou trop de NaN, saut de la détection d'outliers.")
            continue  
        model = IsolationForest(contamination=0.01, random_state=42)
        df_outliers[f'outlier_isolation_forest_{col}'] = model.fit_predict(df[[col]].dropna()) == -1
    return df_outliers

def handle_outliers(df, outliers_df, strategy="remove"):
    """Applique une stratégie pour gérer les outliers détectés."""
    df_copy = df.copy()
    for col in df.columns:
        # Filtrer les lignes avec des outliers détectés
        outliers = outliers_df[f'outlier_iqr_{col}'] | outliers_df[f'outlier_zscore_{col}'] | outliers_df[f'outlier_isolation_forest_{col}']
        
        if strategy == "remove":
            # Supprimer les lignes contenant des outliers
            df_copy = df_copy[~outliers]
        
        elif strategy == "impute":
            # Imputer les valeurs des outliers par la médiane (par exemple)
            imputer = SimpleImputer(strategy="median")
            df_copy[col] = imputer.fit_transform(df_copy[[col]])
        
        elif strategy == "keep":
            # Conserver les outliers (aucune action)
            pass
    
    return df_copy

# Exemple d'utilisation
path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"

# Charger le dataset
df = pd.read_csv(path, nrows=100, sep='\t', encoding="utf-8", low_memory=False, na_filter=True) 

# Sélectionner uniquement les colonnes numériques
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
df_selected = df[numeric_columns].dropna()  # Appliquer dropna ici et ne pas modifier le DataFrame original en place

# Appliquer les méthodes de détection d'outliers
df_outliers_iqr = detect_outliers_iqr(df_selected, numeric_columns)
df_outliers_zscore = detect_outliers_zscore(df_selected, numeric_columns)
df_outliers_isolation_forest = detect_outliers_isolation_forest(df_selected, numeric_columns)

# Fusionner les résultats des différentes méthodes d'outliers
df_combined = pd.concat([df_outliers_iqr, df_outliers_zscore.dropna(axis=1, how='all'), df_outliers_isolation_forest.dropna(axis=1, how='all')], axis=1)

# Afficher un aperçu des valeurs identifiées comme outliers
outlier_columns = [col for col in df_combined.columns if 'outlier_' in col]
print(df_combined[outlier_columns].head())

# Choisir une stratégie de traitement des outliers
strategy = input("Choisissez une stratégie pour traiter les outliers ('remove', 'impute', 'keep'): ")

# Appliquer la stratégie choisie
df_handled = handle_outliers(df_selected, df_combined, strategy=strategy)

# Affichage des distributions après traitement des outliers
for col in df_selected.columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_handled[col])
    plt.title(f"Distribution de {col} après traitement des outliers ({strategy})")
    plt.show() 
