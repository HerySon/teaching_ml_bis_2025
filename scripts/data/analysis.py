from typing import Dict, List

import numpy as np
import pandas as pd

from .correlation_analysis import analyze_correlations

def analyze_data_quality(df: pd.DataFrame) -> Dict:
    """
    Analyse complète de la qualité des données.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Dict: Rapport d'analyse
    """
    # Dimensions du DataFrame
    n_rows, n_cols = df.shape
    
    # Types de données
    dtypes = df.dtypes.value_counts()
    
    # Valeurs manquantes
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / n_rows * 100).round(2)
    columns_with_missing = missing_values[missing_values > 0].index.tolist()
    
    # Valeurs uniques
    unique_counts = df.nunique()
    unique_percentages = (unique_counts / n_rows * 100).round(2)
    
    # Colonnes constantes
    constant_columns = [col for col in df.columns if unique_counts[col] == 1]
    
    # Colonnes avec trop de catégories (pour les variables catégorielles)
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    high_cardinality_columns = [col for col in categorical_columns 
                              if unique_counts[col] / n_rows > 0.5]
    
    # Colonnes avec peu de valeurs uniques (pour les variables numériques)
    numeric_columns = df.select_dtypes(include=np.number).columns
    low_variance_columns = [col for col in numeric_columns 
                          if unique_counts[col] / n_rows < 0.01]
    
    # Analyse de la mémoire
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum() / 1024**2  # en MB
    
    return {
        'dimensions': {'rows': n_rows, 'columns': n_cols},
        'dtypes': dtypes.to_dict(),
        'missing_values': {
            'counts': missing_values.to_dict(),
            'percentages': missing_percentages.to_dict(),
            'columns_with_missing': columns_with_missing
        },
        'unique_values': {
            'counts': unique_counts.to_dict(),
            'percentages': unique_percentages.to_dict()
        },
        'quality_issues': {
            'constant_columns': constant_columns,
            'high_cardinality_columns': high_cardinality_columns,
            'low_variance_columns': low_variance_columns
        },
        'memory_usage': {
            'total_mb': total_memory,
            'per_column': memory_usage.to_dict()
        }
    }

def check_missing_values(df: pd.DataFrame, threshold: float = 50) -> pd.Series:
    """
    Analyse et affiche le pourcentage de valeurs manquantes par colonne.
    
    Args:
        df: DataFrame pandas
        threshold: pourcentage minimum pour afficher une alerte (ex : 50% par défaut)

    Returns:
        pd.Series: Série triée des valeurs manquantes
    """
    missing_values = df.isnull().mean() * 100
    df_missing = missing_values[missing_values > 0].sort_values(ascending=False)

    if df_missing.empty:
        print("✅ Aucune valeur manquante dans le dataset.")
    else:
        print(f"⚠️ {len(df_missing)} colonnes sur {len(df.columns)} ont des valeurs manquantes.")
        print(df_missing[df_missing > threshold])

    return df_missing

def check_data_consistency(df: pd.DataFrame) -> List[str]:
    """
    Vérifie la cohérence des données.
    
    Args:
        df: DataFrame à analyser
    
    Returns:
        List[str]: Liste des incohérences
    """
    inconsistencies = []
    
    if 'energy-kcal_100g' in df.columns:
        energy_issues = df[df['energy-kcal_100g'] > 900]
        inconsistencies.append(f"Energie > 900kcal/100g: {len(energy_issues)} produits")
    
    nutrient_cols = ['carbohydrates_100g', 'proteins_100g', 'fat_100g']
    if all(col in df.columns for col in nutrient_cols):
        sum_over_100 = df[df[nutrient_cols].sum(axis=1) > 100]
        inconsistencies.append(f"Somme nutriments > 100g: {len(sum_over_100)} produits")
    
    return inconsistencies

def detect_outliers(series: pd.Series, method: str = 'zscore', threshold: float = 3.0) -> pd.Series:
    """
    Détecte les valeurs aberrantes dans une série numérique.
    
    Args:
        series: Série à analyser
        method: Méthode de détection ('zscore', 'iqr')
        threshold: Seuil pour la détection
    
    Returns:
        pd.Series: Masque booléen indiquant les outliers
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("La série doit être numérique")
    
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < (Q1 - threshold * IQR)) | (series > (Q3 + threshold * IQR))
    
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'zscore' ou 'iqr'") 