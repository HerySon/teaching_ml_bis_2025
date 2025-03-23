"""
Module de scaling pour le traitement des données nutritionnelles.
Fournit des fonctions pour charger, préparer et appliquer différentes
méthodes de scaling sur les données.
"""

# Standard library imports
from typing import Dict, List, Optional, Tuple, Union

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)

# Constants
VALID_RATIO_THRESHOLD = 0.05
CORRELATION_CHANGE_THRESHOLD = 0.1
PRIORITY_KEYWORDS = [
    '_100g',
    'score',
    'quantity',
    'energy',
    'fat',
    'protein',
    'carbohydrate',
    'sugar',
    'fiber',
    'salt'
]


def load_and_prepare_data(file_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Charge et prépare les données numériques d'un fichier CSV.
    
    Args:
        file_path: Chemin vers le fichier CSV
        nrows: Nombre de lignes à charger (None pour tout charger)
        
    Returns:
        DataFrame avec les colonnes numériques nettoyées
    """
    separators = ['\t', ',', ';']
    df = None
    
    for sep in separators:
        try:
            df_test = pd.read_csv(file_path, sep=sep, nrows=1)
            if len(df_test.columns) > 1:
                df = pd.read_csv(
                    file_path,
                    sep=sep,
                    on_bad_lines='skip',
                    low_memory=False,
                    encoding='utf-8',
                    nrows=nrows
                )
                break
        except Exception as e:
            print(f"Erreur avec le séparateur {sep}: {str(e)}")
            continue
    
    if df is None:
        raise ValueError("Impossible de lire correctement le fichier CSV")
    
    print("Colonnes disponibles dans le fichier :")
    print(df.columns.tolist())
    
    numeric_columns = []
    
    columns_to_check = sorted(
        df.columns,
        key=lambda x: any(keyword in x.lower() for keyword in PRIORITY_KEYWORDS),
        reverse=True
    )
    
    for col in columns_to_check:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_columns.append(col)
                print(f"-> Colonne {col} déjà numérique")
                continue
            
            cleaned_series = (df[col].astype(str)
                            .replace(['', 'NA', 'N/A', 'nan', 'NaN', 'None',
                                    '<NA>', 'undefined', '?'], np.nan)
                            .str.replace(',', '.')
                            .str.strip()
                            .str.replace(r'[^\d.-]', '', regex=True))
            
            numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
            valid_count = numeric_series.notna().sum()
            valid_ratio = valid_count / len(df)
            
            print(f"Colonne {col}: {valid_count} valeurs numériques "
                  f"valides ({valid_ratio:.2%})")
            
            if valid_ratio > VALID_RATIO_THRESHOLD:
                df[col] = numeric_series
                numeric_columns.append(col)
                print(f"-> Colonne {col} retenue comme numérique")
        except Exception as e:
            print(f"Erreur lors du traitement de la colonne {col}: {str(e)}")
            continue
    
    print("\nColonnes numériques identifiées :")
    print(numeric_columns)
    
    if not numeric_columns:
        raise ValueError("Aucune colonne numérique n'a été trouvée dans le fichier")
    
    df_numeric = df[numeric_columns].copy()
    
    for column in df_numeric.columns:
        df_numeric[column] = df_numeric[column].replace([np.inf, -np.inf], np.nan)
        median_value = df_numeric[column].median()
        
        if pd.isna(median_value):
            median_value = 0
        
        df_numeric[column] = df_numeric[column].fillna(median_value)
        
        print(f"\nStatistiques pour {column}:")
        print(f"Nombre de valeurs non-NaN : {df_numeric[column].count()}")
        print(f"Médiane : {median_value}")
    
    if len(df_numeric) == 0:
        raise ValueError("Toutes les lignes ont été supprimées pendant le nettoyage")
    
    return df_numeric


def plot_scaling_comparison(
    df_original: pd.DataFrame,
    df_standard: pd.DataFrame,
    df_minmax: pd.DataFrame,
    df_robust: pd.DataFrame
) -> None:
    """
    Visualise la comparaison des différentes méthodes de scaling.
    
    Args:
        df_original: DataFrame original
        df_standard: DataFrame après StandardScaler
        df_minmax: DataFrame après MinMaxScaler
        df_robust: DataFrame après RobustScaler
    """
    sample_columns = df_original.columns[:3]
    
    for column in sample_columns:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        sns.histplot(data=df_original, x=column, ax=ax1, bins=30)
        ax1.set_title(f'{column} (Original)')
        
        sns.histplot(data=df_standard, x=column, ax=ax2, bins=30)
        ax2.set_title(f'{column} (StandardScaler)')
        
        sns.histplot(data=df_minmax, x=column, ax=ax3, bins=30)
        ax3.set_title(f'{column} (MinMaxScaler)')
        
        sns.histplot(data=df_robust, x=column, ax=ax4, bins=30)
        ax4.set_title(f'{column} (RobustScaler)')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\nStatistiques pour {column}:")
        print("-" * 50)
        stats_df = pd.DataFrame({
            'Original': df_original[column].describe(),
            'StandardScaler': df_standard[column].describe(),
            'MinMaxScaler': df_minmax[column].describe(),
            'RobustScaler': df_robust[column].describe()
        })
        print(stats_df)
    
    plot_correlation_comparison(df_original, df_standard, df_minmax, df_robust)


def analyze_scaling_impact(df_original: pd.DataFrame, 
                         df_scaled: pd.DataFrame, 
                         column: str) -> None:
    """
    Analyse l'impact du scaling sur une colonne.
    
    Args:
        df_original: DataFrame original
        df_scaled: DataFrame après scaling
        column: Nom de la colonne à analyser
    """
    print(f"\nAnalyse pour {column}:")
    print("-" * 50)
    
    # Statistiques descriptives
    print("Statistiques descriptives :")
    print(pd.DataFrame({
        'Original': df_original[column].describe(),
        'Scaled': df_scaled[column].describe()
    }))
    
    # Test de normalité
    _, p_value_original = stats.normaltest(df_original[column])
    _, p_value_scaled = stats.normaltest(df_scaled[column])
    print(f"\nTest de normalité (p-value):")
    print(f"Original: {p_value_original:.4f}")
    print(f"Scaled: {p_value_scaled:.4f}")
    
    # Calcul des outliers (méthode IQR)
    def count_outliers(series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return ((series < lower_bound) | (series > upper_bound)).sum()
    
    outliers_original = count_outliers(df_original[column])
    outliers_scaled = count_outliers(df_scaled[column])
    print(f"\nNombre d'outliers:")
    print(f"Original: {outliers_original}")
    print(f"Scaled: {outliers_scaled}")


def plot_correlation_comparison(df_original, df_standard, df_minmax, df_robust):
    """
    Visualise la comparaison des corrélations de manière plus lisible.
    """
    # Sélectionner les variables les plus importantes/pertinentes
    important_vars = [
        'energy-kcal_100g',
        'proteins_100g',
        'carbohydrates_100g',
        'sugars_100g',
        'fat_100g',
        'fiber_100g',
        'salt_100g',
        'nutrition-score-fr_100g'
    ]
    
    # Créer une figure avec 2x2 sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    fig.suptitle('Comparaison des matrices de corrélation selon la méthode de scaling', fontsize=16)
    
    # Pour chaque méthode de scaling
    datasets = {
        'Original': df_original[important_vars],
        'StandardScaler': df_standard[important_vars],
        'MinMaxScaler': df_minmax[important_vars],
        'RobustScaler': df_robust[important_vars]
    }
    
    for (title, data), ax in zip(datasets.items(), axes.ravel()):
        # Calculer la matrice de corrélation
        corr = data.corr()
        
        # Créer la heatmap
        sns.heatmap(corr, 
                    annot=True,  # Afficher les valeurs
                    fmt='.2f',   # Format à 2 décimales
                    cmap='RdBu_r',  # Rouge-Blanc-Bleu
                    vmin=-1, vmax=1,  # Échelle fixe
                    ax=ax)
        
        ax.set_title(f'Corrélation ({title})')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # Ajouter une analyse des changements de corrélation
    print("\nAnalyse des changements de corrélation :")
    print("----------------------------------------")
    
    original_corr = df_original[important_vars].corr()
    
    for method, scaled_df in [('StandardScaler', df_standard), 
                            ('MinMaxScaler', df_minmax), 
                            ('RobustScaler', df_robust)]:
        scaled_corr = scaled_df[important_vars].corr()
        print(f"\nImpact du {method} sur les corrélations :")
        
        # Identifier les changements significatifs
        changes = (scaled_corr - original_corr).abs()
        significant_changes = changes[changes > 0.1]
        
        if significant_changes.size > 0:
            print("Changements significatifs détectés entre :")
            for i, j in zip(*np.where(changes > 0.1)):
                if i < j:  # Éviter les doublons
                    var1, var2 = important_vars[i], important_vars[j]
                    print(f"- {var1} et {var2}: {original_corr.iloc[i,j]:.2f} → {scaled_corr.iloc[i,j]:.2f}")
        else:
            print("Pas de changement significatif dans les corrélations")


def test_scaling_methods(input_data: Union[str, pd.DataFrame]) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Teste et analyse les méthodes de scaling les plus adaptées.
    
    Args:
        input_data: Chemin vers le fichier CSV ou DataFrame pandas
        
    Returns:
        Tuple[DataFrame, DataFrame, DataFrame]: DataFrames scalés
        (standard, minmax, robust)
    """
    df = (load_and_prepare_data(input_data) if isinstance(input_data, str)
          else input_data)
    
    print("Analyse des variables et recommandations de scaling :")
    print("=" * 50)
    
    scaler_standard = StandardScaler()
    scaler_minmax = MinMaxScaler()
    scaler_robust = RobustScaler()
    
    df_standard = pd.DataFrame(
        scaler_standard.fit_transform(df),
        columns=df.columns
    )
    df_minmax = pd.DataFrame(
        scaler_minmax.fit_transform(df),
        columns=df.columns
    )
    df_robust = pd.DataFrame(
        scaler_robust.fit_transform(df),
        columns=df.columns
    )
    
    plot_scaling_comparison(df, df_standard, df_minmax, df_robust)
    
    return df_standard, df_minmax, df_robust


def plot_scaling_results(df, standard_vars, robust_vars, minmax_vars):
    """
    Visualise les résultats du scaling pour chaque groupe de variables.
    """
    # Visualisations et comparaisons...
    pass 