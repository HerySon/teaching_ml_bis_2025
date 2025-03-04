import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from functools import lru_cache

def get_optimal_numeric_type(min_val: float, max_val: float, has_decimals: bool) -> np.dtype:
    """
    Détermine le type numérique optimal pour une plage de valeurs donnée.
    
    Args:
        min_val: Valeur minimale
        max_val: Valeur maximale
        has_decimals: Si True, utilise des types float
        
    Returns:
        np.dtype: Type de données optimal
    """
    if has_decimals:
        if min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
            return np.float32
        return np.float64
    
    if min_val >= 0:
        if max_val <= 255:
            return np.uint8
        elif max_val <= 65535:
            return np.uint16
        elif max_val <= 4294967295:
            return np.uint32
        return np.uint64
    else:
        if min_val >= -128 and max_val <= 127:
            return np.int8
        elif min_val >= -32768 and max_val <= 32767:
            return np.int16
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return np.int32
        return np.int64

def process_numeric_columns(df: pd.DataFrame, min_unique_ratio: float = 0.01) -> Tuple[pd.DataFrame, Dict]:
    """
    Traite les colonnes numériques du DataFrame.
    
    Args:
        df: DataFrame à traiter
        min_unique_ratio: Ratio minimum de valeurs uniques
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame traité et informations sur les colonnes
    """
    info = {
        'numeric_columns': [],
        'dropped_columns': [],
        'downcasted_columns': []
    }
    
    df_clean = df.copy()
    numeric_columns = df.select_dtypes(include=np.number).columns
    
    for column in numeric_columns:
        # Vérification du ratio de valeurs uniques
        unique_ratio = df[column].nunique() / len(df)
        if unique_ratio < min_unique_ratio:
            info['dropped_columns'].append((column, 'low_variance'))
            continue
            
        # Optimisation du type
        original_dtype = df[column].dtype
        series = df[column]
        has_nan = series.isna().any()
        
        if not has_nan:
            min_val = series.min()
            max_val = series.max()
            has_decimals = not np.all(series == series.astype(int))
            optimal_type = get_optimal_numeric_type(min_val, max_val, has_decimals)
            
            if optimal_type != original_dtype:
                try:
                    df_clean[column] = series.astype(optimal_type)
                    info['downcasted_columns'].append(
                        (column, str(original_dtype), str(optimal_type))
                    )
                except (OverflowError, ValueError):
                    pass
        
        info['numeric_columns'].append(column)
    
    # Suppression des colonnes si nécessaire
    columns_to_drop = [col for col, _ in info['dropped_columns']]
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
    
    return df_clean, info

def detect_ordinal_nature(series: pd.Series) -> bool:
    """
    Détection sophistiquée des variables ordinales.
    
    Args:
        series: Série à analyser
        
    Returns:
        bool: True si la variable est probablement ordinale
    """
    name_indicators = ['_level', '_grade', '_score', '_rating', 'grade_', 'level_', 'score_']
    if any(ind in str(series.name).lower() for ind in name_indicators):
        return True
    
    unique_values = series.dropna().unique()
    if all(str(x).replace('.', '').isdigit() for x in unique_values):
        return True
    
    ordinal_patterns = [
        ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'E+', 'E'],
        ['excellent', 'good', 'average', 'poor', 'bad'],
        ['high', 'medium', 'low'],
        ['1st', '2nd', '3rd', '4th', '5th'],
        ['beginner', 'intermediate', 'advanced', 'expert']
    ]
    
    values_str = series.astype(str).str.lower()
    for pattern in ordinal_patterns:
        pattern_lower = [p.lower() for p in pattern]
        if all(any(p in v for p in pattern_lower) for v in values_str):
            return True
    
    return False

def process_categorical_columns(
    df: pd.DataFrame, 
    max_categories: int, 
    detect_ordinal: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Traite les colonnes catégorielles du DataFrame.
    
    Args:
        df: DataFrame à traiter
        max_categories: Nombre maximum de catégories autorisées
        detect_ordinal: Si True, tente de détecter automatiquement les variables ordinales
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame traité et informations sur les colonnes
    """
    info = {
        'categorical_ordinal': [],
        'categorical_nominal': [],
        'dropped_columns': []
    }
    
    df_clean = df.copy()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    for column in categorical_columns:
        n_unique = df[column].nunique()
        if n_unique > max_categories:
            info['dropped_columns'].append((column, 'too_many_categories'))
            continue
        
        is_ordinal = detect_ordinal and detect_ordinal_nature(df[column])
        if is_ordinal:
            df_clean[column] = pd.Categorical(df[column], ordered=True)
            info['categorical_ordinal'].append(column)
        else:
            df_clean[column] = pd.Categorical(df[column], ordered=False)
            info['categorical_nominal'].append(column)
    
    # Suppression des colonnes si nécessaire
    columns_to_drop = [col for col, _ in info['dropped_columns']]
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
    
    return df_clean, info

def analyze_correlations(df: pd.DataFrame,
                        method: str = 'pearson',
                        threshold: float = 0.7,
                        plot: bool = True) -> Dict:
    """
    Analyse les corrélations entre variables numériques.
    
    Args:
        df: DataFrame à analyser
        method: Méthode de corrélation ('pearson', 'spearman', 'kendall')
        threshold: Seuil pour les corrélations fortes
        plot: Si True, affiche une heatmap des corrélations
    
    Returns:
        Dict: Informations sur les corrélations
    """
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr(method=method)
    
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                strong_correlations.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i,j]
                })
    
    if plot:
        try:
            from ..visualization import plot_correlations
            plot_correlations(df, corr_matrix.columns)
        except ImportError:
            print("Module de visualisation non disponible")
    
    return {
        'correlation_matrix': corr_matrix,
        'strong_correlations': strong_correlations
    } 