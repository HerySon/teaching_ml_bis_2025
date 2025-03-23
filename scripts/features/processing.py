from typing import Dict, List, Tuple, Optional
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.stats import entropy
from ..data.analysis import analyze_data_quality
from ..data.correlation_analysis import analyze_correlations

def get_optimal_numeric_type(min_val: float, max_val: float, has_decimals: bool) -> np.dtype:
    """Détermine le type numérique optimal pour une colonne."""
    if pd.isna(min_val) or pd.isna(max_val) or np.isinf(min_val) or np.isinf(max_val):
        return np.float64
    
    if has_decimals:
        return np.float32 if min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max else np.float64
    
    # Déterminer le type optimal pour les entiers
    dtype = np.int64  # Type par défaut
    
    if min_val >= 0:
        if max_val <= np.iinfo(np.uint8).max:
            dtype = np.uint8
        elif max_val <= np.iinfo(np.uint16).max:
            dtype = np.uint16
        elif max_val <= np.iinfo(np.uint32).max:
            dtype = np.uint32
        else:
            dtype = np.uint64
    else:
        if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
            dtype = np.int8
        elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
            dtype = np.int16
        elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
            dtype = np.int32
    
    return dtype

def optimize_numeric_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Optimise les types numériques du DataFrame."""
    info = {'optimized_columns': []}
    df_optimized = df.copy()
    
    for col in df.select_dtypes(include=np.number).columns:
        try:
            series = df[col]
            if not series.isna().any():
                has_decimals = not np.all(series == series.astype(int))
                optimal_type = get_optimal_numeric_type(series.min(), series.max(), has_decimals)
                df_optimized[col] = series.astype(optimal_type)
                info['optimized_columns'].append((col, str(series.dtype), str(optimal_type)))
        except (ValueError, TypeError, OverflowError) as e:
            print(f"Erreur lors de l'optimisation de {col}: {str(e)}")
            continue
    
    return df_optimized, info

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

def enhance_ordinal_detection(series: pd.Series) -> bool:
    """
    Amélioration de la détection des variables ordinales.
    
    Args:
        series: Série à analyser
        
    Returns:
        bool: True si la variable est probablement ordinale
    """
    # Vérification si c'est déjà détecté comme ordinal
    if detect_ordinal_nature(series):
        return True
    
    # Détection des séquences numériques
    unique_values = pd.to_numeric(series.dropna(), errors='coerce')
    if not unique_values.isna().all():
        sorted_values = unique_values.sort_values()
        if all(sorted_values.diff().dropna() > 0):  # Séquence strictement croissante
            return True
    
    # Détection des dates et temps
    try:
        pd.to_datetime(series, errors='raise')
        return True
    except (ValueError, TypeError):
        pass
    
    # Patterns ordinaux supplémentaires
    additional_patterns = [
        ['très faible', 'faible', 'moyen', 'élevé', 'très élevé'],
        ['débutant', 'intermédiaire', 'avancé', 'expert'],
        ['jamais', 'parfois', 'souvent', 'toujours'],
        ['nul', 'passable', 'assez bien', 'bien', 'très bien', 'excellent'],
        ['pas du tout d\'accord', 'pas d\'accord', 'neutre', 'd\'accord', 'tout à fait d\'accord']
    ]
    
    values_str = series.astype(str).str.lower()
    for pattern in additional_patterns:
        if all(any(p in v for p in pattern) for v in values_str):
            return True
    
    return False

def infer_ordinal_order(series: pd.Series) -> List[str]:
    """
    Détermine l'ordre naturel des catégories ordinales.
    
    Args:
        series: Série à analyser
        
    Returns:
        List[str]: Liste des catégories dans leur ordre naturel
    """
    unique_values = series.dropna().unique()
    
    # Essai de conversion en numérique
    try:
        numeric_values = pd.to_numeric(unique_values)
        return [str(x) for x in sorted(numeric_values)]
    except (ValueError, TypeError):
        pass
    
    # Essai de conversion en dates
    try:
        date_values = pd.to_datetime(unique_values)
        return [str(x) for x in sorted(date_values)]
    except (ValueError, TypeError):
        pass
    
    # Patterns ordinaux connus
    known_patterns = {
        'niveau': ['débutant', 'intermédiaire', 'avancé', 'expert'],
        'accord': ['pas du tout d\'accord', 'pas d\'accord', 'neutre', 'd\'accord', 'tout à fait d\'accord'],
        'qualité': ['mauvais', 'passable', 'moyen', 'bon', 'excellent'],
        'fréquence': ['jamais', 'rarement', 'parfois', 'souvent', 'toujours'],
        'intensité': ['très faible', 'faible', 'moyen', 'fort', 'très fort']
    }
    
    # Recherche du pattern qui correspond le mieux
    values_set = set(v.lower() for v in unique_values)
    for pattern in known_patterns.values():
        pattern_set = set(p.lower() for p in pattern)
        if values_set.issubset(pattern_set):
            return [v for v in pattern if v.lower() in values_set]
    
    # Si aucun pattern ne correspond, retour des valeurs dans l'ordre alphabétique
    return sorted(unique_values)

def optimize_categorical_memory(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Optimisation de la mémoire pour les variables catégorielles.
    
    Args:
        df: DataFrame à optimiser
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame optimisé et informations sur l'optimisation
    """
    df_optimized = df.copy()
    optimization_info = {
        'memory_savings': {},
        'encoding_changes': {},
        'rare_categories': {}
    }
    
    for column in df.select_dtypes(include=['object', 'category']).columns:
        initial_memory = df[column].memory_usage(deep=True)
        
        # Obtenir les catégories uniques non nulles
        unique_non_null = df[column].dropna().unique()
        n_unique = len(unique_non_null)
        
        # Optimisation basée sur le nombre de catégories uniques
        if n_unique <= 2:
            df_optimized[column] = pd.Categorical(df[column], categories=unique_non_null)
        elif n_unique <= 255:
            df_optimized[column] = pd.Categorical(df[column], categories=unique_non_null)
        else:
            # Pour les colonnes avec beaucoup de catégories, on utilise un encodage efficace
            df_optimized[column] = pd.Categorical(df[column])
        
        final_memory = df_optimized[column].memory_usage(deep=True)
        savings = initial_memory - final_memory
        
        optimization_info['memory_savings'][column] = {
            'initial_bytes': initial_memory,
            'final_bytes': final_memory,
            'savings_bytes': savings,
            'savings_percent': (savings / initial_memory) * 100
        }
        
        optimization_info['encoding_changes'][column] = {
            'initial_dtype': str(df[column].dtype),
            'final_dtype': str(df_optimized[column].dtype),
            'n_categories': n_unique,
            'has_null': df[column].isna().any()
        }
    
    return df_optimized, optimization_info

def handle_rare_categories(
    df: pd.DataFrame,
    threshold: float = 0.01,
    strategy: str = 'group',
    other_label: str = 'Other'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Gestion des catégories rares dans les variables catégorielles.
    
    Args:
        df: DataFrame à traiter
        threshold: Seuil minimum de fréquence pour une catégorie
        strategy: 'group' pour regrouper, 'drop' pour supprimer
        other_label: Étiquette pour le groupe 'Other'
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame traité et informations sur les modifications
    """
    df_processed = df.copy()
    info = {
        'modified_columns': {},
        'dropped_categories': {},
        'grouped_categories': {}
    }
    
    for column in df.select_dtypes(include=['object', 'category']).columns:
        value_counts = df[column].value_counts(normalize=True)
        rare_categories = value_counts[value_counts < threshold].index
        
        if len(rare_categories) > 0:
            if strategy == 'group':
                # Convertir en catégorie si ce n'est pas déjà le cas
                if not isinstance(df_processed[column].dtype, pd.CategoricalDtype):
                    df_processed[column] = pd.Categorical(df_processed[column])
                
                # Créer un dictionnaire de mapping pour les catégories
                category_mapping = {cat: other_label if cat in rare_categories else cat 
                                 for cat in df_processed[column].unique() if pd.notna(cat)}
                
                # Appliquer le mapping avec rename_categories
                df_processed[column] = df_processed[column].cat.rename_categories(category_mapping)
                
                info['grouped_categories'][column] = {
                    'n_grouped': len(rare_categories),
                    'grouped_categories': list(rare_categories)
                }
            elif strategy == 'drop':
                df_processed = df_processed[~df_processed[column].isin(rare_categories)]
                info['dropped_categories'][column] = {
                    'n_dropped': len(rare_categories),
                    'dropped_categories': list(rare_categories)
                }
            
            info['modified_columns'][column] = {
                'initial_categories': df[column].nunique(),
                'final_categories': df_processed[column].nunique(),
                'threshold': threshold,
                'strategy': strategy
            }
    
    return df_processed, info

def analyze_category_distributions(df: pd.DataFrame) -> Dict:
    """
    Analyse la distribution des catégories dans les variables catégorielles.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Dict: Informations sur les distributions des catégories
    """
    distribution_info = {}
    
    for column in df.select_dtypes(include=['object', 'category']).columns:
        value_counts = df[column].value_counts()
        frequencies = value_counts / len(df)
        
        # Calcul de l'entropie de la distribution
        category_entropy = entropy(frequencies)
        
        # Calcul du déséquilibre des classes
        imbalance_ratio = value_counts.max() / value_counts.min() if len(value_counts) > 0 else 0
        
        distribution_info[column] = {
            'n_categories': len(value_counts),
            'most_common': value_counts.head(5).to_dict(),
            'least_common': value_counts.tail(5).to_dict(),
            'entropy': category_entropy,
            'imbalance_ratio': imbalance_ratio,
            'missing_ratio': df[column].isna().mean(),
            'unique_ratio': df[column].nunique() / len(df),
            'value_counts': value_counts.to_dict()  # Ajout des counts complets
        }
    
    return distribution_info

def memory_usage_report(df: pd.DataFrame) -> Dict:
    """
    Génère un rapport détaillé sur l'utilisation de la mémoire du DataFrame.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Dict: Rapport d'utilisation de la mémoire
    """
    total_memory = df.memory_usage(deep=True).sum()
    column_memory = df.memory_usage(deep=True)
    
    report = {
        'total_memory_bytes': total_memory,
        'total_memory_mb': total_memory / 1024 / 1024,
        'memory_by_dtype': {},
        'column_details': {},
        'optimization_suggestions': []
    }
    
    # Analyse par type de données
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns
        memory = df[cols].memory_usage(deep=True).sum()
        report['memory_by_dtype'][str(dtype)] = {
            'memory_bytes': memory,
            'memory_mb': memory / 1024 / 1024,
            'percentage': (memory / total_memory) * 100,
            'n_columns': len(cols)
        }
    
    # Analyse par colonne
    for column in df.columns:
        memory = df[column].memory_usage(deep=True)
        dtype = df[column].dtype
        n_unique = df[column].nunique() if dtype != 'datetime64[ns]' else None
        
        report['column_details'][column] = {
            'memory_bytes': memory,
            'memory_mb': memory / 1024 / 1024,
            'percentage': (memory / total_memory) * 100,
            'dtype': str(dtype),
            'n_unique': n_unique
        }
        
        # Suggestions d'optimisation
        if dtype == 'object':
            if n_unique and n_unique <= 255:
                report['optimization_suggestions'].append(
                    f"Column '{column}' could be converted to categorical"
                )
        elif dtype == 'int64':
            if df[column].min() >= 0 and df[column].max() <= 255:
                report['optimization_suggestions'].append(
                    f"Column '{column}' could be downcasted to uint8"
                )
    
    return report

def filter_and_analyze_dataset(df: pd.DataFrame,
                             max_categories: int = 30,
                             missing_threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
    """Filtre et analyse le DataFrame."""
    info = {
        'dropped_columns': [],
        'correlation_info': {},
        'category_info': {}
    }
    
    # 1. Filtrer les colonnes avec trop de valeurs manquantes
    missing_cols = [col for col in df.columns if df[col].isna().mean() > missing_threshold]
    df_filtered = df.drop(columns=missing_cols)
    info['dropped_columns'].extend((col, 'too_many_missing') for col in missing_cols)
    
    # 2. Optimiser les types numériques
    df_filtered, numeric_info = optimize_numeric_columns(df_filtered)
    info.update(numeric_info)
    
    # 3. Gérer les colonnes catégorielles
    cat_cols = df_filtered.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        if df_filtered[col].nunique() > max_categories:
            df_filtered = df_filtered.drop(columns=[col])
            info['dropped_columns'].append((col, 'too_many_categories'))
    
    return df_filtered, info

def optimize_memory_usage(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Optimise l'utilisation de la mémoire du DataFrame.
    """
    info = {
        'initial_memory': df.memory_usage(deep=True).sum() / 1024**2,
        'optimized_columns': []
    }
    
    for column in df.columns:
        col_data = df[column]
        if pd.api.types.is_numeric_dtype(col_data):
            try:
                has_decimals = not np.all(col_data == col_data.astype(int))
                optimal_type = get_optimal_numeric_type(
                    col_data.min(), 
                    col_data.max(), 
                    has_decimals
                )
                df[column] = df[column].astype(optimal_type)
                info['optimized_columns'].append({
                    'column': column,
                    'original_type': str(col_data.dtype),
                    'new_type': str(optimal_type)
                })
            except (ValueError, TypeError, OverflowError) as e:
                print(f"Impossible d'optimiser {column}: {str(e)}")
    
    info['final_memory'] = df.memory_usage(deep=True).sum() / 1024**2
    info['memory_reduction_pct'] = (
        (info['initial_memory'] - info['final_memory']) / 
        info['initial_memory'] * 100
    )
    
    return df, info