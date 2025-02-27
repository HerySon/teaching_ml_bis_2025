import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from .processing import detect_ordinal_nature, get_optimal_numeric_type

def analyze_and_select_features(df: pd.DataFrame, 
                              max_categories: int = 30,
                              min_unique_ratio: float = 0.01,
                              missing_threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
    """
    Détecte et sélectionne automatiquement les colonnes pertinentes du DataFrame.
    
    Args:
        df: DataFrame d'entrée
        max_categories: Nombre maximum de catégories pour les variables catégorielles
        min_unique_ratio: Ratio minimum de valeurs uniques pour les variables numériques
        missing_threshold: Seuil de valeurs manquantes acceptables
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame nettoyé et informations sur les features
    """
    feature_info = {
        'feature_types': {
            'numeric': [],
            'ordinal': [],
            'nominal': []
        },
        'dropped_columns': [],
        'downcasted_columns': []
    }
    
    df_clean = df.copy()
    
    for column in df.columns:
        # Vérification des valeurs manquantes
        missing_ratio = df[column].isna().mean()
        if missing_ratio > missing_threshold:
            feature_info['dropped_columns'].append((column, 'too_many_missing'))
            continue
            
        # Analyse du type de données
        dtype = df[column].dtype
        n_unique = df[column].nunique()
        unique_ratio = n_unique / len(df)
        
        # Traitement des variables numériques
        if pd.api.types.is_numeric_dtype(dtype):
            original_dtype = df[column].dtype
            downcasted = pd.to_numeric(df[column], downcast='integer')
            
            if downcasted.dtype != original_dtype:
                df_clean[column] = downcasted
                feature_info['downcasted_columns'].append(
                    (column, str(original_dtype), str(downcasted.dtype))
                )
            
            if unique_ratio >= min_unique_ratio:
                feature_info['feature_types']['numeric'].append(column)
            else:
                feature_info['dropped_columns'].append((column, 'low_variance'))
                
        # Traitement des variables catégorielles
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            if n_unique > max_categories:
                feature_info['dropped_columns'].append((column, 'too_many_categories'))
                continue
            
            if detect_ordinal_nature(df[column]):
                df_clean[column] = pd.Categorical(df[column], ordered=True)
                feature_info['feature_types']['ordinal'].append(column)
            else:
                df_clean[column] = pd.Categorical(df[column], ordered=False)
                feature_info['feature_types']['nominal'].append(column)
    
    # Suppression des colonnes filtrées
    columns_to_drop = [col for col, _ in feature_info['dropped_columns']]
    df_clean = df_clean.drop(columns=columns_to_drop)
    
    print(f"Colonnes numériques : {len(feature_info['feature_types']['numeric'])}")
    print(f"Colonnes ordinales : {len(feature_info['feature_types']['ordinal'])}")
    print(f"Colonnes nominales : {len(feature_info['feature_types']['nominal'])}")
    print(f"Colonnes supprimées : {len(feature_info['dropped_columns'])}")
    print(f"Colonnes optimisées : {len(feature_info['downcasted_columns'])}")
    
    return df_clean, feature_info

def analyze_feature_importance(df: pd.DataFrame, 
                             variance_weight: float = 0.3,
                             missing_weight: float = 0.3,
                             unique_weight: float = 0.4) -> Dict:
    """
    Analyse l'importance des variables selon plusieurs critères.
    
    Args:
        df: DataFrame à analyser
        variance_weight: Poids de la variance dans le score
        missing_weight: Poids des valeurs manquantes dans le score
        unique_weight: Poids du ratio de valeurs uniques dans le score
        
    Returns:
        Dict: Informations sur l'importance des variables
    """
    importance_metrics = {
        'missing_values': df.isnull().mean(),
        'variance': df.select_dtypes(include=np.number).var(),
        'unique_ratio': df.nunique() / len(df),
        'relevance_scores': {}
    }
    
    for column in df.columns:
        series = df[column]
        
        # Score des valeurs manquantes
        missing_ratio = series.isna().mean()
        missing_score = 1 - missing_ratio
        
        # Score de valeurs uniques
        unique_ratio = series.nunique() / len(series)
        unique_score = min(unique_ratio, 0.5) * 2
        
        # Score de variance
        if pd.api.types.is_numeric_dtype(series):
            variance = series.var()
            max_val = series.max()
            min_val = series.min()
            range_val = max_val - min_val if max_val != min_val else 1
            variance_score = min(variance / (range_val ** 2), 1)
        else:
            variance_score = unique_score
        
        total_score = (
            variance_score * variance_weight +
            missing_score * missing_weight +
            unique_score * unique_weight
        )
        
        importance_metrics['relevance_scores'][column] = {
            'total_score': total_score,
            'variance_score': variance_score,
            'missing_score': missing_score,
            'unique_score': unique_score
        }
    
    return importance_metrics

def select_relevant_features(df: pd.DataFrame, 
                           missing_threshold: float = 0.3,
                           variance_threshold: float = 0.01,
                           unique_ratio_threshold: float = 0.01) -> List[str]:
    """
    Sélectionne automatiquement les variables pertinentes.
    
    Args:
        df: DataFrame à analyser
        missing_threshold: Seuil maximum de valeurs manquantes
        variance_threshold: Seuil minimum de variance
        unique_ratio_threshold: Seuil minimum de ratio de valeurs uniques
        
    Returns:
        List[str]: Liste des features sélectionnées
    """
    metrics = analyze_feature_importance(df)
    selected_features = []
    
    for column in df.columns:
        if metrics['missing_values'][column] > missing_threshold:
            continue
            
        if df[column].dtype in [np.number]:
            if metrics['variance'].get(column, 0) < variance_threshold:
                continue
                
        if metrics['unique_ratio'][column] < unique_ratio_threshold:
            continue
            
        selected_features.append(column)
    
    return selected_features 