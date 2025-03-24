from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from .processing import (
    detect_ordinal_nature, 
    get_optimal_numeric_type,
    process_numeric_columns,
    process_categorical_columns,
    analyze_correlations
)

def analyze_and_select_features(df: pd.DataFrame, 
                              max_categories: int = 30,
                              min_unique_ratio: float = 0.01,
                              missing_threshold: float = 0.5,
                              correlation_threshold: float = 0.7) -> Tuple[pd.DataFrame, Dict]:
    """
    Détecte et sélectionne automatiquement les colonnes pertinentes du DataFrame.
    
    Args:
        df: DataFrame d'entrée
        max_categories: Nombre maximum de catégories pour les variables catégorielles
        min_unique_ratio: Ratio minimum de valeurs uniques pour les variables numériques
        missing_threshold: Seuil de valeurs manquantes acceptables
        correlation_threshold: Seuil pour les corrélations fortes
        
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
        'downcasted_columns': [],
        'correlation_info': {
            'variables_to_drop': [],
            'correlation_matrix': None
        }
    }
    
    df_clean = df.copy()
    
    # Première passe : filtrage des colonnes avec trop de valeurs manquantes
    for column in df.columns:
        missing_ratio = df[column].isna().mean()
        if missing_ratio > missing_threshold:
            feature_info['dropped_columns'].append((column, 'too_many_missing'))
    
    # Suppression des colonnes avec trop de valeurs manquantes
    columns_to_drop = [col for col, reason in feature_info['dropped_columns'] if reason == 'too_many_missing']
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
    
    # Traitement des colonnes numériques
    df_numeric, numeric_info = process_numeric_columns(df_clean, min_unique_ratio=min_unique_ratio)
    feature_info['feature_types']['numeric'].extend(numeric_info['numeric_columns'])
    feature_info['downcasted_columns'].extend(numeric_info['downcasted_columns'])
    feature_info['dropped_columns'].extend(
        (col, 'low_variance') for col in numeric_info.get('dropped_columns', [])
    )
    
    # Traitement des colonnes catégorielles
    df_categorical, categorical_info = process_categorical_columns(
        df_numeric, 
        max_categories=max_categories,
        detect_ordinal=True
    )
    feature_info['feature_types']['ordinal'].extend(categorical_info['categorical_ordinal'])
    feature_info['feature_types']['nominal'].extend(categorical_info['categorical_nominal'])
    feature_info['dropped_columns'].extend(categorical_info['dropped_columns'])
    
    # Mise à jour du DataFrame nettoyé
    df_clean = df_categorical
    
    # Deuxième passe : analyse des corrélations et suppression des variables corrélées
    if feature_info['feature_types']['numeric']:
        correlation_info = analyze_correlations(
            df_clean, 
            threshold=correlation_threshold,
            plot=True
        )
        feature_info['correlation_info'] = correlation_info
        
        # Supprimer les variables fortement corrélées
        if correlation_info['variables_to_drop']:
            df_clean = df_clean.drop(columns=correlation_info['variables_to_drop'])
            for var in correlation_info['variables_to_drop']:
                feature_info['dropped_columns'].append((var, 'high_correlation'))
                if var in feature_info['feature_types']['numeric']:
                    feature_info['feature_types']['numeric'].remove(var)
    
    print(f"Colonnes numériques : {len(feature_info['feature_types']['numeric'])}")
    print(f"Colonnes ordinales : {len(feature_info['feature_types']['ordinal'])}")
    print(f"Colonnes nominales : {len(feature_info['feature_types']['nominal'])}")
    print(f"Colonnes supprimées : {len(feature_info['dropped_columns'])}")
    print(f"Colonnes optimisées : {len(feature_info['downcasted_columns'])}")
    
    if feature_info.get('correlation_info') and feature_info['correlation_info'].get('variables_to_drop'):
        print("\nVariables supprimées pour cause de forte corrélation:")
        for var in feature_info['correlation_info']['variables_to_drop']:
            print(f"- {var}")
    
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