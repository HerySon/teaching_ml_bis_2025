from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from ..features.selection import analyze_and_select_features
from ..features.processing import analyze_correlations

def prepare_features_for_clustering(df: pd.DataFrame, 
                                  max_categories: int = 30, 
                                  min_unique_ratio: float = 0.05) -> pd.DataFrame:
    """
    Prépare les features pour la clusterisation en suivant une approche structurée.
    
    Args:
        df: DataFrame original
        max_categories: Nombre maximum de catégories autorisées pour les variables catégorielles
        min_unique_ratio: Ratio minimum de valeurs uniques pour les variables numériques
    
    Returns:
        DataFrame préparé pour la clusterisation
    """
    print("1. Prétraitement initial du DataFrame")
    df_clean, preprocess_info = analyze_and_select_features(
        df,
        max_categories=max_categories,
        min_unique_ratio=min_unique_ratio
    )
    
    print("\n2. Résumé des colonnes conservées:")
    print(f"- Numériques: {len(preprocess_info['feature_types']['numeric'])}")
    print(f"- Catégorielles ordinales: {len(preprocess_info['feature_types']['ordinal'])}")
    print(f"- Catégorielles nominales: {len(preprocess_info['feature_types']['nominal'])}")
    
    print("\n3. Colonnes numériques optimisées (downcasting):")
    for col, old_type, new_type in preprocess_info['downcasted_columns']:
        print(f"- {col}: {old_type} -> {new_type}")
    
    print("\n4. Colonnes catégorielles filtrées (> {max_categories} catégories supprimées):")
    categorical_dropped = [col for col, reason in preprocess_info['dropped_columns'] 
                         if reason == 'too_many_categories']
    for col in categorical_dropped:
        print(f"- {col}")
    
    print("\n5. Analyse des corrélations entre variables numériques")
    correlations = analyze_correlations(df_clean, plot=True)
    if correlations['strong_correlations']:
        print("\nVariables fortement corrélées à examiner:")
        for corr in correlations['strong_correlations']:
            print(f"- {corr['var1']} - {corr['var2']}: {corr['correlation']:.2f}")
    
    return df_clean 