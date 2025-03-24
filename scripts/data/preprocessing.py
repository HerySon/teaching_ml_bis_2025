from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd

def handle_duplicates(df: pd.DataFrame, strategy: str = 'analyze') -> Tuple[pd.DataFrame, Dict]:
    """
    Gère les doublons selon différentes stratégies.
    
    Args:
        df: DataFrame à nettoyer
        strategy: 'analyze', 'remove_all', 'keep_first', 'keep_last', 'aggregate'

    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame nettoyé et informations sur les doublons
    """
    info = {
        'initial_shape': df.shape,
        'duplicate_count': df.duplicated().sum(),
        'duplicate_rows': None,
        'action_taken': None
    }
    
    if strategy == 'analyze':
        duplicates = df[df.duplicated(keep=False)]
        info['duplicate_rows'] = duplicates
        info['duplicate_patterns'] = {
            'full_duplicates': df.duplicated().sum(),
            'partial_duplicates': {
                col: df.duplicated(subset=[col]).sum()
                for col in df.columns
            }
        }
        return df, info
    
    # Initialiser df_clean avant toute opération
    df_clean = df.copy()
    
    if strategy == 'remove_all':
        df_clean = df_clean.drop_duplicates()
        info['action_taken'] = 'removed_all_duplicates'
    elif strategy == 'keep_first':
        df_clean = df_clean.drop_duplicates(keep='first')
        info['action_taken'] = 'kept_first_occurrence'
    elif strategy == 'keep_last':
        df_clean = df_clean.drop_duplicates(keep='last')
        info['action_taken'] = 'kept_last_occurrence'
    elif strategy == 'aggregate':
        numeric_cols = df_clean.select_dtypes(include=np.number).columns
        categorical_cols = df_clean.select_dtypes(exclude=np.number).columns
        
        agg_rules = {
            **{col: 'mean' for col in numeric_cols},
            **{col: lambda x: x.mode().iloc[0] if not x.mode().empty else None 
               for col in categorical_cols}
        }
        
        df_clean = df_clean.groupby(df_clean.index).agg(agg_rules)
        info['action_taken'] = 'aggregated_duplicates'
    
    info['final_shape'] = df_clean.shape
    info['removed_rows'] = info['initial_shape'][0] - info['final_shape'][0]
    
    return df_clean, info

def process_categorical_columns(
    df: pd.DataFrame, 
    max_categories: int, 
    ordinal_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Traite les colonnes catégorielles du DataFrame de manière optimisée.
    
    Args:
        df: DataFrame à traiter
        max_categories: Nombre maximum de catégories autorisées
        ordinal_columns: Liste des colonnes ordinales
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame traité et informations sur les colonnes
    """
    info = {
        'categorical_ordinal': [],
        'categorical_nominal': [],
        'dropped_columns': []
    }
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if categorical_columns.empty:
        return df, info
    
    n_unique = df[categorical_columns].nunique()
    valid_columns = n_unique[n_unique <= max_categories].index
    
    ordinal_set = set(ordinal_columns or [])
    for col in valid_columns:
        if col in ordinal_set:
            df[col] = pd.Categorical(df[col], ordered=True)
            info['categorical_ordinal'].append(col)
        else:
            df[col] = pd.Categorical(df[col], ordered=False)
            info['categorical_nominal'].append(col)
    
    columns_to_drop = n_unique[n_unique > max_categories].index
    info['dropped_columns'].extend((col, 'too_many_categories') for col in columns_to_drop)
    
    if len(columns_to_drop) > 0:
        df = df.drop(columns=columns_to_drop)
    
    return df, info 