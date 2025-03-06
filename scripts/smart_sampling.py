"""
Module pour le sous-échantillonnage intelligent et stratifié des datasets.

Ce module fournit des fonctions pour analyser et sous-échantillonner des datasets
de manière intelligente en utilisant différentes stratégies, notamment le
sous-échantillonnage stratifié.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from collections import Counter


def _auto_select_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Sélectionne intelligemment les meilleures colonnes pour la stratification
    et l'analyse numérique.
    """
    # Colonnes à exclure automatiquement (par pattern)
    exclude_patterns = [
        '_t$',  # timestamps
        '_datetime$',  # dates
        'url',  # URLs
        'code',  # identifiants
        'creator',  # métadonnées
        'created',  # métadonnées
        'modified',  # métadonnées
        'updated',  # métadonnées
        'id'  # identifiants
    ]
    
    # Identification des types de colonnes
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Filtrage intelligent des colonnes numériques
    valid_numeric_cols = []
    for col in numeric_cols:
        # Vérifie si la colonne doit être exclue
        if any(pattern in col.lower() for pattern in exclude_patterns):
            continue
            
        # Analyse statistique de base
        non_null_ratio = df[col].notna().mean()
        unique_ratio = df[col].nunique() / len(df)
        
        # Conditions pour une bonne colonne numérique
        if (non_null_ratio > 0.3 and  # Au moins 30% de valeurs non nulles
            unique_ratio > 0.01 and    # Au moins 1% de valeurs uniques
            unique_ratio < 0.9 and     # Pas trop de valeurs uniques (évite les IDs)
            '_100g' in col):          # Privilégie les colonnes nutritionnelles
            valid_numeric_cols.append(col)
    
    # Filtrage intelligent des colonnes catégorielles
    valid_categorical_cols = []
    for col in categorical_cols:
        # Vérifie si la colonne doit être exclue
        if any(pattern in col.lower() for pattern in exclude_patterns):
            continue
            
        n_unique = df[col].nunique()
        non_null_ratio = df[col].notna().mean()
        
        # Conditions pour une bonne colonne catégorielle
        if (2 <= n_unique <= 50 and  # Entre 2 et 50 catégories
            non_null_ratio > 0.3 and  # Au moins 30% de valeurs non nulles
            any(key in col.lower() for key in ['grade', 'group', 'category', 'type'])):  # Colonnes de classification
            valid_categorical_cols.append(col)
    
    # Si aucune colonne catégorielle n'est trouvée, créer des bins sur une colonne numérique
    if not valid_categorical_cols and valid_numeric_cols:
        best_numeric = valid_numeric_cols[0]
        df[f'{best_numeric}_binned'] = pd.qcut(df[best_numeric].fillna(df[best_numeric].median()), 
                                              q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        valid_categorical_cols.append(f'{best_numeric}_binned')
    
    print("Colonnes sélectionnées pour la stratification:")
    for col in valid_categorical_cols:
        print(f"- {col}: {df[col].nunique()} catégories uniques")
    
    print("\nColonnes numériques sélectionnées:")
    for col in valid_numeric_cols:
        print(f"- {col}")
    
    return valid_categorical_cols, valid_numeric_cols


def smart_sample(
    df: pd.DataFrame,
    target_size: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Crée automatiquement un échantillon intelligent du dataset.
    
    Args:
        df: DataFrame source
        target_size: Taille souhaitée de l'échantillon (défaut: 25% des données)
        random_state: Graine aléatoire pour la reproductibilité
        verbose: Si True, affiche les informations d'analyse
    
    Returns:
        Tuple (DataFrame échantillonné, métriques et informations)
    """
    if target_size is None:
        target_size = len(df) // 4  # 25% par défaut
    
    # Sélection automatique des colonnes
    stratify_cols, numeric_cols = _auto_select_columns(df)
    
    # Création de la colonne de stratification combinée
    df = df.copy()
    if stratify_cols:
        df['combined_strata'] = df[stratify_cols].astype(str).agg('-'.join, axis=1)
    else:
        # Si pas de colonnes catégorielles, utiliser des bins sur la première colonne numérique
        if numeric_cols:
            df['combined_strata'] = pd.qcut(df[numeric_cols[0]], q=10, labels=False, duplicates='drop')
        else:
            raise ValueError("Aucune colonne appropriée trouvée pour la stratification")
    
    # Initialisation de StratifiedKFold
    n_splits = max(2, len(df) // target_size)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Sélection d'un fold
    X = df.drop('combined_strata', axis=1)
    y = df['combined_strata']
    
    for train_idx, test_idx in skf.split(X, y):
        sample = df.iloc[test_idx]
        if len(sample) >= target_size:
            break
    
    # Nettoyage
    sample = sample.drop('combined_strata', axis=1)
    
    # Calcul des métriques
    metrics = _calculate_metrics(df, sample, stratify_cols, numeric_cols)
    
    # Visualisations si verbose
    if verbose:
        _plot_distributions(df, sample, stratify_cols, numeric_cols)
    
    return sample, metrics


def _calculate_metrics(
    original_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    stratify_cols: List[str],
    numeric_cols: List[str]
) -> Dict:
    """Calcule les métriques de qualité de l'échantillonnage."""
    metrics = {
        'reduction_ratio': len(sampled_df) / len(original_df),
        'columns_used': {
            'stratification': stratify_cols,
            'numerical': numeric_cols
        }
    }
    
    # Métriques pour les colonnes de stratification
    for col in stratify_cols:
        orig_props = original_df[col].value_counts(normalize=True)
        sample_props = sampled_df[col].value_counts(normalize=True)
        diff = np.abs(orig_props - sample_props.reindex(orig_props.index).fillna(0)).mean()
        metrics[f"{col}_distribution_difference"] = diff
    
    # Métriques pour les colonnes numériques
    for col in numeric_cols:
        metrics[f"{col}_mean_difference"] = abs(
            original_df[col].mean() - sampled_df[col].mean()
        ) / original_df[col].std()
    
    return metrics


def _plot_distributions(
    original_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    stratify_cols: List[str],
    numeric_cols: List[str]
) -> None:
    """Génère les visualisations des distributions."""
    # Plot pour les colonnes de stratification
    for col in stratify_cols:
        plt.figure(figsize=(12, 4))
        
        # Distribution des catégories
        plt.subplot(1, 2, 1)
        pd.DataFrame({
            'Original': original_df[col].value_counts(normalize=True),
            'Échantillon': sampled_df[col].value_counts(normalize=True)
        }).plot(kind='bar')
        plt.title(f'Distribution - {col}')
        plt.xticks(rotation=45)
        
        # Distribution cumulative
        plt.subplot(1, 2, 2)
        pd.DataFrame({
            'Original': original_df[col].value_counts(normalize=True).cumsum(),
            'Échantillon': sampled_df[col].value_counts(normalize=True).cumsum()
        }).plot()
        plt.title(f'Distribution cumulative - {col}')
        
        plt.tight_layout()
        plt.show()
    
    # Plot pour les colonnes numériques
    if numeric_cols:
        n_cols = len(numeric_cols)
        fig, axes = plt.subplots(n_cols, 2, figsize=(15, 5 * n_cols))
        if n_cols == 1:
            axes = axes.reshape(1, -1)
        
        for idx, col in enumerate(numeric_cols):
            # Boxplot
            axes[idx, 0].boxplot([
                original_df[col].dropna(),
                sampled_df[col].dropna()
            ], labels=['Original', 'Échantillon'])
            axes[idx, 0].set_title(f'Boxplot - {col}')
            
            # Density plot
            original_df[col].dropna().plot(kind='density', ax=axes[idx, 1], 
                                         label='Original')
            sampled_df[col].dropna().plot(kind='density', ax=axes[idx, 1], 
                                        label='Échantillon')
            axes[idx, 1].set_title(f'Distribution - {col}')
            axes[idx, 1].legend()
        
        plt.tight_layout()
        plt.show() 