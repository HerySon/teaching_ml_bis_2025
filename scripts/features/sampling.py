from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from .processing import analyze_category_distributions

def select_representative_sample(
    df: pd.DataFrame,
    sample_size: float = 0.3,
    stratify_columns: List[str] = None,
    balance_rare_categories: bool = True,
    min_category_size: int = 100
) -> Tuple[pd.DataFrame, Dict]:
    """
    Sélectionne un sous-ensemble représentatif du DataFrame en utilisant plusieurs critères.
    
    Args:
        df: DataFrame source
        sample_size: Proportion de l'échantillon (entre 0 et 1) ou nombre d'observations
        stratify_columns: Colonnes à utiliser pour la stratification
        balance_rare_categories: Si True, suréchantillonne les catégories rares
        min_category_size: Taille minimale pour les catégories rares
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame échantillonné et informations sur l'échantillonnage
    """
    info = {
        'initial_size': len(df),
        'final_size': 0,
        'sampling_stats': {},
        'category_distributions': {},
        'numeric_stats': {}
    }
    
    # Détermination de la taille de l'échantillon
    if 0 < sample_size < 1:
        target_size = int(len(df) * sample_size)
    else:
        target_size = int(sample_size)
    
    # Si aucune colonne de stratification n'est spécifiée, on utilise les colonnes catégorielles
    if stratify_columns is None:
        stratify_columns = df.select_dtypes(include=['category', 'object']).columns.tolist()
    
    sampled_indices = set()
    
    # 1. Stratification sur les variables catégorielles
    for col in stratify_columns:
        # Analyse de la distribution des catégories
        value_counts = df[col].value_counts()
        category_proportions = value_counts / len(df)
        
        # Calcul des tailles d'échantillon par catégorie
        if balance_rare_categories:
            # Suréchantillonnage des catégories rares
            min_prop = max(min_category_size / len(df), 1/len(value_counts))
            adjusted_proportions = category_proportions.clip(lower=min_prop)
            adjusted_proportions = adjusted_proportions / adjusted_proportions.sum()
        else:
            adjusted_proportions = category_proportions
        
        target_counts = (adjusted_proportions * target_size).round().astype(int)
        
        # Échantillonnage par catégorie
        for category, target_count in target_counts.items():
            category_indices = df[df[col] == category].index
            available_indices = list(set(category_indices) - sampled_indices)
            
            if len(available_indices) > 0:
                selected = np.random.choice(
                    available_indices,
                    size=min(target_count, len(available_indices)),
                    replace=False
                )
                sampled_indices.update(selected)
        
        # Stockage des statistiques
        info['sampling_stats'][col] = {
            'original_distribution': value_counts.to_dict(),
            'target_distribution': target_counts.to_dict()
        }
    
    # 2. Complétion avec des points représentatifs pour les variables numériques
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0 and len(sampled_indices) < target_size:
        remaining_size = target_size - len(sampled_indices)
        
        # Normalisation des variables numériques
        scaler = StandardScaler()
        numeric_data = df[numeric_cols].fillna(df[numeric_cols].mean())
        scaled_data = scaler.fit_transform(numeric_data)
        
        # Clustering pour identifier les points représentatifs
        n_clusters = min(remaining_size, len(df) - len(sampled_indices))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        # Sélection des points les plus proches des centroïdes
        available_indices = list(set(range(len(df))) - sampled_indices)
        for cluster_id in range(n_clusters):
            cluster_points = [i for i in available_indices if clusters[i] == cluster_id]
            if cluster_points:
                center = kmeans.cluster_centers_[cluster_id]
                distances = np.linalg.norm(
                    scaled_data[cluster_points] - center.reshape(1, -1),
                    axis=1
                )
                closest_point = cluster_points[np.argmin(distances)]
                sampled_indices.add(closest_point)
        
        # Stockage des statistiques numériques
        info['numeric_stats'] = {
            'n_clusters': n_clusters,
            'variance_explained': kmeans.inertia_
        }
    
    # Création de l'échantillon final
    sampled_df = df.loc[list(sampled_indices)].copy()
    info['final_size'] = len(sampled_df)
    
    # Analyse des distributions finales
    info['category_distributions'] = analyze_category_distributions(sampled_df)
    
    return sampled_df, info

def plot_sampling_results(original_df: pd.DataFrame, 
                         sampled_df: pd.DataFrame, 
                         sampling_info: Dict) -> None:
    """
    Visualise les résultats de l'échantillonnage.
    
    Args:
        original_df: DataFrame original
        sampled_df: DataFrame échantillonné
        sampling_info: Informations sur l'échantillonnage
    """
    # 1. Comparaison des distributions catégorielles
    for col, stats in sampling_info['sampling_stats'].items():
        plt.figure(figsize=(15, 6))
        
        # Préparation des données
        orig_dist = pd.Series(stats['original_distribution'])
        sample_dist = sampled_df[col].value_counts()
        
        # Normalisation pour comparaison
        orig_dist_norm = orig_dist / orig_dist.sum()
        sample_dist_norm = sample_dist / sample_dist.sum()
        
        # Création du graphique
        x = np.arange(len(orig_dist))
        width = 0.35
        
        plt.bar(x - width/2, orig_dist_norm, width, label='Original', color='#3498db', alpha=0.7)
        plt.bar(x + width/2, sample_dist_norm, width, label='Échantillon', color='#e74c3c', alpha=0.7)
        
        plt.xlabel('Catégories')
        plt.ylabel('Proportion')
        plt.title(f'Distribution des catégories - {col}')
        plt.xticks(x, orig_dist.index, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 2. Comparaison des distributions numériques
    numeric_cols = original_df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        plt.figure(figsize=(12, 6))
        
        # Boxplots côte à côte
        data = [original_df[col].dropna(), sampled_df[col].dropna()]
        labels = ['Original', 'Échantillon']
        
        plt.boxplot(data, labels=labels, patch_artist=True,
                   medianprops={"color": "red", "linewidth": 1.5},
                   boxprops={"facecolor": "#3498db", "alpha": 0.7})
        
        plt.title(f'Distribution de {col}')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Statistiques descriptives
        print(f"\nStatistiques pour {col}:")
        orig_stats = original_df[col].describe()
        sample_stats = sampled_df[col].describe()
        stats_comparison = pd.DataFrame({
            'Original': orig_stats,
            'Échantillon': sample_stats
        })
        print(stats_comparison)
    
    # 3. Résumé global
    print("\nRésumé de l'échantillonnage:")
    print(f"Taille originale: {sampling_info['initial_size']}")
    print(f"Taille de l'échantillon: {sampling_info['final_size']}")
    print(f"Ratio: {sampling_info['final_size']/sampling_info['initial_size']*100:.1f}%")

def plot_sampling_comparison(original_data: pd.DataFrame, 
                           sampled_data: pd.DataFrame,
                           feature: str,
                           title: str = "Distribution Comparison") -> None:
    """Plot the distribution comparison between original and sampled data."""
    plt.figure(figsize=(10, 6))
    
    # Plot original data
    plt.hist(original_data[feature], bins=30, alpha=0.5, 
            label='Original', **{"color": "red", "linewidth": 1.5})
    
    # Plot sampled data
    plt.hist(sampled_data[feature], bins=30, alpha=0.7,
            label='Sampled', **{"facecolor": "#3498db", "alpha": 0.7})
    
    plt.title(title)
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show() 