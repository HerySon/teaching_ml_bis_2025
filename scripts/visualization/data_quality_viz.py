from typing import Dict, List, Optional, Any
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def unused(func):
    """Décorateur pour marquer les paramètres comme intentionnellement non utilisés."""
    return func

def _plot_type_distribution(df: pd.DataFrame) -> None:
    """Plot la distribution des types de variables."""
    plt.figure(figsize=(12, 8))
    type_counts = df.dtypes.value_counts()
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
    plt.pie(type_counts.values, 
            labels=type_counts.index, 
            autopct='%1.1f%%',
            colors=colors[:len(type_counts)],
            explode=[0.1] * len(type_counts),
            textprops={'fontsize': 12})
    plt.title('Distribution des types de variables', fontsize=14, pad=20)
    plt.legend(type_counts.index,
              title="Types de données",
              title_fontsize=12,
              fontsize=10,
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.show()

def _plot_correlation_matrix(numeric_df: pd.DataFrame) -> None:
    """Plot la matrice de corrélation."""
    if numeric_df.empty:
        return
        
    plt.figure(figsize=(15, 10))
    corr = numeric_df.corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = np.ma.array(corr, mask=mask)
    
    im = plt.imshow(corr_masked, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Coefficient de corrélation')
    
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha='right', fontsize=10)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=10)
    
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            if i > j:
                text = plt.text(j, i, f'{corr.iloc[i, j]:.2f}',
                              ha='center', va='center',
                              color='white' if abs(corr.iloc[i, j]) > 0.3 else 'black',
                              fontsize=9,
                              fontweight='bold')
    plt.title('Matrice de corrélation des variables numériques', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()

def _plot_numeric_distributions(numeric_df: pd.DataFrame) -> None:
    """Plot les distributions des variables numériques."""
    if numeric_df.empty:
        return
        
    stats = numeric_df.agg(['mean', 'std', 'min', 'max'])
    ranges = (stats.loc['max'] - stats.loc['min']).abs()
    sorted_columns = ranges.sort_values().index
    column_groups = [sorted_columns[i:i+5] for i in range(0, len(sorted_columns), 5)]
    
    for idx, group in enumerate(column_groups):
        plt.figure(figsize=(12, 6))
        data = [numeric_df[col].dropna() for col in group]
        bp = plt.boxplot(data,
                       labels=group,
                       patch_artist=True,
                       medianprops={"color": "red", "linewidth": 2},
                       flierprops={"marker": "o", "markerfacecolor": "gray", "markersize": 4})
        
        for patch in bp['boxes']:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.7)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.ylabel('Valeurs', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        ranges_str = [f"{col}: [{stats.loc['min', col]:.2g}, {stats.loc['max', col]:.2g}]" for col in group]
        plt.title(f'Distribution des variables numériques (Groupe {idx+1})\n' + '\n'.join(ranges_str), 
                 fontsize=12, pad=20)
        
        plt.tight_layout()
        plt.show()
        
        print("\nStatistiques détaillées pour le groupe", idx+1)
        for col in group:
            print(f"\n{col}:")
            print(f"  Moyenne: {stats.loc['mean', col]:.2g}")
            print(f"  Écart-type: {stats.loc['std', col]:.2g}")
            print(f"  Min: {stats.loc['min', col]:.2g}")
            print(f"  Max: {stats.loc['max', col]:.2g}")
            print(f"  Nombre de valeurs non-nulles: {numeric_df[col].count()}")

@unused
def plot_data_quality_dashboard(df: pd.DataFrame, quality_report: Dict) -> None:
    """
    Crée un dashboard de visualisation de la qualité des données.
    
    Args:
        df: DataFrame à analyser
        quality_report: Rapport de qualité des données
    """
    plt.style.use('default')
    
    _plot_type_distribution(df)
    _plot_correlation_matrix(df.select_dtypes(include=[np.number]))
    _plot_numeric_distributions(df.select_dtypes(include=[np.number]))
    
    # Utilisation du quality_report pour afficher des informations supplémentaires
    print("\nRésumé de la qualité des données:")
    dims = quality_report['dimensions']
    print(f"Dimensions: {dims['rows']} lignes, {dims['columns']} colonnes")
    print(f"Types de données: {', '.join(f'{k}: {v}' for k, v in quality_report['dtypes'].items())}")
    print(f"Colonnes avec valeurs manquantes: {len(quality_report['missing_values']['columns_with_missing'])}")
    if quality_report['quality_issues']['constant_columns']:
        print(f"Colonnes constantes: {', '.join(quality_report['quality_issues']['constant_columns'])}")

@unused
def plot_categorical_analysis(df: pd.DataFrame, cat_distributions: Dict) -> None:
    """
    Visualise l'analyse des variables catégorielles.
    
    Args:
        df: DataFrame à analyser
        cat_distributions: Résultats de l'analyse des distributions catégorielles
    """
    # Utilisation de df pour calculer le pourcentage de données manquantes
    total_rows = len(df)
    
    for col, dist_info in cat_distributions.items():
        # Une figure par variable catégorielle
        plt.figure(figsize=(15, 6))
        
        # Distribution des catégories avec pourcentage de données manquantes
        counts = pd.Series(dist_info['most_common'])
        missing_count = df[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        
        plt.bar(range(len(counts)), counts.values, color='#3498db', alpha=0.7)
        plt.xticks(range(len(counts)), counts.index, rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        # Construction du titre en deux lignes
        main_title = f'Distribution des catégories - {col}'
        subtitle = f'Données manquantes: {missing_pct:.1f}%'
        plt.suptitle(main_title, fontsize=14, y=1.02)
        plt.title(subtitle, fontsize=12, pad=20)
        
        plt.ylabel('Nombre d\'occurrences', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Métriques de la variable dans une figure séparée
        plt.figure(figsize=(10, 6))
        metrics = {
            'Entropie': dist_info['entropy'],
            'Ratio déséquilibre': dist_info['imbalance_ratio'],
            'Ratio manquants': dist_info['missing_ratio'],
            'Ratio uniques': dist_info['unique_ratio']
        }
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
        plt.bar(range(len(metrics)), list(metrics.values()), color=colors)
        plt.xticks(range(len(metrics)), list(metrics.keys()), rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.title(f'Métriques de la variable {col}', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_memory_usage(memory_report: Dict) -> None:
    """
    Visualise l'utilisation de la mémoire.
    
    Args:
        memory_report: Rapport d'utilisation de la mémoire
    """
    # Figure 1: Répartition par type de données
    plt.figure(figsize=(12, 8))
    memory_by_dtype = {k: v['memory_mb'] for k, v in memory_report['memory_by_dtype'].items()}
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']
    plt.pie(list(memory_by_dtype.values()), 
            labels=list(memory_by_dtype.keys()), 
            autopct='%1.1f%%',
            colors=colors[:len(memory_by_dtype)],
            textprops={'fontsize': 12})
    plt.title('Répartition de la mémoire par type de données', fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Top 10 des colonnes les plus gourmandes
    plt.figure(figsize=(12, 8))
    column_memory = {k: v['memory_mb'] for k, v in memory_report['column_details'].items()}
    top_memory = pd.Series(column_memory).sort_values(ascending=True)[-10:]
    plt.barh(range(len(top_memory)), top_memory.values, color='#3498db', alpha=0.7)
    plt.yticks(range(len(top_memory)), top_memory.index, fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('Utilisation mémoire (MB)', fontsize=12)
    plt.title('Top 10 des colonnes les plus gourmandes en mémoire', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_missing_values(df: pd.DataFrame) -> None:
    """Plot missing values heatmap."""
    plt.figure(figsize=(12, 6))
    
    missing_values = df.isnull().sum().values
    
    plt.plot(missing_values, **{"color": "red", "linewidth": 2})
    plt.scatter(range(len(df.columns)), missing_values, **{"marker": "o", "markerfacecolor": "gray", "markersize": 4})
    
    plt.title('Missing Values by Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Values Count')
    plt.xticks(range(len(df.columns)), df.columns, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_distribution(series: pd.Series) -> Dict:
    """Analyze the distribution of a series."""
    return {
        'mean': series.mean(),
        'median': series.median(),
        'std': series.std(),
        'skew': series.skew(),
        'kurtosis': series.kurtosis()
    } 