from typing import Dict

import numpy as np
import pandas as pd

def analyze_correlations(df: pd.DataFrame,
                        method: str = 'pearson',
                        threshold: float = 0.7,
                        plot: bool = True) -> Dict:
    """
    Analyse les corrélations entre variables numériques et détermine les variables à supprimer.
    
    Args:
        df: DataFrame à analyser
        method: Méthode de corrélation ('pearson', 'spearman', 'kendall')
        threshold: Seuil pour les corrélations fortes
        plot: Si True, affiche une heatmap des corrélations
    
    Returns:
        Dict: Informations sur les corrélations et variables à supprimer
    """
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr(method=method)
    
    strong_correlations = []
    # Dictionnaire pour compter le nombre de fortes corrélations par variable
    correlation_counts = {col: 0 for col in corr_matrix.columns}
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i,j]
                strong_correlations.append({
                    'var1': var1,
                    'var2': var2,
                    'correlation': correlation
                })
                # Incrémenter le compteur pour les deux variables
                correlation_counts[var1] += 1
                correlation_counts[var2] += 1
    
    # Déterminer les variables à supprimer
    variables_to_drop = []
    processed_pairs = set()  # Pour éviter de traiter plusieurs fois les mêmes paires
    
    # Trier les corrélations par force de corrélation
    sorted_correlations = sorted(strong_correlations, 
                               key=lambda x: abs(x['correlation']), 
                               reverse=True)
    
    for corr in sorted_correlations:
        pair = tuple(sorted([corr['var1'], corr['var2']]))
        if pair not in processed_pairs:
            processed_pairs.add(pair)
            # Choisir la variable à supprimer en fonction du nombre de corrélations
            var1_count = correlation_counts[corr['var1']]
            var2_count = correlation_counts[corr['var2']]
            
            # Supprimer la variable qui a le plus de corrélations fortes
            if var1_count > var2_count:
                if corr['var1'] not in variables_to_drop:
                    variables_to_drop.append(corr['var1'])
            else:
                if corr['var2'] not in variables_to_drop:
                    variables_to_drop.append(corr['var2'])
    
    if plot:
        try:
            from ..visualization import plot_correlations
            plot_correlations(df, corr_matrix.columns)
        except ImportError:
            print("Module de visualisation non disponible")
    
    return {
        'correlation_matrix': corr_matrix,
        'strong_correlations': strong_correlations,
        'variables_to_drop': variables_to_drop,
        'correlation_counts': correlation_counts
    } 