# Tache 3.0
# Try ransac to find outliers

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional

def detect_outliers_ransac(
    df: pd.DataFrame,
    key_columns: Optional[List[str]] = None,
    threshold: float = 3.0,
    random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Detect outliers in a DataFrame using statistical methods.
    
    Parameters:
    -----------
    df : df
        Input Open Food Facts dataset
    key_columns : List[str], optional
        List of columns to visualize. default = nutritional columns
    threshold : float, default=3.0
        Number of standard deviations to use for outlier detection
    random_state : int, default=42
        Random state for reproducibility

    
    Returns:
    --------
    Tuple[df, Dict[str, float], Dict[str, float]]
        - Processed df with outlier information
        - Dict of mean values for each column
        - Dict of outlier percentages for each column
    """
    # Copy pour detect outliers
    df_ransac = df.copy()
    
    # Select num columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    
    outlier_indices = []
    column_stats = {}
    outlier_percentages = {}
    
    # Process chaque cols num
    for column in numerical_columns:
        # Skip cols avec trop de vals manquantes
        if df[column].isnull().sum() > len(df) * 0.5:
            continue
            
        # Prepare data
        y = df[column].values
        
        # Remove vals manquantes
        mask = ~np.isnan(y)
        y = y[mask]
        
        if len(y) < 10:  # Skip cols avec trop peu de vals valides
            continue
        
        # Calcul de stats
        mean = np.mean(y)
        std = np.std(y)
        
        # Store stats
        column_stats[column] = {'mean': mean, 'std': std}
        
        # Find outliers avec z-score
        z_scores = np.abs((y - mean) / std)
        outlier_mask = z_scores > threshold
        
        # Get outlier index
        outlier_idx = np.where(outlier_mask)[0]
        original_indices = df_ransac.index[mask][outlier_idx]
        outlier_indices.extend(original_indices)
        
        # Calculate outlier %
        outlier_percentages[column] = (len(outlier_idx) / len(y)) * 100
    
    # Get unique outlier index
    outlier_indices = list(set(outlier_indices))
    
    # Create new col pour dire si les vals sont outliers
    df_ransac['is_outlier'] = df_ransac.index.isin(outlier_indices)
    
    # Sommaire
    print(f"Total number of outliers detected: {len(outlier_indices)}")
    print(f"Percentage of outliers: {(len(outlier_indices) / len(df)) * 100:.2f}%")
    
    # Plots
    if key_columns is None:
        key_columns = ['energy_100g', 'proteins_100g', 'fat_100g', 'carbohydrates_100g']
    
    plt.figure(figsize=(15, 10))
    
    for i, column in enumerate(key_columns, 1):
        if column not in column_stats:
            continue
            
        plt.subplot(2, 2, i)
        sns.scatterplot(data=df_ransac, x=df_ransac.index, y=column, 
                       hue='is_outlier', alpha=0.6)
        plt.title(f'Outliers - {column}')
        plt.xlabel('Sample Index')
        plt.ylabel(column)
        
        # Add mean and standard deviation lines
        stats = column_stats[column]
        plt.axhline(y=stats['mean'], color='g', linestyle='-', alpha=0.5)
        plt.axhline(y=stats['mean'] + threshold * stats['std'], color='y', linestyle='--', alpha=0.3)
        plt.axhline(y=stats['mean'] - threshold * stats['std'], color='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df_ransac, column_stats, outlier_percentages

