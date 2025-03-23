# tache 5.0
# Try clustering with kmeans

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional

def train_kmeans(
    df: pd.DataFrame,
    features: List[str],
    n_clusters_range: List[int] = range(2, 11),
    random_state: int = 42
) -> Tuple[KMeans, Dict[str, float], pd.DataFrame]:
    """
    Train and optimize K-means clustering on the Open Food Facts dataset.
    
    Parameters:
    -----------
    df : df
        Input Open Food Facts dataset
    features : List[str]
        List of features to use for clustering
    n_clusters_range : List[int], default=range(2, 11)
        Range of number of clusters to try
    random_state : int, default=42
        Random state for reproducibility
    
    Returns:
    --------
    Tuple[KMeans, Dict[str, float], df]
        - Best KMeans model
        - Dictionary of evaluation metrics
        - DataFrame with cluster assignments
    """

    # Check que les features existent bien dans le dataset
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features not found in dataset: {missing_features}")
    
    # Prepare data
    X = df[features].copy()
    
    # valeurs manquantes
    X = X.fillna(X.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertia = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    
    # Try plusieurs nnbs de clusters
    for n_clusters in n_clusters_range:
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
        kmeans.fit(X_scaled)
        
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
        calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, kmeans.labels_))
    
    # Find nb de clusters optimaux
    optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    
    # Train model avec le nb de clusters optimaux
    best_kmeans = KMeans(
        n_clusters=optimal_n_clusters,
        random_state=random_state,
        n_init=10
    )
    best_kmeans.fit(X_scaled)
    
    # Add cluster col au df
    df_clustered = df.copy()
    df_clustered['cluster'] = best_kmeans.labels_
    
    # Calcul des metrics
    metrics = {
        'optimal_n_clusters': optimal_n_clusters,
        'inertia': best_kmeans.inertia_,
        'silhouette_score': silhouette_score(X_scaled, best_kmeans.labels_),
        'calinski_harabasz_score': calinski_harabasz_score(X_scaled, best_kmeans.labels_)
    }
    
    # Plots
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(n_clusters_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    
    plt.subplot(1, 3, 2)
    plt.plot(n_clusters_range, silhouette_scores, 'go-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score')
    
    plt.subplot(1, 3, 3)
    plt.plot(n_clusters_range, calinski_harabasz_scores, 'ro-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Calinski-Harabasz Score')
    plt.title('Calinski-Harabasz Score')
    
    plt.tight_layout()
    plt.show()
    
    # caracteristiques des clusters
    plt.figure(figsize=(15, 10))
    
    # Mean pour chaque feature dans chaque clusters
    cluster_means = df_clustered.groupby('cluster')[features].mean()
    
    # heatmap
    sns.heatmap(cluster_means, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('Cluster Characteristics')
    plt.xlabel('Features')
    plt.ylabel('Cluster')
    plt.tight_layout()
    plt.show()
    
    return best_kmeans, metrics, df_clustered
