import numpy as np
import pandas as pd
from typing import Union, List, Dict, Tuple
from sklearn.impute import SimpleImputer
from scipy import stats
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def detect_outliers_tukey(data: Union[pd.Series, np.ndarray], threshold: float = 1.5) -> Tuple[np.ndarray, Dict]:
    """
    Détecte les valeurs aberrantes en utilisant la méthode de Tukey (IQR).
    
    Args:
        data: Données à analyser
        threshold: Seuil pour le calcul des bornes (par défaut 1.5)
    
    Returns:
        Tuple contenant :
        - Un tableau booléen indiquant les positions des valeurs aberrantes
        - Un dictionnaire avec les statistiques calculées
    """
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    outliers = (data < lower_bound) | (data > upper_bound)
    
    stats_dict = {
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    return outliers, stats_dict

def detect_outliers_zscore(data: Union[pd.Series, np.ndarray], threshold: float = 3.0) -> Tuple[np.ndarray, Dict]:
    """
    Détecte les valeurs aberrantes en utilisant le z-score.
    
    Args:
        data: Données à analyser
        threshold: Seuil de z-score pour considérer une valeur comme aberrante
    
    Returns:
        Tuple contenant :
        - Un tableau booléen indiquant les positions des valeurs aberrantes
        - Un dictionnaire avec les statistiques calculées
    """
    data = np.array(data)
    z_scores = np.abs(stats.zscore(data))
    outliers = z_scores > threshold
    
    stats_dict = {
        'z_scores': z_scores,
        'mean': np.mean(data),
        'std': np.std(data)
    }
    
    return outliers, stats_dict

def detect_outliers_elliptic(data: Union[pd.Series, np.ndarray], contamination: float = 0.1) -> Tuple[np.ndarray, Dict]:
    """
    Détecte les valeurs aberrantes en utilisant l'enveloppe elliptique (EllipticEnvelope).
    
    Args:
        data: Données à analyser
        contamination: Proportion attendue de valeurs aberrantes (entre 0 et 0.5)
    
    Returns:
        Tuple contenant :
        - Un tableau booléen indiquant les positions des valeurs aberrantes
        - Un dictionnaire avec les statistiques calculées
    """
    data = np.array(data).reshape(-1, 1)  # Reshape pour 2D array requis par EllipticEnvelope
    
    # Ajustement de l'enveloppe elliptique
    envelope = EllipticEnvelope(contamination=contamination, random_state=42)
    envelope.fit(data)
    
    # Prédiction des valeurs aberrantes (1 pour inliers, -1 pour outliers)
    predictions = envelope.predict(data)
    outliers = predictions == -1
    
    # Calcul des distances de Mahalanobis
    mahalanobis_distances = envelope.mahalanobis(data)
    
    stats_dict = {
        'mahalanobis_distances': mahalanobis_distances,
        'contamination': contamination,
        'location': envelope.location_,
        'covariance': envelope.covariance_
    }
    
    return outliers, stats_dict

def detect_outliers_isolation_forest(data: Union[pd.Series, np.ndarray], 
                                   contamination: float = 0.1,
                                   n_estimators: int = 100,
                                   random_state: int = 42) -> Tuple[np.ndarray, Dict]:
    """
    Détecte les valeurs aberrantes en utilisant Isolation Forest.
    
    Args:
        data: Données à analyser
        contamination: Proportion attendue de valeurs aberrantes (entre 0 et 0.5)
        n_estimators: Nombre d'arbres dans la forêt
        random_state: Graine aléatoire pour la reproductibilité
    
    Returns:
        Tuple contenant :
        - Un tableau booléen indiquant les positions des valeurs aberrantes
        - Un dictionnaire avec les statistiques calculées
    """
    data = np.array(data).reshape(-1, 1)  # Reshape pour 2D array requis par IsolationForest
    
    # Ajustement de l'Isolation Forest
    iso_forest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state
    )
    iso_forest.fit(data)
    
    # Prédiction des valeurs aberrantes (1 pour inliers, -1 pour outliers)
    predictions = iso_forest.predict(data)
    outliers = predictions == -1
    
    # Calcul des scores d'anomalie
    anomaly_scores = iso_forest.score_samples(data)
    
    stats_dict = {
        'anomaly_scores': anomaly_scores,
        'contamination': contamination,
        'n_estimators': n_estimators,
        'random_state': random_state
    }
    
    return outliers, stats_dict

def detect_outliers_lof(data: Union[pd.Series, np.ndarray], 
                       n_neighbors: int = 20,
                       contamination: float = 0.1,
                       novelty: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Détecte les valeurs aberrantes en utilisant Local Outlier Factor (LOF).
    
    Args:
        data: Données à analyser
        n_neighbors: Nombre de voisins à considérer
        contamination: Proportion attendue de valeurs aberrantes (entre 0 et 0.5)
        novelty: Si True, utilise le mode "novelty detection"
    
    Returns:
        Tuple contenant :
        - Un tableau booléen indiquant les positions des valeurs aberrantes
        - Un dictionnaire avec les statistiques calculées
    """
    data = np.array(data).reshape(-1, 1)  # Reshape pour 2D array requis par LOF
    
    # Ajustement du LOF
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=novelty
    )
    
    # Prédiction des valeurs aberrantes (1 pour inliers, -1 pour outliers)
    predictions = lof.fit_predict(data)
    outliers = predictions == -1
    
    # Calcul des scores d'anomalie (LOF scores)
    lof_scores = -lof.negative_outlier_factor_  # Conversion en scores positifs
    
    stats_dict = {
        'lof_scores': lof_scores,
        'n_neighbors': n_neighbors,
        'contamination': contamination,
        'novelty': novelty
    }
    
    return outliers, stats_dict

def handle_outliers(data: pd.DataFrame, 
                   column: str,
                   strategy: str = 'remove',
                   detection_method: str = 'tukey',
                   **kwargs) -> pd.DataFrame:
    """
    Gère les valeurs aberrantes selon la stratégie choisie.
    
    Args:
        data: DataFrame contenant les données
        column: Nom de la colonne à traiter
        strategy: Stratégie de gestion ('remove', 'impute', 'cap')
        detection_method: Méthode de détection ('tukey', 'zscore', 'elliptic', 'isolation_forest' ou 'lof')
        **kwargs: Arguments supplémentaires pour les méthodes de détection
    
    Returns:
        DataFrame avec les valeurs aberrantes traitées
    """
    df = data.copy()
    
    # Détection des valeurs aberrantes
    if detection_method == 'tukey':
        outliers, stats_dict = detect_outliers_tukey(df[column], **kwargs)
    elif detection_method == 'zscore':
        outliers, stats_dict = detect_outliers_zscore(df[column], **kwargs)
    elif detection_method == 'elliptic':
        outliers, stats_dict = detect_outliers_elliptic(df[column], **kwargs)
    elif detection_method == 'isolation_forest':
        outliers, stats_dict = detect_outliers_isolation_forest(df[column], **kwargs)
    elif detection_method == 'lof':
        outliers, stats_dict = detect_outliers_lof(df[column], **kwargs)
    else:
        raise ValueError("Méthode de détection non supportée")
    
    # Application de la stratégie
    if strategy == 'remove':
        df = df[~outliers]
    elif strategy == 'impute':
        imputer = SimpleImputer(strategy='median')
        df[column] = imputer.fit_transform(df[[column]])
    elif strategy == 'cap':
        if detection_method == 'tukey':
            df.loc[df[column] < stats_dict['lower_bound'], column] = stats_dict['lower_bound']
            df.loc[df[column] > stats_dict['upper_bound'], column] = stats_dict['upper_bound']
        elif detection_method == 'zscore':
            mean = stats_dict['mean']
            std = stats_dict['std']
            threshold = kwargs.get('threshold', 3.0)
            df.loc[df[column] < mean - threshold * std, column] = mean - threshold * std
            df.loc[df[column] > mean + threshold * std, column] = mean + threshold * std
        elif detection_method in ['elliptic', 'isolation_forest', 'lof']:
            # Pour ces méthodes, on utilise la médiane pour les valeurs aberrantes
            median_value = df[column].median()
            df.loc[outliers, column] = median_value
    
    return df

def get_outlier_summary(data: pd.DataFrame, 
                       column: str,
                       detection_method: str = 'tukey',
                       **kwargs) -> Dict:
    """
    Génère un résumé des valeurs aberrantes détectées.
    
    Args:
        data: DataFrame contenant les données
        column: Nom de la colonne à analyser
        detection_method: Méthode de détection ('tukey', 'zscore', 'elliptic', 'isolation_forest' ou 'lof')
        **kwargs: Arguments supplémentaires pour les méthodes de détection
    
    Returns:
        Dictionnaire contenant les statistiques sur les valeurs aberrantes
    """
    if detection_method == 'tukey':
        outliers, stats_dict = detect_outliers_tukey(data[column], **kwargs)
    elif detection_method == 'zscore':
        outliers, stats_dict = detect_outliers_zscore(data[column], **kwargs)
    elif detection_method == 'elliptic':
        outliers, stats_dict = detect_outliers_elliptic(data[column], **kwargs)
    elif detection_method == 'isolation_forest':
        outliers, stats_dict = detect_outliers_isolation_forest(data[column], **kwargs)
    elif detection_method == 'lof':
        outliers, stats_dict = detect_outliers_lof(data[column], **kwargs)
    else:
        raise ValueError("Méthode de détection non supportée")
    
    summary = {
        'total_values': len(data),
        'outlier_count': np.sum(outliers),
        'outlier_percentage': (np.sum(outliers) / len(data)) * 100,
        'detection_method': detection_method,
        'detection_stats': stats_dict
    }
    
    return summary 