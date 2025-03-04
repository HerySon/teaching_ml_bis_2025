from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
    RobustScaler
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer

def compute_feature_importance_pca(df: pd.DataFrame, n_components: int = 2) -> Dict:
    """Calcule l'importance des features en utilisant PCA."""
    # Standardisation des données
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Application de PCA
    pca = PCA(n_components=n_components)
    pca.fit_transform(scaled_data)
    
    # Calcul de l'importance des features
    feature_importance = dict(zip(df.columns, np.abs(pca.components_).mean(axis=0)))
    
    return {
        'feature_importance': feature_importance,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'n_components': n_components
    }

def encode_categorical_features(
    df: pd.DataFrame,
    ordinal_columns: Optional[List[str]] = None,
    nominal_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode les variables catégorielles en utilisant LabelEncoder pour les variables ordinales
    et OneHotEncoder pour les variables nominales.
    
    Args:
        df: DataFrame à encoder
        ordinal_columns: Liste des colonnes ordinales à encoder avec LabelEncoder
        nominal_columns: Liste des colonnes nominales à encoder avec OneHotEncoder
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame encodé et dictionnaire des encodeurs
    """
    df_encoded = df.copy()
    encoders = {
        'label_encoders': {},
        'onehot_encoder': None,
        'onehot_features': []
    }
    
    # Encodage des variables ordinales avec LabelEncoder
    if ordinal_columns:
        for col in ordinal_columns:
            if col in df.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df[col].astype(str))
                encoders['label_encoders'][col] = le
    
    # Encodage des variables nominales avec OneHotEncoder
    if nominal_columns:
        nominal_cols_present = [col for col in nominal_columns if col in df.columns]
        if nominal_cols_present:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            nominal_encoded = ohe.fit_transform(df[nominal_cols_present])
            
            # Création des noms de colonnes pour les variables encodées
            feature_names = []
            for i, col in enumerate(nominal_cols_present):
                categories = ohe.categories_[i]
                feature_names.extend([f"{col}_{cat}" for cat in categories])
            
            # Ajout des colonnes encodées au DataFrame
            nominal_df = pd.DataFrame(
                nominal_encoded,
                columns=feature_names,
                index=df.index
            )
            
            # Suppression des colonnes originales et ajout des colonnes encodées
            df_encoded = df_encoded.drop(columns=nominal_cols_present)
            df_encoded = pd.concat([df_encoded, nominal_df], axis=1)
            
            encoders['onehot_encoder'] = ohe
            encoders['onehot_features'] = feature_names
    
    return df_encoded, encoders

def scale_numeric_features(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, RobustScaler]:
    """
    Normalise les variables numériques en utilisant RobustScaler.
    
    Args:
        df: DataFrame à normaliser
        numeric_columns: Liste des colonnes numériques à normaliser
        
    Returns:
        Tuple[DataFrame, RobustScaler]: DataFrame normalisé et scaler
    """
    df_scaled = df.copy()
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    if numeric_columns:
        scaler = RobustScaler()
        df_scaled[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    else:
        scaler = None
    
    return df_scaled, scaler

def detect_and_handle_outliers(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    method: str = 'iqr',
    threshold: float = 1.5,
    strategy: str = 'clip'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Détecte et traite les outliers dans les variables numériques.
    
    Args:
        df: DataFrame à traiter
        numeric_columns: Liste des colonnes numériques à analyser
        method: Méthode de détection ('iqr' ou 'zscore')
        threshold: Seuil pour la détection (1.5 pour IQR, 3 pour z-score)
        strategy: Stratégie de traitement ('clip' ou 'remove')
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame traité et informations sur les outliers
    """
    df_clean = df.copy()
    outliers_info = {
        'n_outliers': {},
        'outliers_ratio': {},
        'bounds': {}
    }
    
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    for col in numeric_columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
        else:  # zscore
            mean = df[col].mean()
            std = df[col].std()
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
        
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = outliers.sum()
        
        outliers_info['n_outliers'][col] = n_outliers
        outliers_info['outliers_ratio'][col] = n_outliers / len(df)
        outliers_info['bounds'][col] = {'lower': lower_bound, 'upper': upper_bound}
        
        if strategy == 'clip':
            df_clean[col] = df[col].clip(lower_bound, upper_bound)
        elif strategy == 'remove' and n_outliers > 0:
            df_clean = df_clean[~outliers]
    
    return df_clean, outliers_info

def apply_pca_reduction(
    df: pd.DataFrame,
    n_components: float = 0.95,
    numeric_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Applique une réduction de dimensionnalité PCA avec gestion des valeurs manquantes.
    
    Args:
        df: DataFrame à transformer
        n_components: Nombre de composantes ou ratio de variance expliquée
        numeric_columns: Liste des colonnes numériques à utiliser
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame transformé et informations sur la PCA
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    # Imputation des valeurs manquantes avec la médiane
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df[numeric_columns]),
        columns=numeric_columns,
        index=df.index
    )
    
    # Application de la PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df_imputed)
    
    # Création du DataFrame avec les composantes principales
    component_names = [f"PC{i+1}" for i in range(pca_result.shape[1])]
    df_pca = pd.DataFrame(
        pca_result,
        columns=component_names,
        index=df.index
    )
    
    # Informations sur la transformation
    pca_info = {
        'n_components': pca.n_components_,
        'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
        'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_).tolist(),
        'feature_importance': {
            col: importance for col, importance in zip(
                numeric_columns,
                np.abs(pca.components_).mean(axis=0)
            )
        },
        'n_samples_imputed': df[numeric_columns].isna().sum().sum()
    }
    
    return df_pca, pca_info

def apply_tsne_visualization(
    df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    perplexity: float = 30.0,
    n_components: int = 2,
    n_jobs: int = 1,
    method: str = 'barnes_hut',
    learning_rate: str = 'auto',
    early_exaggeration: float = 12.0,
    pca_components: Optional[int] = 50,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, TSNE, Optional[np.ndarray]]:
    """
    Applique t-SNE pour la visualisation des données avec gestion optimisée de la mémoire.
    
    Args:
        df: DataFrame à transformer
        numeric_columns: Liste des colonnes numériques à utiliser
        perplexity: Paramètre de perplexité pour t-SNE (défaut: 30.0)
        n_components: Nombre de composantes (2 ou 3)
        n_jobs: Nombre de jobs en parallèle
        method: Méthode de calcul ('barnes_hut' ou 'exact')
        learning_rate: Taux d'apprentissage ('auto' ou valeur numérique)
        early_exaggeration: Facteur d'exagération précoce
        pca_components: Nombre de composantes PCA pour réduction préalable (None pour désactiver)
        sample_size: Taille de l'échantillon à utiliser (None pour utiliser toutes les données)
        random_state: Graine aléatoire pour la reproductibilité
        
    Returns:
        Tuple[DataFrame, TSNE, Optional[np.ndarray]]: DataFrame transformé, modèle t-SNE et indices utilisés
    """
    if numeric_columns is None:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    # Imputation des valeurs manquantes avec la médiane
    imputer = SimpleImputer(strategy='median')
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df[numeric_columns]),
        columns=numeric_columns,
        index=df.index
    )
    
    # Sous-échantillonnage si nécessaire
    selected_indices = None
    if sample_size is not None and sample_size < len(df_imputed):
        selected_indices = np.random.RandomState(random_state).choice(
            len(df_imputed), 
            size=sample_size, 
            replace=False
        )
        df_imputed = df_imputed.iloc[selected_indices]
    
    # Réduction PCA préalable si demandée
    if pca_components is not None and pca_components < len(numeric_columns):
        pca = PCA(n_components=pca_components, random_state=random_state)
        data_transformed = pca.fit_transform(df_imputed)
        print(f"Variance expliquée après réduction PCA : {sum(pca.explained_variance_ratio_):.2%}")
    else:
        data_transformed = df_imputed.values
    
    # Configuration de t-SNE avec gestion des erreurs
    try:
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(perplexity, len(data_transformed) - 1),
            method=method,
            n_jobs=n_jobs,
            learning_rate=learning_rate,
            early_exaggeration=early_exaggeration,
            random_state=random_state
        )
        tsne_result = tsne.fit_transform(data_transformed)
        
    except (ValueError, RuntimeError, MemoryError) as e:
        print(f"Erreur lors de l'exécution de t-SNE: {str(e)}")
        print("Tentative avec des paramètres plus conservateurs...")
        
        # Réduction supplémentaire si nécessaire
        if isinstance(e, MemoryError) and pca_components is not None:
            pca_components = min(pca_components, 20)
            print(f"Réduction à {pca_components} composantes PCA...")
            pca = PCA(n_components=pca_components, random_state=random_state)
            data_transformed = pca.fit_transform(df_imputed)
        
        # Tentative avec des paramètres plus conservateurs
        tsne = TSNE(
            n_components=n_components,
            perplexity=min(30.0, len(data_transformed) / 5),
            method='barnes_hut',
            n_jobs=1,
            learning_rate=200.0,
            early_exaggeration=12.0,
            random_state=random_state
        )
        tsne_result = tsne.fit_transform(data_transformed)
    
    # Création du DataFrame avec les composantes t-SNE
    component_names = [f"TSNE{i+1}" for i in range(n_components)]
    df_tsne = pd.DataFrame(
        tsne_result,
        columns=component_names,
        index=df_imputed.index
    )
    
    # Si on a sous-échantillonné, on retourne aussi les indices utilisés
    if sample_size is not None and sample_size < len(df):
        print(f"Visualisation basée sur un échantillon de {len(df_tsne)} points")
    
    return df_tsne, tsne, selected_indices 