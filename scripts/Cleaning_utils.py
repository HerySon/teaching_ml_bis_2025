import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from functools import lru_cache
import matplotlib.pyplot as plt
from scipy import stats

@lru_cache(maxsize=128) #va nous servir à stocker les résultats des fonctions pour éviter de les recalculer à chaque fois
def get_optimal_numeric_type(min_val: float, max_val: float, has_decimals: bool) -> np.dtype:
    """
    Détermine le type numérique optimal pour une plage de valeurs donnée.
    Utilise le cache pour éviter de recalculer pour des plages similaires.
    
    Args:
        min_val (float): Valeur minimale
        max_val (float): Valeur maximale
        has_decimals (bool): Si True, utilise des types float
        
    Returns:
        np.dtype: Type de données optimal.

    Logique :
        Si les valeurs contiennent des décimales, on choisit un type float :
        Si les valeurs tiennent dans float32, on l'utilise.
        Sinon, on prend float64.
        Si les valeurs sont des entiers, on vérifie :
        Si toutes les valeurs sont positives, on utilise uint[selon l'intervalle].
        Si les valeurs sont négatives, on utilise int[selon l'intervalle].
    """
    if has_decimals:
        if min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
            return np.float32
        return np.float64
    
    if min_val >= 0:
        if max_val <= 255:
            return np.uint8
        elif max_val <= 65535:
            return np.uint16
        elif max_val <= 4294967295:
            return np.uint32
        return np.uint64
    else:
        if min_val >= -128 and max_val <= 127:
            return np.int8
        elif min_val >= -32768 and max_val <= 32767:
            return np.int16
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return np.int32
        return np.int64

def downcast_numeric(series: pd.Series) -> Tuple[pd.Series, str]:
    """
    Optimise le type de données d'une série numérique de manière vectorisée.
    
    Args:
        series (pd.Series): Série à optimiser
        
    Returns:
        Tuple[pd.Series, str]: Série optimisée et nouveau type de données

    Logique :
        Vérifie si la série est numérique, sinon elle est retournée inchangée.
        Si le type d'origine est déjà optimisé (uint8, uint16, float32), il est conservé.
        Vérifie s'il y a des valeurs NaN, car elles empêchent certains changements de type.
        Utilise get_optimal_numeric_type pour déterminer le type le plus adapté.
        Convertit la série au nouveau type si possible
    """
    if not pd.api.types.is_numeric_dtype(series):
        return series, str(series.dtype)
    
    # Vérification rapide si l'optimisation est nécessaire
    original_dtype = series.dtype
    if original_dtype in [np.uint8, np.uint16, np.float32]:
        return series, str(original_dtype)
    
    # Calcul vectorisé des statistiques
    has_nan = series.isna().any()
    if has_nan:
        return series, str(original_dtype)
    
    min_val = series.min()
    max_val = series.max()
    has_decimals = not np.all(series == series.astype(int))
    
    # Détermination du type optimal
    optimal_type = get_optimal_numeric_type(min_val, max_val, has_decimals)
    
    # Conversion uniquement si nécessaire
    if optimal_type != series.dtype:
        try:
            series = series.astype(optimal_type)
        except (OverflowError, ValueError):
            pass  # Garde le type original si la conversion échoue
    
    return series, str(series.dtype)

def process_numeric_columns(df: pd.DataFrame, min_unique_ratio: float) -> Tuple[pd.DataFrame, Dict]:
    """
    Traite les colonnes numériques du DataFrame de manière optimisée.
    
    Args:
        df (pd.DataFrame): DataFrame à traiter
        min_unique_ratio (float): Ratio minimum de valeurs uniques
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame traité et informations sur les colonnes

    Logique :
        Sélectionne les colonnes numériques.
        Calcule le ratio de valeurs uniques par rapport à la taille du dataset.
        Sélectionne les colonnes dont le ratio est supérieur ou égal à min_unique_ratio.
        Applique downcast_numeric à chaque colonne valide.
        Retourne le DataFrame avec les colonnes numériques optimisées et les informations sur les colonnes.

    """
    info = {
        'numerical_columns': [],
        'dropped_columns': [],
        'downcasted_columns': []
    }
    
    # Sélection vectorisée des colonnes numériques
    numeric_columns = df.select_dtypes(include=np.number).columns
    if not len(numeric_columns):
        return df, info
    
    # Calcul vectorisé des ratios uniques
    unique_ratios = df[numeric_columns].nunique() / len(df)
    valid_columns = unique_ratios[unique_ratios >= min_unique_ratio].index
    
    # Traitement des colonnes valides
    for col in valid_columns:
        original_dtype = str(df[col].dtype)
        df[col], new_dtype = downcast_numeric(df[col])
        
        if new_dtype != original_dtype:
            info['downcasted_columns'].append((col, original_dtype, new_dtype))
        info['numerical_columns'].append(col)
    
    # Colonnes à supprimer
    columns_to_drop = unique_ratios[unique_ratios < min_unique_ratio].index
    info['dropped_columns'].extend((col, 'low_unique_ratio') for col in columns_to_drop)
    
    if len(columns_to_drop):
        df = df.drop(columns=columns_to_drop)
    
    return df, info

def process_categorical_columns(
    df: pd.DataFrame, 
    max_categories: int, 
    ordinal_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Traite les colonnes catégorielles du DataFrame de manière optimisée.
    
    Args:
        df (pd.DataFrame): DataFrame à traiter
        max_categories (int): Nombre maximum de catégories autorisées
        ordinal_columns (Optional[List[str]]): Liste des colonnes ordinales
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame traité et informations sur les colonnes*
    
    Logique :
        Sélectionne les colonnes catégorielles.
        Calcule le nombre de catégories uniques par rapport à la taille du dataset.
        Sélectionne les colonnes dont le nombre de catégories est inférieur ou égal à max_categories.
        Applique pd.Categorical à chaque colonne valide.
        Retourne le DataFrame avec les colonnes catégorielles optimisées et les informations sur les colonnes.
        Si les colonnes sont ordinales, on les applique pd.Categorical avec ordered=True.
        Sinon, on les applique pd.Categorical avec ordered=False.
    """
    info = {
        'categorical_ordinal': [],
        'categorical_nominal': [],
        'dropped_columns': []
    }
    
    # Sélection vectorisée des colonnes catégorielles
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if not len(categorical_columns):
        return df, info
    
    # Calcul vectorisé du nombre de catégories uniques
    n_unique = df[categorical_columns].nunique()
    valid_columns = n_unique[n_unique <= max_categories].index
    
    # Optimisation de la mémoire avec categorical
    ordinal_set = set(ordinal_columns or [])
    for col in valid_columns:
        if col in ordinal_set:
            df[col] = pd.Categorical(df[col], ordered=True)
            info['categorical_ordinal'].append(col)
        else:
            df[col] = pd.Categorical(df[col], ordered=False)
            info['categorical_nominal'].append(col)
    
    # Colonnes à supprimer
    columns_to_drop = n_unique[n_unique > max_categories].index
    info['dropped_columns'].extend((col, 'too_many_categories') for col in columns_to_drop)
    
    if len(columns_to_drop):
        df = df.drop(columns=columns_to_drop)
    
    return df, info

def preprocess_dataframe(
    df: pd.DataFrame,
    max_categories: int = 50,
    min_unique_ratio: float = 0.01,
    ordinal_columns: Optional[List[str]] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Prétraite le DataFrame en détectant et filtrant automatiquement les colonnes pertinentes.
    Version optimisée avec gestion de la mémoire améliorée.
    
    Args:
        df (pd.DataFrame): DataFrame d'entrée
        max_categories (int): Nombre maximum de catégories autorisées pour les variables catégorielles
        min_unique_ratio (float): Ratio minimum de valeurs uniques par rapport à la taille du dataset
        ordinal_columns (Optional[List[str]]): Liste des colonnes catégorielles ordinales à conserver
        verbose (bool): Si True, affiche les informations de prétraitement
    
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame prétraité et dictionnaire des informations de prétraitement

    Logique :
        Copie optimisée du DataFrame.
        Traitement des colonnes numériques.
        Traitement des colonnes catégorielles.
        Fusion des informations.
        Affichage des informations si demandé.
    """
    # Validation des entrées
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df doit être un pandas DataFrame")
    if max_categories <= 0:
        raise ValueError("max_categories doit être positif")
    if not 0 < min_unique_ratio < 1:
        raise ValueError("min_unique_ratio doit être entre 0 et 1")
    
    # Copie optimisée du DataFrame
    df_clean = df.copy(deep=False)  # Copie superficielle pour économiser la mémoire
    
    # Traitement des colonnes
    df_clean, numeric_info = process_numeric_columns(df_clean, min_unique_ratio)
    df_clean, categorical_info = process_categorical_columns(df_clean, max_categories, ordinal_columns)
    
    # Fusion des informations
    info = {
        'numerical_columns': numeric_info['numerical_columns'],
        'categorical_ordinal': categorical_info['categorical_ordinal'],
        'categorical_nominal': categorical_info['categorical_nominal'],
        'dropped_columns': numeric_info['dropped_columns'] + categorical_info['dropped_columns'],
        'downcasted_columns': numeric_info['downcasted_columns']
    }
    
    # Affichage des informations si demandé
    if verbose:
        print(f"Colonnes numériques conservées: {len(info['numerical_columns'])}")
        print(f"Colonnes catégorielles ordinales: {len(info['categorical_ordinal'])}")
        print(f"Colonnes catégorielles nominales: {len(info['categorical_nominal'])}")
        print(f"Colonnes supprimées: {len(info['dropped_columns'])}")
        print(f"Colonnes optimisées (downcasting): {len(info['downcasted_columns'])}")
    
    return df_clean, info

def get_feature_names(info: Dict) -> List[str]:
    """
    Retourne la liste de toutes les colonnes conservées après prétraitement.
    
    Args:
        info (Dict): Dictionnaire d'informations retourné par preprocess_dataframe
    
    Returns:
        List[str]: Liste des noms de colonnes
    
    Logique :
        Retourne la liste de toutes les colonnes conservées après prétraitement.
    """
    return (
        info['numerical_columns'] +
        info['categorical_ordinal'] +
        info['categorical_nominal']
    )

def detect_outliers(series: pd.Series, method: str = 'zscore', threshold: float = 3.0) -> pd.Series:
    """
    Détecte les valeurs aberrantes dans une série numérique.
    
    Args:
        series (pd.Series): Série à analyser
        method (str): Méthode de détection ('zscore', 'iqr')
        threshold (float): Seuil pour la détection
    
    Returns:
        pd.Series: Masque booléen indiquant les outliers
    """
    if not pd.api.types.is_numeric_dtype(series):
        raise TypeError("La série doit être numérique")
    
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    
    elif method == 'iqr':
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        return (series < (Q1 - threshold * IQR)) | (series > (Q3 + threshold * IQR))
    
    else:
        raise ValueError("Méthode non reconnue. Utilisez 'zscore' ou 'iqr'")

def handle_missing_values(df: pd.DataFrame, 
                         strategy: Dict[str, str] = None,
                         threshold: float = 0.5) -> Tuple[pd.DataFrame, Dict]:
    """
    Gère les valeurs manquantes dans le DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame à traiter
        strategy (Dict[str, str]): Stratégie par colonne ('mean', 'median', 'mode', 'drop')
        threshold (float): Seuil de suppression pour les colonnes avec trop de valeurs manquantes
    
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame traité et informations sur le traitement
    
    Logique :
        Copie optimisée du DataFrame.
        Suppression des colonnes avec trop de valeurs manquantes.
        Application des stratégies d'imputation.
        Retourne le DataFrame avec les valeurs manquantes gérées et les informations sur le traitement.
    """
    df_clean = df.copy()
    info = {
        'dropped_columns': [],
        'imputed_columns': {},
        'missing_stats': df.isnull().sum().to_dict()
    }
    
    # Suppression des colonnes avec trop de valeurs manquantes
    missing_ratio = df.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > threshold].index
    if len(cols_to_drop) > 0:
        df_clean = df_clean.drop(columns=cols_to_drop)
        info['dropped_columns'].extend(cols_to_drop)
    
    # Application des stratégies d'imputation
    strategy = strategy or {}
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            col_strategy = strategy.get(col, 'auto')
            
            if col_strategy == 'auto':
                if pd.api.types.is_numeric_dtype(df_clean[col]):
                    col_strategy = 'median'
                else:
                    col_strategy = 'mode'
            
            if col_strategy == 'mean':
                value = df_clean[col].mean()
            elif col_strategy == 'median':
                value = df_clean[col].median()
            elif col_strategy == 'mode':
                value = df_clean[col].mode()[0]
            elif col_strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
                continue
            
            df_clean[col] = df_clean[col].fillna(value)
            info['imputed_columns'][col] = {
                'strategy': col_strategy,
                'value': value
            }
    
    return df_clean, info

def plot_distributions(df):
    """
    Visualise la distribution des variables numériques.
    Utilise matplotlib au lieu de seaborn.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
    """
    numeric_cols = df.select_dtypes(include=np.number).columns
    n_cols = len(numeric_cols)
    
    if n_cols == 0:
        print("Aucune colonne numérique à visualiser")
        return
    
    fig, axes = plt.subplots(1, n_cols, figsize=(15, 5))
    if n_cols == 1:
        axes = [axes]
    
    for ax, col in zip(axes, numeric_cols):
        # Utilisation de matplotlib au lieu de seaborn
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(f'Distribution de {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Fréquence')
    
    plt.tight_layout()
    plt.show()

def analyze_correlations(df: pd.DataFrame,
                        method: str = 'pearson',
                        threshold: float = 0.7,
                        plot: bool = True) -> Dict:
    """
    Analyse les corrélations entre variables numériques.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
        method (str): Méthode de corrélation ('pearson', 'spearman', 'kendall')
        threshold (float): Seuil pour les corrélations fortes
        plot (bool): Si True, affiche une heatmap des corrélations
    
    Returns:
        Dict: Informations sur les corrélations
    
    Logique :
        Sélectionne les colonnes numériques.
        Calcule la matrice de corrélation.
        Identifie les paires fortement corrélées.
        Affiche la heatmap si demandé.
        Retourne les informations sur les corrélations.
    """
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr(method=method)
    
    # Identification des paires fortement corrélées
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                strong_correlations.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i,j]
                })
    
    if plot:
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        ax.set_title('Matrice de corrélation')
        ax.set_xticks(np.arange(len(corr_matrix.columns)))
        ax.set_yticks(np.arange(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns)
        ax.set_yticklabels(corr_matrix.columns)
        plt.colorbar(im)
        plt.show()
    
    return {
        'correlation_matrix': corr_matrix,
        'strong_correlations': strong_correlations
    }

def encode_categorical(df: pd.DataFrame,
                      columns: Optional[List[str]] = None,
                      method: str = 'auto') -> Tuple[pd.DataFrame, Dict]:
    """
    Encode les variables catégorielles.
    
    Args:
        df (pd.DataFrame): DataFrame à traiter
        columns (Optional[List[str]]): Colonnes à encoder
        method (str): Méthode d'encodage ('auto', 'onehot', 'label', 'ordinal')
    
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame encodé et informations sur l'encodage
    
    Logique :
        Sélectionne les colonnes à encoder.
        Applique l'encodage approprié en fonction du nombre de valeurs uniques.
        Retourne le DataFrame encodé et les informations sur l'encodage.
    """
    df_encoded = df.copy()
    info = {
        'encoding_maps': {},
        'encoded_features': {}
    }
    
    if columns is None:
        columns = df.select_dtypes(include=['object', 'category']).columns
    
    for col in columns:
        n_unique = df[col].nunique()
        
        if method == 'auto':
            if n_unique == 2:
                chosen_method = 'label'
            elif n_unique <= 10:
                chosen_method = 'onehot'
            else:
                chosen_method = 'label'
        else:
            chosen_method = method
        
        if chosen_method == 'onehot':
            dummies = pd.get_dummies(df[col], prefix=col)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
            info['encoded_features'][col] = list(dummies.columns)
            
        elif chosen_method == 'label':
            label_map = {val: idx for idx, val in enumerate(df[col].unique())}
            df_encoded[col] = df[col].map(label_map)
            info['encoding_maps'][col] = label_map
            
        elif chosen_method == 'ordinal':
            if not pd.api.types.is_categorical_dtype(df[col]) or not df[col].cat.ordered:
                raise ValueError(f"La colonne {col} doit être une catégorie ordinale")
            df_encoded[col] = df[col].cat.codes
            info['encoding_maps'][col] = {cat: code for code, cat in enumerate(df[col].cat.categories)}
    
    return df_encoded, info

def generate_summary_report(df: pd.DataFrame) -> Dict:
    """
    Génère un rapport détaillé sur le DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
    
    Returns:
        Dict: Rapport d'analyse
    
    Logique :
        Récupère les informations de base du DataFrame.
        Analyse les valeurs manquantes.
        Génère un rapport détaillé.
        Résume les valeurs les plus fréquentes pour les colonnes catégorielles.
    """
    report = {
        'basic_info': {
            'shape': df.shape,
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # En MB
            'dtypes': df.dtypes.value_counts().to_dict()
        },
        'missing_values': {
            'total': df.isnull().sum().sum(),
            'by_column': df.isnull().sum().to_dict(),
            'percentage': (df.isnull().mean() * 100).to_dict()
        },
        'numeric_summary': df.describe().to_dict(),
        'categorical_summary': {
            col: {
                'unique_values': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
            for col in df.select_dtypes(include=['object', 'category']).columns
        }
    }
    
    return report

def analyze_data_quality(df):
    """
    Analyse la qualité des données.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
    
    Returns:
        Dict: Rapport d'analyse
    """
    analysis = {
        # Distribution des valeurs
        'distributions': df.describe(),
        # Doublons
        'duplicates': df.duplicated().sum(),
        # Types de données
        'dtypes': df.dtypes,
        # Valeurs aberrantes pour colonnes numériques
        'outliers': {
            col: detect_outliers(df[col]) 
            for col in df.select_dtypes(include=['float64', 'int64']).columns
        }
    }
    return analysis

def check_data_consistency(df):
    """
    Vérifie la cohérence des données.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser
    
    Returns:
        List[str]: Liste des incohérences
    """
    inconsistencies = []
    
    # Exemple de règles métier pour OpenFoodFacts
    if 'energy-kcal_100g' in df.columns:
        energy_issues = df[df['energy-kcal_100g'] > 900]  # Valeurs suspectes
        inconsistencies.append(f"Energie > 900kcal/100g: {len(energy_issues)} produits")
    
    # Somme des nutriments doit être ≤ 100g
    nutrient_cols = ['carbohydrates_100g', 'proteins_100g', 'fat_100g']
    if all(col in df.columns for col in nutrient_cols):
        sum_over_100 = df[df[nutrient_cols].sum(axis=1) > 100]
        inconsistencies.append(f"Somme nutriments > 100g: {len(sum_over_100)} produits")
    
    return inconsistencies

def standardize_values(df):
    """
    Standardise les valeurs numériques.
    
    Args:
        df (pd.DataFrame): DataFrame à standardiser
    
    Returns:
        pd.DataFrame: DataFrame avec valeurs standardisées
    """
    df_copy = df.copy()
    numeric_cols = df_copy.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numeric_cols:
        # Remplacer les valeurs aberrantes par NaN
        df_copy[col] = df_copy[col].where(
            (df_copy[col] <= df_copy[col].quantile(0.99)) & 
            (df_copy[col] >= df_copy[col].quantile(0.01)), 
            np.nan
        )
    
    return df_copy

# Alias pour la compatibilité
standardize_data = standardize_values

def analyze_feature_importance(df: pd.DataFrame) -> Dict:
    """
    Analyse l'importance des variables selon plusieurs critères

    Args:
        df (pd.DataFrame): DataFrame à analyser

    Returns:
        Dict: Informations sur l'importance des variables

    Logique :
        Calcule les métriques d'importance des variables.
        Calcule les corrélations moyennes absolues pour les variables numériques.
        Retourne les informations sur l'importance des variables.
    """
    importance_metrics = {
        'missing_values': df.isnull().mean(),  # Taux de valeurs manquantes
        'variance': df.select_dtypes(include=np.number).var(),  # Variance pour variables numériques
        'unique_ratio': df.nunique() / len(df),  # Ratio de valeurs uniques
        'correlation_strength': None  # Sera calculé ci-dessous
    }
    
    # Calcul des corrélations moyennes absolues pour chaque variable numérique
    numeric_df = df.select_dtypes(include=np.number)
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr().abs()
        importance_metrics['correlation_strength'] = corr_matrix.mean()
    
    return importance_metrics

def select_relevant_features(df: pd.DataFrame, 
                           missing_threshold: float = 0.3,
                           variance_threshold: float = 0.01,
                           unique_ratio_threshold: float = 0.01) -> List[str]:
    """
    Sélectionne automatiquement les variables pertinentes
    """
    metrics = analyze_feature_importance(df)
    
    selected_features = []
    
    for column in df.columns:
        # Vérifier le taux de valeurs manquantes
        if metrics['missing_values'][column] > missing_threshold:
            continue
            
        # Pour les variables numériques
        if df[column].dtype in [np.number]:
            # Vérifier la variance
            if metrics['variance'].get(column, 0) < variance_threshold:
                continue
                
        # Vérifier le ratio de valeurs uniques
        if metrics['unique_ratio'][column] < unique_ratio_threshold:
            continue
            
        selected_features.append(column)
    
    return selected_features

def handle_duplicates(df: pd.DataFrame, strategy: str = 'analyze') -> Tuple[pd.DataFrame, Dict]:
    """
    Gère les doublons selon différentes stratégies
    
    Args:
        df: DataFrame à nettoyer
        strategy: 'analyze', 'remove_all', 'keep_first', 'keep_last', 'aggregate'

    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame nettoyé et informations sur les doublons

    Logique :
        Analyse les doublons.
        Supprime les doublons en fonction de la stratégie choisie.
        Retourne le DataFrame nettoyé et les informations sur les doublons.
    """
    info = {
        'initial_shape': df.shape,
        'duplicate_count': df.duplicated().sum(),
        'duplicate_rows': None,
        'action_taken': None
    }
    
    if strategy == 'analyze':
        # Analyse détaillée des doublons
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
    
    elif strategy == 'remove_all':
        # Supprime toutes les lignes dupliquées
        df_clean = df.drop_duplicates()
        info['action_taken'] = 'removed_all_duplicates'
        
    elif strategy == 'keep_first':
        # Garde la première occurrence
        df_clean = df.drop_duplicates(keep='first')
        info['action_taken'] = 'kept_first_occurrence'
        
    elif strategy == 'keep_last':
        # Garde la dernière occurrence
        df_clean = df.drop_duplicates(keep='last')
        info['action_taken'] = 'kept_last_occurrence'
        
    elif strategy == 'aggregate':
        # Agrège les doublons en utilisant des règles spécifiques
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(exclude=np.number).columns
        
        agg_rules = {
            **{col: 'mean' for col in numeric_cols},  # Moyenne pour les numériques
            **{col: lambda x: x.mode().iloc[0] if not x.mode().empty else None 
               for col in categorical_cols}  # Mode pour les catégorielles
        }
        
        df_clean = df.groupby(df.index).agg(agg_rules)
        info['action_taken'] = 'aggregated_duplicates'
    
    info['final_shape'] = df_clean.shape
    info['removed_rows'] = info['initial_shape'][0] - info['final_shape'][0]
    
    return df_clean, info