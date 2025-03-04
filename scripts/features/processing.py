import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
from scipy.stats import entropy
from ..data.analysis import analyze_data_quality
from ..data.correlation_analysis import analyze_correlations

def get_optimal_numeric_type(min_val: float, max_val: float, has_decimals: bool) -> np.dtype:
    """
    Détermine le type numérique optimal pour une colonne.
    
    Args:
        min_val: Valeur minimale
        max_val: Valeur maximale
        has_decimals: Si True, la colonne contient des décimales
        
    Returns:
        np.dtype: Type numpy optimal
    """
    if pd.isna(min_val) or pd.isna(max_val) or np.isinf(min_val) or np.isinf(max_val):
        return np.float64
        
    if has_decimals:
        if min_val >= np.finfo(np.float32).min and max_val <= np.finfo(np.float32).max:
            return np.float32
        return np.float64
    
    if min_val >= 0:
        if max_val <= np.iinfo(np.uint8).max:
            return np.uint8
        elif max_val <= np.iinfo(np.uint16).max:
            return np.uint16
        elif max_val <= np.iinfo(np.uint32).max:
            return np.uint32
        return np.uint64
    else:
        if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
            return np.int8
        elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
            return np.int16
        elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
            return np.int32
        return np.int64

def process_numeric_columns(df: pd.DataFrame, min_unique_ratio: float = 0.01) -> Tuple[pd.DataFrame, Dict]:
    """
    Traite les colonnes numériques du DataFrame.
    
    Args:
        df: DataFrame à traiter
        min_unique_ratio: Ratio minimum de valeurs uniques
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame traité et informations sur les colonnes
    """
    info = {
        'numeric_columns': [],
        'dropped_columns': [],
        'downcasted_columns': []
    }
    
    df_clean = df.copy()
    numeric_columns = df.select_dtypes(include=np.number).columns
    
    for column in numeric_columns:
        # Vérification du ratio de valeurs uniques
        unique_ratio = df[column].nunique() / len(df)
        if unique_ratio < min_unique_ratio:
            info['dropped_columns'].append((column, 'low_variance'))
            continue
            
        # Optimisation du type
        original_dtype = df[column].dtype
        series = df[column]
        has_nan = series.isna().any()
        
        if not has_nan:
            min_val = series.min()
            max_val = series.max()
            has_decimals = not np.all(series == series.astype(int))
            optimal_type = get_optimal_numeric_type(min_val, max_val, has_decimals)
            
            if optimal_type != original_dtype:
                try:
                    df_clean[column] = series.astype(optimal_type)
                    info['downcasted_columns'].append(
                        (column, str(original_dtype), str(optimal_type))
                    )
                except (OverflowError, ValueError):
                    pass
        
        info['numeric_columns'].append(column)
    
    # Suppression des colonnes si nécessaire
    columns_to_drop = [col for col, _ in info['dropped_columns']]
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
    
    return df_clean, info

def detect_ordinal_nature(series: pd.Series) -> bool:
    """
    Détection sophistiquée des variables ordinales.
    
    Args:
        series: Série à analyser
        
    Returns:
        bool: True si la variable est probablement ordinale
    """
    name_indicators = ['_level', '_grade', '_score', '_rating', 'grade_', 'level_', 'score_']
    if any(ind in str(series.name).lower() for ind in name_indicators):
        return True
    
    unique_values = series.dropna().unique()
    if all(str(x).replace('.', '').isdigit() for x in unique_values):
        return True
    
    ordinal_patterns = [
        ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D+', 'D', 'E+', 'E'],
        ['excellent', 'good', 'average', 'poor', 'bad'],
        ['high', 'medium', 'low'],
        ['1st', '2nd', '3rd', '4th', '5th'],
        ['beginner', 'intermediate', 'advanced', 'expert']
    ]
    
    values_str = series.astype(str).str.lower()
    for pattern in ordinal_patterns:
        pattern_lower = [p.lower() for p in pattern]
        if all(any(p in v for p in pattern_lower) for v in values_str):
            return True
    
    return False

def process_categorical_columns(
    df: pd.DataFrame, 
    max_categories: int, 
    detect_ordinal: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Traite les colonnes catégorielles du DataFrame.
    
    Args:
        df: DataFrame à traiter
        max_categories: Nombre maximum de catégories autorisées
        detect_ordinal: Si True, tente de détecter automatiquement les variables ordinales
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame traité et informations sur les colonnes
    """
    info = {
        'categorical_ordinal': [],
        'categorical_nominal': [],
        'dropped_columns': []
    }
    
    df_clean = df.copy()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    for column in categorical_columns:
        n_unique = df[column].nunique()
        if n_unique > max_categories:
            info['dropped_columns'].append((column, 'too_many_categories'))
            continue
        
        is_ordinal = detect_ordinal and detect_ordinal_nature(df[column])
        if is_ordinal:
            df_clean[column] = pd.Categorical(df[column], ordered=True)
            info['categorical_ordinal'].append(column)
        else:
            df_clean[column] = pd.Categorical(df[column], ordered=False)
            info['categorical_nominal'].append(column)
    
    # Suppression des colonnes si nécessaire
    columns_to_drop = [col for col, _ in info['dropped_columns']]
    if columns_to_drop:
        df_clean = df_clean.drop(columns=columns_to_drop)
    
    return df_clean, info

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

def enhance_ordinal_detection(series: pd.Series) -> bool:
    """
    Amélioration de la détection des variables ordinales.
    
    Args:
        series: Série à analyser
        
    Returns:
        bool: True si la variable est probablement ordinale
    """
    # Vérification si c'est déjà détecté comme ordinal
    if detect_ordinal_nature(series):
        return True
    
    # Détection des séquences numériques
    unique_values = pd.to_numeric(series.dropna(), errors='coerce')
    if not unique_values.isna().all():
        sorted_values = unique_values.sort_values()
        if all(sorted_values.diff().dropna() > 0):  # Séquence strictement croissante
            return True
    
    # Détection des dates et temps
    try:
        pd.to_datetime(series, errors='raise')
        return True
    except (ValueError, TypeError):
        pass
    
    # Patterns ordinaux supplémentaires
    additional_patterns = [
        ['très faible', 'faible', 'moyen', 'élevé', 'très élevé'],
        ['débutant', 'intermédiaire', 'avancé', 'expert'],
        ['jamais', 'parfois', 'souvent', 'toujours'],
        ['nul', 'passable', 'assez bien', 'bien', 'très bien', 'excellent'],
        ['pas du tout d\'accord', 'pas d\'accord', 'neutre', 'd\'accord', 'tout à fait d\'accord']
    ]
    
    values_str = series.astype(str).str.lower()
    for pattern in additional_patterns:
        if all(any(p in v for p in pattern) for v in values_str):
            return True
    
    return False

def infer_ordinal_order(series: pd.Series) -> List[str]:
    """
    Détermine l'ordre naturel des catégories ordinales.
    
    Args:
        series: Série à analyser
        
    Returns:
        List[str]: Liste des catégories dans leur ordre naturel
    """
    unique_values = series.dropna().unique()
    
    # Essai de conversion en numérique
    try:
        numeric_values = pd.to_numeric(unique_values)
        return [str(x) for x in sorted(numeric_values)]
    except (ValueError, TypeError):
        pass
    
    # Essai de conversion en dates
    try:
        date_values = pd.to_datetime(unique_values)
        return [str(x) for x in sorted(date_values)]
    except (ValueError, TypeError):
        pass
    
    # Patterns ordinaux connus
    known_patterns = {
        'niveau': ['débutant', 'intermédiaire', 'avancé', 'expert'],
        'accord': ['pas du tout d\'accord', 'pas d\'accord', 'neutre', 'd\'accord', 'tout à fait d\'accord'],
        'qualité': ['mauvais', 'passable', 'moyen', 'bon', 'excellent'],
        'fréquence': ['jamais', 'rarement', 'parfois', 'souvent', 'toujours'],
        'intensité': ['très faible', 'faible', 'moyen', 'fort', 'très fort']
    }
    
    # Recherche du pattern qui correspond le mieux
    values_set = set(v.lower() for v in unique_values)
    for pattern in known_patterns.values():
        pattern_set = set(p.lower() for p in pattern)
        if values_set.issubset(pattern_set):
            return [v for v in pattern if v.lower() in values_set]
    
    # Si aucun pattern ne correspond, retour des valeurs dans l'ordre alphabétique
    return sorted(unique_values)

def optimize_categorical_memory(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Optimisation de la mémoire pour les variables catégorielles.
    
    Args:
        df: DataFrame à optimiser
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame optimisé et informations sur l'optimisation
    """
    df_optimized = df.copy()
    optimization_info = {
        'memory_savings': {},
        'encoding_changes': {},
        'rare_categories': {}
    }
    
    for column in df.select_dtypes(include=['object', 'category']).columns:
        initial_memory = df[column].memory_usage(deep=True)
        
        # Obtenir les catégories uniques non nulles
        unique_non_null = df[column].dropna().unique()
        n_unique = len(unique_non_null)
        
        # Optimisation basée sur le nombre de catégories uniques
        if n_unique <= 2:
            df_optimized[column] = pd.Categorical(df[column], categories=unique_non_null)
        elif n_unique <= 255:
            df_optimized[column] = pd.Categorical(df[column], categories=unique_non_null)
        else:
            # Pour les colonnes avec beaucoup de catégories, on utilise un encodage efficace
            df_optimized[column] = pd.Categorical(df[column])
        
        final_memory = df_optimized[column].memory_usage(deep=True)
        savings = initial_memory - final_memory
        
        optimization_info['memory_savings'][column] = {
            'initial_bytes': initial_memory,
            'final_bytes': final_memory,
            'savings_bytes': savings,
            'savings_percent': (savings / initial_memory) * 100
        }
        
        optimization_info['encoding_changes'][column] = {
            'initial_dtype': str(df[column].dtype),
            'final_dtype': str(df_optimized[column].dtype),
            'n_categories': n_unique,
            'has_null': df[column].isna().any()
        }
    
    return df_optimized, optimization_info

def handle_rare_categories(
    df: pd.DataFrame,
    threshold: float = 0.01,
    strategy: str = 'group',
    other_label: str = 'Other'
) -> Tuple[pd.DataFrame, Dict]:
    """
    Gestion des catégories rares dans les variables catégorielles.
    
    Args:
        df: DataFrame à traiter
        threshold: Seuil minimum de fréquence pour une catégorie
        strategy: 'group' pour regrouper, 'drop' pour supprimer
        other_label: Étiquette pour le groupe 'Other'
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame traité et informations sur les modifications
    """
    df_processed = df.copy()
    info = {
        'modified_columns': {},
        'dropped_categories': {},
        'grouped_categories': {}
    }
    
    for column in df.select_dtypes(include=['object', 'category']).columns:
        value_counts = df[column].value_counts(normalize=True)
        rare_categories = value_counts[value_counts < threshold].index
        
        if len(rare_categories) > 0:
            if strategy == 'group':
                # Convertir en catégorie si ce n'est pas déjà le cas
                if not isinstance(df_processed[column].dtype, pd.CategoricalDtype):
                    df_processed[column] = pd.Categorical(df_processed[column])
                
                # Créer un dictionnaire de mapping pour les catégories
                category_mapping = {cat: other_label if cat in rare_categories else cat 
                                 for cat in df_processed[column].unique() if pd.notna(cat)}
                
                # Appliquer le mapping avec rename_categories
                df_processed[column] = df_processed[column].cat.rename_categories(category_mapping)
                
                info['grouped_categories'][column] = {
                    'n_grouped': len(rare_categories),
                    'grouped_categories': list(rare_categories)
                }
            elif strategy == 'drop':
                df_processed = df_processed[~df_processed[column].isin(rare_categories)]
                info['dropped_categories'][column] = {
                    'n_dropped': len(rare_categories),
                    'dropped_categories': list(rare_categories)
                }
            
            info['modified_columns'][column] = {
                'initial_categories': df[column].nunique(),
                'final_categories': df_processed[column].nunique(),
                'threshold': threshold,
                'strategy': strategy
            }
    
    return df_processed, info

def analyze_category_distributions(df: pd.DataFrame) -> Dict:
    """
    Analyse la distribution des catégories dans les variables catégorielles.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Dict: Informations sur les distributions des catégories
    """
    distribution_info = {}
    
    for column in df.select_dtypes(include=['object', 'category']).columns:
        value_counts = df[column].value_counts()
        frequencies = value_counts / len(df)
        
        # Calcul de l'entropie de la distribution
        category_entropy = entropy(frequencies)
        
        # Calcul du déséquilibre des classes
        imbalance_ratio = value_counts.max() / value_counts.min() if len(value_counts) > 0 else 0
        
        distribution_info[column] = {
            'n_categories': len(value_counts),
            'most_common': value_counts.head(5).to_dict(),
            'least_common': value_counts.tail(5).to_dict(),
            'entropy': category_entropy,
            'imbalance_ratio': imbalance_ratio,
            'missing_ratio': df[column].isna().mean(),
            'unique_ratio': df[column].nunique() / len(df),
            'value_counts': value_counts.to_dict()  # Ajout des counts complets
        }
    
    return distribution_info

def memory_usage_report(df: pd.DataFrame) -> Dict:
    """
    Génère un rapport détaillé sur l'utilisation de la mémoire du DataFrame.
    
    Args:
        df: DataFrame à analyser
        
    Returns:
        Dict: Rapport d'utilisation de la mémoire
    """
    total_memory = df.memory_usage(deep=True).sum()
    column_memory = df.memory_usage(deep=True)
    
    report = {
        'total_memory_bytes': total_memory,
        'total_memory_mb': total_memory / 1024 / 1024,
        'memory_by_dtype': {},
        'column_details': {},
        'optimization_suggestions': []
    }
    
    # Analyse par type de données
    for dtype in df.dtypes.unique():
        cols = df.select_dtypes(include=[dtype]).columns
        memory = df[cols].memory_usage(deep=True).sum()
        report['memory_by_dtype'][str(dtype)] = {
            'memory_bytes': memory,
            'memory_mb': memory / 1024 / 1024,
            'percentage': (memory / total_memory) * 100,
            'n_columns': len(cols)
        }
    
    # Analyse par colonne
    for column in df.columns:
        memory = df[column].memory_usage(deep=True)
        dtype = df[column].dtype
        n_unique = df[column].nunique() if dtype != 'datetime64[ns]' else None
        
        report['column_details'][column] = {
            'memory_bytes': memory,
            'memory_mb': memory / 1024 / 1024,
            'percentage': (memory / total_memory) * 100,
            'dtype': str(dtype),
            'n_unique': n_unique
        }
        
        # Suggestions d'optimisation
        if dtype == 'object':
            if n_unique and n_unique <= 255:
                report['optimization_suggestions'].append(
                    f"Column '{column}' could be converted to categorical"
                )
        elif dtype == 'int64':
            if df[column].min() >= 0 and df[column].max() <= 255:
                report['optimization_suggestions'].append(
                    f"Column '{column}' could be downcasted to uint8"
                )
    
    return report

def filter_and_analyze_dataset(df: pd.DataFrame,
                             max_categories: int = 30,
                             min_unique_ratio: float = 0.01,
                             missing_threshold: float = 0.5,
                             correlation_threshold: float = 0.7,
                             rare_threshold: float = 0.01,
                             verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Filtre et analyse un DataFrame selon plusieurs critères.
    
    Args:
        df: DataFrame à analyser
        max_categories: Nombre maximum de catégories pour les variables catégorielles
        min_unique_ratio: Ratio minimum de valeurs uniques
        missing_threshold: Seuil pour les valeurs manquantes
        correlation_threshold: Seuil pour les corrélations
        rare_threshold: Seuil pour les valeurs rares
        verbose: Si True, affiche les détails
        
    Returns:
        Tuple[pd.DataFrame, Dict]: DataFrame filtré et rapport d'analyse
    """
    analysis_report = {}
    df_filtered = df.copy()
    
    # 1. Analyse initiale de la qualité des données
    if verbose: print("1. Analyse de la qualité des données initiales :")
    quality_report = analyze_data_quality(df)
    analysis_report['quality_report'] = quality_report
    if verbose:
        print(f"Dimensions initiales : {df.shape}")
        print(f"Types de données : \n{quality_report['dtypes']}\n")
    
    # 2. Filtrage des colonnes avec trop de valeurs manquantes
    if verbose: print("\n2. Filtrage des colonnes avec trop de valeurs manquantes :")
    missing_cols = [col for col, pct in quality_report['missing_values']['percentages'].items()
                   if pct > missing_threshold * 100]
    df_filtered = df_filtered.drop(columns=missing_cols)
    if verbose:
        print(f"Colonnes supprimées : {len(missing_cols)}")
        print(f"Nouvelles dimensions : {df_filtered.shape}\n")
    
    # 3. Analyse des corrélations
    if verbose: print("\n3. Analyse des corrélations :")
    correlation_report = analyze_correlations(df_filtered, threshold=correlation_threshold, plot=False)
    analysis_report['correlation_report'] = correlation_report
    df_filtered = df_filtered.drop(columns=correlation_report['variables_to_drop'])
    if verbose:
        print(f"Colonnes corrélées supprimées : {len(correlation_report['variables_to_drop'])}")
        print(f"Nouvelles dimensions : {df_filtered.shape}\n")
    
    # 4. Analyse des distributions catégorielles
    if verbose: print("\n4. Analyse des distributions catégorielles :")
    categorical_cols = df_filtered.select_dtypes(include=['object', 'category']).columns
    category_distributions = {}
    cols_to_drop = []
    
    for col in categorical_cols:
        value_counts = df_filtered[col].value_counts()
        frequencies = value_counts / len(df_filtered)
        
        # Calcul de l'entropie
        entropy_val = -np.sum(frequencies * np.log2(frequencies)) if len(frequencies) > 0 else 0
        
        # Calcul du ratio de déséquilibre
        imbalance_ratio = value_counts.max() / value_counts.min() if len(value_counts) > 0 else 0
        
        category_distributions[col] = {
            'value_counts': value_counts.to_dict(),
            'most_common': value_counts.head(5).to_dict(),
            'n_categories': len(value_counts),
            'entropy': entropy_val,
            'imbalance_ratio': imbalance_ratio,
            'missing_ratio': df_filtered[col].isna().mean(),
            'unique_ratio': df_filtered[col].nunique() / len(df_filtered)
        }
        
        # Vérifier le nombre de catégories
        if len(value_counts) > max_categories:
            cols_to_drop.append(col)
            if verbose:
                print(f"'{col}' a trop de catégories : {len(value_counts)}")
        
        # Vérifier les valeurs rares
        elif (frequencies < rare_threshold).any():
            rare_categories = frequencies[frequencies < rare_threshold]
            if verbose:
                print(f"'{col}' a des catégories rares : {len(rare_categories)}")
    
    df_filtered = df_filtered.drop(columns=cols_to_drop)
    analysis_report['category_distributions'] = category_distributions
    if verbose:
        print(f"\nColonnes catégorielles supprimées : {len(cols_to_drop)}")
        print(f"Nouvelles dimensions : {df_filtered.shape}\n")
    
    # 5. Optimisation de la mémoire
    if verbose: print("\n5. Optimisation de la mémoire :")
    initial_memory = df_filtered.memory_usage(deep=True).sum() / 1024**2
    
    # Optimiser les types numériques
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        col_data = df_filtered[col]
        if not col_data.isna().any() and not np.isinf(col_data).any():
            has_decimals = not np.all(col_data == col_data.astype(int))
            optimal_type = get_optimal_numeric_type(col_data.min(), col_data.max(), has_decimals)
            try:
                df_filtered[col] = df_filtered[col].astype(optimal_type)
            except Exception as e:
                if verbose:
                    print(f"Impossible d'optimiser la colonne {col}: {str(e)}")
                continue
    
    # Optimiser les types catégoriels
    categorical_cols = df_filtered.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_filtered[col].nunique() / len(df_filtered) < 0.5:  # Si moins de 50% de valeurs uniques
            df_filtered[col] = df_filtered[col].astype('category')
    
    final_memory = df_filtered.memory_usage(deep=True).sum() / 1024**2
    memory_reduction = (initial_memory - final_memory) / initial_memory * 100
    
    # Générer un rapport détaillé sur l'utilisation de la mémoire
    memory_report = {
        'initial_mb': initial_memory,
        'final_mb': final_memory,
        'reduction_percentage': memory_reduction,
        'memory_by_dtype': {},
        'column_details': {}
    }
    
    # Analyse par type de données
    for dtype in df_filtered.dtypes.unique():
        cols = df_filtered.select_dtypes(include=[dtype]).columns
        memory = df_filtered[cols].memory_usage(deep=True).sum() / 1024**2  # Convertir en MB
        memory_report['memory_by_dtype'][str(dtype)] = {
            'memory_mb': memory,
            'percentage': (memory / final_memory) * 100,
            'n_columns': len(cols)
        }
    
    # Analyse par colonne
    for column in df_filtered.columns:
        memory = df_filtered[column].memory_usage(deep=True) / 1024**2  # Convertir en MB
        memory_report['column_details'][column] = {
            'memory_mb': memory,
            'percentage': (memory / final_memory) * 100,
            'dtype': str(df_filtered[column].dtype)
        }
    
    analysis_report['memory_usage'] = memory_report
    
    if verbose:
        print(f"Mémoire initiale : {initial_memory:.2f} MB")
        print(f"Mémoire finale : {final_memory:.2f} MB")
        print(f"Réduction : {memory_reduction:.2f}%\n")
    
    return df_filtered, analysis_report