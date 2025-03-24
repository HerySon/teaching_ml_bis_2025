"""
Module pour le sous-échantillonnage intelligent et stratifié des datasets.

Ce module fournit des fonctions pour analyser et sous-échantillonner des datasets
de manière intelligente en utilisant différentes stratégies, notamment le
sous-échantillonnage stratifié.
"""

import os
import sys
import pytest  # Packages tiers après les packages standard
import pandas as pd
import numpy as np
from typing import Union, Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from scipy import stats


def _auto_select_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Sélectionne intelligemment les meilleures colonnes pour la stratification
    et l'analyse numérique.

    La sélection est basée sur plusieurs critères :
    1. Pour les colonnes catégorielles :
        - Nombre de catégories uniques (entre 2 et 50)
        - Ratio de valeurs non nulles (> 30%)
        - Distribution équilibrée des catégories
    2. Pour les colonnes numériques :
        - Ratio de valeurs non nulles (> 30%)
        - Ratio de valeurs uniques (entre 1% et 90%)
        - Variance significative
        - Absence de valeurs extrêmes excessives

    Args:
        df: DataFrame à analyser

    Returns:
        Tuple[List[str], List[str]]: (colonnes de stratification, colonnes numériques)
    """
    # Colonnes à exclure automatiquement (par pattern)
    exclude_patterns = [
        '_t$', '_datetime$', 'url', 'code', 'creator', 'created',
        'modified', 'updated', 'id', 'uuid', 'guid', 'hash', 'key',
        'index', 'timestamp', 'date', 'time', 'version'
    ]
    
    def is_excluded(col: str) -> bool:
        """Vérifie si une colonne doit être exclue."""
        col_lower = col.lower()
        return any(pattern in col_lower for pattern in exclude_patterns)
    
    def calculate_distribution_score(series: pd.Series) -> float:
        """Calcule un score d'équilibre de distribution pour une série."""
        counts = series.value_counts(normalize=True)
        entropy = -(counts * np.log(counts)).sum()  # Entropie de Shannon
        max_entropy = np.log(len(counts))  # Entropie maximale possible
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def calculate_numeric_score(series: pd.Series) -> float:
        """Calcule un score de qualité pour une colonne numérique."""
        if series.std() == 0:  # Évite les colonnes constantes
            return 0
            
        # Calcul des z-scores pour détecter les valeurs extrêmes
        z_scores = np.abs((series - series.mean()) / series.std())
        outlier_ratio = (z_scores > 3).mean()
        
        # Normalisation des critères entre 0 et 1
        non_null_score = series.notna().mean()
        unique_score = min(1, series.nunique() / (len(series) * 0.1))  # Max 10% unique
        variation_score = min(1, series.std() / series.mean() if series.mean() != 0 else 0)
        outlier_score = 1 - outlier_ratio
        
        # Combinaison pondérée des scores
        return (non_null_score * 0.3 +
                unique_score * 0.2 +
                variation_score * 0.3 +
                outlier_score * 0.2)
    
    # Identification initiale des types de colonnes
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Analyse approfondie des colonnes numériques
    numeric_scores = {}
    for col in numeric_cols:
        if is_excluded(col):
            continue
            
        series = df[col]
        if series.notna().mean() < 0.3:  # Minimum 30% de valeurs non nulles
            continue
            
        score = calculate_numeric_score(series)
        if score > 0.5:  # Seuil minimal de qualité
            numeric_scores[col] = score
    
    # Analyse approfondie des colonnes catégorielles
    categorical_scores = {}
    for col in categorical_cols:
        if is_excluded(col):
            continue
            
        series = df[col]
        n_unique = series.nunique()
        if not (2 <= n_unique <= 50):  # Entre 2 et 50 catégories
            continue
            
        non_null_ratio = series.notna().mean()
        if non_null_ratio < 0.3:  # Minimum 30% de valeurs non nulles
            continue
            
        distribution_score = calculate_distribution_score(series.dropna())
        if distribution_score > 0.3:  # Distribution suffisamment équilibrée
            categorical_scores[col] = distribution_score
    
    # Sélection des meilleures colonnes
    valid_numeric_cols = sorted(numeric_scores.keys(),
                              key=lambda x: numeric_scores[x],
                              reverse=True)[:5]  # Top 5 colonnes numériques
                              
    valid_categorical_cols = sorted(categorical_scores.keys(),
                                  key=lambda x: categorical_scores[x],
                                  reverse=True)[:3]  # Top 3 colonnes catégorielles
    
    # Si aucune colonne catégorielle n'est trouvée, créer des bins sur les meilleures colonnes numériques
    if not valid_categorical_cols and valid_numeric_cols:
        best_numeric = valid_numeric_cols[0]
        df[f'{best_numeric}_binned'] = pd.qcut(
            df[best_numeric].fillna(df[best_numeric].median()),
            q=min(5, df[best_numeric].nunique()),
            labels=[f'Q{i+1}' for i in range(5)],
            duplicates='drop'
        )
        valid_categorical_cols.append(f'{best_numeric}_binned')
    
    # Affichage des informations
    print("\nColonnes sélectionnées pour la stratification:")
    for col in valid_categorical_cols:
        if col in categorical_scores:
            print(f"- {col}: {df[col].nunique()} catégories, "
                  f"score={categorical_scores[col]:.3f}")
        else:
            print(f"- {col}: {df[col].nunique()} catégories (binned)")
    
    print("\nColonnes numériques sélectionnées:")
    for col in valid_numeric_cols:
        print(f"- {col}: score={numeric_scores[col]:.3f}, "
              f"std={df[col].std():.3f}, "
              f"missing={df[col].isna().mean():.1%}")
    
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
    if len(df) == 0:
        raise ValueError("Le DataFrame est vide")
        
    if target_size is None:
        target_size = len(df) // 4
    
    # Sélection automatique des colonnes
    stratify_cols, numeric_cols = _auto_select_columns(df)
    
    # Création de la colonne de stratification combinée
    df = df.copy()
    
    # Si aucune colonne n'est disponible, utiliser un échantillonnage aléatoire
    if not stratify_cols and not numeric_cols:
        sample = df.sample(n=target_size, random_state=random_state)
        metrics = {'sampling_method': 'random', 'reduction_ratio': len(sample) / len(df)}
        return sample, metrics
    
    # Sinon, utiliser la stratification
    if stratify_cols:
        df['combined_strata'] = df[stratify_cols].astype(str).agg('-'.join, axis=1)
    else:
        # Utiliser des bins sur la première colonne numérique
        first_numeric = numeric_cols[0]
        df['combined_strata'] = pd.qcut(
            df[first_numeric].fillna(df[first_numeric].median()),
            q=min(10, df[first_numeric].nunique()),
            labels=False,
            duplicates='drop'
        )
    
    # Calcul des proportions cibles pour chaque strate
    strata_counts = df['combined_strata'].value_counts()
    total_samples = len(df)
    
    # Calcul du nombre d'échantillons par strate
    samples_per_stratum = {}
    for stratum in strata_counts.index:
        stratum_ratio = strata_counts[stratum] / total_samples
        samples_per_stratum[stratum] = max(1, int(target_size * stratum_ratio))
    
    # Ajustement pour atteindre exactement la taille cible
    total_allocated = sum(samples_per_stratum.values())
    if total_allocated > target_size:
        # Réduire proportionnellement chaque strate
        reduction_ratio = target_size / total_allocated
        for stratum in samples_per_stratum:
            samples_per_stratum[stratum] = max(1, int(samples_per_stratum[stratum] * reduction_ratio))
    
    # Échantillonnage stratifié avec les tailles ajustées
    sampled_dfs = []
    for stratum, size in samples_per_stratum.items():
        stratum_df = df[df['combined_strata'] == stratum]
        if len(stratum_df) > size:
            sampled_dfs.append(stratum_df.sample(n=size, random_state=random_state))
        else:
            sampled_dfs.append(stratum_df)  # Prendre toute la strate si trop petite
    
    # Combinaison des échantillons
    sample = pd.concat(sampled_dfs, axis=0)
    
    # Si l'échantillon est encore trop grand, réduire aléatoirement
    if len(sample) > target_size:
        sample = sample.sample(n=target_size, random_state=random_state)
    
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
    """
    Calcule les métriques de qualité de l'échantillonnage.
    
    Métriques calculées :
    1. Ratio de réduction
    2. Pour chaque colonne de stratification :
        - Différence moyenne des proportions
        - Score de Kullback-Leibler (divergence)
        - Test du chi2 d'indépendance
    3. Pour chaque colonne numérique :
        - Différence relative des moyennes
        - Différence relative des écarts-types
        - Test de Kolmogorov-Smirnov
    """
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
        
        # Différence moyenne des proportions
        diff = np.abs(orig_props - sample_props.reindex(orig_props.index).fillna(0)).mean()
        metrics[f"{col}_prop_difference"] = diff
        
        # Divergence de Kullback-Leibler (avec lissage pour éviter div/0)
        eps = 1e-10
        kl_div = np.sum(orig_props * np.log((orig_props + eps) / (sample_props.reindex(orig_props.index).fillna(eps) + eps)))
        metrics[f"{col}_kl_divergence"] = kl_div
        
        # Test du chi2
        orig_counts = original_df[col].value_counts()
        sample_counts = sampled_df[col].value_counts()
        chi2, pval = stats.chi2_contingency([orig_counts, sample_counts])[0:2]
        metrics[f"{col}_chi2_pvalue"] = pval
    
    # Métriques pour les colonnes numériques
    for col in numeric_cols:
        # Statistiques de base
        orig_mean = original_df[col].mean()
        orig_std = original_df[col].std()
        
        # Différences relatives
        mean_diff = abs(orig_mean - sampled_df[col].mean()) / (orig_std if orig_std != 0 else 1)
        std_diff = abs(orig_std - sampled_df[col].std()) / (orig_std if orig_std != 0 else 1)
        
        metrics[f"{col}_mean_difference"] = mean_diff
        metrics[f"{col}_std_difference"] = std_diff
        
        # Test de Kolmogorov-Smirnov
        ks_stat, ks_pval = stats.ks_2samp(
            original_df[col].dropna(),
            sampled_df[col].dropna()
        )
        metrics[f"{col}_ks_pvalue"] = ks_pval
        
        # Quartiles
        orig_quant = original_df[col].quantile([0.25, 0.5, 0.75])
        sample_quant = sampled_df[col].quantile([0.25, 0.5, 0.75])
        metrics[f"{col}_quartile_differences"] = {
            'Q1': abs(orig_quant[0.25] - sample_quant[0.25]) / (orig_std if orig_std != 0 else 1),
            'Q2': abs(orig_quant[0.5] - sample_quant[0.5]) / (orig_std if orig_std != 0 else 1),
            'Q3': abs(orig_quant[0.75] - sample_quant[0.75]) / (orig_std if orig_std != 0 else 1)
        }
    
    return metrics


def _plot_distributions(
    original_df: pd.DataFrame,
    sampled_df: pd.DataFrame,
    stratify_cols: List[str],
    numeric_cols: List[str]
) -> None:
    """
    Génère des visualisations détaillées des distributions.
    
    Pour les colonnes catégorielles :
    - Distribution des catégories (barplot)
    - Distribution cumulative
    - Heatmap des corrélations entre colonnes catégorielles
    
    Pour les colonnes numériques :
    - Boxplots comparatifs
    - Density plots
    - Q-Q plots
    - Violin plots pour la comparaison des distributions
    """
    try:
        # Configuration du style de base
        plt.rcParams.update({
            'figure.figsize': [10, 6],
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 10,
            'axes.titlesize': 12
        })
        
        colors = ['#2ecc71', '#e74c3c']  # Vert pour original, rouge pour échantillon
        
        # Plot pour les colonnes de stratification
        if stratify_cols:
            n_cols = len(stratify_cols)
            fig = plt.figure(figsize=(15, 5 * n_cols))
            gs = plt.GridSpec(n_cols, 3, figure=fig)
            
            for idx, col in enumerate(stratify_cols):
                try:
                    # Distribution des catégories
                    ax1 = fig.add_subplot(gs[idx, 0])
                    data = pd.DataFrame({
                        'Original': original_df[col].value_counts(normalize=True),
                        'Échantillon': sampled_df[col].value_counts(normalize=True)
                    })
                    data.plot(kind='bar', ax=ax1, color=colors)
                    ax1.set_title(f'Distribution - {col}')
                    ax1.tick_params(axis='x', rotation=45)
                    
                    # Distribution cumulative
                    ax2 = fig.add_subplot(gs[idx, 1])
                    data.cumsum().plot(ax=ax2, color=colors)
                    ax2.set_title(f'Distribution cumulative - {col}')
                    ax2.grid(True, alpha=0.3)
                    
                    # Graphique de parité
                    ax3 = fig.add_subplot(gs[idx, 2])
                    orig_props = original_df[col].value_counts(normalize=True)
                    sample_props = sampled_df[col].value_counts(normalize=True)
                    max_val = max(orig_props.max(), sample_props.max())
                    ax3.scatter(orig_props, sample_props.reindex(orig_props.index), alpha=0.6)
                    ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
                    ax3.set_xlabel('Proportions originales')
                    ax3.set_ylabel('Proportions échantillon')
                    ax3.set_title('Graphique de parité')
                except Exception as e:
                    print(f"Erreur lors de la visualisation de {col}: {str(e)}")
                    continue
            
            plt.tight_layout()
            plt.show()
            
            # Heatmap des corrélations entre variables catégorielles
            if len(stratify_cols) > 1:
                try:
                    plt.figure(figsize=(10, 8))
                    cramers_matrix = np.zeros((len(stratify_cols), len(stratify_cols)))
                    
                    def cramers_v(x, y):
                        confusion_matrix = pd.crosstab(x, y)
                        chi2 = stats.chi2_contingency(confusion_matrix)[0]
                        n = confusion_matrix.sum().sum()
                        min_dim = min(confusion_matrix.shape) - 1
                        return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
                    
                    for i, col1 in enumerate(stratify_cols):
                        for j, col2 in enumerate(stratify_cols):
                            cramers_matrix[i, j] = cramers_v(original_df[col1], original_df[col2])
                    
                    plt.imshow(cramers_matrix, cmap='YlOrRd', aspect='auto')
                    plt.colorbar()
                    plt.xticks(range(len(stratify_cols)), stratify_cols, rotation=45)
                    plt.yticks(range(len(stratify_cols)), stratify_cols)
                    
                    # Ajout des valeurs dans les cellules
                    for i in range(len(stratify_cols)):
                        for j in range(len(stratify_cols)):
                            plt.text(j, i, f'{cramers_matrix[i, j]:.2f}',
                                   ha='center', va='center')
                    
                    plt.title("Corrélations entre variables catégorielles (V de Cramér)")
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Erreur lors de la création de la heatmap: {str(e)}")
        
        # Plot pour les colonnes numériques
        if numeric_cols:
            n_cols = len(numeric_cols)
            fig = plt.figure(figsize=(20, 5 * n_cols))
            gs = plt.GridSpec(n_cols, 4, figure=fig)
            
            for idx, col in enumerate(numeric_cols):
                try:
                    # Boxplot
                    ax1 = fig.add_subplot(gs[idx, 0])
                    data = [original_df[col].dropna(), sampled_df[col].dropna()]
                    ax1.boxplot(data, labels=['Original', 'Échantillon'])
                    ax1.set_title(f'Boxplot - {col}')
                    
                    # Histogramme
                    ax2 = fig.add_subplot(gs[idx, 1])
                    for data, label, color in zip([original_df[col].dropna(), sampled_df[col].dropna()],
                                                ['Original', 'Échantillon'],
                                                colors):
                        ax2.hist(data, bins='auto', alpha=0.5, label=label, color=color)
                    ax2.set_title(f'Distribution - {col}')
                    ax2.legend()
                    
                    # Q-Q plot
                    ax3 = fig.add_subplot(gs[idx, 2])
                    stats.probplot(sampled_df[col].dropna(), dist="norm", plot=ax3)
                    ax3.set_title(f'Q-Q Plot - {col}')
                    
                    # Distribution cumulative empirique
                    ax4 = fig.add_subplot(gs[idx, 3])
                    for data, label, color in zip([original_df[col].dropna(), sampled_df[col].dropna()],
                                                ['Original', 'Échantillon'],
                                                colors):
                        data_sorted = np.sort(data)
                        p = np.arange(len(data_sorted)) / (len(data_sorted) - 1)
                        ax4.plot(data_sorted, p, label=label, color=color)
                    ax4.set_title('Distribution cumulative')
                    ax4.legend()
                except Exception as e:
                    print(f"Erreur lors de la visualisation de {col}: {str(e)}")
                    continue
            
            plt.tight_layout()
            plt.show()
            
            # Matrice de corrélation pour les variables numériques
            if len(numeric_cols) > 1:
                try:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
                    
                    # Corrélations dans le dataset original
                    corr_orig = original_df[numeric_cols].corr()
                    im1 = ax1.imshow(corr_orig, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                    plt.colorbar(im1, ax=ax1)
                    ax1.set_xticks(range(len(numeric_cols)))
                    ax1.set_yticks(range(len(numeric_cols)))
                    ax1.set_xticklabels(numeric_cols, rotation=45)
                    ax1.set_yticklabels(numeric_cols)
                    ax1.set_title("Corrélations - Dataset Original")
                    
                    # Ajout des valeurs dans les cellules
                    for i in range(len(numeric_cols)):
                        for j in range(len(numeric_cols)):
                            ax1.text(j, i, f'{corr_orig.iloc[i, j]:.2f}',
                                   ha='center', va='center')
                    
                    # Corrélations dans l'échantillon
                    corr_sample = sampled_df[numeric_cols].corr()
                    im2 = ax2.imshow(corr_sample, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                    plt.colorbar(im2, ax=ax2)
                    ax2.set_xticks(range(len(numeric_cols)))
                    ax2.set_yticks(range(len(numeric_cols)))
                    ax2.set_xticklabels(numeric_cols, rotation=45)
                    ax2.set_yticklabels(numeric_cols)
                    ax2.set_title("Corrélations - Échantillon")
                    
                    # Ajout des valeurs dans les cellules
                    for i in range(len(numeric_cols)):
                        for j in range(len(numeric_cols)):
                            ax2.text(j, i, f'{corr_sample.iloc[i, j]:.2f}',
                                   ha='center', va='center')
                    
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    print(f"Erreur lors de la création des matrices de corrélation: {str(e)}")
    
    except Exception as e:
        print(f"Erreur lors de la création des visualisations: {str(e)}")
        print("Les visualisations ne sont pas disponibles, mais l'échantillonnage a été effectué.")


def test_column_selection(generic_df):
    """Teste la sélection automatique des colonnes."""
    stratify_cols, numeric_cols = _auto_select_columns(generic_df)
    
    # Vérification de la présence des colonnes
    assert len(stratify_cols) > 0, (
        "Au moins une colonne de stratification doit être sélectionnée"
    )
    assert len(numeric_cols) > 0, (
        "Au moins une colonne numérique doit être sélectionnée"
    )
    
    # Vérification de l'existence des colonnes
    assert all(col in generic_df.columns for col in stratify_cols)
    assert all(col in generic_df.columns for col in numeric_cols)
    
    # Vérification de l'exclusion des métadonnées
    metadata_patterns = ['id', 'timestamp', 'url', 'description']
    assert not any(
        any(pattern in col for pattern in metadata_patterns)
        for col in stratify_cols + numeric_cols
    )
    
    # Vérification des colonnes numériques
    assert all(
        '_100g' in col for col in numeric_cols
    ), "Les colonnes numériques doivent contenir '_100g'"


def test_sampling_properties(generic_df):
    """Teste les propriétés de l'échantillonnage."""
    for ratio in [0.1, 0.25, 0.5]:
        target_size = int(len(generic_df) * ratio)
        sample, metrics = smart_sample(
            generic_df,
            target_size=target_size,
            verbose=False
        )
        
        # Vérification de la taille
        assert len(sample) >= target_size * 0.9, (
            f"Taille minimale non atteinte pour ratio {ratio}"
        )
        assert len(sample) <= target_size * 1.1, (
            f"Taille maximale dépassée pour ratio {ratio}"
        )
        
        # Vérification des colonnes
        assert set(sample.columns) == set(generic_df.columns), (
            "Toutes les colonnes doivent être préservées"
        )
        
        # Vérification des distributions
        for col in [c for c in sample.columns if '_100g' in c]:
            _verify_distribution(generic_df, sample, col)


def _verify_distribution(original, sample, column):
    """Vérifie la préservation de la distribution d'une colonne."""
    orig_stats = original[column].describe()
    sample_stats = sample[column].describe()
    
    for stat in ['mean', 'std']:
        if (not pd.isna(orig_stats[stat]) and 
            not pd.isna(sample_stats[stat])):
            percent_diff = abs(
                orig_stats[stat] - sample_stats[stat]
            ) / orig_stats[stat]
            assert percent_diff < 0.2, (
                f"Statistique {stat} trop différente pour {column}"
            ) 