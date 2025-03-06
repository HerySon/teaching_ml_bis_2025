"""Tests pour le module smart_sampling."""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ajout du répertoire parent au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.smart_sampling import smart_sample, _auto_select_columns

@pytest.fixture
def generic_df():
    """Crée un DataFrame de test générique."""
    np.random.seed(42)
    n_samples = 1000
    
    # Génération de colonnes catégorielles avec différentes cardinalités
    categorical_data = {
        f'cat_{i}': np.random.choice([f'val_{j}' for j in range(n_unique)], n_samples)
        for i, n_unique in enumerate([5, 10, 20, 100])  # Test différentes cardinalités
    }
    
    # Génération de colonnes numériques avec différentes distributions
    numeric_data = {
        f'num_{i}': distribution(n_samples)
        for i, distribution in enumerate([
            lambda n: np.random.normal(100, 20, n),    # Distribution normale
            lambda n: np.random.exponential(50, n),    # Distribution exponentielle
            lambda n: np.random.uniform(0, 100, n),    # Distribution uniforme
            lambda n: np.random.poisson(30, n)         # Distribution de Poisson
        ])
    }
    
    # Ajout de colonnes à exclure typiques
    metadata = {
        'id': range(n_samples),
        'timestamp': np.random.randint(1000000, 9999999, n_samples),
        'url': [f'http://example.com/{i}' for i in range(n_samples)],
        'description': ['text'] * n_samples
    }
    
    # Combinaison de toutes les colonnes
    data = {**categorical_data, **numeric_data, **metadata}
    df = pd.DataFrame(data)
    
    # Ajout de valeurs manquantes de manière aléatoire
    for col in df.columns:
        mask = np.random.random(n_samples) < 0.1  # 10% de valeurs manquantes
        df.loc[mask, col] = np.nan
        
    return df

def test_column_selection(generic_df):
    """Teste la sélection automatique des colonnes de manière générique."""
    stratify_cols, numeric_cols = _auto_select_columns(generic_df)
    
    # Vérifications génériques
    assert len(stratify_cols) > 0, "Au moins une colonne de stratification doit être sélectionnée"
    assert len(numeric_cols) > 0, "Au moins une colonne numérique doit être sélectionnée"
    
    # Vérifie que les colonnes sélectionnées existent
    assert all(col in generic_df.columns for col in stratify_cols)
    assert all(col in generic_df.columns for col in numeric_cols)
    
    # Vérifie que les colonnes à exclure ne sont pas sélectionnées
    metadata_patterns = ['id', 'timestamp', 'url', 'description']
    assert not any(any(pattern in col for pattern in metadata_patterns) 
                  for col in stratify_cols + numeric_cols)

def test_sampling_properties(generic_df):
    """Teste les propriétés générales de l'échantillonnage."""
    # Test avec différentes tailles d'échantillon
    for ratio in [0.1, 0.25, 0.5]:
        target_size = int(len(generic_df) * ratio)
        sample, metrics = smart_sample(generic_df, target_size=target_size, verbose=False)
        
        # Vérifications de base
        assert len(sample) >= target_size * 0.9, f"Taille minimale non atteinte pour ratio {ratio}"
        assert len(sample) <= target_size * 1.1, f"Taille maximale dépassée pour ratio {ratio}"
        assert set(sample.columns) == set(generic_df.columns), "Toutes les colonnes doivent être préservées"
        
        # Vérifie la préservation des distributions pour toutes les colonnes numériques
        for col in sample.select_dtypes(include=['int64', 'float64']).columns:
            if col not in ['id', 'timestamp']:  # Ignore les colonnes de métadonnées
                orig_stats = generic_df[col].describe()
                sample_stats = sample[col].describe()
                
                # Vérifie que les statistiques principales sont similaires
                for stat in ['mean', 'std']:
                    if not pd.isna(orig_stats[stat]) and not pd.isna(sample_stats[stat]):
                        percent_diff = abs(orig_stats[stat] - sample_stats[stat]) / orig_stats[stat]
                        assert percent_diff < 0.2, f"Statistique {stat} trop différente pour {col}"

def test_robustness():
    """Teste la robustesse avec différents types de DataFrames."""
    # Test avec un DataFrame vide
    with pytest.raises(ValueError):
        smart_sample(pd.DataFrame())
    
    # Test avec une seule colonne
    df_single = pd.DataFrame({'A': range(100)})
    sample, _ = smart_sample(df_single)
    assert len(sample) > 0
    
    # Test avec uniquement des colonnes catégorielles
    df_cat = pd.DataFrame({
        'A': ['x', 'y', 'z'] * 33,
        'B': ['a', 'b'] * 50
    })
    sample, _ = smart_sample(df_cat)
    assert len(sample) > 0
    
    # Test avec uniquement des colonnes numériques
    df_num = pd.DataFrame({
        'A': np.random.normal(0, 1, 100),
        'B': np.random.uniform(0, 1, 100)
    })
    sample, _ = smart_sample(df_num)
    assert len(sample) > 0 