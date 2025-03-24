import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
import os
import re
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from enum import Enum

class EncodingStrategy(Enum):
    """Stratégies d'encodage disponibles"""
    ONE_HOT = "one_hot"
    COUNT = "count"
    HASH = "hash"

@dataclass
class EncodingConfig:
    """Configuration pour l'encodage"""
    strategy: EncodingStrategy = EncodingStrategy.ONE_HOT
    min_frequency: float = 0.005
    max_categories: int = 20
    hash_bits: int = 8
    sparse_output: bool = True

def clean_filename(filename: str) -> str:
    """Nettoie le nom de fichier pour le rendre valide.
    
    Args:
        filename: Nom de fichier à nettoyer
        
    Returns:
        Nom de fichier nettoyé
    """
    hash_object = hashlib.md5(str(filename).encode())
    return hash_object.hexdigest()[:8]

def should_encode_column(series: pd.Series) -> bool:
    """Détermine si une colonne doit être encodée.
    
    Args:
        series: Série pandas à analyser
        
    Returns:
        True si la colonne doit être encodée, False sinon
    """
    # Ne pas encoder les colonnes avec des URLs ou des dates
    if any(x in series.name.lower() for x in ['url', 'datetime', 'date', 'time']):
        return False
    # Ne pas encoder les colonnes avec trop de valeurs uniques (>90% des lignes)
    if series.nunique() > 0.9 * len(series):
        return False
    # Ne pas encoder les colonnes avec des URLs d'images
    if any(x in series.name.lower() for x in ['image', 'url']):
        return False
    # Ne pas encoder les colonnes avec des tags en double
    if series.name.endswith('_tags') and series.name.replace('_tags', '') in series.name:
        return False
    # Ne pas encoder les colonnes avec des traductions en double
    if series.name.endswith('_en') and series.name.replace('_en', '') in series.name:
        return False
    return True

def handle_rare_categories(series: pd.Series, 
                         min_frequency: float,
                         max_categories: int) -> pd.Series:
    """Gère les catégories rares dans une série.
    
    Args:
        series: Série pandas à traiter
        min_frequency: Seuil minimum de fréquence
        max_categories: Nombre maximum de catégories à conserver
        
    Returns:
        Série pandas avec les catégories rares gérées
    """
    value_counts = series.value_counts(normalize=True)
    # Garder les catégories fréquentes
    frequent_categories = value_counts[value_counts >= min_frequency].index
    # Si moins de 2 catégories fréquentes, augmenter le seuil
    if len(frequent_categories) < 2:
        frequent_categories = value_counts.head(2).index
    # Si plus de max_categories catégories fréquentes, ne garder que les plus fréquentes
    if len(frequent_categories) > max_categories:
        frequent_categories = value_counts.head(max_categories).index
    return series.apply(lambda x: x if x in frequent_categories else 'Other')

def encode_column(series: pd.Series, 
                 config: EncodingConfig) -> Union[sparse.csr_matrix, np.ndarray]:
    """Encode une colonne selon la stratégie choisie.
    
    Args:
        series: Série pandas à encoder
        config: Configuration d'encodage
        
    Returns:
        Matrice encodée (sparse ou dense)
    """
    if config.strategy == EncodingStrategy.ONE_HOT:
        encoder = OneHotEncoder(sparse_output=config.sparse_output, 
                              handle_unknown='ignore')
        return encoder.fit_transform(series.values.reshape(-1, 1))
    
    elif config.strategy == EncodingStrategy.COUNT:
        counts = series.value_counts()
        return counts[series].values.reshape(-1, 1)
    
    elif config.strategy == EncodingStrategy.HASH:
        # Implémentation simple du hash encoding
        hash_values = series.apply(lambda x: hash(str(x)) % (2**config.hash_bits))
        return hash_values.values.reshape(-1, 1)
    
    raise ValueError(f"Stratégie d'encodage non supportée: {config.strategy}")

def plot_categorical_distributions(df: pd.DataFrame, 
                                output_dir: str = 'encoded_features_auto',
                                top_n: int = 10) -> None:
    """Crée des visualisations pour les colonnes catégorielles.
    
    Args:
        df: DataFrame contenant les données
        output_dir: Dossier de sortie
        top_n: Nombre de catégories à afficher
    """
    viz_dir = os.path.join(output_dir, 'visualizations')
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    summary_stats = []
    
    for col in categorical_columns:
        if not should_encode_column(df[col]):
            continue
            
        value_counts = df[col].value_counts()
        total = len(df)
        percentages = (value_counts / total * 100).round(2)
        
        summary_stats.append({
            'Column': col,
            'Unique Values': len(value_counts),
            'Most Common': value_counts.index[0],
            'Most Common %': percentages.iloc[0],
            'Top 3 Categories': ', '.join([f"{cat} ({pct}%)" for cat, pct in zip(value_counts.index[:3], percentages[:3])])
        })
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=percentages[:top_n], y=percentages[:top_n].index)
        plt.title(f'Distribution des {top_n} catégories les plus fréquentes\npour la colonne {col}')
        plt.xlabel('Pourcentage')
        plt.ylabel('Catégories')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'distribution_{clean_filename(col)}.png'))
        plt.close()
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(viz_dir, 'categorical_summary.csv'), index=False)
    
    plt.figure(figsize=(15, 8))
    sns.barplot(data=summary_df, x='Unique Values', y='Column')
    plt.title('Nombre de catégories uniques par colonne')
    plt.xlabel('Nombre de catégories uniques')
    plt.ylabel('Colonnes')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'unique_categories_comparison.png'))
    plt.close()

def encode_categorical_features(data_path: str,
                              nrows: int = 10000,
                              config: Optional[EncodingConfig] = None) -> Dict:
    """Encode automatiquement toutes les colonnes catégorielles d'un dataset.
    
    Args:
        data_path: Chemin vers le fichier CSV
        nrows: Nombre de lignes à lire
        config: Configuration d'encodage (optionnel)
        
    Returns:
        Dictionnaire contenant les informations d'encodage
    """
    if config is None:
        config = EncodingConfig()
    
    output_dir = 'encoded_features_auto'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Chargement des données...")
    df = pd.read_csv(data_path, 
                    nrows=nrows, 
                    on_bad_lines='skip',
                    sep='\t',
                    encoding='utf-8',
                    low_memory=False)
    
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    print(f"\nColonnes catégorielles identifiées ({len(categorical_columns)}):")
    print(categorical_columns.tolist())
    
    encodings_info = {}
    
    for col in categorical_columns:
        try:
            if not should_encode_column(df[col]):
                print(f"\nColonne ignorée (non pertinente) : {col}")
                continue
                
            print(f"\nTraitement de la colonne : {col}")
            
            clean_col = clean_filename(col)
            
            # Gérer les valeurs manquantes
            df[col] = df[col].fillna('Unknown')
            
            # Gérer les catégories rares
            original_categories = df[col].nunique()
            df[col] = handle_rare_categories(df[col], 
                                          config.min_frequency,
                                          config.max_categories)
            
            # Encoder la colonne
            encoded_data = encode_column(df[col], config)
            
            # Sauvegarder la matrice encodée
            if config.sparse_output:
                sparse_file = os.path.join(output_dir, f'col_{clean_col}_sparse.npz')
                sparse.save_npz(sparse_file, encoded_data)
            else:
                np_file = os.path.join(output_dir, f'col_{clean_col}_dense.npy')
                np.save(np_file, encoded_data)
            
            # Sauvegarder les noms des catégories si OneHotEncoder
            if config.strategy == EncodingStrategy.ONE_HOT:
                feature_names = [f"{clean_col}_{clean_filename(str(cat))}" 
                               for cat in df[col].unique()]
                names_file = os.path.join(output_dir, f'col_{clean_col}_feature_names.npy')
                np.save(names_file, feature_names)
            
            # Collecter les informations
            encodings_info[col] = {
                'original_categories': original_categories,
                'encoded_categories': len(df[col].unique()),
                'sparsity': 1 - (encoded_data.nnz / (encoded_data.shape[0] * encoded_data.shape[1])) 
                           if config.sparse_output else 0,
                'file_prefix': f'col_{clean_col}',
                'strategy': config.strategy.value
            }
            
            print(f"Catégories originales : {original_categories}")
            print(f"Catégories après traitement : {len(df[col].unique())}")
            print(f"Taux de sparsité : {encodings_info[col]['sparsity']:.2%}")
            
        except Exception as e:
            print(f"Erreur lors du traitement de la colonne {col}: {str(e)}")
            continue
    
    stats_file = os.path.join(output_dir, 'encoding_stats.csv')
    pd.DataFrame(encodings_info).T.to_csv(stats_file)
    
    print("\nCréation des visualisations...")
    plot_categorical_distributions(df, output_dir)
    
    print("\nEncodage terminé ! Fichiers sauvegardés dans 'encoded_features_auto'")
    return encodings_info 