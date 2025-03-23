import os
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from category_encoders import HashingEncoder, CountEncoder
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

class CategoricalFeatureEncoder:
    def __init__(
        self,
        sparse_output: bool = True,
        handle_unknown: str = 'ignore',
        n_components: int = 8,
        min_frequency: float = 0.01,
        save_dir: str = 'encoded_features'
    ):
        """
        Initialise l'encodeur de features catégorielles.
        
        Args:
            sparse_output: Si True, utilise des matrices sparses pour le stockage
            handle_unknown: Stratégie pour gérer les catégories inconnues
            n_components: Nombre de composantes pour le HashingEncoder
            min_frequency: Seuil minimum de fréquence pour les catégories
            save_dir: Répertoire pour sauvegarder les features encodées
        """
        self.sparse_output = sparse_output
        self.handle_unknown = handle_unknown
        self.n_components = n_components
        self.min_frequency = min_frequency
        self.save_dir = save_dir
        self.encoders = {}
        self.feature_names = {}
        
        # Créer le répertoire de sauvegarde si nécessaire
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def _save_encoded_feature(
        self,
        feature_name: str,
        encoded_data: Union[np.ndarray, sparse.csr_matrix],
        feature_names: List[str]
    ):
        """Sauvegarde une feature encodée sur le disque."""
        file_path = os.path.join(self.save_dir, f"{feature_name}_encoded.npz")
        if sparse.issparse(encoded_data):
            sparse.save_npz(file_path, encoded_data)
        else:
            np.savez(file_path, data=encoded_data, feature_names=feature_names)
    
    def _load_encoded_feature(self, feature_name: str) -> Tuple[Union[np.ndarray, sparse.csr_matrix], List[str]]:
        """Charge une feature encodée depuis le disque."""
        file_path = os.path.join(self.save_dir, f"{feature_name}_encoded.npz")
        data = np.load(file_path)
        if 'data' in data:
            return data['data'], data['feature_names']
        return sparse.load_npz(file_path), None
    
    def encode_incremental(
        self,
        df: pd.DataFrame,
        columns: List[str],
        method: str = 'onehot'
    ) -> pd.DataFrame:
        """
        Encode les features catégorielles de manière incrémentale.
        
        Args:
            df: DataFrame à encoder
            columns: Liste des colonnes à encoder
            method: Méthode d'encodage ('onehot', 'hashing', 'count')
            
        Returns:
            DataFrame avec les features encodées
        """
        df_encoded = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Vérifier si l'encodage existe déjà
            if os.path.exists(os.path.join(self.save_dir, f"{col}_encoded.npz")):
                encoded_data, feature_names = self._load_encoded_feature(col)
                if feature_names:
                    df_encoded[feature_names] = encoded_data
                else:
                    df_encoded[f"{col}_encoded"] = encoded_data
                continue
            
            # Encodage de la colonne
            if method == 'onehot':
                encoded_data, feature_names = self._encode_onehot(df[col])
            elif method == 'hashing':
                encoded_data, feature_names = self._encode_hashing(df[col])
            elif method == 'count':
                encoded_data, feature_names = self._encode_count(df[col])
            else:
                raise ValueError(f"Méthode d'encodage non supportée: {method}")
            
            # Sauvegarde de l'encodage
            self._save_encoded_feature(col, encoded_data, feature_names)
            
            # Ajout des colonnes encodées au DataFrame
            if feature_names:
                df_encoded[feature_names] = encoded_data
            else:
                df_encoded[f"{col}_encoded"] = encoded_data
        
        return df_encoded
    
    def _encode_onehot(self, series: pd.Series) -> Tuple[Union[np.ndarray, sparse.csr_matrix], List[str]]:
        """Encode une colonne avec OneHotEncoder."""
        ohe = OneHotEncoder(
            sparse_output=self.sparse_output,
            handle_unknown=self.handle_unknown
        )
        encoded = ohe.fit_transform(series.values.reshape(-1, 1))
        feature_names = [f"{series.name}_{cat}" for cat in ohe.categories_[0]]
        self.encoders[series.name] = ohe
        return encoded, feature_names
    
    def _encode_hashing(self, series: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Encode une colonne avec HashingEncoder."""
        hasher = HashingEncoder(n_components=self.n_components)
        encoded = hasher.fit_transform(series)
        feature_names = [f"{series.name}_hash_{i}" for i in range(self.n_components)]
        self.encoders[series.name] = hasher
        return encoded.values, feature_names
    
    def _encode_count(self, series: pd.Series) -> Tuple[np.ndarray, List[str]]:
        """Encode une colonne avec CountEncoder."""
        counter = CountEncoder()
        encoded = counter.fit_transform(series)
        feature_names = [f"{series.name}_count"]
        self.encoders[series.name] = counter
        return encoded.values, feature_names
    
    def handle_rare_categories(
        self,
        df: pd.DataFrame,
        columns: List[str],
        strategy: str = 'group'
    ) -> pd.DataFrame:
        """
        Gère les catégories rares dans les variables catégorielles.
        
        Args:
            df: DataFrame à traiter
            columns: Liste des colonnes à traiter
            strategy: 'group' pour regrouper, 'drop' pour supprimer
            
        Returns:
            DataFrame traité
        """
        df_processed = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            value_counts = df[col].value_counts(normalize=True)
            rare_categories = value_counts[value_counts < self.min_frequency].index
            
            if len(rare_categories) > 0:
                if strategy == 'group':
                    df_processed[col] = df_processed[col].replace(
                        rare_categories,
                        'Other'
                    )
                elif strategy == 'drop':
                    df_processed = df_processed[~df_processed[col].isin(rare_categories)]
        
        return df_processed
    
    def cluster_categories(
        self,
        df: pd.DataFrame,
        columns: List[str],
        n_clusters: int = 5
    ) -> pd.DataFrame:
        """
        Regroupe les catégories en utilisant le clustering.
        
        Args:
            df: DataFrame à traiter
            columns: Liste des colonnes à traiter
            n_clusters: Nombre de clusters
            
        Returns:
            DataFrame traité
        """
        df_processed = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            # Convertir les catégories en vecteurs TF-IDF
            vectorizer = TfidfVectorizer()
            try:
                X = vectorizer.fit_transform(df[col].astype(str))
                
                # Appliquer K-means
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(X)
                
                # Remplacer les catégories par leurs clusters
                df_processed[col] = clusters
            except:
                # Si la conversion échoue, on garde la colonne telle quelle
                continue
        
        return df_processed 