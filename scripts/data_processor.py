"""
Module pour le traitement et l'optimisation des DataFrames pandas.

Ce module fournit une classe DataFrameProcessor qui permet de détecter automatiquement
les types de colonnes, d'optimiser la mémoire et de sélectionner les colonnes pertinentes
pour l'analyse de données.
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataFrameProcessor')

class DataFrameProcessor:
    """
    Classe pour traiter un DataFrame pandas, détecter les types de colonnes,
    optimiser la mémoire et sélectionner les colonnes pertinentes pour l'analyse.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialise le processeur avec un DataFrame pandas.
        
        Args:
            df: Le DataFrame à traiter
        """
        self.df = df.copy()
        self.original_memory = self.df.memory_usage(deep=True).sum()
        self.column_types = {
            'numeric': [],
            'categorical_ordinal': [],
            'categorical_nominal': [],
            'datetime': [],
            'text': [],
            'other': []
        }
        self.ordinal_columns_mapping = {
            'nutriscore_grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5},
            'nova_group': {1: 1, 2: 2, 3: 3, 4: 4},
            'environmental_score_grade': {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5}
            # Ajouter d'autres colonnes ordinales ici si nécessaire
        }
    
    def detect_column_types(self, 
                           text_max_categories: int = 1000,
                           ordinal_columns: Optional[Dict[str, Dict]] = None
                           ) -> Dict[str, List[str]]:
        """
        Détecte automatiquement les types de colonnes dans le DataFrame.
        
        Args:
            text_max_categories: Nombre maximum de catégories pour considérer une colonne 
            comme textuelle
            ordinal_columns: Dictionnaire des colonnes ordinales avec leurs mappages
            
        Returns:
            Dictionnaire contenant les colonnes classées par type
        """
        logger.info("Détection des types de colonnes...")
        
        # Utiliser le mapping fourni ou celui par défaut
        if ordinal_columns:
            self.ordinal_columns_mapping.update(ordinal_columns)
            
        # Réinitialisation des catégories
        for key in self.column_types:
            self.column_types[key] = []
            
        for col in self.df.columns:
            # Ignorer les colonnes avec toutes les valeurs nulles
            if self.df[col].isna().all():
                self.column_types['other'].append(col)
                continue
                
            dtype = self.df[col].dtype
            
            # Colonnes numériques
            if np.issubdtype(dtype, np.number):
                # Vérifier si c'est une colonne ordinale
                if col in self.ordinal_columns_mapping:
                    self.column_types['categorical_ordinal'].append(col)
                else:
                    # Si a peu de valeurs uniques, peut être catégorielle
                    unique_count = self.df[col].nunique()
                    if unique_count < 20 and unique_count / len(self.df) < 0.01:
                        self.column_types['categorical_nominal'].append(col)
                    else:
                        self.column_types['numeric'].append(col)
            
            # Colonnes de type datetime
            elif pd.api.types.is_datetime64_dtype(dtype) or (
                dtype == 'object' and 
                'datetime' in col and 
                self.df[col].str.contains('T.*Z').mean() > 0.5
            ):
                self.column_types['datetime'].append(col)
            
            # Colonnes catégorielles
            elif dtype == 'object' or pd.api.types.is_categorical_dtype(dtype):
                # Colonnes ordinales connues
                if col in self.ordinal_columns_mapping:
                    self.column_types['categorical_ordinal'].append(col)
                else:
                    # Analyser le nombre de catégories uniques
                    unique_count = self.df[col].nunique()
                    
                    # Si trop de catégories, considérer comme texte ou autre
                    if unique_count > text_max_categories:
                        self.column_types['text'].append(col)
                    else:
                        self.column_types['categorical_nominal'].append(col)
            else:
                self.column_types['other'].append(col)
                
        # Afficher un résumé
        for col_type, cols in self.column_types.items():
            logger.info("%s: %d colonnes", col_type, len(cols))
            
        return self.column_types
    
    def optimize_memory(self, 
                       convert_numerics: bool = True, 
                       convert_categories: bool = True,
                       verbose: bool = True) -> pd.DataFrame:
        """
        Optimise l'utilisation de la mémoire du DataFrame par downcasting.
        
        Args:
            convert_numerics: Si True, convertit les types numériques
            convert_categories: Si True, convertit les colonnes catégorielles
            verbose: Si True, affiche les informations de conversion
            
        Returns:
            DataFrame optimisé
        """
        logger.info("Optimisation de la mémoire...")
        
        # Si les types de colonnes n'ont pas été détectés
        if not any(self.column_types.values()):
            self.detect_column_types()
            
        df_optimized = self.df.copy()
        
        # Downcast des colonnes numériques
        if convert_numerics:
            for col in self.column_types['numeric']:
                col_data = df_optimized[col]
                
                # Ignorer les colonnes avec toutes les valeurs NaN
                if col_data.isna().all():
                    continue
                    
                # Convertir les entiers
                if pd.api.types.is_integer_dtype(col_data):
                    # Déterminer le type d'entier minimum requis
                    min_val = col_data.min()
                    max_val = col_data.max()
                    
                    if min_val >= 0:  # Unsigned int
                        if max_val <= 255:
                            df_optimized[col] = col_data.astype(np.uint8)
                        elif max_val <= 65535:
                            df_optimized[col] = col_data.astype(np.uint16)
                        elif max_val <= 4294967295:
                            df_optimized[col] = col_data.astype(np.uint32)
                    else:  # Signed int
                        if min_val >= -128 and max_val <= 127:
                            df_optimized[col] = col_data.astype(np.int8)
                        elif min_val >= -32768 and max_val <= 32767:
                            df_optimized[col] = col_data.astype(np.int16)
                        elif min_val >= -2147483648 and max_val <= 2147483647:
                            df_optimized[col] = col_data.astype(np.int32)
                
                # Convertir les flottants
                elif pd.api.types.is_float_dtype(col_data):
                    # Tester si float32 est suffisant (moins précis mais prend moins de place)
                    df_optimized[col] = col_data.astype(np.float32)
        
        # Convertir les colonnes catégorielles en type 'category'
        if convert_categories:
            for col_type in ['categorical_ordinal', 'categorical_nominal']:
                for col in self.column_types[col_type]:
                    if col in df_optimized.columns:
                        df_optimized[col] = df_optimized[col].astype('category')
        
        # Afficher les économies de mémoire réalisées
        if verbose:
            new_memory = df_optimized.memory_usage(deep=True).sum()
            memory_reduction = 100 * (1 - new_memory / self.original_memory)
            logger.info("Mémoire initiale: %.2f MB", self.original_memory / 1e6)
            logger.info("Mémoire optimisée: %.2f MB", new_memory / 1e6)
            logger.info("Réduction: %.2f%%", memory_reduction)
        
        return df_optimized
    
    def filter_categorical_columns(self, 
                                  min_categories: int = 2, 
                                  max_categories: int = 50,
                                  max_missing_pct: float = 0.8) -> Tuple[List[str], List[str]]:
        """
        Filtre les colonnes catégorielles selon différents critères.
        
        Args:
            min_categories: Nombre minimum de catégories
            max_categories: Nombre maximum de catégories
            max_missing_pct: Pourcentage maximum de valeurs manquantes
            
        Returns:
            Tuple contenant (colonnes_retenues, colonnes_rejetées)
        """
        logger.info("Filtrage des colonnes catégorielles...")
        
        # Si les types de colonnes n'ont pas été détectés
        if not any(self.column_types.values()):
            self.detect_column_types()
            
        retained_columns = []
        rejected_columns = []
        
        # Combiner les colonnes catégorielles ordinales et non-ordinales
        categorical_columns = (self.column_types['categorical_ordinal'] + 
                               self.column_types['categorical_nominal'])
        
        for col in categorical_columns:
            # Calculer le pourcentage de valeurs manquantes
            missing_pct = self.df[col].isna().mean()
            
            # Compter le nombre de catégories uniques (en ignorant les NaN)
            unique_count = self.df[col].dropna().nunique()
            
            # Vérifier les critères
            if (missing_pct <= max_missing_pct and 
                unique_count >= min_categories and 
                unique_count <= max_categories):
                retained_columns.append(col)
            else:
                reason = []
                if missing_pct > max_missing_pct:
                    reason.append(f"trop de valeurs manquantes ({missing_pct:.2%})")
                if unique_count < min_categories:
                    reason.append(f"pas assez de catégories ({unique_count})")
                if unique_count > max_categories:
                    reason.append(f"trop de catégories ({unique_count})")
                    
                rejected_columns.append((col, ", ".join(reason)))
        
        # Journaliser les résultats
        logger.info("Colonnes catégorielles retenues: %d", len(retained_columns))
        logger.info("Colonnes catégorielles rejetées: %d", len(rejected_columns))
        
        # Afficher les raisons de rejet pour les premières colonnes
        for col, reason in rejected_columns[:10]:
            logger.debug("Rejeté: %s - %s", col, reason)
        
        return retained_columns, [col for col, _ in rejected_columns]
    
    def select_relevant_columns(self, 
                               min_categorical_categories: int = 2,
                               max_categorical_categories: int = 50,
                               max_missing_pct: float = 0.8,
                               include_text: bool = False,
                               include_datetime: bool = False,
                               include_url: bool = False) -> pd.DataFrame:
        """
        Sélectionne les colonnes pertinentes selon les critères spécifiés.
        
        Args:
            min_categorical_categories: Nombre minimum de catégories pour les colonnes catégorielles
            max_categorical_categories: Nombre maximum de catégories pour les colonnes catégorielles
            max_missing_pct: Pourcentage maximum de valeurs manquantes
            include_text: Si True, inclut les colonnes textuelles
            include_datetime: Si True, inclut les colonnes de type datetime (False par défaut)
            include_url: Si True, inclut les colonnes contenant "url" dans 
            leur nom (False par défaut)
            
        Returns:
            DataFrame avec uniquement les colonnes pertinentes
        """
        logger.info("Sélection des colonnes pertinentes...")
        
        # Si les types de colonnes n'ont pas été détectés
        if not any(self.column_types.values()):
            self.detect_column_types()
        
        # Filtrer les colonnes numériques avec trop de valeurs manquantes
        numeric_cols = [col for col in self.column_types['numeric'] 
                       if self.df[col].isna().mean() <= max_missing_pct]
        
        # Filtrer les colonnes catégorielles
        categorical_cols, _ = self.filter_categorical_columns(
            min_categories=min_categorical_categories,
            max_categories=max_categorical_categories,
            max_missing_pct=max_missing_pct
        )
        
        # Colonnes de base à inclure
        columns_to_include = numeric_cols + categorical_cols
        
        # Ajouter les colonnes de date si nécessaire
        datetime_cols = []
        if include_datetime:
            datetime_cols = [col for col in self.column_types['datetime'] 
                            if self.df[col].isna().mean() <= max_missing_pct]
            columns_to_include.extend(datetime_cols)
        
        # Ajouter les colonnes textuelles si nécessaire
        text_cols = []
        if include_text:
            text_cols = [col for col in self.column_types['text'] 
                        if self.df[col].isna().mean() <= max_missing_pct]
            columns_to_include.extend(text_cols)
        
        # Filtrer les colonnes URL si nécessaire
        url_cols = []
        if not include_url:
            # Identifier les colonnes qui contiennent "url" dans leur nom
            url_cols = [col for col in columns_to_include if 'url' in col.lower()]
            # Retirer ces colonnes
            columns_to_include = [col for col in columns_to_include if col not in url_cols]
        
        # Créer le dataframe avec les colonnes sélectionnées
        df_relevant = self.df[columns_to_include].copy()
        
        # Statistiques pour le log
        non_url_numeric_cols = [col for col in numeric_cols if col not in url_cols]
        non_url_categorical_cols = [col for col in categorical_cols if col not in url_cols]
        
        logger.info("Colonnes sélectionnées: %d sur %d", 
                    len(columns_to_include), len(self.df.columns))
        logger.info(
            "Numériques: %d, Catégorielles: %d, URL filtrées: %d, Datetime: %d, Texte: %d",
            len(non_url_numeric_cols), 
            len(non_url_categorical_cols),
            len(url_cols),
            len(datetime_cols) if include_datetime else 0,
            len(text_cols) if include_text else 0
        )
        
        return df_relevant

    def get_column_stats(self) -> pd.DataFrame:
        """
        Génère des statistiques sur les colonnes du DataFrame.
        
        Returns:
            DataFrame contenant les statistiques des colonnes
        """
        # Si les types de colonnes n'ont pas été détectés
        if not any(self.column_types.values()):
            self.detect_column_types()
            
        stats = []
        
        for col in self.df.columns:
            # Déterminer le type de colonne
            col_type = next((t for t, cols in self.column_types.items() if col in cols), "unknown")
            
            # Calculer les statistiques de base
            missing_count = self.df[col].isna().sum()
            missing_pct = missing_count / len(self.df)
            unique_count = self.df[col].nunique()
            unique_pct = unique_count / len(self.df)
            
            # Ajouter aux statistiques
            stats.append({
                'column': col,
                'type': col_type,
                'dtype': str(self.df[col].dtype),
                'missing_count': missing_count,
                'missing_pct': missing_pct,
                'unique_count': unique_count,
                'unique_pct': unique_pct
            })
        
        # Créer un DataFrame à partir des statistiques
        stats_df = pd.DataFrame(stats)
        
        # Trier par type et nom de colonne
        stats_df = stats_df.sort_values(['type', 'column'])
        
        return stats_df 
        
    def execute_process(self, 
                        text_max_categories: int = 1000,
                        ordinal_columns: Optional[Dict[str, Dict]] = None,
                        min_categorical_categories: int = 2,
                        max_categorical_categories: int = 50,
                        max_missing_pct: float = 0.8,
                        include_text: bool = False,
                        include_datetime: bool = False,
                        include_url: bool = False,
                        optimize_memory_flag: bool = True,
                        return_stats: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Exécute la chaîne complète de traitement du DataFrame en une seule méthode.
        
        Cette méthode combine les étapes de détection des types de colonnes, d'optimisation
        de la mémoire et de sélection des colonnes pertinentes en une seule opération.
        
        Args:
            text_max_categories: Nombre maximum de catégories pour considérer une colonne 
            comme textuelle
            ordinal_columns: Dictionnaire des colonnes ordinales avec leurs mappages
            min_categorical_categories: Nombre minimum de catégories pour les colonnes catégorielles
            max_categorical_categories: Nombre maximum de catégories pour les colonnes catégorielles
            max_missing_pct: Pourcentage maximum de valeurs manquantes
            include_text: Si True, inclut les colonnes textuelles (False par défaut)
            include_datetime: Si True, inclut les colonnes de type datetime (False par défaut)
            include_url: Si True, inclut les colonnes contenant "url" 
            dans leur nom (False par défaut)
            optimize_memory_flag: Si True, applique l'optimisation de mémoire
            return_stats: Si True, inclut les statistiques des colonnes dans le résultat
            
        Returns:
            Dictionnaire contenant les différents DataFrames générés pendant le processus
        """
        logger.info("Démarrage du processus complet de traitement...")
        
        # Étape 1: Détection des types de colonnes
        logger.info("Étape 1/4: Détection des types de colonnes")
        column_types = self.detect_column_types(
            text_max_categories=text_max_categories,
            ordinal_columns=ordinal_columns
        )
        
        # Étape 2: Optimisation de la mémoire (optionnelle)
        if optimize_memory_flag:
            logger.info("Étape 2/4: Optimisation de la mémoire")
            df_optimized = self.optimize_memory()
            # Mettre à jour le DataFrame principal avec la version optimisée
            self.df = df_optimized
        else:
            logger.info("Étape 2/4: Optimisation de la mémoire (ignorée)")
            df_optimized = self.df.copy()
        
        # Étape 3: Sélection des colonnes pertinentes
        logger.info("Étape 3/4: Sélection des colonnes pertinentes")
        df_relevant = self.select_relevant_columns(
            min_categorical_categories=min_categorical_categories,
            max_categorical_categories=max_categorical_categories,
            max_missing_pct=max_missing_pct,
            include_text=include_text,
            include_datetime=include_datetime,
            include_url=include_url
        )
        
        # Étape 4: Génération des statistiques (optionnelle)
        if return_stats:
            logger.info("Étape 4/4: Génération des statistiques des colonnes")
            stats_df = self.get_column_stats()
        else:
            logger.info("Étape 4/4: Génération des statistiques des colonnes (ignorée)")
            stats_df = None
        
        # Préparer les résultats
        results = {
            'df_optimized': df_optimized,
            'df_relevant': df_relevant,
            'column_types': column_types
        }
        
        if return_stats:
            results['stats_df'] = stats_df
            
        # Génération d'un résumé du processus
        logger.info("Processus de traitement terminé avec succès")
        logger.info("Nombre total de colonnes: %d", len(self.df.columns))
        logger.info("Nombre de colonnes pertinentes: %d", len(df_relevant.columns))
        logger.info(
            "Réduction: %.2f%%", 
            100 * (1 - len(df_relevant.columns) / len(self.df.columns))
        )
        
        # Résumer les types de colonnes sélectionnées
        type_counts = {}
        for col_type, cols in column_types.items():
            type_counts[col_type] = len([c for c in cols if c in df_relevant.columns])
        
        logger.info("Composition du DataFrame filtré:")
        for col_type, count in type_counts.items():
            if count > 0:
                logger.info(" - %s: %d colonnes", col_type, count)
                
        return results 
