"""
Module pour le traitement et l'optimisation des DataFrames pandas.

Ce module fournit une classe DataFrameProcessor qui permet de détecter automatiquement
les types de colonnes, d'optimiser la mémoire et de sélectionner les colonnes pertinentes
pour l'analyse de données.
"""
try:
    import logging
    import numpy as np
    import pandas as pd
    from typing import Dict, List, Tuple, Optional
    from dataclasses import dataclass
except ImportError as e:
    print(f"Erreur lors de l'importation des modules : {e}")

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DataFrameProcessor')


@dataclass
class ColumnTypeConfig:
    """Configuration pour la détection des types de colonnes."""
    text_max_categories: int = 1000
    ordinal_columns: Optional[Dict[str, Dict]] = None


@dataclass
class MemoryConfig:
    """Configuration pour l'optimisation de la mémoire."""
    convert_numerics: bool = True
    convert_categories: bool = True
    verbose: bool = True


@dataclass
class CategoricalConfig:
    """Configuration pour le filtrage des colonnes catégorielles."""
    min_categories: int = 2
    max_categories: int = 50
    max_missing_pct: float = 0.8


@dataclass
class ColumnSelectionConfig:
    """Configuration pour la sélection des colonnes pertinentes."""
    min_categorical_categories: int = 2
    max_categorical_categories: int = 50
    max_missing_pct: float = 0.8
    include_text: bool = False
    include_datetime: bool = False
    include_url: bool = False


@dataclass
class ProcessConfig:
    """Configuration complète pour le traitement du DataFrame."""
    column_type_config: ColumnTypeConfig = ColumnTypeConfig()
    memory_config: MemoryConfig = MemoryConfig()
    categorical_config: CategoricalConfig = CategoricalConfig()
    column_selection_config: ColumnSelectionConfig = ColumnSelectionConfig()
    optimize_memory_flag: bool = True
    return_stats: bool = True


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
        
        # Configuration par défaut
        self.column_type_config = ColumnTypeConfig()
        self.memory_config = MemoryConfig()
        self.categorical_config = CategoricalConfig()
        self.column_selection_config = ColumnSelectionConfig()
        
        # Types de colonnes et mapping
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
        }

    def _is_datetime_column(self, col: str, dtype: str) -> bool:
        """
        Vérifie si une colonne est de type datetime.
        
        Args:
            col: Nom de la colonne
            dtype: Type de données de la colonne
            
        Returns:
            True si la colonne est de type datetime
        """
        return (pd.api.types.is_datetime64_dtype(dtype) or
                (dtype == 'object' and 'datetime' in col and
                 self.df[col].str.contains('T.*Z').mean() > 0.5))

    def _is_categorical_column(self, col: str, dtype: str, unique_count: int) -> bool:
        """
        Vérifie si une colonne est catégorielle.
        
        Args:
            col: Nom de la colonne
            dtype: Type de données de la colonne
            unique_count: Nombre de valeurs uniques
            
        Returns:
            True si la colonne est catégorielle
        """
        return (dtype == 'object' or pd.api.types.is_categorical_dtype(dtype)) and (
                col in self.ordinal_columns_mapping or \
                unique_count <= self.column_type_config.text_max_categories
        )

    def detect_column_types(
            self,
            config: ColumnTypeConfig = ColumnTypeConfig()
    ) -> Dict[str, List[str]]:
        """
        Détecte automatiquement les types de colonnes dans le DataFrame.
        
        Args:
            config: Configuration pour la détection des types
            
        Returns:
            Dictionnaire contenant les colonnes classées par type
        """
        logger.info("Détection des types de colonnes...")
        self.column_type_config = config

        # Utiliser le mapping fourni ou celui par défaut
        if config.ordinal_columns:
            self.ordinal_columns_mapping.update(config.ordinal_columns)

        # Réinitialisation des catégories
        for key in self.column_types:
            self.column_types[key] = []

        for col in self.df.columns:
            if self.df[col].isna().all():
                self.column_types['other'].append(col)
                continue

            dtype = self.df[col].dtype
            unique_count = self.df[col].nunique()

            if np.issubdtype(dtype, np.number):
                self._handle_numeric_column(col)
            elif self._is_datetime_column(col, dtype):
                self.column_types['datetime'].append(col)
            elif self._is_categorical_column(col, dtype, unique_count):
                self._handle_categorical_column(col)
            else:
                self.column_types['other'].append(col)

        for col_type, cols in self.column_types.items():
            logger.info("%s: %d colonnes", col_type, len(cols))

        return self.column_types

    def _handle_numeric_column(self, col: str) -> None:
        """
        Gère la classification d'une colonne numérique.
        
        Args:
            col: Nom de la colonne
        """
        if col in self.ordinal_columns_mapping:
            self.column_types['categorical_ordinal'].append(col)
        else:
            unique_count = self.df[col].nunique()
            if unique_count < 20 and unique_count / len(self.df) < 0.01:
                self.column_types['categorical_nominal'].append(col)
            else:
                self.column_types['numeric'].append(col)

    def _handle_categorical_column(self, col: str) -> None:
        """
        Gère la classification d'une colonne catégorielle.
        
        Args:
            col: Nom de la colonne
        """
        if col in self.ordinal_columns_mapping:
            self.column_types['categorical_ordinal'].append(col)
        else:
            unique_count = self.df[col].nunique()
            if unique_count > self.column_type_config.text_max_categories:
                self.column_types['text'].append(col)
            else:
                self.column_types['categorical_nominal'].append(col)

    def optimize_memory(self, config: MemoryConfig = MemoryConfig()) -> pd.DataFrame:
        """
        Optimise l'utilisation de la mémoire du DataFrame par downcasting.
        
        Args:
            config: Configuration pour l'optimisation de la mémoire
            
        Returns:
            DataFrame optimisé
        """
        logger.info("Optimisation de la mémoire...")
        self.memory_config = config

        if not any(self.column_types.values()):
            self.detect_column_types()

        df_optimized = self.df.copy()

        if config.convert_numerics:
            self._optimize_numeric_columns(df_optimized)

        if config.convert_categories:
            self._optimize_categorical_columns(df_optimized)

        if config.verbose:
            self._log_memory_optimization(df_optimized)

        return df_optimized

    def _optimize_numeric_columns(self, df: pd.DataFrame) -> None:
        """
        Optimise les colonnes numériques.
        
        Args:
            df: DataFrame à optimiser
        """
        for col in self.column_types['numeric']:
            if df[col].isna().all():
                continue

            if pd.api.types.is_integer_dtype(df[col]):
                self._optimize_integer_column(df, col)
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = df[col].astype(np.float32)

    def _optimize_integer_column(self, df: pd.DataFrame, col: str) -> None:
        """
        Optimise une colonne entière.
        
        Args:
            df: DataFrame à optimiser
            col: Nom de la colonne
        """
        col_data = df[col]
        min_val = col_data.min()
        max_val = col_data.max()

        if min_val >= 0:  # Unsigned int
            if max_val <= 255:
                df[col] = col_data.astype(np.uint8)
            elif max_val <= 65535:
                df[col] = col_data.astype(np.uint16)
            elif max_val <= 4294967295:
                df[col] = col_data.astype(np.uint32)
        else:  # Signed int
            if min_val >= -128 and max_val <= 127:
                df[col] = col_data.astype(np.int8)
            elif min_val >= -32768 and max_val <= 32767:
                df[col] = col_data.astype(np.int16)
            elif min_val >= -2147483648 and max_val <= 2147483647:
                df[col] = col_data.astype(np.int32)

    def _optimize_categorical_columns(self, df: pd.DataFrame) -> None:
        """
        Optimise les colonnes catégorielles.
        
        Args:
            df: DataFrame à optimiser
        """
        for col_type in ['categorical_ordinal', 'categorical_nominal']:
            for col in self.column_types[col_type]:
                if col in df.columns:
                    df[col] = df[col].astype('category')

    def _log_memory_optimization(self, df_optimized: pd.DataFrame) -> None:
        """
        Journalise les résultats de l'optimisation de la mémoire.
        
        Args:
            df_optimized: DataFrame optimisé
        """
        new_memory = df_optimized.memory_usage(deep=True).sum()
        memory_reduction = 100 * (1 - new_memory / self.original_memory)
        logger.info("Mémoire initiale: %.2f MB", self.original_memory / 1e6)
        logger.info("Mémoire optimisée: %.2f MB", new_memory / 1e6)
        logger.info("Réduction: %.2f%%", memory_reduction)

    def filter_categorical_columns(self, config: CategoricalConfig = CategoricalConfig()) -> Tuple[
        List[str], List[str]]:
        """
        Filtre les colonnes catégorielles selon différents critères.
        
        Args:
            config: Configuration pour le filtrage des colonnes catégorielles
            
        Returns:
            Tuple contenant (colonnes_retenues, colonnes_rejetées)
        """
        logger.info("Filtrage des colonnes catégorielles...")
        self.categorical_config = config

        if not any(self.column_types.values()):
            self.detect_column_types()

        retained_columns = []
        rejected_columns = []

        categorical_columns = (self.column_types['categorical_ordinal'] +
                               self.column_types['categorical_nominal'])

        for col in categorical_columns:
            if self._is_valid_categorical_column(col):
                retained_columns.append(col)
            else:
                rejected_columns.append(self._get_rejection_reason(col))

        self._log_categorical_filtering(retained_columns, rejected_columns)

        return retained_columns, [col for col, _ in rejected_columns]

    def _is_valid_categorical_column(self, col: str) -> bool:
        """
        Vérifie si une colonne catégorielle est valide.
        
        Args:
            col: Nom de la colonne
            
        Returns:
            True si la colonne est valide
        """
        missing_pct = self.df[col].isna().mean()
        unique_count = self.df[col].dropna().nunique()
        max_missing = self.categorical_config.max_missing_pct
        min_categories = self.categorical_config.min_categories
        max_categories = self.categorical_config.max_categories

        return (missing_pct <= max_missing and
                min_categories <= unique_count <= max_categories)

    def _get_rejection_reason(self, col: str) -> Tuple[str, str]:
        """
        Détermine la raison du rejet d'une colonne.
        
        Args:
            col: Nom de la colonne
            
        Returns:
            Tuple (colonne, raison)
        """
        missing_pct = self.df[col].isna().mean()
        unique_count = self.df[col].dropna().nunique()
        reasons = []

        if missing_pct > self.categorical_config.max_missing_pct:
            reasons.append(f"trop de valeurs manquantes ({missing_pct:.2%})")
        if unique_count < self.categorical_config.min_categories:
            reasons.append(f"pas assez de catégories ({unique_count})")
        if unique_count > self.categorical_config.max_categories:
            reasons.append(f"trop de catégories ({unique_count})")

        return col, ", ".join(reasons)

    def _log_categorical_filtering(
            self,
            retained: List[str],
            rejected: List[Tuple[str, str]]
    ) -> None:
        """
        Journalise les résultats du filtrage des colonnes catégorielles.
        
        Args:
            retained: Liste des colonnes retenues
            rejected: Liste des colonnes rejetées avec leurs raisons
        """
        logger.info("Colonnes catégorielles retenues: %d", len(retained))
        logger.info("Colonnes catégorielles rejetées: %d", len(rejected))

        for col, reason in rejected[:10]:
            logger.debug("Rejeté: %s - %s", col, reason)

    def select_relevant_columns(
            self,
            config: ColumnSelectionConfig = ColumnSelectionConfig()
    ) -> pd.DataFrame:
        """
        Sélectionne les colonnes pertinentes selon les critères spécifiés.
        
        Args:
            config: Configuration pour la sélection des colonnes
            
        Returns:
            DataFrame avec uniquement les colonnes pertinentes
        """
        logger.info("Sélection des colonnes pertinentes...")
        self.column_selection_config = config

        if not any(self.column_types.values()):
            self.detect_column_types()

        columns_to_include = self._get_columns_to_include()
        df_relevant = self.df[columns_to_include].copy()

        self._log_column_selection(columns_to_include)

        return df_relevant

    def _get_columns_to_include(self) -> List[str]:
        """
        Détermine les colonnes à inclure dans le DataFrame final.
        
        Returns:
            Liste des colonnes à inclure
        """
        columns_to_include = []
        max_missing = self.column_selection_config.max_missing_pct

        # Colonnes numériques
        numeric_cols = [col for col in self.column_types['numeric']
                       if self.df[col].isna().mean() <= max_missing]
        columns_to_include.extend(numeric_cols)

        # Colonnes catégorielles
        categorical_cols, _ = self.filter_categorical_columns(
            CategoricalConfig(
                min_categories=self.column_selection_config.min_categorical_categories,
                max_categories=self.column_selection_config.max_categorical_categories,
                max_missing_pct=max_missing
            )
        )
        columns_to_include.extend(categorical_cols)

        # Colonnes de date si nécessaire
        if self.column_selection_config.include_datetime:
            datetime_cols = [col for col in self.column_types['datetime']
                           if self.df[col].isna().mean() <= max_missing]
            columns_to_include.extend(datetime_cols)

        # Colonnes textuelles si nécessaire
        if self.column_selection_config.include_text:
            text_cols = [col for col in self.column_types['text']
                        if self.df[col].isna().mean() <= max_missing]
            columns_to_include.extend(text_cols)

        # Filtrer les colonnes URL si nécessaire
        if not self.column_selection_config.include_url:
            columns_to_include = [col for col in columns_to_include if 'url' not in col.lower()]

        return columns_to_include

    def _log_column_selection(self, columns_to_include: List[str]) -> None:
        """
        Journalise les résultats de la sélection des colonnes.
        
        Args:
            columns_to_include: Liste des colonnes sélectionnées
        """
        url_cols = [col for col in columns_to_include if 'url' in col.lower()]
        non_url_cols = [col for col in columns_to_include if col not in url_cols]

        logger.info("Colonnes sélectionnées: %d sur %d",
                    len(columns_to_include), len(self.df.columns))
        logger.info(
            "Numériques: %d, Catégorielles: %d, URL filtrées: %d, Datetime: %d, Texte: %d",
            len([col for col in non_url_cols if col in self.column_types['numeric']]),
            len([col for col in non_url_cols if col in self.column_types['categorical_nominal'] +
                 self.column_types['categorical_ordinal']]),
            len(url_cols),
            len([col for col in columns_to_include if col in self.column_types['datetime']]),
            len([col for col in columns_to_include if col in self.column_types['text']])
        )

    def get_column_stats(self) -> pd.DataFrame:
        """
        Génère des statistiques sur les colonnes du DataFrame.
        
        Returns:
            DataFrame contenant les statistiques des colonnes
        """
        if not any(self.column_types.values()):
            self.detect_column_types()

        stats = []
        for col in self.df.columns:
            col_type = next((t for t, cols in self.column_types.items() if col in cols), "unknown")
            stats.append(self._get_column_statistics(col, col_type))

        stats_df = pd.DataFrame(stats)
        return stats_df.sort_values(['type', 'column'])

    def _get_column_statistics(self, col: str, col_type: str) -> Dict:
        """
        Calcule les statistiques pour une colonne.
        
        Args:
            col: Nom de la colonne
            col_type: Type de la colonne
            
        Returns:
            Dictionnaire contenant les statistiques
        """
        missing_count = self.df[col].isna().sum()
        missing_pct = missing_count / len(self.df)
        unique_count = self.df[col].nunique()
        unique_pct = unique_count / len(self.df)

        return {
            'column': col,
            'type': col_type,
            'dtype': str(self.df[col].dtype),
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'unique_count': unique_count,
            'unique_pct': unique_pct
        }

    def execute_process(self, config: ProcessConfig = ProcessConfig()) -> Dict[str, pd.DataFrame]:
        """
        Exécute la chaîne complète de traitement du DataFrame.
        
        Args:
            config: Configuration complète du processus
            
        Returns:
            Dictionnaire contenant les différents DataFrames générés
        """
        logger.info("Démarrage du processus complet de traitement...")

        # Étape 1: Détection des types de colonnes
        logger.info("Étape 1/4: Détection des types de colonnes")
        column_types = self.detect_column_types(config.column_type_config)

        # Étape 2: Optimisation de la mémoire
        if config.optimize_memory_flag:
            logger.info("Étape 2/4: Optimisation de la mémoire")
            df_optimized = self.optimize_memory(config.memory_config)
            self.df = df_optimized
        else:
            logger.info("Étape 2/4: Optimisation de la mémoire (ignorée)")
            df_optimized = self.df.copy()

        # Étape 3: Sélection des colonnes pertinentes
        logger.info("Étape 3/4: Sélection des colonnes pertinentes")
        df_relevant = self.select_relevant_columns(config.column_selection_config)

        # Étape 4: Génération des statistiques
        if config.return_stats:
            logger.info("Étape 4/4: Génération des statistiques des colonnes")
            stats_df = self.get_column_stats()
        else:
            logger.info("Étape 4/4: Génération des statistiques des colonnes (ignorée)")
            stats_df = None

        results = self._prepare_results(df_optimized, df_relevant, column_types, stats_df)
        self._log_process_summary(df_relevant, column_types)

        return results

    def _prepare_results(self, df_optimized: pd.DataFrame, df_relevant: pd.DataFrame,
                         column_types: Dict[str, List[str]],
                         stats_df: Optional[pd.DataFrame]
                         ) -> Dict[
        str, pd.DataFrame]:
        """
        Prépare les résultats du processus.
        
        Args:
            df_optimized: DataFrame optimisé
            df_relevant: DataFrame avec colonnes pertinentes
            column_types: Types de colonnes détectés
            stats_df: Statistiques des colonnes
            
        Returns:
            Dictionnaire des résultats
        """
        results = {
            'df_optimized': df_optimized,
            'df_relevant': df_relevant,
            'column_types': column_types
        }

        if stats_df is not None:
            results['stats_df'] = stats_df

        return results

    def _log_process_summary(
            self,
            df_relevant: pd.DataFrame,
            column_types: Dict[str, List[str]]
    ) -> None:
        """
        Journalise un résumé du processus.
        
        Args:
            df_relevant: DataFrame avec colonnes pertinentes
            column_types: Types de colonnes détectés
        """
        logger.info("Processus de traitement terminé avec succès")
        logger.info("Nombre total de colonnes: %d", len(self.df.columns))
        logger.info("Nombre de colonnes pertinentes: %d", len(df_relevant.columns))
        logger.info(
            "Réduction: %.2f%%",
            100 * (1 - len(df_relevant.columns) / len(self.df.columns))
        )

        type_counts = {}
        for col_type, cols in column_types.items():
            type_counts[col_type] = len([c for c in cols if c in df_relevant.columns])

        logger.info("Composition du DataFrame filtré:")
        for col_type, count in type_counts.items():
            if count > 0:
                logger.info(" - %s: %d colonnes", col_type, count)
