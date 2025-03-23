"""
Script pour le traitement automatique des données OpenFoodFacts
Ce script utilise la classe DataFrameProcessor pour filtrer et sélectionner
les colonnes pertinentes du dataset OpenFoodFacts.
"""

import argparse
import logging
import time

import pandas as pd

# Importer notre classe DataFrameProcessor
from data_processor import DataFrameProcessor


def parse_arguments():
    """
    Analyse les arguments de ligne de commande.
    
    Returns:
        Arguments parsés
    """
    parser = argparse.ArgumentParser(description='Traitement automatique des données OpenFoodFacts')
    
    parser.add_argument(
        '--input', type=str,
        default="https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz",
        help='Chemin ou URL vers le fichier de données (CSV)'
    )

    parser.add_argument('--nrows', type=int, default=None,
                        help='Nombre de lignes à charger (None pour toutes)')

    parser.add_argument('--max-missing', type=float, default=0.7,
                        help='Pourcentage maximum de valeurs manquantes (0-1)')

    parser.add_argument('--min-categories', type=int, default=2,
                        help='Nombre minimum de catégories pour colonnes catégorielles')

    parser.add_argument('--max-categories', type=int, default=30,
                        help='Nombre maximum de catégories pour colonnes catégorielles')

    parser.add_argument('--include-text', action='store_true',
                        help='Inclure les colonnes textuelles')

    parser.add_argument('--include-datetime', action='store_true',
                        help='Inclure les colonnes de type datetime')

    parser.add_argument('--include-url', action='store_true',
                        help='Inclure les colonnes contenant "url" dans leur nom')

    parser.add_argument('--no-optimize', action='store_true',
                        help='Désactiver l\'optimisation de mémoire')

    parser.add_argument('--debug', action='store_true',
                        help='Activer le mode debug (affichage détaillé)')

    return parser.parse_args()


def setup_logging(debug_mode=False):
    """
    Configure le système de logging.
    
    Args:
        debug_mode: Si True, active le niveau de logging DEBUG
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def main():
    """
    Fonction principale qui exécute le traitement des données OpenFoodFacts.
    """
    # Analyser les arguments
    args = parse_arguments()

    # Configurer le logging
    setup_logging(args.debug)

    # Démarrer le chronomètre
    start_time = time.time()

    logging.info("Démarrage du traitement des données OpenFoodFacts")
    logging.info("Fichier source: %s", args.input)

    # Charger les données
    logging.info("Chargement des données (nrows=%s)", args.nrows if args.nrows else 'toutes')
    try:
        df = pd.read_csv(args.input, nrows=args.nrows, sep='\t', encoding="utf-8")
        logging.info("Données chargées: %d lignes x %d colonnes", df.shape[0], df.shape[1])
    except (pd.errors.ParserError, pd.errors.EmptyDataError, IOError) as e:
        logging.error("Erreur lors du chargement des données: %s", e)
        return

    # Initialiser le processeur
    processor = DataFrameProcessor(df)

    # Exécuter le processus complet
    logging.info("Exécution du pipeline de traitement")
    results = processor.execute_process(
        min_categorical_categories=args.min_categories,
        max_categorical_categories=args.max_categories,
        max_missing_pct=args.max_missing,
        include_text=args.include_text,
        include_datetime=args.include_datetime,
        include_url=args.include_url,
        optimize_memory_flag=not args.no_optimize,
        return_stats=True
    )

    # Extraire les résultats
    df_relevant = results['df_relevant']
    stats_df = results['stats_df']

    # Afficher un résumé final
    elapsed_time = time.time() - start_time
    logging.info("Traitement terminé en %.2f secondes", elapsed_time)
    logging.info("DataFrame original: %d colonnes", df.shape[1])
    logging.info("DataFrame filtré: %d colonnes", df_relevant.shape[1])
    logging.info(
        "Réduction: %.2f%%",
        100 * (1 - df_relevant.shape[1] / df.shape[1])
    )

    # Afficher les statistiques sur les données manquantes (utiliser stats_df)
    if stats_df is not None:
        missing_avg = stats_df['missing_pct'].mean()
        logging.info("Pourcentage moyen de valeurs manquantes: %.2f%%", missing_avg * 100)

    # Afficher un résumé des types de colonnes dans le DataFrame filtré
    column_types = results['column_types']
    logging.info("Composition du DataFrame filtré:")
    for col_type, cols in column_types.items():
        cols_in_result = [c for c in cols if c in df_relevant.columns]
        if cols_in_result:
            logging.info(" - %s: %d colonnes", col_type, len(cols_in_result))
            if len(cols_in_result) <= 10:  # Afficher les noms si peu nombreux
                logging.info("   %s", ', '.join(cols_in_result))


if __name__ == "__main__":
    main()
