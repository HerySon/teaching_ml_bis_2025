import argparse
import asyncio

from .data_frame_processor import DataFrameProcessor
from .data_frame_filter import DataFrameSelector
from .data_cleaning import FeatureCleaning
from .data_encoder import FeatureEncoder

__all__ = ['DataFrameProcessor']

__version__ = '1.0.0'
__author__ = 'Feurking'

def pandas_display() -> str:
    return """⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⡗⠀⠀⠉⠑⢢⣴⣿⣷⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡶⢢⠀⠀⠀⠀⠀⠀⠀⠀⠻⠋⠀⠀⠀⠀⠀⢿⣿⠿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠳⢝⢧⡀⠀⠀⠀⠀⠀⢠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣼⣄⠙⠄⠀⠀⠀⢀⠇⢠⡄⠀⠀⣄⡀⠀⠀⠀⣴⣄⠘⡀⠀⠀⠀⠀⠀⠀⠀⣀⣤⡴⠶⢚⣫⣵⡶⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿⡅⣤⣀⠃⠀⠘⠀⢿⠇⠀⠸⣿⣿⡄⠀⠀⣿⣿⣷⣷⣄⠀⠀⠀⠀⣀⣸⣿⡀⠸⣿⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⠛⠁⣹⣿⣷⣦⣤⣶⡄⢀⣀⠀⠙⠋⠁⣠⣾⣿⣿⣿⣿⣿⣿⣾⣿⣿⣿⣿⣿⡟⢰⣿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢷⣾⣿⣿⣿⣿⣿⣿⣿⣌⣋⣀⣀⣠⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣈⣯⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠿⠿⣿⣿⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⡿⠟⠿⣿⣿⠿⠛⠁⠀⠀⠀⠀⠈⢿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡗⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿⣦⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⣿⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⣀⣠⣾⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿⣿⣿⣿⡿⠛⠒⠚⠀⠀⠙⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢼⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠛⠛⠛⠋⠉⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⠿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀Feurking⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀"""

def main(
        description: str,
        args_description: str,
        parser: argparse.ArgumentParser
    ) -> argparse.Namespace:

    print(description, " - Version:", __version__)
    
    parser.add_argument("--file_path", type=str, required=True, help="Chemin vers le fichier CSV")
    parser.add_argument("--output_path", type=str, default="cleaned_data.csv", help="Chemin du fichier de sortie")
    parser.add_argument("--missing_threshold", type=float, default=0.5, help="Seuil de valeurs manquantes pour supprimer une colonne")
    parser.add_argument("--pattern_col", type=str, default="serving_size", help="Colonne pour extraire un motif particulier")
    parser.add_argument("--pattern", type=str, default="(\\d+)", help="Motif regex à extraire")
    parser.add_argument("--irrelevant_cols", type=str, nargs='+', default=["id", "unwanted_col"], help="Colonnes non pertinentes")
    
    parser.add_argument("--category_threshold", type=int, default=10, help="Nombre max de catégories pour les variables ordinales")
    parser.add_argument("--limit", type=int, default=None, help="Nombre maximum de lignes à charger")

    parser.add_argument("--sample_size", type=float, default=0.1, help="Proportion de l'échantillon à sélectionner")
    parser.add_argument("--random_state", type=int, default=42, help="Valeur pour initialiser la graine aléatoire")
    parser.add_argument("--plot_type", type=str, default="numeric", help="Type de graphique à générer")
    parser.add_argument("--colunm_name", type=str, default="cities", help="Nom de la colonne à visualiser")

    parser.add_argument('--min_freq', type=float, default=0.01, help="Fréquence minimale pour la suppression des catégories rares")
    parser.add_argument('--n_components', type=int, default=8, help="Nombre de composants pour HashingEncoder")
    parser.add_argument('--n_clusters', type=int, default=5, help="Nombre de clusters pour la fusion des catégories via K-means")
    
    return parser.parse_args()

def get_all_columns(df) -> list:
    return df.columns.tolist()

def pars_args_data_encoder(args=None) -> None:
    parser = argparse.ArgumentParser(description="Commandes de nettoyage de données")

    args = main("Bienvenue dans le DataEncoder", "Commandes de nettoyage de données", parser)
    encoder = FeatureEncoder.from_csv(args.file_path, args.limit)

    encoder.merge_categories_clustering(n_clusters=args.n_clusters)
    # encoder.merge_rare_categories(threshold=args.min_freq)

    # encoder.one_hot_encode_incremental(min_freq=args.min_freq)
    # encoder.count_encode(min_freq=args.min_freq)

    # encoder.hash_encode(n_components=args.n_components)
    # encoder.merge_categories_clustering(n_clusters=args.n_clusters)

    # encoder.save_encoded_df()

def pars_args_data_cleaning(args=None) -> None:
    parser = argparse.ArgumentParser(description="Script de nettoyage de données")
    args = main("Bienvenue dans le DataCleaner", "Commandes de nettoyage de données", parser)

    cleaner = FeatureCleaning.from_csv(args.file_path, args.limit)

    cleaner.drop_uninformative_columns()
    cleaner.drop_irrelevant_columns(args.irrelevant_cols)
    cleaner.handle_missing_values(threshold=args.missing_threshold)
    cleaner.extract_pattern(args.pattern_col, args.pattern, "extracted_pattern")

    cleaner.fix_errors(args.pattern_col, lambda x: x.lower() if isinstance(x, str) else x)

    cleaner.remove_outliers()
    cleaner.clean_whitespace()
    cleaner.normalize_date_columns()
    cleaner.remove_duplicates()

    cleaner.summarize()
    cleaner.display_info()


def pars_args_data_frame_filter(args=None) -> None:

    parser = argparse.ArgumentParser(description="Commandes relatives au DataFrameSelector script")

    args = main("Bienvenue dans le DataFrameFilter", "Commandes relatives au DataFrameSelector script", parser)

    selector = DataFrameSelector(args.limit, args.file_path)

    selector.filter_columns_by_category_count(args.category_threshold)
    selector.filter_columns_by_correlation(min_corr=0.1)

    sample_df = selector.get_representative_sample(sample_size=args.sample_size, random_state=args.random_state)

    selector.save_plot(args.colunm_name, args.plot_type, sample_df, "undersample_plot")

def pars_args_data_frame_processor(args=None) -> None:
    
    print(pandas_display())

    parser = argparse.ArgumentParser(description="Commandes relatives au DataFrameProcessor script")
    
    args = main("Bienvenue dans le DataFrameProcessor", "Commandes relatives au DataFrameProcessor script", parser)

    processor = DataFrameProcessor(args.file_path, args.category_threshold, args.limit)
    asyncio.run(processor.process_dataframe())
