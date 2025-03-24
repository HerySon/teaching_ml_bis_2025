import argparse

from .utilities.data_processing import ApplyFunctionThread

from .encoder.data_encoder import FeatureEncoder
from .scaler.data_scaling import FeatureScaler
from .cleanner.data_missing_values import AdvancedDataFrameProcessor

__all__ = [
    'DataFrameProcessor',
    'ApplyFunctionThread',
    'FeatureEncoder',
    'FeatureScaler',
]

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

def main(description: str, parser: argparse.ArgumentParser) -> argparse.Namespace:
    print(description, " - Version:", __version__)

    parser.add_argument("--file_path", type=str, required=True, help="Chemin vers le fichier CSV")
    parser.add_argument("--output_path", type=str, default="cleaned_data.csv", help="Chemin du fichier de sortie")
    parser.add_argument("--missing_threshold", type=float, default=0.5, help="Seuil de valeurs manquantes pour supprimer une colonne")
    parser.add_argument("--pattern_col", type=str, default="serving_size", help="Colonne pour extraire un motif particulier")
    parser.add_argument("--pattern", type=str, default="(\\d+)", help="Motif regex à extraire")
    parser.add_argument("--irrelevant_cols", type=str, nargs='+', default=["id", "unwanted_col"], help="Colonnes non pertinentes")
    parser.add_argument("--method", type=str, nargs='+', default=["col1", "col2"], help="Colonnes à encoder")
    parser.add_argument("--target_col", type=str, required=True, help="Colonne cible pour certains encodages")
    parser.add_argument("--option", type=int, default="0", help="Méthode d'imputation")

    return parser.parse_args()

def get_all_columns(df) -> list:
    return df.columns.tolist()


def pars_args_data_scaling(args: list) -> None:
    parser = argparse.ArgumentParser()
    args = main("Data Scaling Script", parser)

    scaler = FeatureScaler.from_csv(args.file_path, limit=1000)

    scaled_df = scaler.scale_based_on_distribution(args.option) if args.option else scaler.scale_based_on_distribution()

    scaled_df.to_csv(args.output_path, index=False, encoding='utf-8')

    print(f"✅ Scaled data saved to {args.output_path}")


def pars_args_data_encoder(args: list) -> None:
    parser = argparse.ArgumentParser()
    args = main("Data Encoding Script", parser)

    encoder = FeatureEncoder.from_csv(args.file_path, args.target_col, limit=1000)

    encoder.encoding_selected(args.method[0])

    encoder.save_encoded_df(args.output_path)

    print(f"✅ Encoded data saved to {args.output_path}")

def pars_args_data_frame_processor(args: list) -> None:
    parser = argparse.ArgumentParser()
    args = main("Data Cleaning Script", parser)

    processor = AdvancedDataFrameProcessor(args.file_path)

    processor.impute_missing_values(method='linear_regression', col="France")

    processor.df.to_csv(args.output_path, index=False)

    print(f"✅ Cleaned data saved to {args.output_path}")