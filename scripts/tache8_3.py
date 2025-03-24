"""
Module tache8_3
Ce module propose une fonction pour enrichir le dataset principal avec
des données additionnelles, puis réaliser un feature engineering de base.
"""

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def enrich_dataset(
    main_path: str,
    extra_path: str,
    output_path: str
) -> pd.DataFrame:
    """
    Enrichit le dataset principal avec des données additionnelles,
    et crée de nouvelles features simples.

    Args:
        main_path (str): Chemin du fichier CSV principal.
        extra_path (str): Chemin du fichier CSV additionnel.
        output_path (str): Chemin où sauvegarder le CSV enrichi.

    Returns:
        pd.DataFrame: Le DataFrame enrichi.
    """
    logging.info("Chargement du dataset principal depuis %s", main_path)
    df_main = pd.read_csv(main_path)
    logging.info("Chargement du dataset additionnel depuis %s", extra_path)
    df_extra = pd.read_csv(extra_path)

    logging.info("Dimensions df_main : %s", df_main.shape)
    logging.info("Dimensions df_extra : %s", df_extra.shape)

    # Jointure sur la colonne 'product_id'
    df_merged = pd.merge(
        df_main,
        df_extra,
        on="product_id",
        how="left"  # ou "inner", selon le besoin
    )

    logging.info("Après merge, dimensions : %s", df_merged.shape)

    # Exemple de feature engineering : calculer un ratio
    if "score_x" in df_merged.columns and "score_y" in df_merged.columns:
        df_merged["score_ratio"] = df_merged["score_x"] / (
            df_merged["score_y"] + 1e-9
        )

    # Gérer d'éventuelles valeurs manquantes
    # (ex: remplir par 0, ou supprimer les lignes, etc.)
    # df_merged.fillna(0, inplace=True)

    # Sauvegarde du dataset final
    logging.info("Sauvegarde du dataset enrichi dans %s", output_path)
    df_merged.to_csv(output_path, index=False)

    return df_merged


def main() -> None:
    """
    Exemple d'utilisation de la fonction enrich_dataset.
    """
    df_final = enrich_dataset(
        main_path="data/main_dataset.csv",
        extra_path="data/extra_dataset.csv",
        output_path="data/enriched_dataset.csv"
    )
    logging.info("Dataset final prêt, aperçu :\n%s", df_final.head())


if __name__ == "__main__":
    main()