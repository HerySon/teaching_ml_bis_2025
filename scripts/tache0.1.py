"""
Module tache0_1
Ce module propose une méthode pour filtrer les colonnes d'un DataFrame
selon leur type. Il distingue les colonnes numériques et les colonnes
catégorielles, en séparant ces dernières en variables ordinales et
nominales selon un seuil et la présence de mots-clés dans le nom.
Les colonnes numériques sont downcastées pour économiser de la mémoire.
"""

import logging
from typing import List, Dict

import pandas as pd

# Configuration du logging pour afficher les messages en français.
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


def categoriser_colonne(
    nom_col: str,
    serie: pd.Series,
    mots_cles: List[str],
    seuil_modalites: int
) -> str:
    """
    Détermine le type d'une colonne : 'numerique', 'ordinal', 'nominal' ou 'drop'.

    Args:
        nom_col (str): Nom de la colonne.
        serie (pd.Series): Données de la colonne.
        mots_cles (List[str]): Mots-clés pour détecter les colonnes ordinales.
        seuil_modalites (int): Nombre max d'unicités pour qu'une colonne soit
            considérée comme nominale.

    Returns:
        str: 'numerique', 'ordinal', 'nominal' ou 'drop'.
    """
    if pd.api.types.is_numeric_dtype(serie):
        return 'numerique'
    if (pd.api.types.is_object_dtype(serie) or 
            pd.api.types.is_categorical_dtype(serie)):
        nb_uniques = serie.nunique(dropna=False)
        nom_min = nom_col.lower()
        if any(mot in nom_min for mot in mots_cles):
            return 'ordinal'
        if nb_uniques <= seuil_modalites:
            return 'nominal'
        return 'drop'
    return 'drop'


def optimiser_dataframe(
    df: pd.DataFrame,
    seuil_modalites: int = 10,
    mots_cles: List[str] = None
) -> Dict[str, List[str]]:
    """
    Filtre et catégorise les colonnes d'un DataFrame en colonnes numériques,
    ordinales, nominales ou à ignorer. Les colonnes numériques sont
    downcastées pour réduire l'utilisation de la mémoire.

    Args:
        df (pd.DataFrame): DataFrame en entrée.
        seuil_modalites (int, optional): Seuil max d'unicités pour qu'une
            colonne catégorielle soit nominale. Par défaut 10.
        mots_cles (List[str], optional): Mots-clés pour détecter les colonnes
            ordinales. Par défaut : ['level', 'grade', 'rank', 'score',
            'stage', 'rating'].

    Returns:
        Dict[str, List[str]]: Dictionnaire contenant :
            - 'numerique': liste des colonnes numériques.
            - 'ordinal': liste des colonnes catégorielles ordinales.
            - 'nominal': liste des colonnes catégorielles nominales.
            - 'ignore': liste des colonnes ignorées.
    """
    if mots_cles is None:
        mots_cles = ['level', 'grade', 'rank', 'score', 'stage', 'rating']

    colonnes_numeriques = []
    colonnes_ordinales = []
    colonnes_nominales = []
    colonnes_ignorees = []

    for nom in df.columns:
        type_col = categoriser_colonne(nom, df[nom], mots_cles, seuil_modalites)
        if type_col == 'numerique':
            type_orig = df[nom].dtype
            if pd.api.types.is_float_dtype(df[nom]):
                df[nom] = pd.to_numeric(df[nom], downcast='float')
            else:
                df[nom] = pd.to_numeric(df[nom], downcast='integer')
            colonnes_numeriques.append(nom)
            logging.debug(
                "Colonne %s downcastée de %s vers %s", nom, type_orig,
                df[nom].dtype
            )
        elif type_col == 'ordinal':
            colonnes_ordinales.append(nom)
            logging.debug("Colonne %s reconnue comme ordinale", nom)
        elif type_col == 'nominal':
            colonnes_nominales.append(nom)
            logging.debug("Colonne %s reconnue comme nominale", nom)
        else:
            colonnes_ignorees.append(nom)
            logging.debug("Colonne %s ignorée", nom)

    logging.info("=== Résumé d'optimisation ===")
    logging.info("Colonnes numériques : %s", colonnes_numeriques)
    logging.info("Colonnes ordinales : %s", colonnes_ordinales)
    logging.info("Colonnes nominales : %s", colonnes_nominales)
    logging.info("Colonnes ignorées : %s", colonnes_ignorees)

    return {
        'numerique': colonnes_numeriques,
        'ordinal': colonnes_ordinales,
        'nominal': colonnes_nominales,
        'ignore': colonnes_ignorees
    }


def main() -> None:
    """
    Point d'entrée pour tester la fonction optimiser_dataframe.
    """
    donnees = {
        'age': [25, 32, 47, 51],
        'income': [50000.0, 60000.0, 80000.0, 75000.0],
        'grade': ['A', 'B', 'B', 'C'],
        'city': ['Paris', 'Lyon', 'Paris', 'Marseille'],
        'id': ['a1', 'b2', 'c3', 'd4'],
        'misc': ['x', 'y', 'z', 'w']
    }
    df_exemple = pd.DataFrame(donnees)
    logging.info("Démarrage de l'optimisation du DataFrame d'exemple...")
    resultat = optimiser_dataframe(df_exemple, seuil_modalites=4)
    logging.info("Résultat de l'optimisation : %s", resultat)


if __name__ == "__main__":
    main()
