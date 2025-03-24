"""
Module tache9_1
Exemple de pipeline pour la phase 3, corrigé pour Pylint.
Ce script montre comment charger, nettoyer, enrichir, séparer et
entraîner un modèle de classification avec un pipeline scikit-learn.
"""

# pylint: disable=import-error
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_data(path: str) -> pd.DataFrame:
    """
    Charge le dataset depuis un fichier CSV.

    Args:
        path (str): Chemin du fichier CSV.

    Returns:
        pd.DataFrame: Le DataFrame chargé.
    """
    logging.info("Chargement des données depuis %s", path)
    df = pd.read_csv(path)
    logging.info("Dataset chargé, dimensions : %s", df.shape)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame en supprimant les lignes avec des valeurs
    manquantes dans la colonne 'target'.

    Args:
        df (pd.DataFrame): Données brutes.

    Returns:
        pd.DataFrame: Données nettoyées.
    """
    if "target" in df.columns:
        df = df.dropna(subset=["target"])
    logging.info("Après nettoyage, dimensions : %s", df.shape)
    return df


def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame en créant une nouvelle colonne 'ratio' si les
    colonnes 'col1' et 'col2' existent.

    Args:
        df (pd.DataFrame): Données nettoyées.

    Returns:
        pd.DataFrame: Données enrichies.
    """
    if "col1" in df.columns and "col2" in df.columns:
        df["ratio"] = df["col1"] / (df["col2"] + 1e-9)
    logging.info("Données enrichies, dimensions : %s", df.shape)
    return df


def split_data(df: pd.DataFrame, target_col: str = "target",
               test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, 
                                               pd.Series, pd.Series]:
    """
    Sépare le DataFrame en ensembles d'entraînement et de test.

    Args:
        df (pd.DataFrame): DataFrame complet.
        target_col (str): Nom de la colonne cible.
        test_size (float): Proportion de test.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            x_train, x_test, y_train, y_test.
    """
    x_data = df.drop(columns=[target_col])
    y_data = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size, random_state=42
    )
    logging.info("Données divisées : train (%d lignes) / test (%d lignes)",
                 x_train.shape[0], x_test.shape[0])
    return x_train, x_test, y_train, y_test


def build_pipeline(numeric_cols: list, categorical_cols: list) -> Pipeline:
    """
    Construit un pipeline scikit-learn intégrant le prétraitement et
    un modèle (RandomForestClassifier).

    Args:
        numeric_cols (list): Liste des colonnes numériques.
        categorical_cols (list): Liste des colonnes catégorielles.

    Returns:
        Pipeline: Un pipeline complet prêt à être entraîné.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("clf", model)
    ])

    return pipeline


def evaluate_model(pipeline: Pipeline, x_test: pd.DataFrame,
                   y_test: pd.Series) -> None:
    """
    Évalue le modèle sur l'ensemble de test et affiche un rapport de
    classification ainsi qu'une matrice de confusion.

    Args:
        pipeline (Pipeline): Le pipeline entraîné.
        x_test (pd.DataFrame): Les features de test.
        y_test (pd.Series): La cible de test.
    """
    y_pred = pipeline.predict(x_test)
    logging.info("Rapport de classification :\n%s",
                 classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    logging.info("Matrice de confusion :\n%s", cm)


def main() -> None:
    """
    Exécute le pipeline complet : chargement, nettoyage, enrichissement,
    séparation des données, construction et entraînement du pipeline, puis
    évaluation du modèle.
    """
    df_raw = load_data("data/enriched_dataset.csv")
    df_clean = clean_data(df_raw)
    df_enriched = enrich_data(df_clean)
    x_train, x_test, y_train, y_test = split_data(
        df_enriched, target_col="target"
    )
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in x_train.columns 
                        if col not in numeric_cols]
    pipe = build_pipeline(numeric_cols, categorical_cols)
    pipe.fit(x_train, y_train)
    logging.info("Pipeline entraîné avec succès.")
    evaluate_model(pipe, x_test, y_test)
    # Pour sauvegarder le modèle, décommente ci-dessous :
    # from joblib import dump
    # dump(pipe, "models/final_model.joblib")
    # logging.info("Modèle sauvegardé dans models/final_model.joblib")

if __name__ == "__main__":
    main()
