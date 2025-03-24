"""
Module tache9_1
Ce module propose un template de chaîne de traitement (pipeline) pour la
phase 3 du projet, réutilisant et améliorant les briques développées
dans les tâches précédentes (ex. chargement, enrichissement, analyse d'erreur).
"""

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
    Charge le dataset depuis un chemin CSV.

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
    Nettoie le DataFrame (gestion des NaN, outliers, etc.).

    Args:
        df (pd.DataFrame): Données brutes.

    Returns:
        pd.DataFrame: Données nettoyées.
    """
    # Exemple : supprimer les lignes où target est manquant
    if "target" in df.columns:
        df = df.dropna(subset=["target"])

    # Autres opérations de nettoyage selon le besoin
    # ex: df.fillna(0, inplace=True)

    logging.info("Après nettoyage, dimensions : %s", df.shape)
    return df


def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le DataFrame (ex: merge avec un dataset additionnel,
    création de features, etc.).

    Args:
        df (pd.DataFrame): Données nettoyées.

    Returns:
        pd.DataFrame: Données enrichies.
    """
    # Exemple fictif : créer une colonne ratio si "col1" et "col2" existent
    if "col1" in df.columns and "col2" in df.columns:
        df["ratio"] = df["col1"] / (df["col2"] + 1e-9)

    logging.info("Données enrichies, dimensions : %s", df.shape)
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Sépare les données en ensembles d'entraînement et de test.

    Args:
        df (pd.DataFrame): DataFrame complet.
        target_col (str): Nom de la colonne cible.
        test_size (float): Proportion de test.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    logging.info(
        "Données divisées : train (%d lignes) / test (%d lignes)",
        x_train.shape[0], x_test.shape[0]
    )
    return x_train, x_test, y_train, y_test


def build_pipeline(numeric_cols: list, categorical_cols: list):
    """
    Construit un pipeline scikit-learn complet avec preprocessing
    et un modèle (RandomForest par exemple).

    Args:
        numeric_cols (list): Liste des colonnes numériques.
        categorical_cols (list): Liste des colonnes catégorielles.

    Returns:
        Pipeline: Un pipeline prêt à être entraîné.
    """
    # Prétraitement des variables numériques
    numeric_transformer = StandardScaler()

    # Prétraitement des variables catégorielles
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    # Assemblage des transformations en colonne
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Choix du modèle
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Pipeline global
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("clf", model)
    ])

    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    Évalue le pipeline entraîné sur un jeu de test et affiche
    un rapport de classification et une matrice de confusion.

    Args:
        pipeline (Pipeline): Le pipeline entraîné.
        x_test (pd.DataFrame): Features de test.
        y_test (pd.Series): Cible de test.
    """
    y_pred = pipeline.predict(x_test)
    logging.info("Rapport de classification :\n%s", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    logging.info("Matrice de confusion :\n%s", cm)


def main() -> None:
    """
    Exemple d'utilisation de la chaîne de traitement (phase 3).
    """
    # 1) Charger le dataset
    df_raw = load_data("data/enriched_dataset.csv")

    # 2) Nettoyer
    df_clean = clean_data(df_raw)

    # 3) Enrichir
    df_enriched = enrich_data(df_clean)

    # 4) Split train/test
    x_train, x_test, y_train, y_test = split_data(df_enriched, target_col="target")

    # 5) Déterminer quelles colonnes sont numériques / catégorielles
    numeric_cols = x_train.select_dtypes(include=[np.number]).columns.tolist()
    # Exemple : si on sait qu'une col "city" est catégorielle
    # on l'ajoute manuellement dans categorical_cols
    categorical_cols = [col for col in x_train.columns if col not in numeric_cols]

    # 6) Construire le pipeline
    pipe = build_pipeline(numeric_cols, categorical_cols)

    # 7) Entraîner
    pipe.fit(x_train, y_train)
    logging.info("Pipeline entraîné avec succès.")

    # 8) Évaluer
    evaluate_model(pipe, x_test, y_test)

    # 9) Sauvegarder le modèle (ex: joblib, pickle)
    # from joblib import dump
    # dump(pipe, "models/final_model.joblib")
    # logging.info("Modèle sauvegardé dans models/final_model.joblib")


if __name__ == "__main__":
    main()
