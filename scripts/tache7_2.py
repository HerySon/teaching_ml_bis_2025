"""
Module tache7_2
Ce module propose une fonction pour analyser les erreurs de classification
en identifiant les échantillons mal classés et en fournissant des indicateurs
de performance (matrice de confusion, rapport de classification). Il affiche
également les features les plus importantes pour le modèle.
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def analyze_errors(
    X: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyse les erreurs de classification en comparant y_pred à y_true,
    et produit des indicateurs de performance.

    Args:
        X (pd.DataFrame): Features du jeu de données.
        y_true (pd.Series): Vraies étiquettes de classe.
        y_pred (np.ndarray): Prédictions du modèle.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Un DataFrame des échantillons mal classés,
            - Un DataFrame de la matrice de confusion.
    """
    # 1) Identification des erreurs
    mask_errors = (y_pred != y_true)
    df_errors = X[mask_errors].copy()
    df_errors['y_true'] = y_true[mask_errors].values
    df_errors['y_pred'] = y_pred[mask_errors]

    logging.info(
        "Nombre d'erreurs : %d sur %d (%.2f%%)",
        df_errors.shape[0], X.shape[0],
        100 * df_errors.shape[0] / X.shape[0]
    )

    # 2) Matrice de confusion
    labels = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    logging.info("Matrice de confusion :\n%s", cm_df)

    # 3) Rapport de classification
    classif_report = classification_report(y_true, y_pred, labels=labels)
    logging.info("Rapport de classification :\n%s", classif_report)

    return df_errors, cm_df


def main() -> None:
    """
    Exemple d'utilisation : on génère un petit dataset factice,
    on entraîne un RandomForest, puis on analyse les erreurs.
    """
    # Génération d'un dataset factice
    rng = np.random.RandomState(42)
    X_example = pd.DataFrame({
        'feat1': rng.normal(loc=0, scale=1, size=100),
        'feat2': rng.normal(loc=5, scale=2, size=100),
        'feat3': rng.randint(0, 2, size=100)
    })
    y_example = (X_example['feat1'] + X_example['feat2'] > 5).astype(int)

    # Séparation en train/test (simple split 80/20)
    idx_train = rng.choice(X_example.index, size=80, replace=False)
    idx_test = [i for i in X_example.index if i not in idx_train]
    X_train, y_train = X_example.loc[idx_train], y_example.loc[idx_train]
    X_test, y_test = X_example.loc[idx_test], y_example.loc[idx_test]

    # Entraînement d'un modèle
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Prédiction
    y_pred = model.predict(X_test)

    # Analyse d'erreur
    df_errors, cm_df = analyze_errors(X_test, y_test, y_pred)

    # Affiche les features les plus importantes
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    logging.info("Features importantes (ordre décroissant) :")
    for idx in sorted_idx:
        logging.info(
            "%s: importance %.3f", X_test.columns[idx], importances[idx]
        )

    # Visualisation rapide de la matrice de confusion
    plt.figure(figsize=(5, 4))
    plt.imshow(cm_df, cmap='Blues')
    plt.title("Matrice de confusion")
    plt.colorbar()
    plt.xticks(ticks=range(len(cm_df)), labels=cm_df.columns)
    plt.yticks(ticks=range(len(cm_df)), labels=cm_df.index)
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
