"""
Module tache7_2
Ce module propose une fonction pour analyser les erreurs de classification
en identifiant les échantillons mal classés, et en fournissant des indicateurs
de performance (matrice de confusion, rapport de classification, etc.).
"""

import logging
from typing import Tuple  # standard library d'abord

# pylint: disable=import-error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def analyze_errors(
    x_data: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyse les erreurs de classification en comparant y_pred à y_true,
    et produit des indicateurs de performance.

    Args:
        x_data (pd.DataFrame): Features du jeu de données.
        y_true (pd.Series): Vraies étiquettes de classe.
        y_pred (np.ndarray): Prédictions du modèle.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - Un DataFrame des échantillons mal classés,
            - Un DataFrame représentant la matrice de confusion.
    """
    mask_errors = (y_pred != y_true)
    df_errors = x_data[mask_errors].copy()
    df_errors['y_true'] = y_true[mask_errors].values
    df_errors['y_pred'] = y_pred[mask_errors]

    logging.info(
        "Nombre d'erreurs : %d sur %d (%.2f%%)",
        df_errors.shape[0], x_data.shape[0],
        100 * df_errors.shape[0] / x_data.shape[0]
    )

    labels = np.unique(y_true)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(matrix, index=labels, columns=labels)
    logging.info("Matrice de confusion :\n%s", cm_df)

    classif_report = classification_report(y_true, y_pred, labels=labels)
    logging.info("Rapport de classification :\n%s", classif_report)

    return df_errors, cm_df


def main() -> None:  # pylint: disable=too-many-locals
    """
    Exemple d'utilisation : génère un dataset, entraîne un RandomForest,
    puis analyse les erreurs.
    """
    rng = np.random.RandomState(42)
    x_example = pd.DataFrame({
        'feat1': rng.normal(loc=0, scale=1, size=100),
        'feat2': rng.normal(loc=5, scale=2, size=100),
        'feat3': rng.randint(0, 2, size=100)
    })
    y_example = (x_example['feat1'] + x_example['feat2'] > 5).astype(int)

    idx_train = rng.choice(x_example.index, size=80, replace=False)
    idx_test = [i for i in x_example.index if i not in idx_train]
    x_train, y_train = x_example.loc[idx_train], y_example.loc[idx_train]
    x_test, y_test = x_example.loc[idx_test], y_example.loc[idx_test]

    model = RandomForestClassifier(random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    df_err, cm_result = analyze_errors(x_test, y_test, y_pred)
    # On "utilise" df_err pour éviter l'alerte unused-variable
    logging.info("Exemples d'erreurs :\n%s", df_err.head())

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    logging.info("Features importantes (ordre décroissant) :")
    for idx in sorted_idx:
        logging.info(
            "%s: importance %.3f",
            x_test.columns[idx],
            importances[idx]
        )

    plt.imshow(cm_result, cmap='Blues')
    plt.title("Matrice de confusion")
    plt.colorbar()
    plt.xticks(ticks=range(len(cm_result)), labels=cm_result.columns)
    plt.yticks(ticks=range(len(cm_result)), labels=cm_result.index)
    plt.xlabel("Prédit")
    plt.ylabel("Vrai")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
