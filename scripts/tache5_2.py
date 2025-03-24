"""
Module tache5_2
Ce module propose des fonctions pour interpréter et visualiser les résultats
d'un clustering. Il fournit des statistiques descriptives et des graphiques
pour mieux comprendre les clusters formés.
"""

import logging
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def interpret_clusters(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    numeric_vars: List[str] = None
) -> pd.DataFrame:
    """
    Fournit un résumé statistique par cluster et un barplot comparant
    les moyennes des variables numériques.

    Args:
        df (pd.DataFrame): DataFrame contenant les données et une colonne
            identifiant le cluster.
        cluster_col (str): Nom de la colonne identifiant le cluster.
        numeric_vars (List[str]): Variables numériques à comparer. Si None,
            toutes les colonnes numériques sont utilisées.

    Returns:
        pd.DataFrame: DataFrame contenant la moyenne des variables numériques
            par cluster.
    """
    if numeric_vars is None:
        # Sélectionne toutes les colonnes numériques (sauf la colonne cluster)
        numeric_vars = df.select_dtypes(include=['int', 'float']).columns
        numeric_vars = list(numeric_vars)
        if cluster_col in numeric_vars:
            numeric_vars.remove(cluster_col)

    if cluster_col not in df.columns:
        raise ValueError(f"La colonne '{cluster_col}' est absente du DataFrame.")

    # Calcul des moyennes par cluster
    cluster_means = df.groupby(cluster_col)[numeric_vars].mean()
    logging.info(
        "Moyennes par cluster :\n%s",
        cluster_means
    )

    # Barplot des moyennes
    cluster_means.plot(
        kind='bar',
        figsize=(8, 5)
    )
    plt.title("Comparaison des moyennes par cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Moyenne")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return cluster_means


def plot_clusters_pca(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    n_components: int = 2
) -> None:
    """
    Effectue une PCA sur les variables numériques et affiche un scatter plot
    coloré par cluster.

    Args:
        df (pd.DataFrame): DataFrame contenant les données et une colonne
            identifiant le cluster.
        cluster_col (str): Nom de la colonne identifiant le cluster.
        n_components (int): Dimension cible pour la PCA (2 ou 3).
    """
    if cluster_col not in df.columns:
        raise ValueError(f"La colonne '{cluster_col}' est absente du DataFrame.")

    # On ne garde que les colonnes numériques (hors cluster)
    df_num = df.select_dtypes(include=['int', 'float']).copy()
    if cluster_col in df_num.columns:
        df_num.drop(columns=[cluster_col], inplace=True)

    if df_num.shape[1] == 0:
        raise ValueError("Aucune variable numérique à projeter en PCA.")

    pca = PCA(n_components=n_components)
    pca_res = pca.fit_transform(df_num)

    plt.figure(figsize=(6, 5))
    labels = df[cluster_col].values

    if n_components == 2:
        plt.scatter(
            pca_res[:, 0],
            pca_res[:, 1],
            c=labels,
            cmap='viridis',
            alpha=0.7
        )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA - Projection en 2D des clusters")
    elif n_components == 3:
        # Projection 3D
        ax = plt.axes(projection='3d')
        ax.scatter(
            pca_res[:, 0],
            pca_res[:, 1],
            pca_res[:, 2],
            c=labels,
            cmap='viridis',
            alpha=0.7
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.title("PCA - Projection en 3D des clusters")
    else:
        raise ValueError("n_components doit être 2 ou 3 pour la visualisation.")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Exemple d'utilisation des fonctions d'interprétation de clusters.
    """
    # Exemple de DataFrame avec une colonne 'cluster'
    data = {
        'age': [25, 32, 47, 51, 29],
        'income': [50000, 60000, 80000, 75000, 40000],
        'score': [0.2, 0.4, 0.8, 0.9, 0.3],
        'cluster': [0, 1, 0, 1, 0]
    }
    df_example = pd.DataFrame(data)
    logging.info("DataFrame exemple :\n%s", df_example)

    # Interprétation basique
    interpret_clusters(df_example, cluster_col="cluster")

    # Visualisation en 2D via PCA
    plot_clusters_pca(df_example, cluster_col="cluster", n_components=2)


if __name__ == "__main__":
    main()
