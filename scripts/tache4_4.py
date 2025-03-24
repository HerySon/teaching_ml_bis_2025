"""
Module tache4_4
Ce module propose une fonction pour tester différentes méthodes de 
plongement non linéaire (Isomap, MDS, t-SNE, UMAP, etc.) afin de visualiser 
les données en 2D. L'objectif est d'explorer la structure du jeu de données 
de manière plus intuitive qu'avec un simple PCA linéaire.
"""

import logging
import pandas as pd
from sklearn.manifold import TSNE, Isomap, MDS

try:
    import umap  # Nécessite "pip install umap-learn"
    HAVE_UMAP = True
except ImportError:
    HAVE_UMAP = False

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def reduire_dimension(
    df: pd.DataFrame,
    method: str = "tsne",
    n_components: int = 2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Réduit la dimension d'un DataFrame pour une visualisation en 2D à l'aide 
    d'une méthode de plongement non linéaire.
    
    Args:
        df (pd.DataFrame): Données d'entrée (colonnes numériques).
        method (str): Méthode parmi "isomap", "mds", "tsne", "umap".
        n_components (int): Dimension cible (2 pour la visualisation).
        random_state (int): Graine aléatoire pour la reproductibilité.
    
    Returns:
        pd.DataFrame: DataFrame contenant les coordonnées 2D résultantes, avec 
                      les mêmes index que df.
    """
    df_num = df.select_dtypes(include=['int', 'float'])
    if df_num.shape[1] == 0:
        raise ValueError("Aucune colonne numérique trouvée dans le DataFrame.")

    method = method.lower()
    logging.info("Réduction de dimension en 2D avec la méthode : %s", method)

    if method == "isomap":
        model = Isomap(n_components=n_components)
    elif method == "mds":
        model = MDS(n_components=n_components, random_state=random_state)
    elif method == "tsne":
        model = TSNE(n_components=n_components, random_state=random_state)
    elif method == "umap":
        if not HAVE_UMAP:
            raise ImportError(
                "Le package 'umap-learn' n'est pas installé. "
                "Veuillez l'installer pour utiliser UMAP."
            )
        model = umap.UMAP(n_components=n_components,
                          random_state=random_state)
    else:
        raise ValueError("Méthode inconnue. Choisir parmi 'isomap', "
                         "'mds', 'tsne', 'umap'.")

    embedding = model.fit_transform(df_num)
    colonnes = [f"comp_{i+1}" for i in range(n_components)]
    df_embedded = pd.DataFrame(embedding, columns=colonnes, index=df_num.index)
    logging.info("Réduction terminée. Dimensions finales : %s", 
                 df_embedded.shape)
    return df_embedded


def main() -> None:
    """
    Exemple d'utilisation de la fonction reduire_dimension.
    """
    data = {
        'feat1': [1.0, 2.5, 3.3, 4.1],
        'feat2': [10, 9, 12, 15],
        'feat3': [0.2, 0.3, 0.9, 0.1],
        'feat4': [100, 200, 300, 400],
        'cat': ['A', 'B', 'A', 'C']
    }
    df_example = pd.DataFrame(data)
    logging.info("Données initiales :\n%s", df_example)
    df_tsne = reduire_dimension(df_example, method="tsne", n_components=2)
    logging.info("Coordonnées t-SNE :\n%s", df_tsne)
    # Pour tester UMAP, décommente les lignes suivantes :
    # df_umap = reduire_dimension(df_example, method_
