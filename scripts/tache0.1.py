import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

# Configuration du logging pour afficher les messages de debug et d'info
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

def tache_01(
    df: pd.DataFrame, 
    cat_unique_threshold: int = 10, 
    ord_keywords: List[str] = None, 
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Détecte, filtre et sélectionne automatiquement les colonnes pertinentes d'un DataFrame.

    La fonction analyse chaque colonne et les classe en 3 catégories :
      - Colonnes numériques : elles sont optimisées par downcasting pour réduire l'utilisation mémoire.
      - Colonnes catégorielles ordinales : détectées par la présence de mots-clés (ex. 'grade', 'score') dans leur nom.
      - Colonnes catégorielles nominales : celles dont le nombre d'unicités est inférieur ou égal à un seuil donné (cat_unique_threshold).

    Les colonnes ayant trop de modalités (supérieures au seuil) ou dont le type n'est pas supporté sont ignorées.

    Paramètres:
    -----------
    df : pd.DataFrame
        Le DataFrame à traiter.
    cat_unique_threshold : int, optionnel, default=10
        Seuil maximal de modalités pour qu'une colonne catégorielle soit considérée comme nominale.
    ord_keywords : List[str], optionnel
        Liste de mots-clés servant à identifier les colonnes ordinales.
        Par défaut : ['level', 'grade', 'rank', 'score', 'stage', 'rating'].
    verbose : bool, optionnel, default=True
        Si True, affiche des messages détaillés sur le traitement effectué.

    Retourne:
    ---------
    Dict[str, Any]
        Un dictionnaire contenant :
          - 'numerical' : liste des colonnes numériques optimisées.
          - 'categorical_ordinal' : liste des colonnes catégorielles ordinales.
          - 'categorical_nominal' : liste des colonnes catégorielles nominales.
          - 'dropped_columns' : liste des colonnes ignorées (trop de modalités ou type non supporté).
    """
    # Si aucun mot-clé ordinal n'est fourni, utiliser la liste par défaut
    if ord_keywords is None:
        ord_keywords = ['level', 'grade', 'rank', 'score', 'stage', 'rating']

    # Initialisation des listes pour stocker les noms de colonnes selon leur catégorie
    numerical_cols = []              # Colonnes numériques optimisées
    categorical_ordinal_cols = []    # Colonnes catégorielles ordinales
    categorical_nominal_cols = []    # Colonnes catégorielles nominales (nombre de modalités acceptable)
    dropped_columns = []             # Colonnes ignorées (trop de modalités ou type non supporté)

    # Parcours de chaque colonne du DataFrame
    for col in df.columns:
        # Vérifie si la colonne est de type numérique
        if pd.api.types.is_numeric_dtype(df[col]):
            original_dtype = df[col].dtype  # Sauvegarde du type original pour le log
            # Si la colonne est un float, on tente de downcaster en float plus petit (ex: float32)
            if pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast='float')
            else:
                # Pour les entiers, downcast en integer plus petit (ex: int32 ou int16)
                df[col] = pd.to_numeric(df[col], downcast='integer')
            numerical_cols.append(col)
            # Log du downcasting effectué
            if verbose:
                logging.debug(f"Colonne numérique optimisée: {col} (de {original_dtype} à {df[col].dtype})")
        
        # Vérifie si la colonne est de type catégoriel (objet ou déjà catégorique)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            # Calcul du nombre d'unicités, en incluant les NaN comme catégorie à part
            n_unique = df[col].nunique(dropna=False)
            # Vérifie si le nom de la colonne contient un mot-clé indiquant une variable ordinale
            if any(keyword in col.lower() for keyword in ord_keywords):
                categorical_ordinal_cols.append(col)
                if verbose:
                    logging.debug(f"Colonne catégorielle ordinale identifiée: {col} (modalités: {n_unique})")
            # Si la colonne n'est pas ordinale, vérifie si le nombre de modalités est acceptable
            elif n_unique <= cat_unique_threshold:
                categorical_nominal_cols.append(col)
                if verbose:
                    logging.debug(f"Colonne catégorielle nominale identifiée: {col} (modalités: {n_unique})")
            else:
                # Si trop de modalités, la colonne est ignorée
                dropped_columns.append(col)
                if verbose:
                    logging.debug(f"Colonne catégorielle ignorée: {col} (trop de modalités: {n_unique})")
        else:
            # Pour tout autre type non supporté, ignorer la colonne
            dropped_columns.append(col)
            if verbose:
                logging.debug(f"Colonne ignorée: {col} (type non supporté: {df[col].dtype})")

    # Affiche un résumé du filtrage via logging
    if verbose:
        logging.info("\n=== Résumé du filtrage ===")
        logging.info(f"Colonnes numériques : {numerical_cols}")
        logging.info(f"Colonnes catégorielles ordinales : {categorical_ordinal_cols}")
        logging.info(f"Colonnes catégorielles nominales : {categorical_nominal_cols}")
        logging.info(f"Colonnes ignorées : {dropped_columns}")

    # Retourne un dictionnaire avec les listes de colonnes classées
    return {
        'numerical': numerical_cols,
        'categorical_ordinal': categorical_ordinal_cols,
        'categorical_nominal': categorical_nominal_cols,
        'dropped_columns': dropped_columns
    }

# Bloc d'exécution si ce script est lancé directement (utile pour les tests)
if __name__ == "__main__":
    # Création d'un DataFrame d'exemple pour tester la fonction
    data = {
        'age': [25, 32, 47, 51],                          # Colonne numérique
        'income': [50000.0, 60000.0, 80000.0, 75000.0],    # Colonne numérique
        'grade': ['A', 'B', 'B', 'C'],                    # Colonne catégorielle ordinale (ex: niveau, note)
        'city': ['Paris', 'Lyon', 'Paris', 'Marseille'],   # Colonne catégorielle nominale (peu de modalités)
        'id': ['a1', 'b2', 'c3', 'd4'],                    # Colonne catégorielle nominale
        'misc': ['x', 'y', 'z', 'w']                      # Colonne catégorielle nominale
    }
    # Conversion du dictionnaire en DataFrame
    df_example = pd.DataFrame(data)
    
    # Début du filtrage avec un seuil de 4 modalités pour les colonnes catégorielles nominales
    logging.info("Début du filtrage du DataFrame d'exemple...")
    result = tache_01(df_example, cat_unique_threshold=4, verbose=True)
    
    # Affiche le résultat final du filtrage
    logging.info(f"\nRésultat du filtrage : {result}")
