import pandas as pd
import numpy as np

def Classement_colonnes(data, ordinal_feature=['nutriscore_grade'], threshold=999):
    """
    Classe les colonnes d'un DataFrame et optimise les variables numériques.

    Paramètres :
    - data : DataFrame Pandas
    - ordinal_feature : Liste des variables ordinales à considérer
    - threshold : Nombre maximal de catégories avant d'être considéré comme "trop dispersé"

    Retourne :
    - Un dictionnaire des colonnes classées par type.
    - Un DataFrame optimisé avec downcasting des variables numériques.
    """
    colonnes_numeriques = []
    colonnes_categorielles_ordinales = []
    colonnes_categorielles_non_ordinales = []
    colonnes_trop_dispersées = []

    for col in data.columns:
        if np.issubdtype(data[col].dtype, np.number):
            colonnes_numeriques.append(col)
        elif data[col].dtype == "object" or data[col].dtype.name == "category":
            nb_categories = data[col].nunique()

            if col in ordinal_feature:
                colonnes_categorielles_ordinales.append(col)
            else:
                colonnes_categorielles_non_ordinales.append(col)

            if nb_categories > threshold:
                colonnes_trop_dispersées.append(col)

    def downcast_column(col):
        if np.issubdtype(col.dtype, np.integer):
            return pd.to_numeric(col, downcast="integer")
        elif np.issubdtype(col.dtype, np.floating):
            return pd.to_numeric(col, downcast="float")
        return col

    for col in colonnes_numeriques:
        data[col] = downcast_column(data[col])

    resultat = {
        "Numeriques": colonnes_numeriques,
        "Categorielles ordinales": colonnes_categorielles_ordinales,
        "Categorielles non ordinales": colonnes_categorielles_non_ordinales,
        "Colonnes trop dispersées": colonnes_trop_dispersées
    }

    return resultat, data