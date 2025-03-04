import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def sous_echantillonner(data, 
                        col_categorie, 
                        colonnes_a_garder=None, 
                        taille_echantillon=None, 
                        random_state=42):
    """
    Fonction pour sous-échantillonner un DataFrame en équilibrant les catégories.

    Paramètres :
    - data : DataFrame Pandas
    - col_categorie : (str) Nom de la colonne catégorielle pour équilibrer l’échantillon
    - colonnes_a_garder : (list) Liste des colonnes à garder (None = toutes les colonnes)
    - taille_echantillon : (int) Nombre de lignes par catégorie (None = plus petite catégorie)
    - random_state : (int) Seed pour la reproductibilité

    Retourne :
    - Un DataFrame sous-échantillonné et équilibré
    """

    df_filtre = data.copy()
    
    if colonnes_a_garder:
        df_filtre = df_filtre[[col_categorie] + colonnes_a_garder]

    # 🔹 Étape 2 : Déterminer la taille d’échantillon max possible
    min_cat = df_filtre[col_categorie].value_counts().min()  # Taille de la plus petite catégorie
    taille_echantillon = taille_echantillon or min_cat  # Par défaut, on prend la taille la plus faible

    # 🔹 Étape 3 : Prendre un sous-échantillon de chaque catégorie
    df_sample = df_filtre.groupby(col_categorie, group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), taille_echantillon), random_state=random_state)
    )

    # 🔹 Étape 4 : Réinitialiser l’index
    df_sample = df_sample.reset_index(drop=True)

    # 🔹 Étape 5 : Visualisation des proportions avant/après
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(y=data[col_categorie], order=data[col_categorie].value_counts().index, ax=axes[0])
    axes[0].set_title("Distribution originale")

    sns.countplot(y=df_sample[col_categorie], order=data[col_categorie].value_counts().index, ax=axes[1])
    axes[1].set_title("Distribution après sous-échantillonnage")

    plt.show()

    return df_sample