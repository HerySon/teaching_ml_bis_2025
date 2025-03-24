import seaborn as sns
import matplotlib.pyplot as plt

def sous_echantillonner(data, col_categorie, colonnes_a_garder=None, taille_echantillon=None):
    """
    Fonction pour sous-échantillonner un DataFrame en équilibrant les catégories.

    Paramètres :
    - data : DataFrame Pandas
    - col_categorie : (str) Nom de la colonne catégorielle pour équilibrer l’échantillon
    - colonnes_a_garder : (list) Liste des colonnes à garder (None = toutes les colonnes)
    - taille_echantillon : (int) Nombre de lignes par catégorie (None = plus petite catégorie)

    Retourne :
    - Un DataFrame sous-échantillonné et équilibré
    """
    if colonnes_a_garder:
        data = data[[col_categorie] + colonnes_a_garder]
    
    min_cat = data[col_categorie].value_counts().min()
    taille_echantillon = min_cat if taille_echantillon is None else taille_echantillon
    
    df_sample = data.groupby(col_categorie, group_keys=False).head(taille_echantillon)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.countplot(y=data[col_categorie], order=data[col_categorie].value_counts().index, ax=axes[0])
    axes[0].set_title("Distribution originale")
    
    sns.countplot(y=df_sample[col_categorie], order=data[col_categorie].value_counts().index, ax=axes[1])
    axes[1].set_title("Distribution après sous-échantillonnage")
    
    plt.show()
    
    return df_sample