#Machine Learning Open Foods Facts - Alex Gueydan 25 ©
#Import de toutes les librairies
import pandas as pd
import re
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder


class Tache1:
    """
    Classe permettant de manipuler et traiter un dataset Open Food Facts.
    Elle inclut des méthodes pour sélectionner des colonnes numériques, ordinales et non ordinales,
    effectuer des downcasts pour optimiser la mémoire et filtrer les variables en fonction du nombre
    de catégories uniques.
    """

    def __init__(self):
        """
        Initialise la classe et charge un échantillon de 10 000 lignes du dataset Open Food Facts
        en utilisant le séparateur tabulation.

        Attributs :
            df (DataFrame): Un DataFrame Pandas contenant l'échantillon du dataset.
        """
        #Forcage de pandas a afficher autant de caractère qu'il peut sur une ligne
        pd.set_option("display.max_columns", None)

        #Meme chose avec le nombre de lignes
        pd.set_option("display.max_rows", None)

        #Chargement du dataset, en prenant un échantillon de 100000 lignes pour ne pas trop surcharger
        self.df = pd.read_csv("datasets/en.openfoodfacts.org.products.csv", sep="\t", on_bad_lines='skip', nrows=10000, low_memory=False)

    def remove_irrelevant_columns(self):
        """
        Supprime les colonnes non pertinentes pour l'analyse.
        
        Retour :
            pd.DataFrame : Le DataFrame nettoyé.
        """
        columns_to_drop = [
            "code", "url", "creator", "created_t", "created_datetime",
            "last_modified_t", "last_modified_datetime", "packaging", "packaging_tags",
            "brands_tags", "categories_tags", "categories_fr",
            "origins_tags", "manufacturing_places", "manufacturing_places_tags",
            "labels_tags", "labels_fr", "emb_codes", "emb_codes_tags",
            "first_packaging_code_geo", "cities", "cities_tags", "purchase_places",
            "countries_tags", "countries_fr", "image_ingredients_url",
            "image_ingredients_small_url", "image_nutrition_url", "image_nutrition_small_url",
            "image_small_url", "image_url", "last_updated_t", "last_updated_datetime", "last_modified_by", "last_image_t"
        ]

        self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns], errors='ignore', inplace=True)
        return self.df
    
    def remove_duplicates(self):
        """
        Supprime les doublons en considérant toutes les colonnes.

        Retour :
            pd.DataFrame : Le DataFrame sans doublons.
        """
        self.df.drop_duplicates(keep="first", inplace=True)
        
        return self.df
    
    def remove_high_nan_columns(self, threshold=90):
        """
        Supprime les colonnes ayant un pourcentage de valeurs manquantes supérieur au seuil.

        Arguments :
            threshold (float) : Pourcentage de valeurs manquantes au-dessus duquel une colonne est supprimée.

        Retour :
            pd.DataFrame : Le DataFrame sans les colonnes trop incomplètes.
        """
        # Calcul du pourcentage de valeurs manquantes par colonne
        nan_ratio = self.df.isna().mean() * 100

        # Colonnes à supprimer (taux de NaN supérieur au seuil)
        cols_to_remove = nan_ratio[nan_ratio > threshold].index.tolist()

        # Suppression des colonnes
        self.df.drop(columns=cols_to_remove, inplace=True)
        
        return self.df
    
    def get_column_count(self):
        """Retourne le nombre total de colonnes du DataFrame."""
        return self.df.shape[1]
    
    def clean_column(self, nom_colonne):
        """
        Nettoie la colonne 'serving_size' :
        - Extrait la quantité principale en grammes ou millilitres.
        - Convertit les unités en valeurs standardisées.
        - Supprime les valeurs aberrantes.
        
        Retour :
            pd.DataFrame : Dataset avec une colonne 'serving_size_clean' nettoyée.
        """

        # Expression régulière pour détecter la quantité et l'unité, c'est pour detecter les g, kg, ml, l etc.
        pattern = r'(\d+[\.,]?\d*)\s*(g|kg|kilogram|kilograms|l|litre|litres|cl|ml)'

        def extract_serving(value):

            # Trouver tous les nombres suivis d'unités dans la colonne
            matches = re.findall(pattern, str(value))
            if not matches:
                return None

            # Prendre le premier nombre trouvé
            quantity, unit = matches[0]
            print(matches)

            # Remplacer les virgules par des points et convertir en float
            quantity = quantity.replace(",", ".")
            try:
                quantity = float(quantity)
            except ValueError:
                return None  # Si conversion impossible

            # Convertir les unités en standard
            unit = unit.lower()
            if unit in ["kg", "kilogram", "kilograms"]:
                quantity *= 1000
                unit = "g"
            elif unit in ["mg"]:
                quantity /= 1000
                unit = "g"
            elif unit in ["l", "litre", "litres"]:
                quantity *= 1000
                unit = "ml"
            elif unit in ["cl"]:
                quantity *= 10
                unit = "ml"
            elif unit == "":
                unit = "g"

            # Éliminer les valeurs aberrantes
            if quantity <= 0 or quantity > 10000:
                return None

            return f"{quantity}"

        # Appliquer la transformation
        self.df[nom_colonne] = self.df['serving_size'].apply(extract_serving)

        return self.df

    def knn_imputer(self, n_neighbors):
        """
        Impute les valeurs manquantes d'une colonne spécifique en utilisant KNN Imputer,
        en se basant sur les colonnes corrélées.
        
        :param n_neighbors: Nombre de voisins à utiliser pour l'imputation
        """
        # Calcul de la matrice de corrélation
        df_numeric = self.df.select_dtypes(include=['float64', 'int64'])
        print(df_numeric.isnull().sum())

        # Appliquer le KNN Imputer uniquement sur les colonnes sélectionnées
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(df_numeric)
        df_subset_imputed = pd.DataFrame(imputed_data, columns=df_numeric.columns)

        print(f"Dimensions avant imputation: {df_numeric.shape}")
        print(f"Dimensions après imputation: {imputed_data.shape}")

        return df_subset_imputed
    
def clean_data_set():
    tache = Tache1()
    tache.remove_irrelevant_columns()
    tache.remove_high_nan_columns()
    tache.remove_duplicates()
    return tache.df
    
tache = Tache1()
tache.remove_irrelevant_columns()
tache.remove_high_nan_columns()
tache.remove_duplicates()
df_imputed = tache.knn_imputer(5)
# df_imputed = tache.knn_impute_column('energy_100g', 5, 0.8)
print(df_imputed[["energy-kcal_100g", "energy_100g"]].head(300))
