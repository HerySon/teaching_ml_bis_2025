#Machine Learning Open Foods Facts - Alex Gueydan 25 ©
#Import de toutes les librairies
import pandas as pd


class Tache0:
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

    def select_numeric_columns(self):
        """
        Sélectionne les colonnes du DataFrame ayant un type numérique (int64, float64).

        Retour :
            DataFrame : Un DataFrame contenant uniquement les colonnes numériques.
        """
        #isolement des colonnes qui ont un type entier ou float
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        return self.df[numeric_cols]
    
    def select_ordinal_columns(self, cols):
        """
        Sélectionne les colonnes ordinales spécifiées dans l'argument `cols`.

        Arguments :
            cols (list) : Liste des noms de colonnes ordinales.

        Retour :
            DataFrame : Un DataFrame contenant uniquement les colonnes ordinales spécifiées.
        """
        # Sélection des colonnes spécifiées dans l'argument cols
        return self.df[cols]

    def select_non_ordinal_columns(self, cols):
        """
        Sélectionne les colonnes non ordinales du DataFrame. Exclut les colonnes ordinales spécifiées
        dans l'argument `cols`.

        Arguments :
            cols (list) : Liste des noms de colonnes ordinales à exclure.

        Retour :
            DataFrame : Un DataFrame contenant uniquement les colonnes non ordinales.
        """
        # Sélection des colonnes de type 'object'
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        # Exclusion des colonnes ordinales
        non_ordinal_cols = [col for col in categorical_cols if col not in cols]
        
        return self.df[non_ordinal_cols]
    
    def select_non_ordinal_columns_without_date(self, cols):
        """
        Sélectionne les colonnes non ordinales sans les colonnes contenant des dates. 
        Exclut les colonnes ordinales spécifiées dans l'argument `cols` et les colonnes dont le nom 
        contient "date".

        Arguments :
            cols (list) : Liste des noms de colonnes ordinales à exclure.

        Retour :
            DataFrame : Un DataFrame contenant uniquement les colonnes non ordinales sans les colonnes de type "date".
        """
        # Sélection des colonnes de type 'object'
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Exclusion des colonnes dont le nom contient 'date'
        non_ordinal_cols = [col for col in categorical_cols if col not in cols and 'date' not in col.lower()]
        
        return self.df[non_ordinal_cols]

    def downcast_numerics(self):
        """
        Effectue un downcasting sur toutes les colonnes numériques pour réduire l'utilisation de la mémoire.
        Les types `int64` sont downcastés vers des types d'entiers plus petits (int32, int16, int8),
        et les types `float64` sont downcastés vers `float32`.

        Retour :
            DataFrame : Le DataFrame avec les colonnes numériques downcastées.
        """
        # Itération sur toutes les colonnes du dataset
        for col in self.df.select_dtypes(include=['int64', 'float64']).columns:
            # Si la colonne est de type int64, essayer de la downcaster en int32, int16 ou int8
            if self.df[col].dtype == 'int64':
                self.df[col] = pd.to_numeric(self.df[col], downcast='integer')
            
            # Si la colonne est de type float64, essayer de la downcaster en float32
            elif self.df[col].dtype == 'float64':
                self.df[col] = pd.to_numeric(self.df[col], downcast='float')
        
        return self.df

    def numbers_variables(self, threshold):
        """
        Filtre les variables catégorielles qui ont un nombre de catégories uniques inférieur ou égal
        au seuil spécifié dans l'argument `threshold`.

        Arguments :
            threshold (int) : Le nombre maximum de catégories uniques pour qu'une variable soit incluse.

        Retour :
            DataFrame : Le DataFrame avec les colonnes catégorielles ayant un nombre de catégories 
            inférieur ou égal au seuil.
        """
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Liste pour stocker les colonnes filtrées
        filtered_cols = []
        
        # Compter le nombre de catégories uniques pour chaque colonne catégorielle ordinale ou non
        for col in categorical_cols:
            unique_count = self.df[col].nunique()
            
            # Si le nombre de catégories uniques est inférieur ou égal a la variable threshold, on garde la colonne
            if unique_count <= threshold:
                filtered_cols.append(col)
        
        return self.df[filtered_cols]
    
    def unique_categories_count(self, column_name):
        """
        Renvoie le nombre de catégories uniques dans la colonne spécifiée.

        Arguments :
            column_name (str) : Le nom de la colonne dont on veut obtenir le nombre de catégories uniques.

        Retour :
            int : Le nombre de catégories uniques dans la colonne spécifiée.
        """
        # Vérifier si la colonne existe
        return self.df[column_name].nunique()
    

    
    
tache = Tache0()
data = tache.downcast_numerics()
category_df = tache.numbers_variables(100)
print(category_df.head())