import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from sklearn.impute import KNNImputer

class Tache43:
    """
    Classe permettant d'appliquer l'Analyse en Composantes Principales (ACP) sur le dataset Open Food Facts.

    Cette classe propose plusieurs méthodes pour :
    - Nettoyer le dataset en supprimant les colonnes non pertinentes.
    - Éliminer les doublons afin d'assurer la qualité des données.
    - Supprimer les colonnes contenant un taux trop élevé de valeurs manquantes.

    Attributs :
        df (DataFrame) : DataFrame Pandas contenant un échantillon du dataset Open Food Facts.
        scaled_data (ndarray) : Données standardisées prêtes pour l'ACP.
        pca_components (ndarray) : Composantes principales après application de l'ACP.
        explained_variance_ratio (ndarray) : Pourcentage de variance expliquée par chaque composante principale.
    """

    def __init__(self):
        """
        Initialise la classe en chargeant un échantillon du dataset Open Food Facts.

        Le dataset est chargé avec un maximum de 10 000 lignes pour optimiser la performance.
        """
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        self.df = pd.read_csv("datasets/en.openfoodfacts.org.products.csv", sep="\t", on_bad_lines='skip', 
                              nrows=10000, low_memory=False)

        self.scaled_data = None
        self.pca_components = None
        self.explained_variance_ratio = None

    def remove_irrelevant_columns(self):
        """
        Supprime les colonnes non pertinentes pour l'analyse.

        Retour :
            pd.DataFrame : Le DataFrame nettoyé, sans les colonnes jugées inutiles.
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
            "image_small_url", "image_url", "last_updated_t", "last_updated_datetime",
            "last_modified_by", "last_image_t", "last_image_datetime"
        ]

        self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns], 
                     errors='ignore', inplace=True)
        return self.df

    def remove_duplicates(self):
        """
        Supprime les doublons dans le DataFrame.

        Retour :
            pd.DataFrame : Le DataFrame sans doublons.
        """
        self.df.drop_duplicates(keep="first", inplace=True)
        return self.df

    def remove_high_nan_columns(self, threshold=70):
        """
        Supprime les colonnes contenant un pourcentage de valeurs manquantes supérieur au seuil défini.

        Arguments :
            threshold (float) : Pourcentage maximal de valeurs manquantes toléré dans une colonne.
                                Par défaut, ce seuil est fixé à 70%.

        Retour :
            pd.DataFrame : Le DataFrame après suppression des colonnes trop incomplètes.
        """
        nan_ratio = self.df.isna().mean() * 100  # Calcul du pourcentage de valeurs NaN par colonne
        cols_to_remove = nan_ratio[nan_ratio > threshold].index.tolist()  # Sélection des colonnes à supprimer

        self.df.drop(columns=cols_to_remove, inplace=True)  # Suppression des colonnes sélectionnées
        return self.df
    
    def get_column_count(self):
        """
        Retourne le nombre total de colonnes dans le DataFrame.

        Retour :
            int : Nombre total de colonnes présentes dans le DataFrame.
        """
        return self.df.shape[1]
    
    def clean_column(self, nom_colonne):
        """
        Nettoie la colonne 'serving_size' en extrayant les quantités et en les convertissant en unités standardisées.

        Cette méthode effectue les étapes suivantes :
        - Extrait la quantité et l'unité de mesure à l'aide d'une expression régulière.
        - Convertit les valeurs en grammes (g) ou millilitres (ml) selon le type de produit.
        - Supprime les valeurs aberrantes (quantités inférieures ou égales à 0 ou supérieures à 10 000).

        Arguments :
            nom_colonne (str) : Nom de la colonne où stocker les valeurs nettoyées.

        Retour :
            pd.DataFrame : Le DataFrame avec une colonne contenant les valeurs normalisées.
        """

        # Expression régulière pour détecter les quantités et unités (g, kg, ml, etc.)
        pattern = r'(\d+[\.,]?\d*)\s*(g|kg|kilogram|kilograms|l|litre|litres|cl|ml)'

        def extract_serving(value):
            """
            Extrait la quantité et l'unité, convertit les valeurs dans une unité standardisée et filtre les valeurs aberrantes.

            Arguments :
                value (str) : Valeur brute de la colonne.

            Retour :
                str ou None : Valeur nettoyée en grammes ou millilitres, ou None si la donnée est invalide.
            """
            matches = re.findall(pattern, str(value))
            if not matches:
                return None

            # Sélection de la première correspondance trouvée
            quantity, unit = matches[0]

            # Remplacement des virgules par des points et conversion en float
            quantity = quantity.replace(",", ".")
            try:
                quantity = float(quantity)
            except ValueError:
                return None

            # Conversion des unités en valeurs standardisées
            unit = unit.lower()
            if unit in ["kg", "kilogram", "kilograms"]:
                quantity *= 1000  # Conversion en grammes
                unit = "g"
            elif unit in ["mg"]:
                quantity /= 1000  # Conversion en grammes
                unit = "g"
            elif unit in ["l", "litre", "litres"]:
                quantity *= 1000  # Conversion en millilitres
                unit = "ml"
            elif unit in ["cl"]:
                quantity *= 10  # Conversion en millilitres
                unit = "ml"
            elif unit == "":
                unit = "g"

            # Suppression des valeurs aberrantes
            if quantity <= 0 or quantity > 10000:
                return None

            return f"{quantity}"

        # Appliquer la transformation à la colonne 'serving_size'
        self.df[nom_colonne] = self.df['serving_size'].apply(extract_serving)

        return self.df

    def knn_imputer(self, n_neighbors):
        """
        Impute les valeurs manquantes en utilisant l'algorithme KNN Imputer.

        Cette méthode :
        - Sélectionne uniquement les colonnes numériques du DataFrame.
        - Remplit les valeurs manquantes en utilisant la moyenne des 'n_neighbors' voisins les plus proches.

        Arguments :
            n_neighbors (int) : Nombre de voisins à utiliser pour l'imputation des valeurs manquantes.

        Retour :
            pd.DataFrame : Le DataFrame avec les valeurs imputées.
        """
        # Sélection des colonnes numériques uniquement
        df_numeric = self.df.select_dtypes(include=['float64', 'int64'])

        # Application du KNN Imputer sur les colonnes sélectionnées
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(df_numeric)

        # Remplacement des valeurs dans le DataFrame
        self.df[df_numeric.columns] = imputed_data

        return self.df

    def apply_pca(self, n_components=5):
        """
        Applique la standardisation des données et effectue une Analyse en Composantes Principales (ACP).

        Cette méthode suit les étapes suivantes :
        - Sélectionne les colonnes contenant '_100g' (variables nutritionnelles pour 100g).
        - Supprime les colonnes présentant une forte corrélation (> 0.8) pour éviter la redondance.
        - Supprime les lignes contenant des valeurs manquantes.
        - Standardise les données afin d’assurer une ACP correcte.
        - Applique l’ACP pour réduire la dimensionnalité des données.
        - Stocke les nouvelles coordonnées dans l’espace des composantes principales ainsi que la variance expliquée.

        Arguments :
            n_components (int) : Nombre de composantes principales à conserver (par défaut 5).

        Retour :
            tuple :
                - np.ndarray : Les données projetées dans l'espace des composantes principales.
                - np.ndarray : La proportion de variance expliquée par chaque composante.
        """

        # Sélection des colonnes contenant '_100g' dans leur nom (variables nutritionnelles)
        col100g = [col for col in self.df.columns if '_100g' in col]
        
        # Extraction des colonnes pertinentes
        data_100g = self.df[col100g]

        # Calcul de la matrice de corrélation
        corr_matrix = data_100g.corr().abs()

        # Détermination des colonnes fortement corrélées (seuil > 0.8)
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)]

        # Suppression des colonnes redondantes
        data_100g.drop(columns=to_drop, inplace=True)

        # Suppression des lignes contenant des valeurs manquantes
        data_100g.dropna(inplace=True) 
        
        # Standardisation des données pour assurer une ACP correcte
        scaler = StandardScaler()
        scaled_data_100g = scaler.fit_transform(data_100g)
        
        # Application de l'ACP
        pca = PCA(n_components=n_components)
        self.pca_components = pca.fit_transform(scaled_data_100g)
        
        # Récupération de la variance expliquée par chaque composante principale
        self.explained_variance_ratio = pca.explained_variance_ratio_

        print(data_100g.columns)

        return self.pca_components, self.explained_variance_ratio

    def plot_explained_variance(self):
        """
        Génère un graphique illustrant la variance expliquée par chaque composante principale.

        Cette méthode permet de visualiser l'importance de chaque composante dans la réduction de la dimensionnalité.
        Un graphique en barres est affiché, montrant la proportion de variance expliquée par chaque composante principale.
        """

        plt.figure(figsize=(8, 6))
        plt.bar(range(1, len(self.explained_variance_ratio) + 1), self.explained_variance_ratio, alpha=0.7)
        plt.xlabel('Composantes principales')
        plt.ylabel('Variance expliquée')
        plt.title('Variance expliquée par chaque composante principale')
        plt.show()

    def cumulative_variance_plot(self):
        """
        Affiche un graphique de la variance expliquée cumulée.

        Cette méthode permet de visualiser la proportion de variance expliquée en fonction du nombre 
        de composantes principales retenues. Elle trace également deux lignes de référence :
        - Une ligne horizontale rouge au niveau de 95% de variance expliquée.
        - Une ligne verticale indiquant le nombre minimal de composantes nécessaires pour atteindre ce seuil.

        Le but est d’aider à choisir un nombre optimal de composantes principales pour l’ACP.
        """

        # Calcul de la variance cumulée
        cumulative_variance = np.cumsum(self.explained_variance_ratio)

        # Tracé du graphique
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                 marker='o', linestyle='--', color='b', label='Variance cumulée')
        plt.xlabel('Nombre de composantes principales')
        plt.ylabel('Variance cumulée expliquée')
        plt.title('Variance cumulée expliquée')

        # Ajout de lignes de référence (95% de variance expliquée)
        plt.axhline(y=0.95, color='r', linestyle='-', label='95% Variance expliquée')
        plt.axvline(x=np.argmax(cumulative_variance >= 0.95) + 1, color='r', linestyle='--', 
                    label=f'{np.argmax(cumulative_variance >= 0.95) + 1} composantes')

        plt.legend()
        plt.show()

    def plot_pca_biplot(self, truncate_outliers=True, threshold=1.5):
        """
        Affiche un biplot de l'ACP combinant :
        - La projection des observations sur les deux premières composantes principales.
        - Les vecteurs des variables contribuant à ces composantes.

        Ce graphique permet de visualiser l'impact des variables sur la réduction de dimension.
        """

        # Vérifier que l'ACP a été appliquée
        if self.pca_components is None:
            raise ValueError("L'ACP n'a pas encore été appliquée. Exécutez `apply_pca()` d'abord.")
        
        # Récupérer les colonnes de données _100g
        cols_100g = [col for col in self.df.columns if '_100g' in col]
        data_100g = self.df[cols_100g].dropna()
        
        # Filtrage des outliers avant l'ACP
        if truncate_outliers:
            # Calcul des quartiles et de l'IQR pour chaque colonne
            Q1 = data_100g.quantile(0.25)
            Q3 = data_100g.quantile(0.75)
            IQR = Q3 - Q1
            
            # Définir les bornes inférieures et supérieures pour chaque colonne
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Filtrer les données
            data_100g = data_100g[~((data_100g < lower_bound) | (data_100g > upper_bound)).any(axis=1)]

        # Standardisation des données
        scaled_data_100g = StandardScaler().fit_transform(data_100g)

        # Appliquer l'ACP
        pca = PCA(n_components=2)
        pca_scores = pca.fit_transform(scaled_data_100g)  # Projection des observations
        loadings = pca.components_.T  # Matrice des charges (impact des variables)

        # Création du graphique
        plt.figure(figsize=(10, 7))

        # Nuage de points des observations
        sns.scatterplot(x=pca_scores[:, 0], y=pca_scores[:, 1], alpha=0.5, label="Observations")

        # Ajout des vecteurs des variables
        for i, var in enumerate(cols_100g):
            plt.arrow(0, 0, loadings[i, 0] * 3, loadings[i, 1] * 3, color='r', alpha=0.5, head_width=0.05)
            plt.text(loadings[i, 0] * 3.2, loadings[i, 1] * 3.2, var, color='r', fontsize=10)

        # Personnalisation du graphique
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.title("Biplot de l'ACP")
        plt.legend()
        plt.show()

    def plot_pca_scatter(self, truncate_outliers=True, threshold=1.5):
        """
        Affiche un nuage de points des données projetées sur les deux premières composantes principales.

        Cette visualisation permet d’analyser la distribution des observations dans le nouvel espace réduit 
        par l’Analyse en Composantes Principales (ACP). Une dispersion bien définie peut indiquer des 
        regroupements naturels dans les données.
        """

        # Vérifier que l'ACP a été appliquée
        if self.pca_components is None:
            raise ValueError("L'ACP n'a pas encore été appliquée. Exécutez `apply_pca()` d'abord.")
        
        # Récupérer les colonnes de données _100g
        cols_100g = [col for col in self.df.columns if '_100g' in col]
        data_100g = self.df[cols_100g].dropna()

        # Filtrage des outliers avant l'ACP
        if truncate_outliers:
            # Calcul des quartiles et de l'IQR pour chaque colonne
            Q1 = data_100g.quantile(0.25)
            Q3 = data_100g.quantile(0.75)
            IQR = Q3 - Q1
            
            # Définir les bornes inférieures et supérieures pour chaque colonne
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Filtrer les données
            data_100g = data_100g[~((data_100g < lower_bound) | (data_100g > upper_bound)).any(axis=1)]

        # Standardisation des données
        scaled_data_100g = StandardScaler().fit_transform(data_100g)

        # Appliquer l'ACP
        pca = PCA(n_components=2)
        pca_scores = pca.fit_transform(scaled_data_100g)  # Projection des observations

        # Création du graphique
        plt.figure(figsize=(8, 6))

        # Nuage de points des observations
        sns.scatterplot(x=pca_scores[:, 0], y=pca_scores[:, 1], alpha=0.5)
        plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.title("Nuage de points des données après ACP")
        plt.show()

    def heatmap(self, threshold):
        """
        Affiche une heatmap des corrélations entre les variables numériques du dataset,
        en filtrant celles qui sont supérieures ou inférieures à un seuil donné.

        La heatmap permet d’identifier les relations entre les différentes variables et de 
        visualiser les groupes de caractéristiques fortement corrélées.

        Arguments :
            threshold (float) : Seuil minimal de corrélation à afficher. Seules les corrélations 
                                absolues supérieures à ce seuil apparaîtront sur la heatmap.
        """

        # Sélection des colonnes numériques
        numeric_cols = self.df.select_dtypes(include=['int64', 'float64'])

        # Calcul de la matrice de corrélation
        correlation_matrix = numeric_cols.corr().abs()

        # Application du seuil pour filtrer les valeurs faibles
        correlation_matrix = correlation_matrix[(correlation_matrix > threshold)]

        # Création de la heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, linewidths=0.5)
        plt.title(f'Carte de chaleur des corrélations (Seuil: {threshold})')
        plt.show()


tache = Tache43()
tache.remove_irrelevant_columns()
tache.remove_high_nan_columns()
tache.apply_pca()
tache.plot_explained_variance()
tache.cumulative_variance_plot()
tache.plot_pca_biplot()
tache.plot_pca_scatter()
