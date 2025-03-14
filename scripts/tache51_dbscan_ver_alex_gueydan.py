import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Tache51:
    def __init__(self):
        """
        Initialise la classe en chargeant un échantillon du dataset Open Food Facts.

        Le dataset est chargé avec un maximum de 10 000 lignes pour optimiser la performance.
        """
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)

        self.df = pd.read_csv("datasets/en.openfoodfacts.org.products.csv", sep="\t", on_bad_lines='skip', 
                              nrows=100000, low_memory=False)
        
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

    def pre_processing(self):
        """
        Effectue les étapes de prétraitement suivantes sur le DataFrame :
        1. Sélectionne les colonnes avec '_100g' dans leur nom.
        2. Supprime les colonnes fortement corrélées (seuil > 0.8).
        3. Supprime les lignes avec des valeurs manquantes.
        4. Standardise les données (normalisation).

        Arguments:
        - df (pd.DataFrame): Le DataFrame contenant les données à prétraiter.

        Retourne:
        - scaled_data_100g (np.array): Les données normalisées et traitées.
        """
        # Sélectionner les colonnes '_100g'
        col100g = [col for col in self.df.columns if "_100g" in col]
        data_100g = self.df[col100g]

        # Calcul de la matrice de corrélation
        corr_matrix = data_100g.corr().abs()

        # Déterminer les colonnes fortement corrélées (seuil > 0.8)
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            column for column in upper_triangle.columns if any(upper_triangle[column] > 0.8)
        ]

        # Supprimer les colonnes redondantes
        data_100g = data_100g.drop(columns=to_drop)

        features_to_drop = [
            'saturated-fat_100g', 'trans-fat_100g', 
            'fruits-vegetables-nuts-estimate-from-ingredients_100g',
            'nutrition-score-fr_100g'
        ]

        data_100g = data_100g.drop(columns=features_to_drop)

        # Supprimer les lignes contenant des valeurs manquantes
        data_100g = data_100g.dropna()

        print(data_100g.columns)

        return data_100g
        
    def apply_dbscan(self, eps=0.5, min_samples=5, threshold=1.5):
        """
        Applique DBSCAN après normalisation des données et traitement des outliers.
        """

        # Prétraitement des données
        self.data100 = self.pre_processing()

        # Suppression des outliers avec IQR
        # Q1 = self.data100.quantile(0.25)
        # Q3 = self.data100.quantile(0.75)
        # IQR = Q3 - Q1
        # lower_bound = Q1 - threshold * IQR
        # upper_bound = Q3 + threshold * IQR

        # self.data100 = self.data100[~((self.data100 < lower_bound) | (self.data100 > upper_bound)).any(axis=1)].reset_index(drop=True)

        # Standardisation des données
        scaler = StandardScaler()
        scaled_data_100g = scaler.fit_transform(self.data100)

        # Appliquer t-SNE pour réduire à 2D
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(scaled_data_100g)  # Données standardisées

        # Appliquer DBSCAN sur les résultats de t-SNE
        dbscan = DBSCAN(eps=2.5, min_samples=30, metric="euclidean")
        clusters = dbscan.fit_predict(X_tsne)

        # Ajouter les clusters au DataFrame
        self.data100["tsne1"], self.data100["tsne2"] = X_tsne[:, 0], X_tsne[:, 1]
        self.data100["cluster"] = clusters

        # Visualisation des clusters après t-SNE + DBSCAN
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.data100["tsne1"], y=self.data100["tsne2"], hue=self.data100["cluster"], palette="viridis")
        plt.title("Clusters DBSCAN après t-SNE")
        plt.show()

    def get_kdist_plot(self, k, radius_nbrs=1.0):

        nbrs = NearestNeighbors(n_neighbors=k, radius=radius_nbrs).fit(self.data100)

        # For each point, compute distances to its k-nearest neighbors
        distances, indices = nbrs.kneighbors(self.data100) 
                                        
        distances = np.sort(distances, axis=0)
        distances = distances[:, k-1]

        # Plot the sorted K-nearest neighbor distance for each point in the dataset
        plt.figure(figsize=(8,8))
        plt.plot(distances)
        plt.xlabel('Points/Objects in the dataset', fontsize=12)
        plt.ylabel('Sorted {}-nearest neighbor distance'.format(k), fontsize=12)
        plt.grid(True, linestyle="--", color='black', alpha=0.4)
        plt.show()
        plt.close()

    def visualize_clusters(self):
        """
        Réduit les dimensions avec PCA et affiche les clusters détectés par DBSCAN.
        """

        # Vérification que 'cluster' est bien dans self.data100
        if "cluster" not in self.data100.columns:
            print("Erreur : DBSCAN doit être appliqué avant de visualiser les clusters.")
            return

        # Standardisation des données
        scaler = StandardScaler()
        scaled_data_100g = scaler.fit_transform(self.data100.drop(columns=["cluster"], errors="ignore"))

        # Réduction de dimension avec PCA
        pca = PCA(n_components=2)
        pca_scores = pca.fit_transform(scaled_data_100g)

        # Ajout des composantes principales
        self.data100["pca1"] = pca_scores[:, 0]
        self.data100["pca2"] = pca_scores[:, 1]

        # Affichage des clusters avec Seaborn
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=self.data100["pca1"], y=self.data100["pca2"], 
            hue=self.data100["cluster"], palette=sns.color_palette("viridis"), alpha=0.4
        )

        plt.axhline(0, color="grey", linestyle="--", linewidth=0.5)
        plt.axvline(0, color="grey", linestyle="--", linewidth=0.5)
        plt.xlabel("Composante principale 1")
        plt.ylabel("Composante principale 2")
        plt.title("Clusters DBSCAN après réduction PCA")
        plt.show()

        # Vérification du nombre de points considérés comme bruit (-1)
        print("Nombre de points considérés comme bruit (-1) :", (self.data100["cluster"] == -1).sum())

    def apply_tsne_and_plot(self, perplexity=30, learning_rate=200, n_iter=1000):
        """
        Applique t-SNE aux données standardisées et visualise les clusters détectés par DBSCAN.
        """

        if "cluster" not in self.data100.columns:
            print("Erreur : DBSCAN doit être appliqué avant t-SNE.")
            return
        
        # On ignore la colonne des clusters pour t-SNE
        features = self.data100.drop(columns=["cluster"])

        # Appliquer t-SNE pour réduction à 2D
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter, random_state=42)
        tsne_results = tsne.fit_transform(features)

        # Ajouter les colonnes t-SNE dans le DataFrame
        self.data100["tsne1"] = tsne_results[:, 0]
        self.data100["tsne2"] = tsne_results[:, 1]

        # 🔹 **Visualisation avec seaborn**
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x=self.data100["tsne1"], 
            y=self.data100["tsne2"], 
            hue=self.data100["cluster"], 
            palette="tab10",  # Palette de couleurs pour mieux voir les clusters
            alpha=0.7
        )
        
        plt.title("Visualisation des Clusters DBSCAN avec t-SNE")
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.legend(title="Cluster", loc="upper right", bbox_to_anchor=(1.2, 1))
        plt.show()

# Exemple d'utilisation :
dbscan = Tache51()
dbscan.remove_irrelevant_columns()
dbscan.remove_high_nan_columns()
clusters = dbscan.apply_dbscan(eps=1, min_samples=30)
dbscan.apply_tsne_and_plot()
# dbscan.get_kdist_plot(40)
# dbscan.visualize_clusters()

