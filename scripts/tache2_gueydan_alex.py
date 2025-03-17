from sklearn.preprocessing import MinMaxScaler, Normalizer, RobustScaler, StandardScaler, PowerTransformer
import pandas as pd

class Tache2:
    def __init__(self):
        """
        Initialise la classe en chargeant un échantillon du dataset Open Food Facts.
        
        Le dataset est chargé avec un maximum de 100 000 lignes pour optimiser la performance.
        """
        pd.set_option("display.max_columns", None)  # Afficher toutes les colonnes
        pd.set_option("display.max_rows", None)  # Afficher toutes les lignes

        # Chargement du dataset en spécifiant des options pour éviter les erreurs de format et optimiser les performances
        self.df = pd.read_csv("datasets/en.openfoodfacts.org.products.csv", sep="\t", on_bad_lines='skip', 
                            nrows=100000, low_memory=False)
        
        print(self.df.shape)  # Affichage de la taille du DataFrame après chargement

    def remove_high_nan_columns(self, threshold=90):
        """
        Supprime les colonnes avec un pourcentage de valeurs manquantes supérieur au seuil.
        
        Arguments :
            threshold (float) : Pourcentage de NaN au-dessus duquel une colonne est supprimée.

        Retour :
            pd.DataFrame : DataFrame après suppression des colonnes avec trop de NaN.
        """
        # Calcul du pourcentage de NaN par colonne
        nan_ratio = self.df.isna().mean() * 100

        # Identification des colonnes à supprimer
        cols_to_remove = nan_ratio[nan_ratio > threshold].index.tolist()

        # Suppression des colonnes avec trop de NaN
        self.df.drop(columns=cols_to_remove, inplace=True)
        
        return self.df  # Retourner le DataFrame modifié
    
    def normalization(self):
        """
        Applique la normalisation (Normalizer) sur les colonnes numériques.
        
        La normalisation transforme chaque ligne pour qu'elle ait une norme égale à 1.
        
        Retour :
            pd.DataFrame : DataFrame normalisé avec les colonnes numériques modifiées.
        """
        
        # Sélection des colonnes numériques
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])

        # Création de l'objet Normalizer
        scaler = Normalizer()

        # Appliquer la normalisation
        scaled_data = scaler.fit_transform(numeric_df)

        # Retourner un DataFrame avec les données normalisées
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

        return scaled_df  # DataFrame normalisé

    def min_max_scaling(self):
        """
        Applique le MinMaxScaler sur les colonnes numériques pour les ramener dans la plage [0, 1].
        
        Retour :
            pd.DataFrame : DataFrame avec les colonnes numériques normalisées.
        """
        # Sélection des colonnes numériques
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])

        # Application du MinMaxScaler pour normaliser les données
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Création d'un DataFrame avec les données normalisées
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

        return scaled_df  # Retourne les données normalisées
    
    def standard_scaling(self):
        """
        Applique le StandardScaler sur les colonnes numériques pour les centrer (moyenne = 0) et réduire leur écart-type à 1.
        
        Retour :
            pd.DataFrame : DataFrame avec les colonnes numériques standardisées.
        """
        # Sélection des colonnes numériques
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])

        # Application du StandardScaler pour centrer et réduire les données
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Création d'un DataFrame avec les données standardisées
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

        return scaled_df  # Retourne les données standardisées
    
    def robust_scaling(self):
        """
        Applique le RobustScaler sur les colonnes numériques pour les rendre robustes aux valeurs aberrantes.
        Les valeurs NaN sont remplacées par la médiane avant la transformation.
        
        Retour :
            pd.DataFrame : DataFrame avec les colonnes numériques robustes.
        """
        # Sélection des colonnes numériques
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])

        # Application du RobustScaler pour traiter les données avec des valeurs aberrantes
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Création d'un DataFrame avec les données robustes
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

        return scaled_df  # Retourne les données robustes
    
    def power_scaling(self):
        """
        Applique le PowerTransformer sur les colonnes numériques pour transformer les données 
        en une distribution plus proche de la normale.
        
        Retour :
            pd.DataFrame : DataFrame avec les colonnes numériques transformées.
        """
        # Sélection des colonnes numériques
        numeric_df = self.df.select_dtypes(include=["int64", "float64"])

        # Application du PowerTransformer pour rendre les données plus normales
        scaler = PowerTransformer(method='yeo-johnson')  # 'yeo-johnson' permet de traiter aussi les valeurs négatives
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Création d'un DataFrame avec les données transformées
        scaled_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)

        return scaled_df  # Retourne les données transformées:
    
    
tache = Tache2()
tache.remove_high_nan_columns()

# df = tache.min_max_scaling()
# print(df.describe())

# df1 = tache.standard_scaling()
# print(df1.describe())

# df2 = tache.robust_scaling()
# print(df2.describe())

df3 = tache.power_scaling()
print(df3.describe())
