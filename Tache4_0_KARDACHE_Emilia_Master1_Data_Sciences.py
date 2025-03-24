import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset
path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(path, nrows=100, sep='\t', encoding="utf-8", low_memory=False, na_filter=True)

# Supprimer les doublons
df = df.drop_duplicates() 

# Sélectionner les colonnes numériques pour les visualisations univariées
numeric_columns = df.select_dtypes(include=['number']).columns
 
# Fonction de visualisation univariée
def plot_univariate_data(df, columns):
    for col in columns:
        # Enlever les NaN avant de faire la visualisation
        df_cleaned = df[col].dropna()

        # Vérifier s'il y a des valeurs manquantes après nettoyage
        if df_cleaned.isnull().sum() > 0:
            print(f"Des valeurs manquantes dans la colonne : {col}")
        
        plt.figure(figsize=(12, 6))

        # 1. Histogramme
        plt.subplot(1, 3, 1)
        sns.histplot(df_cleaned, kde=True, bins=30, color='skyblue')
        plt.title(f'Histogramme de {col}')
        plt.xlabel(col)
        plt.ylabel('Fréquence')

        # 2. Boxplot
        plt.subplot(1, 3, 2)
        sns.boxplot(x=df_cleaned, color='orange')
        plt.title(f'Boxplot de {col}')
        plt.xlabel(col)

        # 3. Diagramme en violon
        plt.subplot(1, 3, 3)
        sns.violinplot(x=df_cleaned, color='lightgreen')
        plt.title(f'Diagramme en violon de {col}')
        plt.xlabel(col)

        plt.tight_layout()
        plt.show()

# Exécuter la fonction de visualisation pour les colonnes numériques
plot_univariate_data(df, numeric_columns)
  