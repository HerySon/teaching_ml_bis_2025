import pandas as pd

# Chargement du fichier CSV (assurez-vous que le chemin du fichier est correct)
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=1000, sep='\t', encoding="utf-8")

# 1. Supprimer les variables non pertinentes
columns_to_remove = ['url', 'creator', 'created_t', 'created_datetime', 'last_modified_t', 'last_modified_datetime']
df.drop(columns=columns_to_remove, axis=1, inplace=True)

# 2. Gérer les valeurs manquantes
# a. Calcul du pourcentage de valeurs manquantes pour chaque colonne
missing_percentage = df.isnull().mean() * 100
# b. Suppression des colonnes avec plus de 50% de valeurs manquantes
columns_to_drop = missing_percentage[missing_percentage > 50].index.tolist()
df.drop(columns=columns_to_drop, axis=1, inplace=True)

# c. Imputation des valeurs manquantes
# Imputer les variables numériques avec la médiane
df.fillna(df.median(), inplace=True)

# Imputer les variables catégorielles avec la valeur la plus fréquente
df['categories'].fillna(df['categories'].mode()[0], inplace=True)

# 3. Extraction des motifs dans certaines variables
# Extraction des valeurs numériques de la colonne 'serving_size'
df['serving_size_numeric'] = df['serving_size'].str.extract('(\d+)', expand=False).astype(float)

# 4. Traitement des erreurs dans les variables
# Par exemple, traiter des erreurs dans la colonne 'energy_100g' (en kJ)
df['energy_100g'] = pd.to_numeric(df['energy_100g'], errors='coerce')  # Transforme en NaN les erreurs

# Remplacer les NaN ou valeurs incohérentes (par exemple, énergie égale à 0) par une valeur par défaut
df['energy_100g'].fillna(df['energy_100g'].median(), inplace=True)
df['energy_100g'].replace(0, df['energy_100g'].median(), inplace=True)

# 5. Vérification des types de données
# Conversion de certaines colonnes en types appropriés
df['energy_100g'] = df['energy_100g'].astype(float)
df['categories'] = df['categories'].astype('category')

# Vérification du type des données après transformation
print(df.dtypes)

# Aperçu des données après nettoyage
print(df.head())

# Sauvegarder le DataFrame nettoyé dans un fichier CSV
df.to_csv("cleaned_open_food_facts.csv", index=False)
