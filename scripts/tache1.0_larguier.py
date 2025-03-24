import pandas as pd

# Charger le dataset avec 1000 lignes depuis l'URL
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=1000, sep='\t', encoding="utf-8")

# 1. Voir le pourcentage de valeurs manquantes pour chaque colonne
missing_ratio = df.isnull().sum() / len(df)

# Colonnes avec plus de 50% de valeurs manquantes
colonnes_a_supprimer = missing_ratio[missing_ratio > 0.5].index.tolist()
print("Colonnes avec trop de valeurs manquantes :", colonnes_a_supprimer)

# 2. Identifier les colonnes ayant une seule valeur unique (peut-être inutiles)
colonnes_uniques = [col for col in df.columns if df[col].nunique() == 1]
print("Colonnes avec une seule valeur unique :", colonnes_uniques)

# 3. Liste des colonnes généralement non pertinentes dans OpenFoodFacts
colonnes_non_pertinentes = ["url", "code", "creator", "created_t", "last_modified_t", "product_name"]
print("Colonnes potentiellement non pertinentes :", colonnes_non_pertinentes)

# Combiner toutes les colonnes à supprimer
toutes_colonnes_a_supprimer = list(set(colonnes_a_supprimer + colonnes_uniques + colonnes_non_pertinentes))
print("Colonnes à supprimer (non pertinentes) :", toutes_colonnes_a_supprimer)

# Vérifier quelles colonnes existent réellement dans le DataFrame
colonnes_existantes = [col for col in toutes_colonnes_a_supprimer if col in df.columns]
print("Colonnes existantes à supprimer :", colonnes_existantes)

# Créer un nouveau DataFrame nettoyé en supprimant les colonnes inutiles
df_nettoye = df.drop(columns=colonnes_existantes)

# Affichage du DataFrame nettoyé
print("DataFrame nettoyé :")
print(df_nettoye.head())
