import pandas as pd

# Charger le dataset
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=1000, sep='\t', encoding="utf-8")

# Sélectionner les colonnes catégorielles
categorical_cols = df.select_dtypes(include=['object']).columns

# Appliquer l'encodage (One-Hot Encoding sur les colonnes catégorielles)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Sauvegarder le dataset encodé
df_encoded.to_csv("encoded_openfood.csv", index=False)

print("Encodage terminé. Fichier enregistré sous 'encoded_openfood.csv'.")
