import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

# Charger le dataset de base
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=100000, sep='\t', encoding="utf-8", low_memory=False)

# Sélectionner uniquement les colonnes numériques
numerical_cols = df.select_dtypes(include=['number']).columns

# Remplacer les NaN par la médiane de chaque colonne
df[numerical_cols] = df[numerical_cols].apply(lambda x: x.fillna(x.median()))

# Remplacer les valeurs infinies par la médiane
df[numerical_cols] = df[numerical_cols].replace([np.inf, -np.inf], np.nan)
df[numerical_cols] = df[numerical_cols].apply(lambda x: x.fillna(x.median()))

# Appliquer le scaling (StandardScaler)
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Sauvegarder le dataset normalisé
df_scaled.to_csv("scaled_openfood.csv", index=False)

print("Scaling terminé et fichier enregistré sous 'scaled_openfood.csv'.")
