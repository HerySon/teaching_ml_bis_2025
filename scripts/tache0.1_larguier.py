import pandas as pd

# Charger un sous-ensemble du dataset (100 000 lignes pour éviter les temps de chargement excessifs)
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=100000, sep='\t', encoding="utf-8")

# Méthode 1 : Échantillon aléatoire de 10 000 lignes
df_sample_random = df.sample(n=10000, random_state=42)

# Méthode 2 : Échantillonnage stratifié sur le Nutri-Score
if "nutriscore_grade" in df.columns:
    df_sample_strat = df.groupby("nutriscore_grade", group_keys=False).apply(lambda x: x.sample(frac=0.1, random_state=42))
else:
    df_sample_strat = df_sample_random  # Fallback si Nutri-Score non dispo

# Méthode 3 : Filtrage des produits avec des valeurs nutritionnelles renseignées
cols_nutrition = ["energy-kcal_100g", "fat_100g", "sugars_100g", "salt_100g"]
df_filtered = df.dropna(subset=cols_nutrition)
df_sample_filtered = df_filtered.sample(n=min(10000, len(df_filtered)), random_state=42)

# Sauvegarde des fichiers
df_sample_random.to_csv("sample_random.csv", index=False)
df_sample_strat.to_csv("sample_stratified.csv", index=False)
df_sample_filtered.to_csv("sample_filtered.csv", index=False)

print("Sous-échantillonnage terminé !")
print(f"Random: {df_sample_random.shape}, Stratifié: {df_sample_strat.shape}, Filtré: {df_sample_filtered.shape}")
