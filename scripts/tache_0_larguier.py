import pandas as pd

# Charger le dataset avec 1000 lignes depuis l'URL
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=1000, sep='\t', encoding="utf-8")
print(df.columns)  # Affiche toutes les colonnes du DataFrame

print(df.head())  # Affichage des premières lignes du DataFrame pour vérifier les données

# 1. Filtrer par type d'aliment (par exemple, les produits laitiers)
# Vérifier la colonne 'categories' pour les produits laitiers
df_laitiers = df[df['categories'].str.contains("dairy", case=False, na=False)]
print("\nDataFrame avec uniquement les produits laitiers :")
print(df_laitiers.head())  # Affichage des premières lignes

# 2. Filtrer par Nutri-Score supérieur à "B"
# Vérifier si la colonne 'nutrition_grade_fr' existe et appliquer le filtrage
df_nutriscore_b = df[df['nutriscore_grade'].isin(['A', 'B'])]
print("\nDataFrame avec Nutri-Score supérieur ou égal à 'B' :")
print(df_nutriscore_b.head())  # Affichage des premières lignes

# 3. Filtrer par une valeur nutritionnelle (par exemple, moins de 10g de sucres)
# Vérifier si la colonne 'sugars_100g' existe et appliquer le filtrage
df_sucres = df[df['sugars_100g'] < 10]
print("\nDataFrame avec moins de 10g de sucres pour 100g :")
print(df_sucres.head())  # Affichage des premières lignes

# 4. Filtrer les produits sans gluten
# Vérifier si la colonne 'ingredients_text' contient "gluten" ou non
df_sans_gluten = df[~df['ingredients_text'].str.contains("gluten", case=False, na=False)]
print("\nDataFrame avec uniquement les produits sans gluten :")
print(df_sans_gluten.head())  # Affichage des premières lignes

# 5. Exclure les produits avec des valeurs manquantes dans certaines colonnes
# Par exemple, exclure les produits pour lesquels le Nutri-Score est manquant
df_nettoye = df.dropna(subset=['nutriscore_grade'])
print("\nDataFrame après exclusion des produits avec des valeurs manquantes dans le Nutri-Score :")
print(df_nettoye.head())  # Affichage des premières lignes

# 6. Filtrer par une marque spécifique (par exemple, 'Nestlé')
df_nestle = df[df['brands'].str.contains('Nestlé', case=False, na=False)]
print("\nDataFrame avec uniquement les produits de la marque Nestlé :")
print(df_nestle.head())  # Affichage des premières lignes

# 7. Filtrer les produits selon la présence d'un certain ingrédient (par exemple, 'chocolat')
df_chocolat = df[df['ingredients_text'].str.contains('chocolat', case=False, na=False)]
print("\nDataFrame avec uniquement les produits contenant du chocolat :")
print(df_chocolat.head())  # Affichage des premières lignes
