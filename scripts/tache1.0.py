import pandas as pd
import numpy as np

def clean_dataset(df):
    # 1. Suppression des variables non pertinentes
    non_pertinent_cols = ['url', 'creator', 'created_t', 'created_datetime', 'last_modified_t', 'last_modified_datetime', 'image_url', 'image_small_url']
    df = df.drop(columns=non_pertinent_cols, errors='ignore')
    
    # 2. Gestion des valeurs manquantes
    # Suppression des colonnes ayant plus de 50% de valeurs manquantes
    threshold = 0.5
    df = df.loc[:, df.isnull().mean() < threshold]
    
    # Imputation des valeurs manquantes pour certaines colonnes numériques avec la moyenne
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    
    # Imputation des valeurs manquantes pour certaines colonnes catégorielles avec la mode
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    # 3. Extraction des motifs particuliers
    # Extraction de la quantité de la variable 'serving_size'
    # Supposons que serving_size contienne des données du type "100 g", "200 ml", etc.
    df['serving_size_value'] = df['serving_size'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['serving_size_unit'] = df['serving_size'].str.extract(r'([a-zA-Z]+)').fillna('')
    
    # Extraction de la quantité de la variable 'quantity' (par exemple, "500 g" ou "1 L")
    df['quantity_value'] = df['quantity'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['quantity_unit'] = df['quantity'].str.extract(r'([a-zA-Z]+)').fillna('')
    
    # 4. Traitement des erreurs dans les données
    # Suppression des lignes où les quantités ou serving_size sont nulles ou négatives
    df = df[df['serving_size_value'].notna() & (df['serving_size_value'] > 0)]
    df = df[df['quantity_value'].notna() & (df['quantity_value'] > 0)]
    
    # 5. Traitement de certaines variables spécifiques
    # Par exemple, ajuster les valeurs de nutriments qui peuvent être aberrantes (si nécessaire)
    # Pour cette tâche, on peut supposer que les valeurs de nutriments comme "energy_100g" ne peuvent pas être négatives
    df = df[df['energy_100g'] >= 0]
    
    return df

# Exemple d'utilisation avec un DataFrame
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=1000, sep='\t', encoding="utf-8")

# Application de la fonction de nettoyage
cleaned_df = clean_dataset(df)

# Affichage du DataFrame nettoyé
print(cleaned_df.head())
