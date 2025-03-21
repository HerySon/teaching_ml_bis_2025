import pandas as pd
import numpy as np

def detect_variables(df, max_categories=50, sample_size=1000):
    """
    Cette fonction détecte et catégorise les variables d'un DataFrame :
    - Identifie les variables numériques et catégorielles.
    - Classe les variables catégorielles en ordinales et non ordinales.
    - Optimise la mémoire des variables numériques en effectuant un downcasting.
    - Filtre les variables catégorielles selon le nombre de catégories.
    - Sélectionne aléatoirement un sous-ensemble des lignes.

    Paramètres :   
    - df : DataFrame
        Le DataFrame à analyser.
    - max_categories : int, optionnel (par défaut 50)
        Le nombre maximal de catégories pour qu'une variable catégorielle soit conservée.
    - sample_size : int, optionnel (par défaut 1000)
        Le nombre de lignes à sélectionner aléatoirement.

    Retourne :
    - Un dictionnaire avec les résultats de l'analyse des variables :
        - "Types de variables" : un DataFrame des types et noms de variables.
        - "Colonnes conservées (catégorielles)" : une liste des colonnes catégorielles conservées.
        - "Colonnes exclues (catégorielles)" : une liste des colonnes catégorielles exclues.
    """

    # 1. Sélectionner un échantillon aléatoire
    df_sample = df.sample(n=sample_size, random_state=42)  # `random_state` pour reproductibilité
    print("Échantillon aléatoire sélectionné :")
    print(df_sample.head())  # Affiche les premières lignes de l'échantillon sélectionné

    # 2. Détection des types de variables sur df_sample
    numeric_columns = df_sample.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df_sample.select_dtypes(include=['object', 'category']).columns.tolist()
    
    predefined_ordinal_categories = {
        "nutri_score": ["e", "d", "c", "b", "a"]
    } 
    
    ordinal_columns = []
    non_ordinal_columns = []
 
    for col in categorical_columns:
        unique_values = df_sample[col].dropna().astype(str).str.lower().str.strip().unique()
        
        for predefined_values in predefined_ordinal_categories.values():
            matching_values = [val for val in unique_values if val in predefined_values]
            if len(matching_values) / len(unique_values) > 0.7:
                ordinal_columns.append(col)
                break
        else:
            non_ordinal_columns.append(col)

    # 3. Optimisation de la mémoire avec le downcasting pour les variables numériques
    def downcast_variables(df_sample, numeric_columns):
        for col in numeric_columns:
            if pd.api.types.is_integer_dtype(df_sample[col]):
                df_sample[col] = pd.to_numeric(df_sample[col], downcast="integer")
            elif pd.api.types.is_float_dtype(df_sample[col]):
                df_sample[col] = pd.to_numeric(df_sample[col], downcast="float")
        return df_sample
    
    df_sample = downcast_variables(df_sample, numeric_columns)

    # 4. Filtrage des colonnes catégorielles avec un nombre de catégories supérieur à la limite
    def filter_categorical_columns(df_sample, categorical_columns, max_categories):
        kept_columns = []
        excluded_columns = []
        for col in categorical_columns:
            num_unique = df_sample[col].nunique()
            if num_unique <= max_categories:
                kept_columns.append(col)
            else:
                excluded_columns.append(col)
        return kept_columns, excluded_columns
    
    kept_columns, excluded_columns = filter_categorical_columns(df_sample, categorical_columns, max_categories)
    
    # 5. Création du tableau pour afficher le type des variables
    result_df = pd.DataFrame({
        "Type de variable": ["Catégorielle Ordinale"] * len(ordinal_columns) + 
                            ["Numérique"] * len(numeric_columns) + 
                            ["Catégorielle Non Ordinale"] * len(non_ordinal_columns),
        "Nom de la colonne": ordinal_columns + numeric_columns + non_ordinal_columns
    })
    
    # Résultats
    print("Mémoire avant downcasting :", df_sample.memory_usage(deep=True).sum(), "bytes")
    print("Mémoire après downcasting :", df_sample.memory_usage(deep=True).sum(), "bytes")
    
    return {
        "Types de variables": result_df,
        "Colonnes conservées (catégorielles)": kept_columns,
        "Colonnes exclues (catégorielles)": excluded_columns
    }

# Exemple d'utilisation
path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(path, nrows=10000, sep='\t', encoding="utf-8", low_memory=False, na_filter=True)

# Appliquer la fonction sur un échantillon de 1000 lignes
result = detect_variables(df, sample_size=1000)

# Afficher les résultats
print(result["Types de variables"])
print("Colonnes conservées : ", result["Colonnes conservées (catégorielles)"])
print("Colonnes exclues : ", result["Colonnes exclues (catégorielles)"])

