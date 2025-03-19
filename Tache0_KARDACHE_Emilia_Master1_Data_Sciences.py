import pandas as pd
import numpy as np

def detect_variables(df, max_categories=50):

    """
    Cette fonction détecte et catégorise les variables d'un DataFrame :
    - Identifie les variables numériques et catégorielles.
    - Classe les variables catégorielles en ordinales et non ordinales.
    - Optimise la mémoire des variables numériques en effectuant un downcasting.
    - Filtre les variables catégorielles selon le nombre de catégories.
 
    Paramètres :
    - df : DataFrame
        Le DataFrame à analyser.
    - max_categories : int, optionnel (par défaut 50)
        Le nombre maximal de catégories pour qu'une variable catégorielle soit conservée.

    Retourne :
    - Un dictionnaire avec les résultats de l'analyse des variables :
        - "Types de variables" : un DataFrame des types et noms de variables.
        - "Colonnes conservées (catégorielles)" : une liste des colonnes catégorielles conservées.
        - "Colonnes exclues (catégorielles)" : une liste des colonnes catégorielles exclues.
    """

    # 1. Détection des types de variables
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    predefined_ordinal_categories = {
        "nutri_score": ["e", "d", "c", "b", "a"]
    } 
    
    ordinal_columns = []
    non_ordinal_columns = []
 
    for col in categorical_columns:
        unique_values = df[col].dropna().astype(str).str.lower().str.strip().unique()
        
        for predefined_values in predefined_ordinal_categories.values():
            matching_values = [val for val in unique_values if val in predefined_values]
            if len(matching_values) / len(unique_values) > 0.7:
                ordinal_columns.append(col)
                break
        else:
            non_ordinal_columns.append(col)

    # 2. Optimisation de la mémoire avec le downcasting pour les variables numériques
    def downcast_variables(df, numeric_columns):
        for col in numeric_columns:
            if pd.api.types.is_integer_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast="integer")
            elif pd.api.types.is_float_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], downcast="float")
        return df
    
    df = downcast_variables(df, numeric_columns)

    # 3. Filtrage des colonnes catégorielles avec un nombre de catégories supérieur à la limite
    def filter_categorical_columns(df, max_categories):
        kept_columns = []
        excluded_columns = []
        for col in categorical_columns:
            num_unique = df[col].nunique()
            if num_unique <= max_categories:
                kept_columns.append(col)
                excluded_columns.append(col)
        return kept_columns, excluded_columns
    
    kept_columns, excluded_columns = filter_categorical_columns(df, max_categories)
    
    # 4. Création du tableau pour afficher le type des variables
    result_df = pd.DataFrame({
        "Type de variable": ["Catégorielle Ordinale"] * len(ordinal_columns) + 
                            ["Numérique"] * len(numeric_columns) + 
                            ["Catégorielle Non Ordinale"] * len(non_ordinal_columns),
        "Nom de la colonne": ordinal_columns + numeric_columns + non_ordinal_columns
    })
    
    # Résultats
    print("Mémoire avant downcasting :", df.memory_usage(deep=True).sum(), "bytes")
    print("Mémoire après downcasting :", df.memory_usage(deep=True).sum(), "bytes")
    
    return {
        "Types de variables": result_df,
        "Colonnes conservées (catégorielles)": kept_columns,
        "Colonnes exclues (catégorielles)": excluded_columns
    }

# Exemple d'utilisation
path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(path, nrows=10000, sep='\t', encoding="utf-8", low_memory=False, na_filter=True)
result = detect_variables(df)


# Afficher les résultats
print(result["Types de variables"])
print("Colonnes conservées : ", result["Colonnes conservées (catégorielles)"])
print("Colonnes exclues : ", result["Colonnes exclues (catégorielles)"]) 
