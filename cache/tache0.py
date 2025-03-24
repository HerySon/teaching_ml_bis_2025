import pandas as pd

def detect_and_filter_columns(df, max_categories=10):
    # Identifications des types de colonnes
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Initialisation des listes pour les types de variables
    ordinal_cols = []
    non_ordinal_cols = []

    # Détection des variables ordinales/non-ordinales
    for col in categorical_cols:
        unique_vals = df[col].nunique()
        if unique_vals <= max_categories:
            # Si le nombre de catégories est inférieur au seuil, on les considère comme ordonnables
            ordinal_cols.append(col)
        else:
            non_ordinal_cols.append(col)

    # Downcasting des colonnes numériques
    for col in numeric_cols:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')

    # Affichage des résultats
    print(f"Colonnes numériques : {numeric_cols}")
    print(f"Colonnes ordinales : {ordinal_cols}")
    print(f"Colonnes non ordinales : {non_ordinal_cols}")
    
    # Afficher le DataFrame après downcasting
    print("\nDtypes du DataFrame après downcasting :")
    print(df.dtypes)

    return {
        'numeric_cols': numeric_cols,
        'ordinal_cols': ordinal_cols,
        'non_ordinal_cols': non_ordinal_cols,
        'downcasted_df': df
    }

# Exemple d'utilisation avec un DataFrame
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=50, sep='\t', encoding="utf-8")

result = detect_and_filter_columns(df)

# Affichage du DataFrame après downcasting (tout affiché)
print("\nAperçu du DataFrame après downcasting :")
print(result['downcasted_df'].head())  # Affiche les premières lignes du DataFrame modifié
