import pandas as pd
import numpy as np
from typing import Tuple, Dict, List

def detect_and_filter_columns(df: pd.DataFrame, ordinal_cols: List[str] = None, cat_threshold: int = 10) -> Tuple[Dict[str, List[str]], pd.DataFrame]:
    """
    Analyse et optimise un DataFrame en classifiant les colonnes par type et en réduisant la mémoire utilisée.
    
    Paramètres :
        df (pd.DataFrame) : Le DataFrame d'entrée.
        ordinal_cols (List[str], optional) : Liste des colonnes catégorielles ordinales.
        cat_threshold (int, optional) : Nombre maximal de catégories pour considérer une colonne comme catégorielle.
    
    Retourne :
        Tuple[Dict[str, List[str]], pd.DataFrame] :
            - Un dictionnaire contenant les colonnes classées par type.
            - Un DataFrame optimisé en mémoire.
    """
    
    # Initialisation des catégories de colonnes
    column_types = {
        "numerical": [],
        "ordinal_categorical": [],
        "non_ordinal_categorical": []
    }
    
    df_optimized = df.copy()
    
    for col in df.columns:
        col_dtype = df[col].dtype
        
        if np.issubdtype(col_dtype, np.number):
            column_types["numerical"].append(col)
            
            # Downcasting pour optimiser la mémoire
            if col_dtype == np.int64:
                df_optimized[col] = pd.to_numeric(df[col], downcast='integer')
            elif col_dtype == np.float64:
                df_optimized[col] = pd.to_numeric(df[col], downcast='float')
        
        elif col_dtype == 'object' or col_dtype.name == 'category':
            unique_vals = df[col].nunique()
            
            if ordinal_cols and col in ordinal_cols:
                column_types["ordinal_categorical"].append(col)
            else:
                if unique_vals <= cat_threshold:
                    df_optimized[col] = df[col].astype("category")
                    column_types["non_ordinal_categorical"].append(col)
    
    return column_types, df_optimized


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    """Sauvegarde un DataFrame sous format CSV."""
    df.to_csv(path, index=False)
    print(f"✅ DataFrame enregistré dans : {path}")


def main():
    """Fonction principale pour exécuter l'analyse et l'optimisation des colonnes d'un DataFrame."""
    input_path = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
    output_path = r"C:\Users\nakka\Machine_learning\teaching_ml_bis_2025\data\Processed_Data.csv"
    
    try:
        df = pd.read_csv(input_path, nrows=100, sep='\t',encoding="utf-8")
        types_detectes, df_optimized = detect_and_filter_columns(df, ordinal_cols=None, cat_threshold=15)
        save_dataframe(df_optimized, output_path)
        
        # Affichage des résultats
        print("\n Résumé des colonnes détectées :")
        for category, columns in types_detectes.items():
            print(f"  ➜ {category}: {columns}")
        
        print("\n Aperçu du DataFrame optimisé :")
        print(df_optimized.head())
    
    except FileNotFoundError:
        print(f" Fichier introuvable : {input_path}")
    except Exception as e:
        print(f" Une erreur s'est produite : {e}")


if __name__ == "__main__":
    main()