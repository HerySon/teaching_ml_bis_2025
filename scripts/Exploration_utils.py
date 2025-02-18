import pandas as pd
import os

print("Loading scripts module...")

def load_data(path):
    """
    Charge les données depuis un fichier CSV. 
    Si le fichier n'existe pas, le script renvoie un message d'erreur.
    Args:
        path (str): Chemin vers le fichier CSV.

    Returns:
        pd.DataFrame: DataFrame contenant les données.
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier {path} n'existe pas.")
    
    df = pd.read_csv(path, 
                 sep='\t',  # tabulation comme separateur
                 encoding='utf-8',
                 low_memory=False,
                 on_bad_lines='skip'  # Ignore les lignes problématiques
                )
    return df

def check_missing_values(df, threshold=50):
    """
    Analyse et affiche le pourcentage de valeurs manquantes par colonne.
    
    Arguments :
    - df : DataFrame pandas
    - threshold : float, pourcentage minimum pour afficher une alerte (ex : 50% par défaut)

    Retourne :
    - df_missing : DataFrame trié des valeurs manquantes
    """
    missing_values = df.isnull().mean() * 100
    df_missing = missing_values[missing_values > 0].sort_values(ascending=False)

    if df_missing.empty:
        print("✅ Aucune valeur manquante dans le dataset.")
    else:
        print(f"⚠️ {len(df_missing)} colonnes sur {len(df.columns)} ont des valeurs manquantes.")
        print(df_missing[df_missing > threshold])

    return df_missing

__all__ = ['load_data', 'check_missing_values']
