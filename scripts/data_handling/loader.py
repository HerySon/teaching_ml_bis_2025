import pandas as pd
import os

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

def save_data(df, path, index=False):
    """
    Sauvegarde un DataFrame dans un fichier CSV.
    
    Args:
        df (pd.DataFrame): DataFrame à sauvegarder
        path (str): Chemin où sauvegarder le fichier
        index (bool): Si True, sauvegarde l'index
    """
    df.to_csv(path, sep='\t', index=index, encoding='utf-8')