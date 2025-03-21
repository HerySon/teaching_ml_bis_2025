import pandas as pd
import numpy as np

def filter_data(data):
    """
    Process and filter the Open Food Facts dataset.
    
    Parameters:
    -----------
    data : df
        The input Open Food Facts dataset
        
    Returns:
    --------
    df
        Processed df with:
        - Numeric columns
        - Categorical columns (ordinal/non-ordinal)
        - Downcasted datatypes for memory efficiency
        - Filtered categorical columns based on cardinality
    """
    # Copy pour pas modif le df original
    df = data.copy()
    
    # On identifie le type des colonnes
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # Colonnes numériques
    # ON downcast les valeurs nums pour save de la memoire
    for col in numeric_cols:
        if df[col].dtype == 'float64':
            df[col] = pd.to_numeric(df[col], downcast='float')
        elif df[col].dtype == 'int64':
            df[col] = pd.to_numeric(df[col], downcast='integer')
    
    # Colonnes catégoriques
    for col in categorical_cols:
        # On calcule le nb de val unique
        n_unique = df[col].nunique()
        n_total = len(df)
        
        # On filtre les colonnes catégoriques avec trop de uniques
        if n_unique / n_total > 0.5:
            df = df.drop(columns=[col])
            continue
            
        # Reste des colonnes catégoriques : on identifie les colonnes ordinales potentielles (genre nutriscore_grade)
        if col == 'nutriscore_grade':
            df[col] = pd.Categorical(df[col], categories=['a', 'b', 'c', 'd', 'e'], ordered=True)
        else:
            # Sinon on convertit en catégorique pour save de la memoire
            df[col] = pd.Categorical(df[col])
    
    # Val manquantes
    # Pour les numeriques on met la mediane
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Pour les categoriques on met la + frequente
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df



