import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
import numpy as np
import matplotlib.pyplot as plt

def apply_scaling(df):
    # Sélectionner les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Initialisation des scalers
    standard_scaler = StandardScaler()
    min_max_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()
    max_abs_scaler = MaxAbsScaler()

    # Dictionnaire pour stocker les résultats
    scaled_data = {}

    # Appliquer StandardScaler
    df_standard = df.copy()
    df_standard[numeric_cols] = standard_scaler.fit_transform(df_standard[numeric_cols])
    scaled_data['StandardScaler'] = df_standard

    # Appliquer MinMaxScaler
    df_minmax = df.copy()
    df_minmax[numeric_cols] = min_max_scaler.fit_transform(df_minmax[numeric_cols])
    scaled_data['MinMaxScaler'] = df_minmax

    # Appliquer RobustScaler
    df_robust = df.copy()
    df_robust[numeric_cols] = robust_scaler.fit_transform(df_robust[numeric_cols])
    scaled_data['RobustScaler'] = df_robust

    # Appliquer MaxAbsScaler
    df_maxabs = df.copy()
    df_maxabs[numeric_cols] = max_abs_scaler.fit_transform(df_maxabs[numeric_cols])
    scaled_data['MaxAbsScaler'] = df_maxabs

    # Affichage des résultats avec des graphiques
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    for i, (scaler_name, scaled_df) in enumerate(scaled_data.items()):
        ax[i//2, i%2].boxplot([df[numeric_cols].values.flatten(), scaled_df[numeric_cols].values.flatten()],
                              labels=['Original', scaler_name])
        ax[i//2, i%2].set_title(f"Comparaison avec {scaler_name}")
    
    plt.tight_layout()
    plt.show()

    return scaled_data

# Exemple d'utilisation avec un DataFrame
url = "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
df = pd.read_csv(url, nrows=1000, sep='\t', encoding="utf-8")

# Appliquer le scaling sur les données
scaled_data = apply_scaling(df)
