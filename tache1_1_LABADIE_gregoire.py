import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
import scipy.sparse as sp
import pandas as pd

def encodage_categoriel(data, threshold=0.01, keep_columns=None, use_hashing=False, use_count_encoder=False, save_encoded=False):
    resultat, data = Classement_colonnes(data)
     
    colonnes_categorielles = resultat["Categorielles ordinales"] + resultat["Categorielles non ordinales"]

    # Suppression des catégories rares
    for col in colonnes_categorielles:
        frequence = data[col].value_counts(normalize=True)
        rare_categories = frequence[frequence < threshold].index
        data[col] = data[col].replace(rare_categories, 'Autre')

    # Fonction d'encodage
    def encode_column_and_save(col, encoder, save=False):
        encoded = encoder.fit_transform(data[[col]])
        encoded_df = pd.DataFrame.sparse.from_spmatrix(encoded, columns=encoder.get_feature_names_out([col]))
        
        # Sauvegarde si requis
        if save:
            sp.save_npz(f"{col}_encoded.npz", encoded)
        
        return encoded_df

    # Encodage et sauvegarde incrémentale avec HashingEncoder, CountEncoder ou OneHotEncoder
    for col in colonnes_categorielles:
        if use_hashing:
            encoder = ce.HashingEncoder()
        elif use_count_encoder:
            encoder = ce.CountEncoder()
        else:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)
        
        encoded_df = encode_column_and_save(col, encoder, save_encoded)
        data = pd.concat([data, encoded_df], axis=1).drop(columns=[col])

    return data