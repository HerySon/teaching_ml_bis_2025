from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scaling_variables(data, method="standard", columns_to_scale=None, ordinal_feature=['nutriscore_grade'], threshold=999):
    """
    Applique des méthodes de scaling aux variables numériques d'un DataFrame, tout en réutilisant les fonctions précédentes.
    
    Paramètres :
    - data : DataFrame Pandas
    - method : (str) Méthode de scaling à utiliser ('standard', 'minmax', 'robust')
    - columns_to_scale : (list) Liste des colonnes à scaler, si None toutes les colonnes numériques sont scalées.
    - ordinal_feature : Liste des variables ordinales (utilisée dans la fonction Classement_colonnes)
    - threshold : Limite du nombre de catégories pour les variables catégorielles
    
    Retourne :
    - DataFrame avec les variables scalées
    """
    
    # Classification des colonnes avec la fonction 'Classement_colonnes' pour récupérer les colonnes numériques
    classification_result, data = Classement_colonnes(data, ordinal_feature=ordinal_feature, threshold=threshold)
    
    # Si aucune colonne n'est spécifiée, prendre toutes les colonnes numériques
    if columns_to_scale is None:
        columns_to_scale = classification_result["Numeriques"]
    
    # Initialiser le scaler en fonction de la méthode choisie
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Méthode de scaling inconnue : {method}. Choisissez parmi 'standard', 'minmax', ou 'robust'.")
    
    # Appliquer le scaler uniquement sur les colonnes numériques spécifiées
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    return data
