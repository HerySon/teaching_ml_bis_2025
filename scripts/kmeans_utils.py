"""
Module d'utilitaires pour l'analyse et l'optimisation de K-means.
Fournit des fonctions pour l'entraînement, l'optimisation et la prédiction avec K-means.
"""
try:
    import os

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        calinski_harabasz_score,
        davies_bouldin_score,
        silhouette_score,
    )
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    import joblib
except ImportError as e:
    print(f"Erreur lors de l'importation des modules : {e}")


def find_optimal_clusters(data, k_max=10, k_min=2, method='silhouette'):
    """
    Détermine le nombre optimal de clusters en utilisant différentes métriques.
    
    Paramètres:
    -----------
    data : array-like
        Les données à analyser
    k_max : int
        Nombre maximum de clusters à tester
    k_min : int
        Nombre minimum de clusters à tester
    method : str
        Méthode à utiliser parmi :
        - 'silhouette' : Score de silhouette (plus élevé = meilleur)
        - 'calinski_harabasz' : Score de Calinski-Harabasz (plus élevé = meilleur)
        - 'davies_bouldin' : Score de Davies-Bouldin (plus bas = meilleur)
        - 'inertia' : Inertie du clustering (plus bas = meilleur)
    
    Retourne:
    --------
    dict : Dictionnaire contenant les scores pour chaque nombre de clusters
    """
    scores = {}

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)

        if method == 'silhouette':
            score = silhouette_score(data, labels)
        elif method == 'calinski_harabasz':
            score = calinski_harabasz_score(data, labels)
        elif method == 'davies_bouldin':
            score = -davies_bouldin_score(data, labels)  # On inverse car plus bas = meilleur
        elif method == 'inertia':
            score = -kmeans.inertia_  # On inverse car plus bas = meilleur
        else:
            raise ValueError(
                "Méthode non supportée. Utilisez 'silhouette', 'calinski_harabasz', "
                "'davies_bouldin' ou 'inertia'"
            )

        scores[k] = score

    return scores


def plot_cluster_scores(scores, method):
    """
    Visualise les scores pour différents nombres de clusters.
    
    Paramètres:
    -----------
    scores : dict
        Dictionnaire des scores pour chaque nombre de clusters
    method : str
        Nom de la méthode utilisée pour le titre du graphique
    """
    plt.figure(figsize=(10, 6))
    plt.plot(list(scores.keys()), list(scores.values()), marker='o')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Score')
    plt.title(f'Analyse du nombre optimal de clusters - {method}')
    plt.grid(True)
    plt.show()


def optimize_kmeans(data, n_clusters, method='multiple_init', **kwargs):
    """
    Optimise les paramètres de K-means en utilisant différentes stratégies.
    
    Paramètres:
    -----------
    data : array-like
        Les données à analyser
    n_clusters : int
        Nombre de clusters souhaité
    method : str
        Méthode d'optimisation à utiliser :
        - 'multiple_init' : Teste plusieurs initialisations (défaut)
        - 'grid_search' : Utilise GridSearchCV pour tester plusieurs combinaisons
        - 'elkan' : Compare les algorithmes 'elkan' et 'full'
        - 'custom_init' : Utilise des centres initiaux personnalisés
    
    **kwargs : arguments additionnels selon la méthode
        Pour 'multiple_init' :
            - n_init : int (défaut=10)
            - max_iter : int (défaut=300)
        Pour 'grid_search' :
            - param_grid : dict (défaut=None)
            - cv : int (défaut=3)
        Pour 'custom_init' :
            - init_centers : array-like
    
    Retourne:
    --------
    KMeans : Le meilleur modèle K-means trouvé
    """
    # Configuration par défaut pour chaque méthode
    default_configs = {
        'multiple_init': {
            'n_init': 10,
            'max_iter': 300,
            'init': 'k-means++'
        },
        'grid_search': {
            'param_grid': {
                'init': ['k-means++', 'random'],
                'n_init': [10, 20],
                'max_iter': [200, 300, 400],
                'algorithm': ['elkan', 'full']
            },
            'cv': 3
        },
        'elkan': {
            'algorithms': ['elkan', 'full']
        }
    }

    # Mise à jour des configurations avec les kwargs
    config = default_configs.get(method, {})
    config.update(kwargs)

    if method == 'multiple_init':
        best_model = None
        best_score = -np.inf

        for _ in range(config['n_init']):
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=config['init'],
                max_iter=config['max_iter'],
                random_state=None
            )
            kmeans.fit(data)

            score = silhouette_score(data, kmeans.labels_)

            if score > best_score:
                best_score = score
                best_model = kmeans

        return best_model

    if method == 'grid_search':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        grid_search = GridSearchCV(
            kmeans,
            param_grid=config['param_grid'],
            cv=config['cv'],
            scoring='silhouette'
        )
        grid_search.fit(data)

        return grid_search.best_estimator_

    if method == 'elkan':
        best_model = None
        best_score = -np.inf

        for algo in config['algorithms']:
            kmeans = KMeans(
                n_clusters=n_clusters,
                algorithm=algo,
                random_state=42
            )
            kmeans.fit(data)

            score = silhouette_score(data, kmeans.labels_)

            if score > best_score:
                best_score = score
                best_model = kmeans

        return best_model

    if method == 'custom_init':
        init_centers = kwargs.get('init_centers')
        if init_centers is None:
            raise ValueError("init_centers doit être fourni pour la méthode 'custom_init'")

        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init_centers,
            n_init=1,
            random_state=42
        )
        kmeans.fit(data)

        return kmeans

    raise ValueError(
        "Méthode non supportée. Utilisez 'multiple_init', 'grid_search', "
        "'elkan' ou 'custom_init'"
    )


def train_kmeans(data, n_clusters=None, optimize=True, save=True, **kwargs):
    """
    Fonction principale pour entraîner un modèle K-means.
    
    Paramètres:
    -----------
    data : array-like
        Les données à analyser
    n_clusters : int, optional
        Nombre de clusters souhaité. Si None, sera déterminé automatiquement
    optimize : bool
        Si True, détermine le nombre optimal de clusters
    save : bool
        Si True, sauvegarde le modèle final et le scaler
    **kwargs : arguments additionnels
        - method : str (défaut='silhouette')
            Méthode pour déterminer le nombre optimal de clusters
        - k_max : int (défaut=10)
            Nombre maximum de clusters à tester
        - k_min : int (défaut=2)
            Nombre minimum de clusters à tester
        - optimize_method : str (défaut='multiple_init')
            Méthode d'optimisation des paramètres
        - model_path : str (défaut='models/kmeans_model.pkl')
            Chemin pour sauvegarder le modèle
    
    Retourne:
    --------
    tuple : (KMeans, dict)
        - Le modèle K-means entraîné
        - Dictionnaire contenant les informations sur l'entraînement
    """
    # Configuration par défaut
    default_config = {
        'method': 'silhouette',
        'k_max': 10,
        'k_min': 2,
        'optimize_method': 'multiple_init',
        'model_path': 'models/kmeans_model.pkl'
    }

    # Mise à jour de la configuration avec les kwargs
    config = default_config.copy()
    config.update(kwargs)

    # Création du dossier models si nécessaire
    if save:
        os.makedirs('models', exist_ok=True)

    # Standardisation des données
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Détermination du nombre optimal de clusters si nécessaire
    if optimize and n_clusters is None:
        scores = find_optimal_clusters(
            scaled_data,
            k_max=config['k_max'],
            k_min=config['k_min'],
            method=config['method']
        )
        plot_cluster_scores(scores, config['method'])

        # Sélection du nombre optimal de clusters
        n_clusters = max(scores.items(), key=lambda x: x[1])[0]
        print(f"Nombre optimal de clusters trouvé : {n_clusters}")

    # Optimisation des paramètres
    model = optimize_kmeans(
        scaled_data,
        n_clusters=n_clusters,
        method=config['optimize_method'],
        **kwargs
    )

    # Sauvegarde du modèle et du scaler si demandé
    if save:
        scaler_path = os.path.join(os.path.dirname(config['model_path']), 'scaler.pkl')

        save_model(model, config['model_path'])
        joblib.dump(scaler, scaler_path)
        print(f"Modèle sauvegardé dans {config['model_path']}")
        print(f"Scaler sauvegardé dans {scaler_path}")

    # Création du dictionnaire d'informations
    info = {
        'n_clusters': n_clusters,
        'optimize_method': config['optimize_method'],
        'scaler': scaler,
        'silhouette_score': silhouette_score(scaled_data, model.labels_),
        'calinski_harabasz_score': calinski_harabasz_score(scaled_data, model.labels_),
        'davies_bouldin_score': davies_bouldin_score(scaled_data, model.labels_),
        'inertia': model.inertia_
    }

    return model, info


def predict_clusters(model, data, scaler=None):
    """
    Prédit les clusters pour de nouvelles données.
    
    Paramètres:
    -----------
    model : KMeans
        Le modèle K-means chargé
    data : array-like
        Les données à prédire
    scaler : StandardScaler, optional
        Le scaler utilisé pour l'entraînement. Si None, les données ne seront pas standardisées
    
    Retourne:
    --------
    array-like
        Les labels des clusters prédits
    """
    if scaler is not None:
        data = scaler.transform(data)
    return model.predict(data)


def save_model(model, path="models/kmeans_model.pkl"):
    """
    Sauvegarde le modèle K-means entraîné.
    
    Paramètres:
    -----------
    model : KMeans
        Le modèle à sauvegarder
    path : str
        Chemin où sauvegarder le modèle
    """
    joblib.dump(model, path)


def load_kmeans(model_path, scaler_path=None):
    """
    Charge un modèle K-means pré-entraîné et son scaler associé.
    
    Paramètres:
    -----------
    model_path : str
        Chemin vers le fichier du modèle K-means sauvegardé
    scaler_path : str, optional
        Chemin vers le fichier du scaler sauvegardé. Si None, cherchera dans le même dossier
    
    Retourne:
    --------
    tuple : (KMeans, StandardScaler)
        - Le modèle K-means chargé
        - Le scaler associé (si trouvé)
    
    Raises:
    -------
    FileNotFoundError
        Si le fichier du modèle n'existe pas
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier du modèle n'existe pas : {model_path}")

    # Chargement du modèle
    model = joblib.load(model_path)

    # Chargement du scaler
    scaler = None
    if scaler_path is None:
        # Cherche le scaler dans le même dossier que le modèle
        scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.pkl')

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    return model, scaler
