from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def plot_pca_results(pca_info: Dict, figsize: tuple = (12, 5)):
    """
    Visualise les résultats de la PCA.
    
    Args:
        pca_info: Dictionnaire contenant les informations de la PCA
        figsize: Taille de la figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot de la variance expliquée cumulée
    n_components = len(pca_info['explained_variance_ratio'])
    components = range(1, n_components + 1)
    
    ax1.plot(components, pca_info['cumulative_variance_ratio'], 'b-', marker='o')
    ax1.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax1.set_xlabel('Nombre de composantes')
    ax1.set_ylabel('Variance expliquée cumulée')
    ax1.set_title('Variance expliquée cumulée par composante')
    ax1.grid(True)
    ax1.legend()
    
    # Plot de l'importance des features
    feature_importance = pd.Series(pca_info['feature_importance']).sort_values(ascending=True)
    feature_importance.plot(kind='barh', ax=ax2)
    ax2.set_xlabel('Importance moyenne')
    ax2.set_title('Importance des features dans la PCA')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_tsne_results(
    df_tsne: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Visualise les résultats de t-SNE avec gestion des labels.
    
    Args:
        df_tsne: DataFrame contenant les composantes t-SNE
        labels: Series contenant les labels pour la coloration (optionnel)
        figsize: Taille de la figure (largeur, hauteur)
    """
    fig = plt.figure(figsize=figsize)
    
    # Préparation des labels
    if labels is not None:
        # S'assurer que les labels correspondent aux indices de df_tsne
        labels = labels.loc[df_tsne.index]
        
        # Conversion des labels en valeurs numériques si nécessaire
        if labels.dtype == 'object' or labels.dtype.name == 'category':
            label_encoder = LabelEncoder()
            label_colors = label_encoder.fit_transform(labels)
            unique_labels = label_encoder.classes_
        else:
            label_colors = labels.values
            unique_labels = np.unique(labels)
        
        # Création d'une colormap personnalisée
        n_classes = len(unique_labels)
        cmap = plt.cm.get_cmap('tab20' if n_classes > 10 else 'tab10')
    else:
        label_colors = None
        cmap = None
    
    # Visualisation 2D ou 3D
    if 'TSNE3' in df_tsne.columns:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            df_tsne['TSNE1'],
            df_tsne['TSNE2'],
            df_tsne['TSNE3'],
            c=label_colors,
            cmap=cmap,
            alpha=0.6
        )
        ax.set_xlabel('TSNE1')
        ax.set_ylabel('TSNE2')
        ax.set_zlabel('TSNE3')
    else:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(
            df_tsne['TSNE1'],
            df_tsne['TSNE2'],
            c=label_colors,
            cmap=cmap,
            alpha=0.6
        )
        ax.set_xlabel('TSNE1')
        ax.set_ylabel('TSNE2')
    
    # Ajout de la légende si des labels sont fournis
    if labels is not None:
        legend_elements = [
            plt.scatter([], [], c=[cmap(i / (n_classes - 1))], label=label)
            for i, label in enumerate(unique_labels)
        ]
        ax.legend(handles=legend_elements, title='Classes')
    
    plt.title('Visualisation t-SNE')
    plt.tight_layout()
    plt.show()

def plot_outliers_detection(
    df: pd.DataFrame,
    outliers_info: Dict,
    columns: Optional[List[str]] = None,
    figsize: tuple = (15, 5)
):
    """
    Visualise la détection des outliers pour les variables sélectionnées.
    
    Args:
        df: DataFrame original
        outliers_info: Dictionnaire contenant les informations sur les outliers
        columns: Liste des colonnes à visualiser (si None, utilise les 3 premières)
        figsize: Taille de la figure
    """
    if columns is None:
        columns = list(outliers_info['n_outliers'].keys())[:3]
    
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=figsize)
    if n_cols == 1:
        axes = [axes]
    
    for ax, col in zip(axes, columns):
        bounds = outliers_info['bounds'][col]
        
        # Création du boxplot
        ax.boxplot(df[col].dropna())
        
        # Ajout des limites
        ax.axhline(y=bounds['lower'], color='r', linestyle='--', label='Limites')
        ax.axhline(y=bounds['upper'], color='r', linestyle='--')
        
        # Annotations
        n_outliers = outliers_info['n_outliers'][col]
        ratio = outliers_info['outliers_ratio'][col]
        ax.set_title(f'{col}\n{n_outliers} outliers ({ratio:.1%})')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_encoding_distribution(
    df_original: pd.DataFrame,
    df_encoded: pd.DataFrame,
    encoders: Dict,
    figsize: tuple = (15, 5)
):
    """
    Visualise la distribution des variables avant et après encodage.
    
    Args:
        df_original: DataFrame original
        df_encoded: DataFrame encodé
        encoders: Dictionnaire des encodeurs utilisés
        figsize: Taille de la figure
    """
    # Visualisation des variables ordinales (LabelEncoder)
    if encoders['label_encoders']:
        n_ordinal = len(encoders['label_encoders'])
        fig, axes = plt.subplots(2, n_ordinal, figsize=figsize, squeeze=False)
        
        for i, (col, encoder) in enumerate(encoders['label_encoders'].items()):
            # Distribution originale
            df_original[col].value_counts().plot(
                kind='bar',
                ax=axes[0, i],
                title=f'{col} (Original)'
            )
            axes[0, i].tick_params(axis='x', rotation=45)
            
            # Distribution encodée
            df_encoded[col].value_counts().plot(
                kind='bar',
                ax=axes[1, i],
                title=f'{col} (Encodé)'
            )
            axes[1, i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    # Visualisation des variables nominales (OneHotEncoder)
    if encoders['onehot_features']:
        plt.figure(figsize=figsize)
        
        # Sélection des colonnes one-hot
        onehot_cols = encoders['onehot_features']
        df_encoded[onehot_cols].sum().plot(kind='bar')
        
        plt.title('Distribution des variables one-hot encodées')
        plt.xlabel('Catégories')
        plt.ylabel('Nombre d\'occurrences')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_scaling_distribution(
    df_original: pd.DataFrame,
    df_scaled: pd.DataFrame,
    numeric_columns: List[str],
    figsize: tuple = (15, 5)
):
    """
    Visualise la distribution des variables avant et après scaling.
    
    Args:
        df_original: DataFrame original
        df_scaled: DataFrame normalisé
        numeric_columns: Liste des colonnes numériques
        figsize: Taille de la figure
    """
    n_cols = len(numeric_columns)
    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    
    for i, col in enumerate(numeric_columns):
        # Distribution originale
        axes[0, i].hist(df_original[col].dropna(), bins=30)
        axes[0, i].set_title(f'{col} (Original)')
        axes[0, i].grid(True)
        
        # Distribution normalisée
        axes[1, i].hist(df_scaled[col].dropna(), bins=30)
        axes[1, i].set_title(f'{col} (Normalisé)')
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.show() 