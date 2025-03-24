"""
Module de nettoyage et d'analyse de données.

Ce module fournit des fonctions pour nettoyer et analyser des DataFrames,
avec des visualisations des résultats.
"""

# Imports standards
from typing import Dict, Tuple, List, Optional

# Imports tiers requis
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

# Imports optionnels
try:
    import seaborn as sns
except ImportError:
    print("Warning: seaborn not installed. Some visualizations might be limited.")

# Imports internes
try:
    from ..data.analysis import analyze_data_quality
    from ..data.preprocessing import handle_duplicates
    from ..features.processing import filter_and_analyze_dataset
except ImportError:
    print("Warning: Some internal modules could not be imported. Some features might be limited.")

# Constants
MISSING_THRESHOLD = 0.8
FIGURE_SIZE = (15, 5)
COLORS = {
    'before': '#ff9999',
    'after': '#66b3ff',
    'problems': ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
}

# Fonctions principales
def process_and_analyze_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Fonction principale de nettoyage avec fonctionnalités avancées.
    
    Args:
        df: DataFrame à nettoyer
        
    Returns:
        Tuple[DataFrame, Dict]: DataFrame nettoyé et informations
    """
    # Nettoyage de base
    df_clean, cleaning_info = clean_dataset(df)
    
    # Identification des colonnes problématiques
    problematic_cols = identify_problematic_columns(df_clean)
    
    # Validation nutritionnelle
    df_clean = validate_nutritional_values(df_clean)
    
    # Détection des anomalies
    anomalies = detect_category_anomalies(df_clean)
    
    # Enrichissement
    df_clean = enrich_dataset(df_clean)
    
    # Validation des codes-barres
    df_clean = validate_barcodes(df_clean)
    
    # Standardisation des langues
    df_clean = standardize_language(df_clean)
    
    # Création du dictionnaire d'information complet
    info = {
        'final_stats': cleaning_info['final_stats'],
        'problematic_columns': problematic_cols,
        'anomalies': anomalies,
        'cleaning_info': cleaning_info,
        'data_summary': {
            'missing_values_before': df.isna().sum().sum(),
            'missing_values_after': df_clean.isna().sum().sum(),
            'initial_shape': df.shape,
            'final_shape': df_clean.shape
        }
    }
    
    return df_clean, info


def display_cleaning_summary(info: Dict) -> None:
    """
    Affiche un résumé détaillé du nettoyage des données.
    
    Args:
        info: Dictionnaire contenant les informations du nettoyage
    """
    print("=== Résumé du nettoyage ===")
    print(f"Dimensions initiales: {info['final_stats']['initial_rows']} lignes, {info['final_stats']['initial_columns']} colonnes")
    print(f"Dimensions finales: {info['final_stats']['final_rows']} lignes, {info['final_stats']['final_columns']} colonnes")

    print("\n=== Colonnes problématiques ===")
    for category, columns in info['problematic_columns'].items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for col in columns:
            if isinstance(col, tuple):
                try:
                    ratio = float(col[1])
                    print(f"  - {col[0]}: {ratio:.2%}")
                except (ValueError, TypeError):
                    print(f"  - {col[0]}: {col[1]}")
            else:
                print(f"  - {col}")

    print("\n=== Modifications effectuées ===")
    print("Colonnes ajoutées:", ", ".join(info['final_stats']['columns_added']) or "Aucune")
    print("Colonnes supprimées:", ", ".join(info['final_stats']['columns_removed']) or "Aucune")

    # Afficher les traitements spécifiques
    if info.get('specific_treatments'):
        print("\n=== Traitements spécifiques ===")
        for col, treatment in info['specific_treatments'].items():
            print(f"  - {col}: {treatment}")

    # Afficher les erreurs corrigées
    if info.get('errors_found'):
        print("\n=== Erreurs corrigées ===")
        for col, error_info in info['errors_found'].items():
            print(f"  - {col}: {error_info}")

    # Ajouter les visualisations
    plot_cleaning_summary(info)


# Fonctions de visualisation
def plot_cleaning_summary(info: Dict) -> None:
    """
    Crée des visualisations du résumé du nettoyage des données.
    
    Args:
        info: Dictionnaire contenant les informations du nettoyage
    """
    # Créer une figure avec 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # 1. Graphique des valeurs manquantes avant/après
    missing_data = {
        'Avant nettoyage': info['data_summary']['missing_values_before'],
        'Après nettoyage': info['data_summary']['missing_values_after']
    }
    bars1 = ax1.bar(missing_data.keys(), missing_data.values(), color=[COLORS['before'], COLORS['after']])
    ax1.set_title('Valeurs manquantes avant/après nettoyage')
    ax1.set_ylabel('Nombre de valeurs manquantes')
    
    # Ajouter les valeurs sur les barres
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # 2. Graphique des colonnes problématiques
    problem_categories = []
    counts = []
    
    for category, columns in info['problematic_columns'].items():
        problem_categories.append(category.replace('_', ' ').title())
        counts.append(len(columns))
    
    bars2 = ax2.bar(problem_categories, counts, color=COLORS['problems'][:len(problem_categories)])
    ax2.set_title('Distribution des problèmes détectés')
    ax2.set_ylabel('Nombre de colonnes')
    ax2.tick_params(axis='x', rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                str(int(height)),
                ha='center', va='bottom')
    
    # Ajuster la mise en page
    plt.tight_layout()
    plt.show()


# Fonctions utilitaires
def _format_number(n: int) -> str:
    """Formate un nombre pour l'affichage."""
    return f"{n:,}"

def extract_quantity_from_serving_size(serving_size: str) -> Tuple[float, str]:
    """
    Extrait la quantité et l'unité d'une chaîne de caractères de taille de portion.
    
    Args:
        serving_size: Chaîne de caractères contenant la taille de portion
        
    Returns:
        Tuple[float, str]: Quantité et unité extraites
    """
    if pd.isna(serving_size):
        return np.nan, np.nan
        
    # Pattern pour extraire le nombre et l'unité
    pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)'
    match = re.search(pattern, str(serving_size))
    
    if match:
        quantity = float(match.group(1))
        unit = match.group(2).lower()
        return quantity, unit
    return np.nan, np.nan

def clean_text_column(text: str) -> str:
    """
    Nettoie une chaîne de caractères en retirant les caractères spéciaux et en normalisant l'espacement.
    
    Args:
        text: Chaîne de caractères à nettoyer
        
    Returns:
        str: Chaîne nettoyée
    """
    if pd.isna(text):
        return text
    # Convertir en minuscules
    text = str(text).lower()
    # Remplacer les caractères spéciaux par des espaces
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normaliser les espaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def identify_problematic_columns(df: pd.DataFrame, 
                              missing_threshold: float = 0.5,
                              unique_threshold: float = 0.95) -> Dict:
    """
    Identifie les colonnes problématiques dans le dataset.
    
    Args:
        df: DataFrame à analyser
        missing_threshold: Seuil pour les valeurs manquantes
        unique_threshold: Seuil pour les valeurs uniques
        
    Returns:
        Dict: Informations sur les colonnes problématiques
    """
    problematic_columns = {
        'high_missing': [],
        'high_unique': [],
        'constant': [],
        'potential_errors': []
    }
    
    for column in df.columns:
        # Vérifier les valeurs manquantes
        missing_ratio = df[column].isna().mean()
        if missing_ratio > missing_threshold:
            problematic_columns['high_missing'].append((column, missing_ratio))
            
        # Vérifier les valeurs uniques
        unique_ratio = df[column].nunique() / len(df)
        if unique_ratio > unique_threshold:
            problematic_columns['high_unique'].append((column, unique_ratio))
            
        # Vérifier les colonnes constantes
        if unique_ratio == 1:
            problematic_columns['constant'].append(column)
            
        # Vérifier les potentiels problèmes de format
        if df[column].dtype == 'object':
            # Vérifier les valeurs qui ne suivent pas le format attendu
            if column == 'serving_size':
                invalid_values = df[~df[column].isna() & ~df[column].str.match(r'^\d+(?:\.\d+)?\s*[a-zA-Z]+$', na=False)]
                if not invalid_values.empty:
                    problematic_columns['potential_errors'].append((column, 'format_error'))
    
    return problematic_columns

def clean_dataset(
    df: pd.DataFrame,
    max_categories: int = 30,
    missing_threshold: float = 0.8,
    extract_quantities: bool = True,
    clean_text: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Nettoie le dataset en identifiant et traitant les différents types de problèmes.
    
    Args:
        df: DataFrame à nettoyer
        max_categories: Nombre maximum de catégories pour les variables catégorielles
        missing_threshold: Seuil de valeurs manquantes acceptables
        extract_quantities: Si True, extrait les quantités des colonnes de taille de portion
        clean_text: Si True, nettoie les colonnes textuelles
    """
    cleaning_info = {
        'removed_columns': [],
        'imputed_columns': [],
        'extracted_patterns': {},
        'errors_found': {},
        'specific_treatments': {}
    }
    
    df_clean = df.copy()
    
    # 1. Variables non pertinentes
    redundant_columns = [
        'image_url', 'image_small_url',  # Garder uniquement image_front_url
        'created_datetime', 'last_modified_datetime',  # Garder uniquement les timestamps
        'creator', 'last_modified_by'  # Informations non pertinentes pour l'analyse
    ]
    df_clean = df_clean.drop(columns=[col for col in redundant_columns if col in df_clean.columns])
    cleaning_info['removed_columns'].extend(redundant_columns)

    # 2. Traitement des valeurs manquantes selon le type de variable
    # Variables nutritionnelles : imputation par 0
    nutrition_cols = [col for col in df_clean.columns if any(x in col for x in 
                     ['energy', 'fat', 'protein', 'carbohydrates', 'sugar', 'fiber', 'salt'])]
    for col in nutrition_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0)
            cleaning_info['imputed_columns'].append((col, 'zero_imputation'))

    # Variables catégorielles : imputation par mode si < 50% manquantes
    categorical_cols = [col for col in df_clean.columns if df_clean[col].dtype == 'object']
    for col in categorical_cols:
        missing_ratio = df_clean[col].isnull().mean()
        if missing_ratio < 0.5:
            mode_value = df_clean[col].mode().iloc[0]
            df_clean[col] = df_clean[col].fillna(mode_value)
            cleaning_info['imputed_columns'].append((col, 'mode_imputation'))

    # 3. Extraction de motifs
    if 'serving_size' in df_clean.columns:
        # Extraction des quantités et unités de serving_size
        serving_info = df_clean['serving_size'].str.extract(r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)')
        df_clean['serving_quantity'] = pd.to_numeric(serving_info[0], errors='coerce')
        df_clean['serving_unit'] = serving_info[1]
        cleaning_info['extracted_patterns']['serving_size'] = {
            'successful_extractions': serving_info[0].notna().sum()
        }

    if 'ingredients_text' in df_clean.columns:
        # Nettoyage basique des ingrédients
        df_clean['ingredients_clean'] = df_clean['ingredients_text'].str.lower()
        df_clean['ingredients_clean'] = df_clean['ingredients_clean'].str.replace(r'[^\w\s]', ' ')
        cleaning_info['extracted_patterns']['ingredients'] = 'basic_cleaning'

    # 4. Détection et correction des erreurs
    # Vérification des valeurs négatives dans les colonnes nutritionnelles
    for col in nutrition_cols:
        if col in df_clean.columns:
            negative_values = (df_clean[col] < 0).sum()
            if negative_values > 0:
                df_clean[col] = df_clean[col].clip(lower=0)
                cleaning_info['errors_found'][col] = f"corrected_{negative_values}_negative_values"

    # 5. Traitements spécifiques
    # Standardisation des unités nutritionnelles (tout en 100g)
    per_serving_cols = [col for col in df_clean.columns if '_serving' in col]
    for col in per_serving_cols:
        if col in df_clean.columns and 'serving_quantity' in df_clean.columns:
            base_col = col.replace('_serving', '_100g')
            if base_col in df_clean.columns:
                df_clean[base_col] = df_clean[base_col].fillna(
                    df_clean[col] * 100 / df_clean['serving_quantity']
                )
                cleaning_info['specific_treatments'][col] = 'converted_to_100g'

    # Analyse finale
    cleaning_info['final_stats'] = {
        'initial_columns': len(df.columns),
        'final_columns': len(df_clean.columns),
        'initial_rows': len(df),
        'final_rows': len(df_clean),
        'columns_added': list(set(df_clean.columns) - set(df.columns)),
        'columns_removed': list(set(df.columns) - set(df_clean.columns))
    }

    return df_clean, cleaning_info

def standardize_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise les unités de mesure en convertissant tout en grammes ou millilitres.
    """
    df = df.copy()
    
    # Conversion des onces en grammes
    mask_oz = df['serving_unit'].str.contains('oz', case=False, na=False)
    df.loc[mask_oz, 'serving_quantity'] = df.loc[mask_oz, 'serving_quantity'] * 28.35
    
    # Conversion des livres en grammes
    mask_lb = df['serving_unit'].str.contains('lb', case=False, na=False)
    df.loc[mask_lb, 'serving_quantity'] = df.loc[mask_lb, 'serving_quantity'] * 453.592
    
    # Conversion des cuillères à soupe en millilitres
    mask_tbsp = df['serving_unit'].str.contains('tbsp', case=False, na=False)
    df.loc[mask_tbsp, 'serving_quantity'] = df.loc[mask_tbsp, 'serving_quantity'] * 15
    
    # Standardisation des unités
    df.loc[mask_oz | mask_lb, 'serving_unit'] = 'g'
    df.loc[mask_tbsp, 'serving_unit'] = 'ml'
    
    return df

def validate_nutritional_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Valide et corrige les valeurs nutritionnelles selon des règles métier.
    
    Args:
        df: DataFrame contenant les données nutritionnelles
        
    Returns:
        DataFrame avec les valeurs nutritionnelles validées
    """
    # Définir des plages valides pour chaque nutriment
    valid_ranges = {
        'energy_100g': (0, 3000),  # kcal
        'proteins_100g': (0, 100),
        'carbohydrates_100g': (0, 100),
        'fat_100g': (0, 100),
        'fiber_100g': (0, 50),
        'salt_100g': (0, 100)
    }
    
    # Vérifier la cohérence des macronutriments
    macro_cols = ['proteins_100g', 'carbohydrates_100g', 'fat_100g']
    df['total_macros'] = df[macro_cols].sum(axis=1)
    
    # Identifier les lignes incohérentes (somme > 100g)
    inconsistent_rows = df['total_macros'] > 100
    
    if inconsistent_rows.any():
        # Normaliser les valeurs pour qu'elles somment à 100
        for col in macro_cols:
            df.loc[inconsistent_rows, col] = (
                df.loc[inconsistent_rows, col] / 
                df.loc[inconsistent_rows, 'total_macros'] * 100
            )
    
    return df

def detect_category_anomalies(df: pd.DataFrame) -> Dict:
    """
    Détecte les anomalies dans les valeurs nutritionnelles par catégorie.
    
    Args:
        df: DataFrame avec les données
        
    Returns:
        Dict contenant les anomalies détectées par catégorie
    """
    anomalies = {}
    
    for category in df['categories'].unique():
        category_data = df[df['categories'] == category]
        
        # Calculer les statistiques par catégorie
        for col in ['energy_100g', 'proteins_100g', 'fat_100g']:
            if col in df.columns:
                Q1 = category_data[col].quantile(0.25)
                Q3 = category_data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Définir les limites
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Identifier les anomalies
                outliers = category_data[
                    (category_data[col] < lower_bound) | 
                    (category_data[col] > upper_bound)
                ]
                
                if not outliers.empty:
                    if category not in anomalies:
                        anomalies[category] = {}
                    anomalies[category][col] = outliers.index.tolist()
    
    return anomalies

def enrich_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrichit le dataset avec des features dérivées.
    
    Args:
        df: DataFrame original
        
    Returns:
        DataFrame enrichi
    """
    df = df.copy()
    
    # Ratio protéines/calories
    if 'proteins_100g' in df.columns and 'energy_100g' in df.columns:
        df['protein_ratio'] = (df['proteins_100g'] * 4) / df['energy_100g']
    
    # Catégorisation nutritionnelle
    if 'fat_100g' in df.columns:
        df['fat_category'] = pd.cut(
            df['fat_100g'],
            bins=[0, 3, 20, float('inf')],
            labels=['low_fat', 'medium_fat', 'high_fat']
        )
    
    # Complexité des ingrédients
    if 'ingredients_text' in df.columns:
        df['ingredients_count'] = df['ingredients_text'].str.count(',') + 1
        df['ingredients_complexity'] = pd.cut(
            df['ingredients_count'],
            bins=[0, 5, 10, float('inf')],
            labels=['simple', 'medium', 'complex']
        )
    
    return df

def validate_barcodes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Valide et nettoie les codes-barres.
    
    Args:
        df: DataFrame contenant les codes-barres
        verbose: Si True, affiche les messages de debug
        
    Returns:
        DataFrame avec codes-barres validés
    """
    def check_ean13(code: str) -> bool:
        if not isinstance(code, str) or len(code) != 13:
            return False
        
        try:
            digits = list(map(int, code))
            checksum = digits[-1]
            calc = sum(digits[::2][:-1]) + sum(d * 3 for d in digits[1::2])
            return (10 - (calc % 10)) % 10 == checksum
        except ValueError:
            return False
    
    if 'code' in df.columns:
        df['valid_barcode'] = df['code'].apply(check_ean13)
        invalid_codes = df[~df['valid_barcode']].index
        
        if len(invalid_codes) > 0 and verbose:
            print(f"Found {len(invalid_codes)} invalid barcodes")
    
    return df

def standardize_language(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise les champs textuels en fonction de la langue.
    
    Args:
        df: DataFrame avec les données
        
    Returns:
        DataFrame avec textes standardisés
    """
    try:
        from langdetect import detect
    except ImportError:
        print("Warning: langdetect not installed. Language detection will be skipped.")
        return df
    
    text_columns = ['product_name', 'ingredients_text', 'generic_name']
    
    for col in text_columns:
        if col in df.columns:
            # Détecter la langue
            df[f'{col}_lang'] = df[col].apply(
                lambda x: detect(str(x)) if pd.notna(x) else 'unknown'
            )
            
            # Marquer les incohérences
            if 'lang' in df.columns:
                df[f'{col}_consistent'] = (
                    df[f'{col}_lang'] == df['lang']
                )
    
    return df

if __name__ == "__main__":
    # Exemple d'utilisation
    df = pd.read_csv("../data/en.openfoodfacts.org.products.csv", sep='\t', encoding='utf-8', nrows=300000)
    df_clean, info = process_and_analyze_dataset(df)
    
    display_cleaning_summary(info)
    
    if 'quantity_extraction' in info['cleaning_info']:
        print("\nExtraction des quantités:")
        print(f"  Succès: {info['cleaning_info']['quantity_extraction']['successful_extractions']}")
        print(f"  Échecs: {info['cleaning_info']['quantity_extraction']['failed_extractions']}") 