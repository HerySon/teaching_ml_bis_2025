import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Optional
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def fetch_usda_data(api_key: str, food_name: str) -> Optional[Dict]:
    """
    Fetch nutritional data from USDA's Food Database API.
    
    Parameters:
    -----------
    api_key : str
        USDA API key
    food_name : str
        Name of the food to search for
    
    Returns:
    --------
    Optional[Dict]
        Nutritional data if found, None otherwise
    """
    # Search endpoint
    search_url = f"https://api.nal.usda.gov/fdc/v1/foods/search?api_key={api_key}&query={food_name}"
    
    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            if data.get('foods'):
                # Get the first food item
                food = data['foods'][0]
                
                # Extract nutritional values
                nutrients = {}
                print(f"\nNutrients found for {food_name}:")
                for nutrient in food.get('foodNutrients', []):
                    if nutrient.get('nutrientName'):
                        nutrient_name = nutrient['nutrientName'].lower()
                        value = nutrient.get('value', 0)
                        nutrients[nutrient_name] = value
                        print(f"- {nutrient_name}: {value}")
                
                if not nutrients:
                    print(f"No nutrient data found for {food_name}")
                    return None
                
                return {
                    'food_name': food.get('description', food_name),
                    'brand': food.get('brandOwner', 'Unknown'),
                    'serving_size': food.get('servingSize', 100),
                    'serving_unit': food.get('servingSizeUnit', 'g'),
                    **nutrients
                }
            else:
                print(f"No results found for {food_name}")
        else:
            print(f"API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error fetching data for {food_name}: {str(e)}")
    
    return None

def enrich_dataset(
    df: pd.DataFrame,
    api_key: str,
    sample_size: int = 100,
    delay: float = 1.0
) -> pd.DataFrame:
    """
    Enrich the Open Food Facts dataset with additional nutritional data from USDA.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original Open Food Facts dataset
    api_key : str
        USDA API key
    sample_size : int, default=100
        Number of products to enrich
    delay : float, default=1.0
        Delay between API calls in seconds
    
    Returns:
    --------
    pd.DataFrame
        Enriched dataset
    """
    # Create a copy of the original dataframe
    df_enriched = df.copy()
    
    # Print available columns
    print("\nAvailable columns in the dataset:")
    print(df.columns.tolist())
    
    # Try different product name columns in order of preference
    product_name_columns = ['abbreviated_product_name', 'generic_name', 'product_name_en']
    product_name_col = None
    
    for col in product_name_columns:
        if col in df.columns:
            valid_names = df[col].notna().sum()
            if valid_names > 0:
                print(f"\nUsing '{col}' column with {valid_names} valid product names")
                product_name_col = col
                break
    
    if product_name_col is None:
        print("No valid product name columns found. Available columns:")
        for col in df.columns:
            if 'name' in col.lower() or 'product' in col.lower():
                print(f"- {col}")
        return df_enriched
    
    # Sample unique products to enrich
    unique_products = df[product_name_col].dropna().unique()
    if len(unique_products) < sample_size:
        print(f"\nWarning: Only {len(unique_products)} unique products found. Using all of them.")
        sample_products = df[df[product_name_col].isin(unique_products)]
    else:
        sample_products = df[df[product_name_col].isin(np.random.choice(unique_products, sample_size, replace=False))]
    
    # Initialize new columns for USDA data
    usda_columns = [
        'usda_energy_100g', 'usda_proteins_100g', 'usda_fat_100g', 'usda_carbohydrates_100g',
        'usda_fiber_100g', 'usda_sugars_100g', 'usda_sodium_100g', 'usda_calcium_100g',
        'usda_iron_100g', 'usda_vitamin_c_100g'
    ]
    for col in usda_columns:
        df_enriched[col] = np.nan
    
    # Track successful matches
    successful_matches = 0
    total_attempts = 0
    
    # Print Open Food Facts columns that we'll try to match
    print("\nOpen Food Facts columns we'll try to match:")
    original_columns = [
        'energy_100g', 'proteins_100g', 'fat_100g', 'carbohydrates_100g',
        'fiber_100g', 'sugars_100g', 'sodium_100g', 'calcium_100g',
        'iron_100g', 'vitamin_c_100g'
    ]
    for col in original_columns:
        if col in df.columns:
            print(f"- {col} (found in dataset)")
        else:
            print(f"- {col} (not found in dataset)")
    
    # Print sample of original data
    print("\nSample of original Open Food Facts data:")
    available_columns = [col for col in original_columns if col in df.columns]
    if available_columns:
        print(df[available_columns].head())
    else:
        print("No matching columns found in the dataset")
    
    # Fetch USDA data for each product
    print(f"\nFetching USDA data for {len(sample_products)} unique products...")
    for _, row in tqdm(sample_products.iterrows(), total=len(sample_products)):
        # Use the selected product name column
        food_name = row[product_name_col]
        
        # Skip if product name is missing
        if pd.isna(food_name):
            continue
            
        total_attempts += 1
        # Fetch USDA data
        usda_data = fetch_usda_data(api_key, food_name)
        
        if usda_data:
            successful_matches += 1
            # Map USDA nutrients to our columns
            nutrient_mapping = {
                'energy': 'usda_energy_100g',
                'protein': 'usda_proteins_100g',
                'total lipid (fat)': 'usda_fat_100g',
                'carbohydrate, by difference': 'usda_carbohydrates_100g',
                'fiber, total dietary': 'usda_fiber_100g',
                'sugars, total': 'usda_sugars_100g',
                'sodium, na': 'usda_sodium_100g',
                'calcium, ca': 'usda_calcium_100g',
                'iron, fe': 'usda_iron_100g',
                'vitamin c, total ascorbic acid': 'usda_vitamin_c_100g'
            }
            
            print(f"\nMapping nutrients for {food_name}:")
            for usda_nutrient, our_column in nutrient_mapping.items():
                if usda_nutrient in usda_data:
                    # Use the original index from the row
                    df_enriched.loc[row.name, our_column] = usda_data[usda_nutrient]
                    print(f"- {usda_nutrient} -> {our_column}: {usda_data[usda_nutrient]}")
                else:
                    print(f"- {usda_nutrient} not found in USDA data")
        
        # Respect API rate limits
        time.sleep(delay)
    
    # Print summary statistics
    print(f"\nSummary of USDA data enrichment:")
    print(f"Total unique products attempted: {total_attempts}")
    print(f"Successful matches: {successful_matches}")
    
    if total_attempts > 0:
        success_rate = (successful_matches/total_attempts)*100
        print(f"Success rate: {success_rate:.1f}%")
        
        # Print sample of enriched data
        print("\nSample of enriched data:")
        print(df_enriched[[product_name_col] + usda_columns].head())
        
        # Print non-null counts for USDA columns
        print("\nNumber of non-null values in USDA columns:")
        print(df_enriched[usda_columns].count())
        
        # Calculate correlations between Open Food Facts and USDA data
        correlations = {}
        for col in usda_columns:
            original_col = col.replace('usda_', '')
            if original_col in df.columns:
                # Filter out NaN values and ensure we have enough data points
                mask = ~(df[original_col].isna() | df_enriched[col].isna())
                if mask.sum() >= 2:  # Need at least 2 points for correlation
                    correlation = df.loc[mask, original_col].corr(df_enriched.loc[mask, col])
                    correlations[original_col] = correlation
                    print(f"\nCorrelation for {original_col}:")
                    print(f"Number of valid pairs: {mask.sum()}")
                    print(f"Open Food Facts values: {df.loc[mask, original_col].head()}")
                    print(f"USDA values: {df_enriched.loc[mask, col].head()}")
    
        if correlations:
            print("\nCorrelations between Open Food Facts and USDA data:")
            print("-" * 30)
            for nutrient, correlation in correlations.items():
                print(f"{nutrient}: {correlation:.3f}")
        else:
            print("\nNo correlations could be calculated. This might be because:")
            print("1. No USDA data was successfully retrieved")
            print("2. No matching columns were found between Open Food Facts and USDA data")
            print("3. All values were missing in one or both datasets")
            print("4. Not enough valid data pairs (need at least 2) for correlation")
    else:
        print("No products were attempted. This might be because:")
        print("1. All product names in the sample were missing")
        print("2. The sample size was 0")
        print("3. The dataset is empty")
    
    return df_enriched

def analyze_data_quality(df: pd.DataFrame) -> None:
    """
    Analyze the quality of the enriched dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Enriched dataset
    """
    # Calculate missing value percentages
    missing_pct = df.isnull().mean() * 100
    
    print("\nMissing Value Analysis:")
    print("-" * 30)
    for col, pct in missing_pct.items():
        print(f"{col}: {pct:.1f}%")
    
    # Calculate basic statistics for numerical columns
    print("\nBasic Statistics:")
    print("-" * 30)
    print(df.describe())
    
    # Calculate correlations between original and USDA data
    usda_cols = [col for col in df.columns if col.startswith('usda_')]
    original_cols = [col.replace('usda_', '') for col in usda_cols]
    
    print("\nCorrelations between Original and USDA Data:")
    print("-" * 30)
    for orig_col, usda_col in zip(original_cols, usda_cols):
        if orig_col in df.columns and usda_col in df.columns:
            correlation = df[orig_col].corr(df[usda_col])
            print(f"{orig_col} vs {usda_col}: {correlation:.3f}")

def visualize_nutrient_comparisons(df: pd.DataFrame) -> None:
    """
    Create visualizations comparing Open Food Facts and USDA nutrient data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Enriched dataset
    """
    # Set style
    plt.style.use('default')
    
    # Get USDA columns
    usda_cols = [col for col in df.columns if col.startswith('usda_')]
    original_cols = [col.replace('usda_', '') for col in usda_cols]
    
    # Filter out columns with no valid data
    valid_pairs = []
    for orig_col, usda_col in zip(original_cols, usda_cols):
        if orig_col in df.columns and usda_col in df.columns:
            # Check if we have any non-null values
            if not df[orig_col].isna().all() and not df[usda_col].isna().all():
                valid_pairs.append((orig_col, usda_col))
    
    if not valid_pairs:
        print("No valid data pairs found for visualization")
        return
    
    # Create subplots
    n_cols = 2
    n_rows = (len(valid_pairs) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, (orig_col, usda_col) in enumerate(valid_pairs):
        # Create scatter plot
        ax = axes[idx]
        # Filter out NaN values
        mask = ~(df[orig_col].isna() | df[usda_col].isna())
        if mask.any():
            ax.scatter(df.loc[mask, orig_col], df.loc[mask, usda_col], alpha=0.5)
            
            # Calculate max value for the perfect match line
            max_val = max(df.loc[mask, orig_col].max(), df.loc[mask, usda_col].max())
            ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect match')
            
            # Calculate correlation
            correlation = df.loc[mask, orig_col].corr(df.loc[mask, usda_col])
            
            ax.set_xlabel(f'Open Food Facts {orig_col}')
            ax.set_ylabel(f'USDA {orig_col}')
            ax.set_title(f'{orig_col}\nCorrelation: {correlation:.3f}')
            ax.legend()
    
    # Remove empty subplots if any
    for idx in range(len(valid_pairs), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

def visualize_nutrient_distributions(df: pd.DataFrame) -> None:
    """
    Create box plots comparing nutrient distributions between Open Food Facts and USDA data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Enriched dataset
    """
    # Set style
    plt.style.use('default')
    
    # Get USDA columns
    usda_cols = [col for col in df.columns if col.startswith('usda_')]
    original_cols = [col.replace('usda_', '') for col in usda_cols]
    
    # Filter out columns with no valid data
    valid_pairs = []
    for orig_col, usda_col in zip(original_cols, usda_cols):
        if orig_col in df.columns and usda_col in df.columns:
            # Check if we have any non-null values
            if not df[orig_col].isna().all() and not df[usda_col].isna().all():
                valid_pairs.append((orig_col, usda_col))
    
    if not valid_pairs:
        print("No valid data pairs found for visualization")
        return
    
    # Create subplots
    n_cols = 2
    n_rows = (len(valid_pairs) + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten()
    
    for idx, (orig_col, usda_col) in enumerate(valid_pairs):
        # Create box plot
        ax = axes[idx]
        # Filter out NaN values
        mask = ~(df[orig_col].isna() | df[usda_col].isna())
        if mask.any():
            data = [df.loc[mask, orig_col], df.loc[mask, usda_col]]
            ax.boxplot(data, labels=['Open Food Facts', 'USDA'])
            
            ax.set_title(f'{orig_col} Distribution')
            ax.set_ylabel('Value')
            
            # Add statistical test
            try:
                t_stat, p_value = stats.ttest_ind(df.loc[mask, orig_col], df.loc[mask, usda_col])
                ax.text(0.02, 0.98, f'p-value: {p_value:.3f}', 
                       transform=ax.transAxes, 
                       verticalalignment='top')
            except:
                ax.text(0.02, 0.98, 'Statistical test not available', 
                       transform=ax.transAxes, 
                       verticalalignment='top')
    
    # Remove empty subplots if any
    for idx in range(len(valid_pairs), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()

def visualize_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Create a correlation heatmap between Open Food Facts and USDA nutrient data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Enriched dataset
    """
    # Set style
    plt.style.use('default')
    
    # Get USDA columns and corresponding original columns
    usda_cols = [col for col in df.columns if col.startswith('usda_')]
    original_cols = [col.replace('usda_', '') for col in usda_cols]
    
    # Filter out columns with no valid data
    valid_pairs = []
    for orig_col, usda_col in zip(original_cols, usda_cols):
        if orig_col in df.columns and usda_col in df.columns:
            # Check if we have any non-null values
            if not df[orig_col].isna().all() and not df[usda_col].isna().all():
                valid_pairs.append((orig_col, usda_col))
    
    if not valid_pairs:
        print("No valid data pairs found for visualization")
        return
    
    # Create correlation matrix
    corr_matrix = pd.DataFrame()
    
    for orig_col, usda_col in valid_pairs:
        # Filter out NaN values
        mask = ~(df[orig_col].isna() | df[usda_col].isna())
        if mask.any():
            corr_matrix.loc[orig_col, 'Open Food Facts'] = 1.0
            corr_matrix.loc[orig_col, 'USDA'] = df.loc[mask, orig_col].corr(df.loc[mask, usda_col])
    
    if not corr_matrix.empty:
        # Create heatmap
        plt.figure(figsize=(10, len(valid_pairs)))
        sns.heatmap(corr_matrix, 
                    annot=True, 
                    cmap='RdYlBu_r', 
                    center=0,
                    fmt='.3f',
                    square=True)
        
        plt.title('Correlation between Open Food Facts and USDA Data')
        plt.tight_layout()
        plt.show()

def visualize_missing_data(df: pd.DataFrame) -> None:
    """
    Create a visualization of missing data in the enriched dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Enriched dataset
    """
    # Set style
    plt.style.use('default')
    
    # Calculate missing percentages
    missing_pct = df.isnull().mean() * 100
    
    # Filter out columns with no missing values
    missing_pct = missing_pct[missing_pct > 0]
    
    if missing_pct.empty:
        print("No missing data found in the dataset")
        return
    
    # Create bar plot
    plt.figure(figsize=(15, 8))
    missing_pct.plot(kind='bar')
    
    plt.title('Missing Data Percentage by Column')
    plt.xlabel('Columns')
    plt.ylabel('Missing Data (%)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

# Example usage in notebook:
"""
# Import required libraries
import pandas as pd
from scripts.additional_data import (
    enrich_dataset, 
    analyze_data_quality,
    visualize_nutrient_comparisons,
    visualize_nutrient_distributions,
    visualize_correlation_heatmap,
    visualize_missing_data
)

# Your USDA API key
api_key = "YOUR_API_KEY"

# Load your dataset (assuming it's already loaded in the notebook)
# df = pd.read_csv('your_data.csv')  # or however you loaded your data

# Enrich the dataset
df_enriched = enrich_dataset(
    df=df,
    api_key=api_key,
    sample_size=100,  # Number of products to enrich
    delay=1.0  # Delay between API calls to respect rate limits
)

# Analyze the quality of the enriched dataset
analyze_data_quality(df_enriched)

# Create visualizations
visualize_nutrient_comparisons(df_enriched)  # Scatter plots comparing Open Food Facts and USDA data
visualize_nutrient_distributions(df_enriched)  # Box plots showing distributions
visualize_correlation_heatmap(df_enriched)  # Correlation heatmap
visualize_missing_data(df_enriched)  # Missing data visualization

# Now you can use df_enriched directly in your notebook
# For example, to see the enriched data:
print(df_enriched.head())
"""
