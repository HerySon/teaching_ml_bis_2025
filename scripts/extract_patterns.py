"""
Utility functions for extracting patterns from text fields in the OpenFoodFacts dataset.
Focuses on quantity, serving_size, and similar fields that contain structured information.
"""

import re

try:
    import numpy as np
    import pandas as pd
except ImportError:
    raise ImportError("numpy and pandas are required for this module")


def extract_quantity(text: str) -> float | None:
    """
    Extract quantity from product text fields.

    Args:
        text: Text containing quantity information

    Returns:
        Extracted quantity as float, or None if extraction fails
    """
    if pd.isna(text) or not isinstance(text, str):
        return None

    # Common patterns for quantities: "100g", "2x50g", "250 ml", etc.
    pattern = r'(\d+(?:\.\d+)?)\s*(?:x\s*)?(\d+(?:\.\d+)?)?\s*(g|ml|l|cl|kg|oz)'
    match = re.search(pattern, text.lower())

    if match:
        # Handle 2x50g format
        if match.group(2):
            value = float(match.group(1)) * float(match.group(2))
        else:
            value = float(match.group(1))

        # Convert to grams/ml for consistency
        unit = match.group(3)
        if unit == 'kg':
            value *= 1000  # kg to g
        elif unit == 'l':
            value *= 1000  # l to ml
        elif unit == 'cl':
            value *= 10    # cl to ml
        elif unit == 'oz':
            value *= 28.35  # oz to g (approximate)

        return value

    return None


def extract_nutrients(ingredients_text: str) -> dict[str, float]:
    """
    Extract nutrient information from ingredients text.

    Args:
        ingredients_text: Ingredients text field

    Returns:
        Dictionary of nutrient names and their quantities
    """
    if pd.isna(ingredients_text) or not isinstance(ingredients_text, str):
        return {}

    nutrients = {}

    # Pattern for nutrient mentions: "10g of protein", "5% sugar", etc.
    pattern = r'(\d+(?:\.\d+)?)\s*(?:g|%)\s+(?:of\s+)?(\w+)'
    matches = re.finditer(pattern, ingredients_text.lower())

    for match in matches:
        value = float(match.group(1))
        nutrient = match.group(2)
        nutrients[nutrient] = value

    return nutrients


def extract_brands(brands_text: str) -> list[str]:
    """
    Extract individual brands from brands field.

    Args:
        brands_text: Brands text field

    Returns:
        List of brand names
    """
    if pd.isna(brands_text) or not isinstance(brands_text, str):
        return []

    # Split by common separators and clean
    brands = [b.strip() for b in re.split(r'[,;]', brands_text)]

    # Remove empty strings
    brands = [b for b in brands if b]

    return brands


def parse_serving_size(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse serving_size column to extract quantity and unit.

    Args:
        df: Input DataFrame with serving_size column

    Returns:
        DataFrame with added serving_quantity and serving_unit columns
    """
    if 'serving_size' not in df.columns:
        print("serving_size column not found in DataFrame")
        return df

    df_parsed = df.copy()

    # Initialize new columns
    df_parsed['serving_quantity'] = np.nan
    df_parsed['serving_unit'] = None

    # Process only non-null serving_size values
    mask = ~df['serving_size'].isna()

    # Regex pattern for serving size
    pattern = r'(\d+(?:\.\d+)?)\s*(?:x\s*)?(\d+(?:\.\d+)?)?\s*(g|ml|l|cl|kg|oz)'

    # Create temporary lists to hold results
    quantities = []
    units = []
    indices = []

    for idx, serving_size in df.loc[mask, 'serving_size'].items():
        if not isinstance(serving_size, str):
            continue

        match = re.search(pattern, serving_size.lower())
        if match:
            # Extract quantity
            if match.group(2):  # Format like "2 x 100g"
                quantity = float(match.group(1)) * float(match.group(2))
            else:
                quantity = float(match.group(1))

            # Extract unit
            unit = match.group(3)

            # Convert to standard unit
            if unit == 'kg':
                quantity *= 1000
                unit = 'g'
            elif unit == 'l':
                quantity *= 1000
                unit = 'ml'
            elif unit == 'cl':
                quantity *= 10
                unit = 'ml'

            # Store for batch update
            indices.append(idx)
            quantities.append(quantity)
            units.append(unit)

    # Batch update the DataFrame (more efficient and avoids loc issues)
    if indices:
        df_parsed.loc[indices, 'serving_quantity'] = quantities
        df_parsed.loc[indices, 'serving_unit'] = units

    print(f"Extracted serving quantities for"
          f"{df_parsed['serving_quantity'].notna().sum()} products")
    return df_parsed


def parse_quantity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse quantity column to extract value and unit.

    Args:
        df: Input DataFrame with quantity column

    Returns:
        DataFrame with added quantity_value and quantity_unit columns
    """
    if 'quantity' not in df.columns:
        print("quantity column not found in DataFrame")
        return df

    df_parsed = df.copy()

    # Initialize new columns
    df_parsed['quantity_value'] = np.nan
    df_parsed['quantity_unit'] = None

    # Process only non-null quantity values
    mask = ~df['quantity'].isna()

    # Regex pattern for quantity
    pattern = r'(\d+(?:\.\d+)?)\s*(?:x\s*)?(\d+(?:\.\d+)?)?\s*(g|ml|l|cl|kg|oz)'

    # Create temporary lists to hold results
    values = []
    units = []
    indices = []

    for idx, quantity in df.loc[mask, 'quantity'].items():
        if not isinstance(quantity, str):
            continue

        match = re.search(pattern, quantity.lower())
        if match:
            # Extract value
            if match.group(2):  # Format like "2 x 100g"
                value = float(match.group(1)) * float(match.group(2))
            else:
                value = float(match.group(1))

            # Extract unit
            unit = match.group(3)

            # Convert to standard unit
            if unit == 'kg':
                value *= 1000
                unit = 'g'
            elif unit == 'l':
                value *= 1000
                unit = 'ml'
            elif unit == 'cl':
                value *= 10
                unit = 'ml'

            # Store for batch update
            indices.append(idx)
            values.append(value)
            units.append(unit)

    # Batch update the DataFrame (more efficient and avoids loc issues)
    if indices:
        df_parsed.loc[indices, 'quantity_value'] = values
        df_parsed.loc[indices, 'quantity_unit'] = units

    print(f"Extracted quantity values for {df_parsed['quantity_value'].notna().sum()} products")
    return df_parsed


def extract_patterns(
    df: pd.DataFrame,
    process_serving_size: bool = True,
    process_quantity: bool = True,
    process_brands: bool = True
) -> pd.DataFrame:
    """
    Process multiple pattern extractions from various columns.

    Args:
        df: Input DataFrame
        process_serving_size: Whether to process serving_size column
        process_quantity: Whether to process quantity column
        process_brands: Whether to process brands column

    Returns:
        DataFrame with added columns for extracted information
    """
    df_processed = df.copy()

    # Process serving_size if requested
    if process_serving_size and 'serving_size' in df.columns:
        df_processed = parse_serving_size(df_processed)

    # Process quantity if requested
    if process_quantity and 'quantity' in df.columns:
        df_processed = parse_quantity(df_processed)

    # Process brands if requested
    if process_brands and 'brands' in df.columns:
        # Create a new column with brand lists
        df_processed['brand_list'] = df_processed['brands'].apply(extract_brands)
        # Count number of brands
        df_processed['brand_count'] = df_processed['brand_list'].apply(len)
        # Extract main brand (first one)
        df_processed['main_brand'] = df_processed['brand_list'].apply(
            lambda x: x[0] if len(x) > 0 else None
        )

        print(f"Processed brands for {df_processed['brand_count'].sum()} products")

    return df_processed
