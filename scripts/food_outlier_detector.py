"""
Food-specific outlier detection and handling for OpenFoodFacts dataset.
"""

try:
    import pandas as pd
    from outlier_handler import OutlierDetector
except ImportError as exc:
    raise ImportError('pandas and outlier_handler are required for this module') from exc


# Define reasonable ranges for various nutrients (per 100g)
NUTRIENT_RANGES = {
    # Values in g per 100g of product (0-100g)
    'fat_100g': (0, 100),
    'saturated-fat_100g': (0, 100),
    'carbohydrates_100g': (0, 100),
    'sugars_100g': (0, 100),
    'fiber_100g': (0, 50),
    'proteins_100g': (0, 100),
    'salt_100g': (0, 100),

    # Values are in kJ or kcal
    'energy_100g': (0, 3900),  # Max theoretical ~ 900 kcal per 100g
    'energy-kj_100g': (0, 16000),  # ~ 3900 kJ

    # Values in mg per 100g
    'sodium_100g': (0, 40000),  # 40g sodium per 100g would be extreme
    'calcium_100g': (0, 2000),  # 2g calcium per 100g would be extreme
    'iron_100g': (0, 150),      # 150mg iron per 100g would be extreme
}


def detect_food_outliers_by_nutrient_range(data: pd.DataFrame) -> pd.Series:
    """
    Detect outliers based on nutrient ranges that are biologically implausible.

    Args:
        data: OpenFoodFacts DataFrame

    Returns:
        Boolean series indicating outliers
    """
    # Initialize outlier mask as all False
    outlier_mask = pd.Series(False, index=data.index)

    # Check each nutrient
    for nutrient, (min_val, max_val) in NUTRIENT_RANGES.items():
        if nutrient in data.columns:
            # Mark as outlier if outside valid range and not NaN
            current_mask = (
                ((data[nutrient] < min_val) | (data[nutrient] > max_val))
                & ~data[nutrient].isna()
            )
            outlier_mask = outlier_mask | current_mask

    return outlier_mask


def detect_food_outliers_by_nutrient_sum(data: pd.DataFrame, threshold: float = 105) -> pd.Series:
    """
    Detect outliers where the sum of macronutrients exceeds possible physical limits.

    Args:
        data: OpenFoodFacts DataFrame
        threshold: Maximum sum of macronutrients in g per 100g (default 105 to allow small measurement errors)

    Returns:
        Boolean series indicating outliers
    """
    # Define columns to sum (main macronutrients)
    macro_cols = ['fat_100g', 'carbohydrates_100g', 'proteins_100g', 'fiber_100g', 'salt_100g']

    # Filter to only columns that exist in the dataset
    existing_cols = [col for col in macro_cols if col in data.columns]

    if len(existing_cols) < 2:
        # Not enough columns to calculate meaningful sum
        return pd.Series(False, index=data.index)

    # Calculate the sum for each row, ignoring NaN values
    nutrient_sum = data[existing_cols].sum(axis=1, skipna=True)

    # Mark as outlier if sum exceeds threshold and at least two nutrient values are present
    non_nan_count = data[existing_cols].count(axis=1)
    outlier_mask = (nutrient_sum > threshold) & (non_nan_count >= 2)

    return outlier_mask


def detect_food_outliers_by_category_norms(
    data: pd.DataFrame,
    category_col: str = 'pnns_groups_1'
) -> pd.Series:
    """
    Detect outliers based on nutrient values that are far from the category norms.

    Args:
        data: OpenFoodFacts DataFrame
        category_col: Column name for food category

    Returns:
        Boolean series indicating outliers
    """
    if category_col not in data.columns:
        return pd.Series(False, index=data.index)

    # Define nutrients to check
    nutrients = [
        'energy_100g', 'fat_100g', 'saturated-fat_100g',
        'carbohydrates_100g', 'sugars_100g', 'proteins_100g'
    ]

    # Filter to nutrients that exist in the dataset
    nutrients = [n for n in nutrients if n in data.columns]

    if not nutrients:
        return pd.Series(False, index=data.index)

    # Initialize outlier mask
    outlier_mask = pd.Series(False, index=data.index)

    # Get categories with at least 10 products
    categories = data[category_col].value_counts()
    valid_categories = categories[categories >= 10].index

    # Process each valid category
    for category in valid_categories:
        category_mask = data[category_col] == category

        for nutrient in nutrients:
            # Convert column to numeric, force non-numeric to NaN
            numeric_data = pd.to_numeric(data[nutrient], errors='coerce')

            # Get only the data for this category
            cat_nutrient_data = numeric_data[category_mask].dropna()

            if len(cat_nutrient_data) < 10:
                continue

            # Calculate z-scores within this category
            mean = cat_nutrient_data.mean()
            std = cat_nutrient_data.std()

            if std == 0 or pd.isna(std):
                continue

            # Calculate z-scores for the entire column
            all_z_scores = (numeric_data - mean) / std

            # Create a mask for this nutrient's outliers in this category
            z_outlier_mask = (abs(all_z_scores) > 3) & category_mask & ~numeric_data.isna()

            # Update the overall outlier mask
            outlier_mask = outlier_mask | z_outlier_mask

    return outlier_mask


class FoodOutlierDetector(OutlierDetector):
    """
    Class for detecting outliers in food datasets, extending OutlierDetector.
    """

    def detect_outliers(self, method: str = "tukey", **kwargs) -> pd.Series:
        """
        Detect outliers using the specified method.

        Args:
            method: Method to use ('tukey', 'zscore', 'isolation_forest', 'lof',
                    'nutrient_range', 'nutrient_sum', 'category_norms')
            **kwargs: Additional parameters for the detection method

        Returns:
            Boolean series indicating outliers
        """
        # Handle food-specific methods
        if method == "nutrient_range":
            outlier_mask = detect_food_outliers_by_nutrient_range(self.data)

        elif method == "nutrient_sum":
            threshold = kwargs.get('threshold', 105)
            outlier_mask = detect_food_outliers_by_nutrient_sum(self.data, threshold)

        elif method == "category_norms":
            category_col = kwargs.get('category_col', 'pnns_groups_1')
            outlier_mask = detect_food_outliers_by_category_norms(self.data, category_col)

        else:
            # Use the parent class method for standard statistical methods
            return super().detect_outliers(method, **kwargs)

        self.outlier_masks[method] = outlier_mask
        return outlier_mask

    def detect_all_food_outliers(self) -> pd.Series:
        """
        Detect outliers using all food-specific methods combined.

        Returns:
            Boolean series indicating outliers
        """
        methods = ["nutrient_range", "nutrient_sum", "category_norms"]
        combined_mask = pd.Series(False, index=self.data.index)

        for method in methods:
            outlier_mask = self.detect_outliers(method)
            combined_mask = combined_mask | outlier_mask

        self.outlier_masks["food_combined"] = combined_mask
        return combined_mask
