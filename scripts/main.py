"""Testing module for outlier detection and handling functionality."""

try:
    import pandas as pd
    import numpy as np
    from .outlier_handler import OutlierDetector, get_outlier_strategies
    from .food_outlier_detector import FoodOutlierDetector, NUTRIENT_RANGES
except ImportError as exc:
    raise ImportError(
        'pandas, numpy, outlier_handler and food_outlier_detector are required'
    ) from exc


def test_generic_outlier_detection():
    """Test classical outlier detection methods."""
    # Create a sample DataFrame
    np.random.seed(42)
    data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100)
    })

    # Add some outliers
    data.loc[0, 'feature1'] = 10  # outlier
    data.loc[1, 'feature2'] = -8  # outlier

    # Create detector
    detector = OutlierDetector(data)

    # Test different methods
    print("--- Testing generic detection methods ---")
    methods = ["tukey", "zscore", "isolation_forest", "lof"]

    for method in methods:
        outliers = detector.detect_outliers(method=method)
        outlier_count = outliers.sum()
        print(f"Method {method}: {outlier_count} outliers detected")

    # Test handling strategies
    print("\n--- Testing handling strategies ---")
    strategies = get_outlier_strategies()

    for strategy_name, description in strategies.items():
        print(f"Strategy: {strategy_name} ({description})")
        result = detector.handle_outliers("tukey", strategy_name)
        if strategy_name == "remove":
            print(
                f"  Remaining rows: {len(result)} "
                f"(removed {len(data) - len(result)} rows)"
            )
        elif strategy_name.startswith("impute_"):
            # Check if outliers have been handled
            original_value = data.loc[0, 'feature1']
            new_value = result.loc[0, 'feature1']
            print(f"  Replacement example: {original_value:.2f} -> {new_value:.2f}")


def test_food_outlier_detection():
    """Test food-specific outlier detection methods."""
    # Create a DataFrame simulating OpenFoodFacts data
    data = pd.DataFrame({
        'pnns_groups_1': ['Snacks'] * 50 + ['Beverages'] * 50,
        'fat_100g': np.random.uniform(0, 30, 100),
        'proteins_100g': np.random.uniform(0, 15, 100),
        'carbohydrates_100g': np.random.uniform(0, 60, 100),
        'sugars_100g': np.random.uniform(0, 30, 100),
        'salt_100g': np.random.uniform(0, 2, 100),
        'energy_100g': np.random.uniform(0, 2000, 100)
    })

    # Add outliers
    # 1. Impossible nutritional value
    data.loc[0, 'fat_100g'] = 120  # > 100g for 100g is impossible

    # 2. Sum of nutrients > 100g
    data.loc[1, 'fat_100g'] = 60
    data.loc[1, 'proteins_100g'] = 30
    data.loc[1, 'carbohydrates_100g'] = 30

    # 3. Value very different from others in the same category
    data.loc[2, 'energy_100g'] = 5000  # Very high energy value

    # Create detector
    detector = FoodOutlierDetector(data)

    # Test food-specific methods
    print("\n--- Testing food-specific detection methods ---")
    methods = ["nutrient_range", "nutrient_sum", "category_norms"]

    for method in methods:
        outliers = detector.detect_outliers(method=method)
        outlier_count = outliers.sum()
        print(f"Method {method}: {outlier_count} outliers detected")

    # Test combined detection
    outliers = detector.detect_all_food_outliers()
    print(f"Combined method: {outliers.sum()} outliers detected")

    # Test handling strategies
    print("\n--- Testing handling strategies for food data ---")
    strategies = ["keep", "remove", "impute_median"]

    for strategy_name in strategies:
        result = detector.handle_outliers("food_combined", strategy_name)
        if strategy_name == "remove":
            print(
                f"Strategy {strategy_name}: {len(result)} remaining rows "
                f"(removed {len(data) - len(result)} rows)"
            )
        elif strategy_name == "impute_median":
            # Check if outliers have been handled
            original_value = data.loc[0, 'fat_100g']
            new_value = result.loc[0, 'fat_100g']
            print(
                f"Strategy {strategy_name}: "
                f"fat_100g example {original_value:.2f} -> {new_value:.2f}"
            )
        else:
            print(f"Strategy {strategy_name}: {len(result)} rows")


def print_nutrient_ranges():
    """Display the normal ranges for nutrients."""
    print("\n--- Nutrient value ranges ---")
    print(f"{'Nutrient':<25} {'Min':<10} {'Max':<10}")
    print("-" * 50)
    for nutrient, (min_val, max_val) in sorted(NUTRIENT_RANGES.items()):
        print(f"{nutrient:<25} {min_val:<10} {max_val:<10}")


if __name__ == "__main__":
    print("Testing outlier detection and handling module")
    print("=" * 60)
    test_generic_outlier_detection()
    test_food_outlier_detection()
    print_nutrient_ranges()
    print("\nTests completed.")
