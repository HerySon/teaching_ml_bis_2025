"""
Main script for the data scaling task on the OpenFoodFacts dataset.
This script provides a command-line interface to run various scaling operations.
"""

try:
    import os
    import argparse
except ImportError as exc:
    raise ImportError('pandas is required for this module') from exc

from .data_scaling import (
    standard_scaler,
    minmax_scaler,
    robust_scaler,
    power_transformer,
    quantile_transformer,
    scale_dataframe
)
from .test_scaling import (
    load_data,
    preprocess_data,
    analyze_distributions,
    compare_scaling_methods,
    evaluate_scaling_method,
    recommend_scaling_methods
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Data scaling for OpenFoodFacts dataset')

    parser.add_argument(
        '--data-url',
        type=str,
        default="https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz",
        help='URL or file path of the dataset'
    )

    parser.add_argument(
        '--nrows',
        type=int,
        default=1000,
        help='Number of rows to load from the dataset'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--operation',
        type=str,
        default='all',
        choices=['analyze', 'compare', 'recommend', 'scale', 'all'],
        help='Operation to perform'
    )

    parser.add_argument(
        '--method',
        type=str,
        default='standard',
        choices=['standard', 'minmax', 'robust', 'power', 'quantile'],
        help='Scaling method to use'
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load and preprocess data
    print(f"Loading data from {args.data_url}...")
    df = load_data(args.data_url, args.nrows)

    print("Preprocessing data...")
    processed_df = preprocess_data(df)

    # Save preprocessed data
    preprocessed_path = os.path.join(args.output_dir, 'preprocessed_data.csv')
    processed_df.to_csv(preprocessed_path, index=False)
    print(f"Preprocessed data saved to {preprocessed_path}")

    # Perform requested operation
    if args.operation in ['analyze', 'all']:
        print("Analyzing data distributions...")
        save_dir = os.path.join(args.output_dir, 'distributions')
        analyze_distributions(processed_df, save_dir=save_dir)
        print(f"Distribution plots saved to {save_dir}")

    if args.operation in ['compare', 'all']:
        print("Comparing scaling methods...")
        save_dir = os.path.join(args.output_dir, 'scaling_comparison')
        compare_scaling_methods(processed_df, save_dir=save_dir)
        print(f"Comparison plots saved to {save_dir}")

    if args.operation in ['recommend', 'all']:
        print("Recommending scaling methods...")
        recommendations = recommend_scaling_methods(processed_df)
        recommendations_path = os.path.join(args.output_dir, 'scaling_recommendations.csv')
        recommendations.to_csv(recommendations_path, index=False)
        print(f"Scaling recommendations saved to {recommendations_path}")

    if args.operation in ['scale', 'all']:
        print("Scaling data...")

        if args.method == 'recommend':
            # First get recommendations
            recommendations = recommend_scaling_methods(processed_df)

            # Group columns by recommended scaling method
            grouped_cols = recommendations.groupby('recommended_scaling')['column'].apply(list).to_dict()

            # Apply recommended scaling methods to each group
            final_scaled_df = processed_df.copy()

            for method, cols in grouped_cols.items():
                if method == 'standard':
                    final_scaled_df[cols] = standard_scaler(processed_df, cols)[cols]
                elif method == 'minmax':
                    final_scaled_df[cols] = minmax_scaler(processed_df, cols)[cols]
                elif method == 'robust':
                    final_scaled_df[cols] = robust_scaler(processed_df, cols)[cols]
                elif method == 'power':
                    final_scaled_df[cols] = power_transformer(processed_df, columns=cols)[cols]
                elif method == 'quantile':
                    final_scaled_df[cols] = quantile_transformer(processed_df, columns=cols)[cols]

            # Save the final scaled dataframe
            scaled_path = os.path.join(args.output_dir, 'scaled_data_recommended.csv')
            final_scaled_df.to_csv(scaled_path, index=False)
            print(f"Scaled data (using recommended methods) saved to {scaled_path}")

        else:
            # Use the specified method for all numeric columns
            scaled_df = scale_dataframe(processed_df, method=args.method)

            # Save the scaled dataframe
            scaled_path = os.path.join(args.output_dir, f'scaled_data_{args.method}.csv')
            scaled_df.to_csv(scaled_path, index=False)
            print(f"Scaled data (using {args.method}) saved to {scaled_path}")

    print("Task completed successfully!")


if __name__ == "__main__":
    main()
