import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
import category_encoders as ce

class DataEncoder:
    """
    This class provides methods for loading and understanding a dataset, followed by various filtering techniques:
    
    1. Loading the data.
    2. Exploring and understanding the data.
    3. Column filtering:
        - 3.1) Filtering based on missing values.
        - 3.2) Filtering based on unique values.
        - 3.3) Filtering by selecting one or more specific columns.
        - 3.4) Filtering based on a specific data type (datatype).
    4. Row filtering:
        - 4.1) Filtering based on missing values.
        - 4.2) Filtering based on a specific column and value.
        - 4.3) Filtering based on the absence of a required value for data analysis.
    """
    
    def __init__(self, data_path, nrows=100):
        """
        Initialize the DataEncoder class with the path to the dataset and the number of rows to load.

        :param data_path: Path to the dataset file
        :param nrows: Number of rows to load from the dataset (default is 100)
        """
        self.data_path = data_path
        self.nrows = nrows
        self.df = None
        self.df_filtered = None
        self.cat_cols = None
        self.ordinal_data = None
        self.non_ordinal_data = None
        
    def load_data(self):
        """
        Load the dataset from the specified path and get a sample of the data.
        """
        self.df = pd.read_csv(self.data_path, nrows=self.nrows, sep='\t', encoding="utf-8", parse_dates=True)
        print(f"Data loaded successfully with {len(self.df)} rows.")
        return self.df

    def filter_data(self, threshold=0.3):
        """
        Filter out columns with more than a certain percentage of missing values.
        
        :param threshold: Minimum percentage of non-null values to keep a column (default 30%)
        """
        self.df_filtered = self.df.loc[:, self.df.notna().mean() > threshold]
        print(f"Data filtered: {len(self.df_filtered.columns)} columns remaining.")
        return self.df_filtered

    def get_categorical_columns(self):
        """
        Select only the categorical columns (of type 'object') from the filtered data.
        """
        self.cat_cols = self.df_filtered.select_dtypes(include=['object'])
        print(f"Categorical columns selected: {self.cat_cols.columns.tolist()}")
        return self.cat_cols

    def calculate_cardinality(self):
        """
        Calculate the cardinality (number of unique values) for each categorical column.
        """
        cardinality_cat = self.cat_cols.nunique(dropna=False).sort_values(ascending=False)
        return cardinality_cat

    def group_columns_by_cardinality(self, low_threshold=10, high_threshold=80):
        """
        Group categorical columns by their cardinality.

        :param low_threshold: Maximum number of unique values for low cardinality columns (default 10)
        :param high_threshold: Maximum number of unique values for medium cardinality columns (default 80)
        """
        cardinality_cat = self.calculate_cardinality()
        
        low_cardinality = cardinality_cat[cardinality_cat <= low_threshold]
        medium_cardinality = cardinality_cat[(cardinality_cat > low_threshold) & (cardinality_cat <= high_threshold)]
        high_cardinality = cardinality_cat[cardinality_cat > high_threshold]

        return low_cardinality, medium_cardinality, high_cardinality

    def identify_unique_value_columns(self):
        """
        Identify columns with only unique values and store them in a list.
        """
        cols_with_unique_values = [col for col in self.cat_cols.columns if self.cat_cols[col].nunique(dropna=False) == len(self.cat_cols[col])]
        print("Columns with only unique values:", cols_with_unique_values)
        return cols_with_unique_values

    def plot_top_frequent_categories(self, col):
        """
        Plot the top 10 most frequent categories for a given categorical column.

        :param col: The column name to plot
        """
        counts = self.cat_cols[col].value_counts().head(10)
        plt.figure(figsize=(10, 5))
        counts.plot(kind='barh', color='skyblue')
        plt.xlabel("Number of occurrences")
        plt.ylabel("Categories")
        plt.title(f"Top 10 Values in {col}")
        plt.gca().invert_yaxis()  # Reverse the order for readability
        plt.show()

    def detect_rare_categories(self, threshold=0.02):
        """
        Detect categories that appear in less than a certain percentage of the data (e.g., 2%).
        
        :param threshold: Rarity threshold (default 2%)
        """
        rare_categories = {}
        for col in self.cat_cols:
            rare_values = self.cat_cols[col].value_counts(normalize=True)[self.cat_cols[col].value_counts(normalize=True) < threshold]
            if not rare_values.empty:
                rare_categories[col] = rare_values
        return rare_categories

    def handle_rare_categories(self, col, threshold=0.01):
        """
        Handle rare categories by replacing them with a common "Other" category.
        
        :param col: The column name to handle rare categories in
        :param threshold: The threshold for considering categories as rare (default 1%)
        """
        counts = self.df[col].value_counts(normalize=True)
        rare_values = counts[counts < threshold].index
        self.df[col] = self.df[col].replace(rare_values, "Other")
        print(f"Rare categories in '{col}' handled and replaced with 'Other'.")
        return self.df

    def separate_ordinal_and_nonordinal(self):
        """
        Separate the columns into ordinal and non-ordinal categorical columns.
        """
        ordinal_cols = []
        non_ordinal_cols = []

        # Example: Assuming 'datetime' or 'grade' columns are ordinal
        for col in self.cat_cols.columns:
            if 'datetime' in col or 'grade' in col:
                ordinal_cols.append(col)
            else:
                non_ordinal_cols.append(col)

        # Extract ordinal and non-ordinal data
        self.ordinal_data = self.df[ordinal_cols]
        self.non_ordinal_data = self.df[non_ordinal_cols]

        print("Ordinal and non-ordinal columns separated.")
        return self.ordinal_data, self.non_ordinal_data

    def encode_ordinal_data(self):
        """
        Encode the ordinal data using OrdinalEncoder from scikit-learn.
        """
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        ordinal_data_transform = oe.fit_transform(self.ordinal_data)
        return pd.DataFrame(ordinal_data_transform, columns=self.ordinal_data.columns)

    def encode_non_ordinal_data(self):
        """
        Encode the non-ordinal data using CountEncoder from the category_encoders library.
        """
        encoder = ce.CountEncoder(cols=self.non_ordinal_data.columns, handle_unknown='value', handle_missing='value')
        df_encoded = encoder.fit_transform(self.non_ordinal_data)
        return df_encoded

    def combine_encoded_data(self):
        """
        Combine the encoded ordinal and non-ordinal data into one DataFrame.
        """
        ordinal_encoded_df = self.encode_ordinal_data()
        non_ordinal_encoded_df = self.encode_non_ordinal_data()

        # Combine both encoded dataframes
        df_combined = pd.concat([non_ordinal_encoded_df, ordinal_encoded_df], axis=1)
        print("Encoded data combined successfully.")
        return df_combined

    def save_combined_data(self, filename="df_combined.csv"):
        """
        Save the combined encoded data to a CSV file.
        
        :param filename: The name of the output CSV file
        """
        df_combined = self.combine_encoded_data()
        df_combined.to_csv(filename, index=False)
        print(f"Combined data saved to {filename}.")

# Example usage:
if __name__ == "__main__":
    # Initialize the DataEncoder class with the dataset path and number of rows to sample
    data_path = "data/en.openfoodfacts.org.products.csv"
    encoder = DataEncoder(data_path, nrows=100)

    # Load data
    encoder.load_data()

    # Filter data
    encoder.filter_data()

    # Get categorical columns
    encoder.get_categorical_columns()

    # Group columns by cardinality
    low_cardinality, medium_cardinality, high_cardinality = encoder.group_columns_by_cardinality()

    # Identify columns with only unique values
    encoder.identify_unique_value_columns()

    # Plot top frequent categories for 'brands' column
    encoder.plot_top_frequent_categories('brands')

    # Detect rare categories
    rare_categories = encoder.detect_rare_categories()

    # Handle rare categories in 'brands' column
    encoder.handle_rare_categories('brands')

    # Separate ordinal and non-ordinal columns
    ordinal_data, non_ordinal_data = encoder.separate_ordinal_and_nonordinal()

    # Encode and combine the data
    encoder.save_combined_data("df_combined.csv")
