import pandas as pd

class DataFrameFilter:
    """
    This class helps to perform various data filtering tasks on a DataFrame, including:

    1) Data loading
    2) Understanding the data
    3) Column-based filtering:
        3.1) Filtering based on missing values
        3.2) Filtering based on unique values
        3.3) Filtering by selecting one or more specific columns
        3.4) Filtering by selecting a specific data type (datatype)
    4) Row-based filtering:
        4.1) Filtering based on missing values
        4.2) Filtering based on a specific column and value
        4.3) Filtering based on the absence of essential values for data analysis
    """

    def __init__(self, file_path, nrows=100):
        """
        Initializes the class with the dataset.

        Parameters:
            file_path (str): Path to the CSV dataset.
            nrows (int): Number of rows to read from the dataset.
        """
        self.df = pd.read_csv(file_path, nrows=nrows, sep='\t', encoding="utf-8", parse_dates=True)
        self.filtered_df = self.df.copy()  # To maintain original DataFrame
        print(f"Dataset loaded with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")

    def get_basic_info(self):
        """
        Prints basic information about the dataset, such as data types and missing values.
        """
        print("Dataset Info:")
        self.df.info()
        
        print("\nDescriptive Statistics:")
        print(self.df.describe())

    def filter_missing_values(self, threshold=0.3):
        """
        Filters columns based on the percentage of non-null values.

        Parameters:
            threshold (float): Minimum percentage of non-null values per column to keep.
        """
        print(f"\nFiltering columns with less than {threshold * 100}% non-null values...")
        self.filtered_df = self.df.loc[:, self.df.notna().mean() > threshold]
        print(f"Remaining columns after filtering: {self.filtered_df.shape[1]}")

    def filter_unique_values(self):
        """
        Filters columns with more than 1 unique value.

        Excludes columns with only 1 unique value.
        """
        print("\nFiltering columns with only 1 unique value...")
        nunique = self.filtered_df.nunique()
        self.filtered_df = self.filtered_df.loc[:, nunique > 1]
        print(f"Remaining columns after filtering: {self.filtered_df.shape[1]}")

    def filter_missing_values_by_row(self, threshold=0.3):
        """
        Filters rows based on the percentage of non-null values.

        Parameters:
            threshold (float): Minimum percentage of non-null values per row to keep.
        """
        print(f"\nFiltering rows with less than {threshold * 100}% non-null values...")
        columns_number = self.filtered_df.shape[1]
        self.filtered_df = self.filtered_df.dropna(thresh=int(threshold * columns_number))
        print(f"Remaining rows after filtering: {self.filtered_df.shape[0]}")

    def focus_on_column_value(self, column, value):
        """
        Focuses on filtering rows based on a specific value in a column.

        Parameters:
            column (str): The column name to filter by.
            value (str): The value to filter for in the column.
        """
        print(f"\nFiltering rows where '{column}' contains '{value}'...")
        self.filtered_df = self.filtered_df[self.filtered_df[column].str.contains(value, na=False)]
        print(f"Remaining rows after filtering: {self.filtered_df.shape[0]}")

    def exclude_invalid_products(self, required_columns=["product_name", "code"]):
        """
        Excludes rows with missing values in critical columns such as 'product_name' and 'code'.

        Parameters:
            required_columns (list): List of columns to check for non-null values.
        """
        print(f"\nExcluding rows with missing values in required columns: {required_columns}...")
        self.filtered_df = self.filtered_df.dropna(subset=required_columns)
        print(f"Remaining rows after excluding invalid products: {self.filtered_df.shape[0]}")

    def show_filtered_data(self):
        """
        Prints the first 5 rows of the filtered DataFrame.
        """
        print("\nFiltered DataFrame (First 5 rows):")
        print(self.filtered_df.head())

    def get_filtered_column_info(self):
        """
        Prints the column names and the number of unique values for each column in the filtered DataFrame.
        """
        print("\nFiltered DataFrame Column Info:")
        print(self.filtered_df.nunique())


# Example usage:
if __name__ == "__main__":
    # Initialize the DataFrameFilter class with the dataset path
    file_path = "../data/en.openfoodfacts.org.products.csv"
    data_filter = DataFrameFilter(file_path, nrows=100)

    # Show basic information about the dataset
    data_filter.get_basic_info()

    # Apply filters to the DataFrame
    data_filter.filter_missing_values(threshold=0.3)  # Remove columns with more than 70% missing values
    data_filter.filter_unique_values()  # Remove columns with only one unique value
    data_filter.filter_missing_values_by_row(threshold=0.3)  # Remove rows with more than 70% missing values

    # Focus on a specific value in a column (example: 'labels' column)
    data_filter.focus_on_column_value(column="labels", value="Point Vert, Fabriqué en France")

    # Exclude rows with missing critical columns
    data_filter.exclude_invalid_products(required_columns=["product_name", "code"])

    # Show the filtered data
    data_filter.show_filtered_data()

    # Get column information of the filtered data
    data_filter.get_filtered_column_info()
