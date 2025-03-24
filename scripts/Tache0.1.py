import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import scipy.stats as stats

class DatasetSampler:
    def __init__(self, file_path, nrows=100):
        """
        Initializes the DatasetSampler class and loads a limited number of rows from the dataset.

        Parameters:
            file_path (str): Path to the dataset CSV file.
            nrows (int): Number of rows to read from the dataset.
        """
        self.file_path = file_path
        self.nrows = nrows
        self.df = None
        self.filtered_df = None

        print("Initializing DatasetSampler...")

        # Load data
        self.load_data()

    def load_data(self):
        """
        Loads the dataset from the file_path, limiting to the first 'nrows' rows.
        
        This method reads data with tab delimiters, utf-8 encoding, and handles possible date parsing.
        """
        try:
            print(f"Loading data from {self.file_path}...")
            self.df = pd.read_csv(self.file_path, nrows=self.nrows, sep='\t', encoding="utf-8", parse_dates=True)
            self.filtered_df = self.df.copy()  # Maintain the original DataFrame
            print(f"Dataset loaded with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        except Exception as e:
            print(f"Error loading data: {e}")
    
    def random_sampling(self, sample_size):
        """
        Perform random sampling on the entire dataset.
        
        Args:
            sample_size (int): The number of samples to select.
            
        Returns:
            pd.DataFrame: A random sample of the dataset.
        """
        return self.df.sample(n=sample_size, random_state=42)

    def chunk_sampling(self, chunk_size, sample_size):
        """
        Perform chunk sampling on the dataset.
        
        Args:
            chunk_size (int): Number of rows to load at once.
            sample_size (int): The number of samples to select from each chunk.
            
        Returns:
            pd.DataFrame: A concatenated sample of the dataset.
        """
        chunks = pd.read_csv(self.file_path, chunksize=chunk_size)
        sampled_data = []
        for chunk in chunks:
            sampled_chunk = chunk.sample(n=sample_size, random_state=42)
            sampled_data.append(sampled_chunk)
        return pd.concat(sampled_data, ignore_index=True)

    def stratified_sampling(self, target_column, test_size):
        """
        Perform stratified sampling to ensure each class is proportionally represented.
        
        Args:
            target_column (str): The column to stratify on.
            test_size (float): The proportion of the data to sample (between 0 and 1).
            
        Returns:
            pd.DataFrame: A stratified sample of the dataset.
        """
        _, stratified_sample = train_test_split(self.df, test_size=test_size, stratify=self.df[target_column], random_state=42)
        return stratified_sample

    def sampling_with_replacement(self, sample_size):
        """
        Perform sampling with replacement on the dataset.
        
        Args:
            sample_size (int): The number of samples to select.
            
        Returns:
            pd.DataFrame: A sample with replacement from the dataset.
        """
        return self.df.sample(n=sample_size, replace=True, random_state=42)

    def weighted_sampling(self, sample_size, weights_column):
        """
        Perform weighted sampling on the dataset, where each observation has a weight.
        
        Args:
            sample_size (int): The number of samples to select.
            weights_column (str): The column containing the weights for sampling.
            
        Returns:
            pd.DataFrame: A weighted sample from the dataset.
        """
        return self.df.sample(n=sample_size, weights=self.df[weights_column], random_state=42)

    def chi_square_test_on_sample(self, column, expected_probs):
        """
        Perform a Chi-Square test on a sampled dataset to evaluate if the sample
        fits the expected distribution.
        
        Args:
            column (str): The column on which the test is performed.
            expected_probs (dict): The expected probabilities for each category.
            
        Returns:
            tuple: Chi-Square statistic, p-value, degrees of freedom, and expected frequencies.
        """
        observed_frequencies = self.df[column].value_counts().values
        categories = self.df[column].value_counts().index
        expected_frequencies = [expected_probs.get(cat, 0) * len(self.df) for cat in categories]
        
        chi2_stat, p_value = stats.chisquare(observed_frequencies, expected_frequencies)
        
        dof = len(categories) - 1  # Degrees of freedom
        return chi2_stat, p_value, dof, expected_frequencies


# Example usage
if __name__ == "__main__":
    print("Starting the script...")

    # Create an instance of the DatasetSampler class
    dataset_path = "data/en.openfoodfacts.org.products.csv"  # Modify with your file path
    sampler = DatasetSampler(dataset_path, nrows=100)  # Load first 100 rows for testing

    print("Attempting to load data...")
    # Load the dataset
    sampler.load_data()

    if sampler.df is not None:
        print("Data loaded successfully.")
    else:
        print("Failed to load data.")
        exit()

    # Perform random sampling
    print("Performing random sampling...")
    random_sample = sampler.random_sampling(10)  # Sample 10 rows randomly
    print(f"Random Sample Shape: {random_sample.shape}")

    # Perform stratified sampling
    print("Performing stratified sampling...")
    stratified_sample = sampler.stratified_sampling("labels", 0.2)  # Sample 20% of the data based on 'category'
    print(f"Stratified Sample Shape: {stratified_sample.shape}")

    # Perform chi-square test on a sample
    print("Performing Chi-Square test...")
    expected_probs = {'category1': 0.2, 'category2': 0.3, 'category3': 0.5}  # Example probabilities
    chi2_stat, p_value, dof, expected_frequencies = sampler.chi_square_test_on_sample("category", expected_probs)
    print(f"Chi-Square Statistic: {chi2_stat}")
    print(f"P-value: {p_value}")
    print(f"Degrees of Freedom: {dof}")
    print(f"Expected Frequencies: {expected_frequencies}")

    print("Finished executing the script.")
