import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

class FeatureSelection:
    """
    A class to perform feature selection on a large dataset, focusing only on numerical features.
    
    Methods:
        __init__(self, data: pd.DataFrame, target: str): Initializes the class with the dataset and the target variable.
        remove_correlated_features(self, correlation_threshold: float): Removes features with high correlation above the threshold.
        apply_statistical_tests(self, k: int): Selects the top k features using statistical tests (e.g., chi-squared test).
        model_based_selection(self, n_estimators: int = 100, max_features: str = 'auto'): Selects features using a model-based approach (RandomForest).
        visualize_feature_importance(self): Plots a bar chart of feature importances.
    """
    
    def __init__(self, data: pd.DataFrame, target: str):
        """
        Initializes the FeatureSelection class with the dataset and target variable.
        
        Parameters:
            data (pd.DataFrame): The dataset to be analyzed.
            target (str): The name of the target variable (dependent variable).
        """
        self.data = data
        self.target = target
        # Selecting only numerical columns for feature selection
        self.X = data.select_dtypes(include=[np.number]).drop(columns=[target])  # Independent numerical features
        self.y = data[target]  # Target variable
        self.selected_features = self.X.columns.tolist()

    def remove_correlated_features(self, correlation_threshold: float = 0.9):
        """
        Removes features that are highly correlated with others above a specified threshold.
        
        Parameters:
            correlation_threshold (float): The correlation value above which features will be removed.
            
        Returns:
            pd.DataFrame: The dataset with uncorrelated features.
        """
        corr_matrix = self.X.corr()
        # Get the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Find index of features with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
        print(f"Dropping features with correlation above {correlation_threshold}: {to_drop}")
        self.X = self.X.drop(columns=to_drop)
        self.selected_features = self.X.columns.tolist()

    def apply_statistical_tests(self, k: int = 10):
        """
        Applies statistical tests to select the top k features based on their relationship with the target.
        
        Parameters:
            k (int): The number of top features to select based on the statistical test (e.g., chi-squared).
            
        Returns:
            pd.DataFrame: The dataset with the selected top k features.
        """
        # Perform a train-test split to apply chi-squared test
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        # Standardize the data before applying the chi-squared test
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Select the k best features based on the chi-squared test
        selector = SelectKBest(score_func=chi2, k=k)
        selector.fit(X_train_scaled, y_train)
        
        # Get the selected features
        selected_columns = self.X.columns[selector.get_support()]
        print(f"Selected top {k} features based on statistical test: {selected_columns}")
        self.X = self.X[selected_columns]
        self.selected_features = selected_columns.tolist()

    def model_based_selection(self, n_estimators: int = 100, max_features: str = 'auto'):
        """
        Selects important features using a model-based approach (e.g., RandomForest).
        
        Parameters:
            n_estimators (int): The number of trees in the random forest model.
            max_features (str or int): The number of features to consider when looking for the best split.
            
        Returns:
            pd.DataFrame: The dataset with selected important features.
        """
        # Train a RandomForest model to rank features based on importance
        model = RandomForestClassifier(n_estimators=n_estimators, max_features=max_features, random_state=42)
        model.fit(self.X, self.y)
        
        # Get the importance of each feature
        feature_importances = model.feature_importances_
        
        # Get indices of features sorted by importance
        important_indices = np.argsort(feature_importances)[::-1]
        
        # Select top features based on importance
        num_top_features = 10  # Select top 10 features by default
        top_features = self.X.columns[important_indices[:num_top_features]]
        print(f"Top {num_top_features} features selected based on model: {top_features}")
        self.X = self.X[top_features]
        self.selected_features = top_features.tolist()

    def visualize_feature_importance(self):
        """
        Visualizes feature importance based on the model (RandomForest).
        
        This method should be called after model_based_selection to plot the feature importance.
        """
        if not hasattr(self, 'X') or self.X.empty:
            print("No features to visualize.")
            return
        
        model = RandomForestClassifier(n_estimators=100, max_features='auto', random_state=42)
        model.fit(self.X, self.y)
        
        # Create a DataFrame of feature importances
        importance_df = pd.DataFrame({
            'Feature': self.X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        
        # Plot feature importances
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance from Random Forest Model')
        plt.show()


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv('data/en.openfoodfacts.org.products.csv', sep="\t", on_bad_lines='skip', 
                    nrows=100000, low_memory=False)

    # Instantiate the FeatureSelection class
    fs = FeatureSelection(df, target='serving_quantity')

    # Remove correlated features
    fs.remove_correlated_features(correlation_threshold=0.9)

    # Apply statistical tests to select top 10 features
    fs.apply_statistical_tests(k=10)

    # Perform model-based feature selection
    fs.model_based_selection(n_estimators=100, max_features='auto')

    # Visualize the feature importance
    fs.visualize_feature_importance()
