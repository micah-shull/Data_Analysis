
#-----------------------------------------------------------#
#--------------      DATA CLEANING TOOLKIT     -------------#
#-----------------------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import KNNImputer
from scipy import stats


#----------   1. Initial Data Import and Preliminary Exploration. ------------#


def load_data(filepath):
    return pd.read_csv(filepath)

def explore_data(df):
    # Display the first few rows of the dataframe
    print(df.head())

    # Display dataframe info to understand data types and missing values
    print(df.info())

    # Display basic statistical details
    print(df.describe().T)


def missing_values_summary(df):
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_summary = pd.DataFrame({'Number Missing': missing_values,
                                    'Percentage': missing_percentage})

    data = missing_summary[missing_summary['Percentage']>5].sort_values(
    by='Number Missing', ascending=False)['Number Missing'].reset_index()
    sns.barplot(data,x='Number Missing', y='index', orient='h')
    plt.title('Largest Missing Values')
    plt.ylabel('')
    plt.show()

    print(missing_summary[missing_summary['Number Missing'] > 0].sort_values(
        by='Number Missing',ascending=False))

    return missing_summary

def plot_distributions(df):
    # Select numeric columns only
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        plt.figure(figsize=(10, 4))

        # Plot histogram
        plt.subplot(1, 2, 1)
        df[col].hist(bins=20)
        plt.title(f'Histogram of {col}')

        # Plot boxplot
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')

        plt.tight_layout()
        plt.show()


# -------------          2. Categorical Data Discovery       -------------#

def analyze_column_categories_auto(df, max_unique_values=25):
    """
    Automatically analyzes columns in a DataFrame to determine the unique number of categories
    for each column (both numeric and object types), includes data type information, saves the results
    in a new DataFrame, and creates a bar plot limited to columns with less than a specified number
    of unique categories.

    Parameters:
    - df: Pandas DataFrame to analyze.
    - max_unique_values: Maximum number of unique values a column can have to be included in the plot.

    Returns:
    - category_counts_df: DataFrame with columns 'Column', 'Unique_Values', and 'Data_Type' indicating
      the number of unique values and data type in each column.
    """
    # Adjusted to include object data types
    columns = df.select_dtypes(include=['int64', 'float64', 'object']).columns
    data = []

    for column in columns:
        unique_count = df[column].nunique()
        data_type = df[column].dtype
        data.append({'Column': column, 'Unique_Values': unique_count, 'Data_Type': data_type})

    category_counts_df = pd.DataFrame(data).sort_values(by='Unique_Values', ascending=False)
    # filter by max_unique_values and create list
    category_counts_list = category_counts_df[
        category_counts_df['Unique_Values'] < max_unique_values]['Column'].to_list()

    # Filtering for plotting
    plot_data = category_counts_df[category_counts_df['Unique_Values'] <= max_unique_values]

    # Plotting
    sns.set_style('whitegrid')
    plt.figure(figsize=(10, len(plot_data) * 0.5))
    sns.barplot(x='Unique_Values', y='Column', data=plot_data,
                hue='Unique_Values', palette='crest')
    plt.title(f'Unique Category Counts per Column (<= {max_unique_values} Categories)')
    plt.xlabel('Number of Unique Values')
    plt.ylabel('Columns')
    plt.tight_layout()
    plt.show()
    sns.set()

    return category_counts_df, category_counts_list


def convert_to_categorical(df, columns):
    """
    Converts specified columns in a DataFrame to categorical data types.

    Parameters:
    - df: Pandas DataFrame.
    - columns: List of column names to convert to categorical.

    Returns:
    - A DataFrame with the specified columns converted to categorical types.
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype('category')
        else:
            print(f"Column '{column}' not found in DataFrame.")
    return df


# -----------------   3. Missing Data Identification and Replacement  ---------------#


def plot_missing_data_matrix(df):
    """Visualize the pattern of missingness in the dataframe"""
    msno.matrix(df)
    plt.show()

def plot_missing_data_heatmap(df):
    """Heatmap to show correlations of missingness between columns"""
    msno.heatmap(df)
    plt.show()

def analyze_missingness_correlation(df, column_with_missing, other_column):
    """Compare the distribution of `other_column` for missing vs non-missing in `column_with_missing`."""
    is_missing = df[column_with_missing].isnull()
    sns.boxplot(x=is_missing, y=other_column, data=df)
    plt.title(f"Distribution of {other_column} for Missing vs. Non-Missing in {column_with_missing}")
    plt.show()

# Function to perform mode imputation on all categorical columns
def mode_imputation(df):
    for column in df.select_dtypes(include=['object', 'category']).columns:
        # Calculate the mode of the current column
        mode_value = df[column].mode()[0]

        # Fill missing values with the mode
        df[column].fillna(mode_value, inplace=True)

#--------------------  Outlier Detection and Removal  ------------------#

def detect_outliers_iqr(df):
    outliers_dict = {}

    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter and save outliers for the column
        outliers_df = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        if not outliers_df.empty:
            outliers_dict[column] = outliers_df

    return outliers_dict


def visualize_and_summarize_outliers(outliers_dict, df, orient='h', palette='coolwarm', figsize_multiplier=0.5):
    """
    Visualizes the distribution of columns with outliers in a standardized form using boxplots and
    returns a summary of the outlier analysis.

    Args:
        outliers_dict (dict): Dictionary containing DataFrames of outliers for each numeric column.
        df (pd.DataFrame): The original DataFrame from which outliers were detected.
        orient (str): Orientation of the boxplot, 'h' for horizontal or 'v' for vertical.
        palette (str): Color palette for the boxplots.
        figsize_multiplier (float): Multiplier to adjust the figure size dynamically based on the number of columns.

    Returns:
        pd.DataFrame: A summary DataFrame containing the count and percentage of outliers for each column.
    """
    outlier_cols = list(outliers_dict.keys())

    if not outlier_cols:
        print("No outliers detected in numeric columns.")
        return pd.DataFrame()  # Return an empty DataFrame if no outliers are detected

    # Visualization
    outlier_data_subset = df[outlier_cols].copy()
    std = outlier_data_subset.std()
    mean = outlier_data_subset.mean()
    outlier_data_subset = outlier_data_subset.subtract(mean, axis=1).div(std.replace(0, np.nan), axis=1)

    sns.set_style("whitegrid")
    sns.set_context("notebook")
    plt.figure(figsize=(10, max(1, len(outlier_cols)) * figsize_multiplier))
    sns.boxplot(data=outlier_data_subset, orient=orient, palette=palette)
    plt.title("Boxplot of Features with Outliers (Standardized Data)")
    plt.show()

    # Summary
    summary_data = []
    for col in outlier_cols:
        total_values = df[col].shape[0]
        num_outliers = outliers_dict[col].shape[0]
        percentage_outliers = (num_outliers / total_values) * 100
        summary_data.append({'Column': col, 'Number of Outliers': num_outliers, 'Percentage of Outliers': percentage_outliers})

    summary_df = pd.DataFrame(summary_data)
    summary_df.sort_values(by='Percentage of Outliers', ascending=False, inplace=True)

    return summary_df


def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Only keep rows in dataframe that do not contain any outliers
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_zscore(df, column):
    df = df[(np.abs(stats.zscore(df[column])) < 3)]
    return df

def remove_outliers_iqr_all(df):
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_cols:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    return df


