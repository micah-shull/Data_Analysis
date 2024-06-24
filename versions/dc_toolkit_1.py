#-----------------------------------------------------------#
#--------------      DATA CLEANING TOOLKIT     -------------#
#-----------------------------------------------------------#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix

    
#-------------- DATA CHECK AND CLEANING ------------#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def comprehensive_data_checks(df):
    """
    Performs comprehensive data checks on a Pandas DataFrame, including:
        - Missing values
        - Duplicate rows
        - Data types
        - Outlier detection
        - Descriptive statistics
        - Visualizations

    Args:
        df (pd.DataFrame): The DataFrame to be checked.

    Returns:
        pd.DataFrame: The original DataFrame with additional information appended.
        pd.DataFrame: A DataFrame containing all duplicate rows.
        dict: A dictionary containing DataFrames of outliers for each numeric column.
    """

    def iqr_outlier_detection(data, col):
        """Identifies outliers in a numeric column using the IQR method."""
        q1 = data[col].quantile(0.25)
        q3 = data[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        return data[col][(data[col] < lower_bound) | (data[col] > upper_bound)]

    print(f"\nTotal missing values: {df.isnull().sum().sum()}")

    duplicate_count = len(df[df.duplicated()])
    print(f"Total duplicate rows: {duplicate_count}")

    duplicate_data = df[df.duplicated()]

    outlier_data = {}
    for col in df.select_dtypes(include=[np.number]):
        outliers = iqr_outlier_detection(df, col)
        if len(outliers) > 0:
            outlier_data[col] = outliers

    # Add visualizations
    # Set the text size globally
    plt.rcParams.update({'font.size': 8})

    # Now create your plot
    df.hist(figsize=((df.shape[1] // 3)+2,(df.shape[1] // 3)+2))
    plt.show()

    # Combine outlier boxplots into a single wide Seaborn boxplot
    outlier_cols = list(outlier_data.keys())
    outlier_data_subset = df[outlier_cols].copy()  # Create a copy to avoid modifying original DataFrame

    # Standardize the outlier data before plotting
    outlier_data_subset = (outlier_data_subset - outlier_data_subset.mean()) / outlier_data_subset.std()

    sns.set_style("ticks")
    sns.set_context("talk")
    plt.figure(figsize=(7, len(outlier_cols)//2))
    sns.boxplot(data=outlier_data_subset, orient="h", palette="Set2")
    plt.title("Combined Boxplot of Features with Outliers (Standardized Data)")
    plt.show()

    if len(outlier_data) > 0:
        # Create a summary DataFrame
        outlier_summary = pd.DataFrame({'Column': list(outlier_data.keys()),
                                         'Number of Outliers': [len(outliers) for outliers in outlier_data.values()]})
        outlier_summary = outlier_summary.sort_values(by='Number of Outliers', ascending=False)

        # Create the horizontal barplot with column names and adjusted font size
        plt.figure(figsize=(7, len(outlier_cols)//3))
        plt.barh(outlier_summary['Column'], outlier_summary['Number of Outliers'], color='lightblue')
        plt.xlabel('Number of Outliers')
        plt.title('Outlier Summary for Columns')
        plt.subplots_adjust(left=0.3) # Adjust spacing for labels
        plt.tick_params(axis='y', which='major', labelsize=8)  # Adjust label size as needed
        plt.show()

        print("\nSummary of Outlier Columns:")
        print(outlier_summary.to_string())

    # reset params
    sns.set()

    return df, duplicate_data, outlier_data

#-------------- DATA CHECK AND CLEANING ------------#
# ------------- ADD OUTLIER TAGS TO DATA -----------#

def add_outlier_flags(df, outlier_data):
    """Adds outlier flags to the original DataFrame based on outlier data.

    Args:
        df (pd.DataFrame): The original DataFrame.
        outlier_data (dict): Dictionary of outlier DataFrames for each column.

    Returns:
        pd.DataFrame: The original DataFrame with an additional "outlier" column.
    """

    for col, outliers in outlier_data.items():
        df["outlier"] = df.index.isin(outliers.index)  # True for outlier rows

    for col, outliers in outlier_data.items():
        df.loc[:, "outlier"] = df.index.isin(outliers.index) 

    return df

#--------------         DATA CHECK AND CLEANING     ------------#
# ------------ REMOVE OUTLIER RETURN CLEANED DATAFRAME  --------#

def handle_outliers(df, outlier_data):
    """Removes outliers, provides descriptive stats, creates side-by-side boxplots, and returns the cleaned data.

    Args:
        df (pd.DataFrame): The original DataFrame with the "outlier" column.
        outlier_data (dict): Dictionary of outlier DataFrames for each column.

    Returns:
        pd.DataFrame: The cleaned DataFrame with outliers removed.
    """

    outlier_cols = list(outlier_data.keys())

    # Remove outlier rows
    df_cleaned = df[~df["outlier"]].copy()
    # Remove outlier Column
    df_cleaned = df_cleaned.drop('outlier',axis=1)

    # Standardize data before plotting
    df_standardized = (df[outlier_cols] - df[outlier_cols].mean()) / df[outlier_cols].std()
    df_cleaned_standardized = (df_cleaned[outlier_cols] - df_cleaned[outlier_cols].mean()) / df_cleaned[outlier_cols].std()

    # Side-by-side boxplots using separate plots
    sns.set_style("ticks")
    sns.set_context("talk")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, len(outlier_cols)//2))  # Create a figure with 2 subplots
    fig.suptitle("Boxplots of Standardized Features Before and After Outlier Removal")

    sns.boxplot(
        data=df_standardized,
        orient="h",
        ax=ax1,
        palette="Set2",
        showmeans=True,
    )
    ax1.set_title("Before Removal")

    sns.boxplot(
        data=df_cleaned_standardized,
        orient="h",
        ax=ax2,
        palette="Set2",
        showmeans=True,
    )
    ax2.set_title("After Removal")
    ax2.set_ylabel("")  # Remove y-axis label for the "After Removal" plot
    ax2.get_yaxis().set_visible(False)  # Also hide the y-axis ticks
    plt.show()

    print('shape of original dataframe =', df.shape)
    print('shape after outlier removal =', df_cleaned.shape)
    print('number of rows removed =', len(outlier_data))

    return df_cleaned


#-------------- DATA CHECK AND CLEANING ------------#
# --------------- MISSING VALUE REMOVAL    ---------#

def impute_missing_values(df, numeric_strategy='mean', categorical_strategy='most_frequent'):
    """Imputes missing values in numerical and categorical columns separately, preserving all columns.

    Args:
        df (pd.DataFrame): The DataFrame to impute.
        numeric_strategy (str, optional): Strategy for numerical columns. Defaults to 'mean'.
            The numeric_strategy parameter accepts either 'mean' or 'median'
        categorical_strategy (str, optional): Strategy for categorical columns. Defaults to 'most_frequent'.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """

    # Preserve original column order
    original_columns = df.columns

    # Separate numerical and categorical columns
    numeric_df = df.select_dtypes(include=[np.number])
    categorical_df = df.select_dtypes(exclude=[np.number])

    # Impute numerical columns
    numeric_imputer = SimpleImputer(strategy=numeric_strategy)
    numeric_df_imputed = pd.DataFrame(numeric_imputer.fit_transform(numeric_df), columns=numeric_df.columns)

    # Impute categorical columns
    categorical_imputer = SimpleImputer(strategy=categorical_strategy)
    categorical_df_imputed = pd.DataFrame(categorical_imputer.fit_transform(categorical_df), columns=categorical_df.columns)

    # Combine imputed DataFrames, preserving original column order
    df_imputed = pd.concat([numeric_df_imputed, categorical_df_imputed], axis=1)[original_columns]
    return df_imputed

#--------------    DATA CHECK AND CLEANING     ------------#
# ------- CATEGORICAL AND ORDINAL DATA IDENTIFICATON ------#

def analyze_categorical_counts(df):
    """Analyzes categorical variable category counts and creates a bar plot.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        tuple: A tuple containing:
            - cats (pd.DataFrame): DataFrame with variable names and category counts.
            - ax (matplotlib.axes.Axes): The generated bar plot.
    """

    categorical_columns = df.select_dtypes(exclude='number').columns
    ords = categorical_columns.to_list()
    counts = [len(df[c].value_counts()) for c in ords]

    cats = pd.DataFrame(zip(ords, counts), columns=['Variable', 'Category Count'])
    cats = cats.sort_values(by=['Category Count'], ascending=False)

    # Create the bar plot
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(data=cats, x='Category Count', y='Variable', orient='h',
                     hue='Category Count', palette='Greens', legend=False)
    plt.title('Categorical Variable Category Counts')
    plt.subplots_adjust(left=0.4)  # Adjust spacing for labels
    plt.tick_params(axis='y', which='major', labelsize=8)  # Adjust label size
    plt.ylabel('')

    return cats, ax

#--------------    DATA CHECK AND CLEANING     ------------#
#--------------     CORRELATION DATA PREP  ----------------#

import pandas as pd

def convert_categorical_to_numeric(df, cat_columns):
    """
    Converts categorical columns in a DataFrame to numeric values.

    Args:
        df (pd.DataFrame): The DataFrame to modify.
        cat_columns (list): A list of column names to convert.

    Returns:
        pd.DataFrame: A copy of the DataFrame with categorical columns converted.
    """

    cat_to_num = {}
    for col in cat_columns:
        unique_cats = df[col].unique()
        num_codes = range(len(unique_cats))
        cat_to_num[col] = dict(zip(unique_cats, num_codes))

    df_copy = df.copy()
    df_copy = df_copy.replace(cat_to_num)

    return df_copy



#-----------------   Template   -------------#

# df = pd.read_csv('/content/sample_data/california_housing_train.csv')
# df, duplicate_data, outlier_data = dc.comprehensive_data_checks(df)
# # outlier percentage
# for key in outlier_data.keys():
#   print(key, round((len(outlier_data[key])/df.shape[0]),3)*100, "%")
# # add outlier flag
# df = dc.add_outlier_flags(df, outlier_data)
# print('Percentage of OUtliers = ',df[df['outlier']== True].shape[0]/df.shape[0])

# # # Visualize Outlier Distributions
# # outlier_cols = list(outlier_data.keys())
# # df[outlier_cols].hist(figsize=(12,12),bins=50,color=color)
# # plt.show()
# # # Visualize Non Outlier Distributions
# # not_outliers = [i for i in df.columns if i not in outlier_cols]
# # df[not_outliers].hist(figsize=(12,9),bins=50,color=color)
# # plt.show()

# # Remove outliers and outlier tag
# df = df[df['outlier']==False]
# df = df.drop('outlier', axis=1)

# # view distributions after outlier removal
# df.hist(figsize=(15,15),bins=50,color=color)
# plt.show()










