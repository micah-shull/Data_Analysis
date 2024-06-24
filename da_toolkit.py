

#-----------------------------------------------------------#
#--------------      DATA ANALYSIS TOOLKIT     -------------#
#-----------------------------------------------------------#


# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import KNNImputer
from scipy import stats


# ---------- UNIVARIATE ANALYSIS - NUMERIC FEATURES ---------------#

def visualize_numerical_features(df, numerical_columns,color='green'):
    """
    Visualizes distributions of numerical features using histograms and boxplots.

    Args:
    - df (pd.DataFrame): The DataFrame containing the data.
    - numerical_columns (list): List of names of the numerical columns to visualize.
    """
    for column in numerical_columns:
        sns.set_context('talk')
        sns.set_style('whitegrid')
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))

        # Histogram
        sns.histplot(df[column], kde=True, ax=ax[0],color=color)
        ax[0].set_title(f'Histogram of {column}')

        # Boxplot
        sns.boxplot(x=df[column], ax=ax[1],color=color)
        ax[1].set_title(f'Boxplot of {column}')

        plt.tight_layout()
        plt.show()


# ---------- UNIVARIATE ANALYSIS - CATEGORICAL FEATURES ---------------#

def visualize_categorical_features(df, categorical_columns, color='yellowgreen'):
    """
    Visualizes distributions of categorical features using bar charts.

    Args:
    - df (pd.DataFrame): The DataFrame containing the data.
    - categorical_columns (list): List of names of the categorical columns to visualize.
    """
    for column in categorical_columns:
        sns.set_style('whitegrid')
        plt.figure(figsize=(10, 4))
        total = float(len(df[column]))  # for calculating percentages

        ax = sns.countplot(x=df[column], order = df[column].value_counts().index, color=color)
        ax.set_title(f'Distribution of {column}')

        # Annotate percentage
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2., height + 3, '{:1.2f}%'.format(100*height/total), ha="center")

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        sns.set() # reset the plot params to default


# ---------- BIVARIATE ANALYSIS - CORRELATIONS ---------------#

def plot_correlation_matrix(df, method='pearson', cmap=sns.color_palette("Spectral", as_cmap=True)):
    """
    Plots a heatmap of the correlation matrix for numerical features in the DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame containing the data.
    - method (str): Method of correlation, default is 'pearson'. Other options include 'kendall' and 'spearman'.
    """
    corr = df.select_dtypes(include=['float64', 'int64']).corr(method=method)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, cbar_kws={'shrink': .8})
    plt.title("Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



# ---------- UNIVARIATE ANALYSIS -CATEGORICAL ---------------#

def plot_categorical_comparisons(df, categorical_column, numerical_column,
                                 plot_type='box',palette='crest',legend=True):
    """
    Explores relationships between a categorical and a numerical column using specified plot types.

    Args:
    - df (pd.DataFrame): The DataFrame containing the data.
    - categorical_column (str): The name of the categorical column.
    - numerical_column (str): The name of the numerical column.
    - plot_type (str): Type of plot to generate. Options: 'stacked', 'violin', 'box'. Default is 'box'.
    """
    if plot_type == 'stacked':
        # Note: Implementing a true "stacked" bar chart for a single numerical comparison might not be straightforward.
        # Consider using a count plot or pivot table for aggregations if going this route.
        pass  # Placeholder for custom implementation
    elif plot_type == 'violin':
        sns.set_style('whitegrid')
        sns.violinplot(x=categorical_column, y=numerical_column, data=df, legend=legend,
                       hue=categorical_column, palette=palette)
    elif plot_type == 'box':
        sns.set_style('whitegrid')
        sns.boxplot(x=categorical_column, y=numerical_column, data=df, legend=legend,
                    hue=categorical_column, palette=palette)
    else:
        print(f"Plot type '{plot_type}' not recognized. Please choose 'stacked', 'violin', or 'box'.")

    plt.title(f"{plot_type.title()} Plot: {categorical_column} vs. {numerical_column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    sns.set()



# ---------------- FEATURE ENGINEERING - DATA BINNING ----------------------#


def generate_bins(df, feature, start_adjust=0, stop_adjust=0, step=5):
    """
    Generates bins for a given feature in a DataFrame based on start, stop, and step values,
    with adjustments to the start and stop values.

    Args:
    - df (pd.DataFrame): The DataFrame containing the data.
    - feature (str): The name of the feature (column) to bin.
    - start_adjust (int): Adjustment to the starting value of the bins.
    - stop_adjust (int): Adjustment to the stopping value of the bins.
    - step (int): The step size between bins.

    Returns:
    - list: A list of bin edges.
    """
    # Ensure the feature exists in the DataFrame
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame.")

    # Calculate start and stop based on the feature's min and max, adjusted as specified
    start = df[feature].min().astype('int') + start_adjust
    stop = df[feature].max().astype('int') + stop_adjust

    # Generate bins
    bins = list(range(start, stop, step))

    # Print proposed bins
    print("Proposed bins:", bins)

    return bins


def bin_data_from_dict(df, feature_bins_dict):
    """
    Bins data for specified features based on provided bins and adds new columns to the DataFrame
    with binned data labeled according to the bins.

    Args:
    - df (pd.DataFrame): The DataFrame containing the data.
    - feature_bins_dict (dict): A dictionary where keys are feature names and values are lists of bin edges.

    Returns:
    - pd.DataFrame: The DataFrame with additional columns for the binned features.
    """
    for feature, bins in feature_bins_dict.items():
        # Ensure the feature exists in the DataFrame
        if feature not in df.columns:
            print(f"Feature '{feature}' not found in DataFrame. Skipping...")
            continue

        # Generate labels based on bins
        labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]

        # Bin the data using pd.cut and assign to a new column
        df[f"{feature}_bins"] = pd.cut(df[feature], bins=bins, labels=labels, include_lowest=True, right=False)

    return df





















