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


def analyze_categorical_counts(df, categorical_columns=None):
    """Analyzes categorical variable category counts and creates a bar plot.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        categorical_columns (list, optional): List of column names to treat as categorical. 
                                               If None, all non-numeric columns are used.

    Returns:
        tuple: A tuple containing:
            - cats (pd.DataFrame): DataFrame with variable names and category counts.
            - ax (matplotlib.axes.Axes): The generated bar plot.
    """

    if categorical_columns is None:
        categorical_columns = df.select_dtypes(exclude='number').columns
    else:
        # Ensure the provided categorical_columns are in the DataFrame
        categorical_columns = [col for col in categorical_columns if col in df.columns]

    counts = [len(df[c].value_counts()) for c in categorical_columns]

    cats = pd.DataFrame(zip(categorical_columns, counts), columns=['Variable', 'Category Count'])
    cats = cats.sort_values(by=['Category Count'], ascending=False)

    # Create the bar plot
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(data=cats, x='Category Count', y='Variable', orient='h',
                     hue='Category Count', palette='Greens', dodge=False)
    plt.title('Categorical Variable Category Counts')
    plt.subplots_adjust(left=0.4)  # Adjust spacing for labels
    plt.tick_params(axis='y', which='major', labelsize=8)  # Adjust label size
    plt.ylabel('')
    plt.legend([],[], frameon=False)  # Remove legend
    plt.show()

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

#-------- DESCRIPTIVE STATISTICS -----------#

def descriptive_statistics_continuous(data):
    """
    Calculates descriptive statistics for the continuous data columns in a Pandas DataFrame,
    including additional statistics like skewness and kurtosis. Adds skew and kurtosis as a row instead of a column. Handles missing values and outliers.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: A DataFrame summarizing the descriptive statistics for the continuous columns.
    """
    try:
        continuous_columns = data.select_dtypes(include=[np.number]).columns
        data_clean = data[continuous_columns].dropna()

        # Check for minimum sample size before calculating skewness and kurtosis
        min_sample_size = 3
        for col in continuous_columns:
            if len(data_clean[col]) < min_sample_size:
                print(f"Warning: Insufficient data for skewness and kurtosis calculation in column '{col}'. Skipping...")
                data_clean.drop(col, axis=1, inplace=True)

        # Remove outliers before calculating skewness and kurtosis (optional)
        # You can implement specific outlier detection techniques here

        statistics = data_clean.describe(percentiles=[0.25, 0.5, 0.75])
        statistics.loc["Min"] = data_clean.min()
        statistics.loc["Max"] = data_clean.max()
        statistics.loc["Skewness"] = data_clean.skew()
        statistics.loc["Kurtosis"] = data_clean.kurt()

        # Convert Skewness and Kurtosis to a row
        statistics = statistics.T
        return statistics.iloc[1:]
    except Exception as e:
        print(f"Error calculating descriptive statistics: {e}")
        return None



#------------  CORRELATION PLOT AND SORTED  ----------------#

def plot_correlations_and_return_sorted(df, target_column, palette='Greens'):
    """
    Plots correlations of a DataFrame with a target column,
    returns sorted correlations, and handles potential errors.

    Args:
        df (pd.DataFrame): DataFrame to analyze.
        target_column (str): Name of the target column for sorting correlations.
        palette (str, optional): Color palette for plots. Defaults to 'Greens'.

    Returns:
        pd.Series: Sorted correlations of non-target columns with the target column.
    """

    try:
        # Select numeric columns
        df_numeric = df.select_dtypes(include=[np.number])

        # Calculate correlations
        corr = df_numeric.corr()

        # Sort correlations by target column (excluding self-correlation)
        sorted_corr = corr[target_column].drop(target_column).sort_values(ascending=False)

        # Plot heatmap of correlations
        plt.figure(figsize=(8, 8), dpi=200)
        sns.heatmap(corr, cmap=palette, annot=True, fmt=".2f", annot_kws={"size": 10})
        plt.title('Correlation Heatmap')
        plt.xticks(rotation=45, ha="right", fontsize=8)  # Rotate x-axis labels and adjust font size
        plt.yticks(fontsize=8)  # Adjust y-axis labels font size
        plt.show()

        # Plot bar chart of sorted correlations using seaborn
        plt.figure(figsize=(10, 6))  # Adjust figure size as needed
        sns.barplot(x=sorted_corr.index, y=sorted_corr.values, 
                    hue=sorted_corr.values, palette=palette)
        plt.title(f'Correlation to {target_column}')
        plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees
        plt.show()

        return sorted_corr

    except KeyError as e:
        print(f"Error: Target column '{target_column}' not found in DataFrame.")
        return None

    except (ValueError, TypeError) as e:
        print(f"Error: Input data is not a valid DataFrame or contains non-numeric columns.")
        return None


# ----------- CATEGORICAL DATA ---------------#

def create_categorical_dataframe(data, include_cols=None, exclude_cols=None):
    """
    Analyzes the categorical data columns in a Pandas DataFrame and converts them to categorical data type. Prints the top 3 categories for each column.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        include_cols (list, optional): List of columns to include in categorical analysis.
        exclude_cols (list, optional): List of columns to exclude from categorical conversion.

    Returns:
        pd.DataFrame: A DataFrame with the categorical columns converted to categorical data type.
    """
    categorical_columns = data.select_dtypes(exclude=[np.number]).columns
    analysis_data = {}

    # Identify columns for conversion
    columns_to_convert = []
    for col in categorical_columns:
        if len(data[col].unique()) < 15:
            columns_to_convert.append(col)

    # Add specified include columns
    if include_cols:
        for col in include_cols:
            if col not in analysis_data and col in data.columns:
                columns_to_convert.append(col)

    # Remove specified exclude columns
    if exclude_cols:
        for col in exclude_cols:
            if col in columns_to_convert:
                columns_to_convert.remove(col)

    # Convert identified columns to categorical
    for col in columns_to_convert:
        data[col] = data[col].astype("category")

    # Print summary and return converted data
    for col in data.select_dtypes("category"):
        print(f"\nAnalysis of {col}:")
        print(f"Number of unique categories: {len(data[col].unique())}")
        print(f"Top 3 categories:")
        print(data[col].value_counts().head(3))

    return data[data.select_dtypes("category").columns]

# ----------- CATEGORICAL DATA PLOT ---------------#

# use the new categorical data from create_categorical_dataframe
def plot_categorical_barplots(df,color='black'):
    """
    Plots barplots for all categorical columns in a DataFrame in a grid.

    Args:
        df (pd.DataFrame): The DataFrame containing categorical data.
    """

    n_rows = int(np.ceil(len(df.columns) / 3))  # Calculate rows for grid
    fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(15, n_rows * 4))  # Create grid

    for i, col in enumerate(df.columns):
        ax = axes.flat[i]  # Get the appropriate subplot
        df[col].value_counts().plot(kind='bar', ax=ax,color=color)  # Create barplot
        ax.set_title(col)  # Set title for the plot
        ax.set_xlabel(col)  # Set x-axis label
        ax.set_ylabel('Count')  # Set y-axis label

    plt.tight_layout()  # Adjust spacing between plots
    plt.show()


#------------------------------------------------------------------------#
# ------------------ MACHINE LEARNING AND MODEL TRAINING ----------------#
#------------------------------------------------------------------------#


# Import Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    

# --------- Model Comparison - Classification Reports -------------#

from sklearn.metrics import classification_report
import pandas as pd

def evaluate_and_report(model, model_name, X_test, y_test):
    # Make predictions
    predictions = model.predict(X_test)

    # Generate classification report
    report_dict = classification_report(y_test, predictions, output_dict=True)

    # Convert report to DataFrame and add a 'Model' column
    report_df = pd.DataFrame(report_dict)
    report_df['Model'] = model_name

    return report_df

# visualize reports for comparison 

def plot_model_comparison(reports_df, metric='precision', scoring='weighted avg'):
    # Set the aesthetic style of the plots
    sns.set_style("ticks")
    sns.set_context("talk")  # Larger fonts and elements
    palette = 'crest'

    # Create a 1x3 subplot
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # Plotting loop
    for i, metric_to_plot in enumerate(['precision', 'recall', 'f1-score']):
        # Filter data for the specific metric
        metric_data = reports_df[reports_df['Metrics'] == metric_to_plot]

        # Plot
        sns.barplot(x='Model', y=scoring, data=metric_data, ax=axes[i], hue=scoring, palette=palette)
        axes[i].set_title(f'{metric_to_plot.capitalize()} ({scoring})')
        axes[i].set_xlabel('')
        axes[i].set_ylabel(metric_to_plot.capitalize())
        axes[i].tick_params(axis='x', rotation=0)
        axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Adjust layout for readability
    plt.tight_layout()

    # Show the plot
    plt.show()

# ------------- Regression Model Comparison----------------#

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

def evaluate_model(best_model, model_name, X_test, y_test):
    # Predict on test data using the best model from grid search
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        'Model': model_name,
        'R2': r2,
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse
    }, index=[0])

    return metrics_df


def process_and_plot_metrics(raw_metrics_df):
    """
    Processes raw metrics data into long format and plots the metrics.

    Parameters:
    raw_metrics_df (DataFrame): A DataFrame containing the raw metrics data.
    """
    # Reshape the DataFrame from wide to long format
    model_metrics_long = pd.melt(raw_metrics_df, id_vars=['Model'], 
                                 value_vars=['R2', 'MSE', 'MAE', 'RMSE'],
                                 var_name='Metric', value_name='Value')

    # Set the aesthetic style of the plots
    sns.set_style("ticks")
    sns.set_context("talk")  # Larger fonts and elements
    palette = 'crest'

    # Metrics to plot
    metrics = ['R2', 'MSE', 'MAE', 'RMSE']

    # Create a separate plot for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Model', y='Value', hue='Value',
                         data=model_metrics_long[model_metrics_long['Metric'] == metric], palette='crest')
        plt.title(f'{metric} Comparison Across Models')
        plt.xticks(rotation=45)
        plt.ylabel(metric)
        plt.xlabel('')

        # Place the legend outside of the plot on the right
        plt.legend(title=metric, bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.show()

# Usage example
# process_and_plot_metrics(model_metrics_df)


#---------  Pred v Actual Metrics  ----------#


def evaluate_regression(model_name, y_true, y_pred):
    """Calculates regression metrics and returns them as a DataFrame with model name."""

    metrics = {
        "Mean Squared Error (MSE)": mean_squared_error(y_true, y_pred),
        "Root Mean Squared Error (RMSE)": mean_squared_error(y_true, y_pred, squared=False),
        "Mean Absolute Error (MAE)": mean_absolute_error(y_true, y_pred),
        "R-squared": r2_score(y_true, y_pred)
    }

    results_df = pd.DataFrame(metrics.items(), columns=["Metric", "Score"])
    results_df["Model"] = model_name  # Add model name directly as a string column

    return results_df


def evaluate_classification(model_name, y_true, y_pred):
    """Calculates classification metrics and returns them as a DataFrame with model name."""

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred)
    }

    results_df = pd.DataFrame(metrics.items(), columns=["Metric", "Score"])
    results_df["Model"] = model_name  # Add model name directly as a string column

    return results_df

#---------  Pred v Actual Plots  ----------#


def plot_combined_grid(y_true, y_pred, color="black"):
    """Creates a 1x2 grid of plots with scatter plot and residual plot, with optional color customization."""

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Scatter plot on the left
    axes[0].scatter(y_true, y_pred, s=20, color=color)  # Apply color
    axes[0].plot([0, 1], [0, 1], color=color, linestyle='--', linewidth=1)  # Reference line
    axes[0].set_xlabel("Actual Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].set_title("Scatter Plot")
    axes[0].grid(True)

    # Residual plot on the right
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, s=20, color=color)  # Apply color
    axes[1].axhline(y=0, color=color, linestyle='--', linewidth=1)  # Zero-residual line
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residual Plot")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_scatter_density(y_true, y_pred, title="Scatter Plot with Density Overlay",color='black'):
    """Creates a scatter plot with density overlay."""

    sns.jointplot(x=y_true, y=y_pred, kind="scatter", s=20, linewidth=0,color=color)
    plt.subplots_adjust(top=0.9)  # Adjust spacing for title
    plt.suptitle(title)
    plt.show()
    
#-------------   Compare Model Metrics ---------------#

def combine_model_metrics(model_results_list):
    """Combines a list of model metric results into a single DataFrame."""

    combined_df = pd.concat(model_results_list, ignore_index=True)

    # Set both "Model" and "Metric" as categorical
    combined_df["Model"] = combined_df["Model"].astype("category")
    combined_df["Metric"] = combined_df["Metric"].astype("category")

    # Rearrange columns
    combined_df = combined_df[["Model", "Metric", "Score"]]

    return combined_df



def plot_categorical_bar_grid_matplotlib(combined_df):
    """Creates a 2x2 grid of categorical bar plots with color-coded models and refined formatting."""

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    colors = plt.cm.tab20(range(len(combined_df["Model"].unique())))  # Generate custom color cycle

    for i, (metric, group_df) in enumerate(combined_df.groupby("Metric")):
        row, col = i // 2, i % 2

        axes[row, col].bar(group_df["Model"], group_df["Score"], color=colors)  # Use the custom colors
        axes[row, col].set_title(metric, fontsize=16)

        if group_df["Score"].max() - group_df["Score"].min() < 0.4:
            axes[row, col].set_ylim(group_df["Score"].min() - 0.05, group_df["Score"].max() + 0.05)

        for p in axes[row, col].patches:
            axes[row, col].annotate(
                f"{p.get_height():.2f}",  # Ensure value annotations are included
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=12,
                color="white",
            )

    plt.tight_layout()
    plt.show()

# -------------- CROSS VALIDATION -------------#

def save_cv_results_to_dataframe(results):
    """Saves cross-validation results to a DataFrame with positive scores.

    Args:
        results (dict): Dictionary of cross-validation results.

    Returns:
        pd.DataFrame: DataFrame containing train and test metric results.
    """

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate through metrics and create columns for train and test scores
    for metric in results:
        if metric.startswith('test_'):
            test_metric = metric.replace('test_', '')

            # Convert scores to positive for MSE, RMSE, and MAE
            if test_metric in ['MSE', 'RMSE', 'MAE']:
                df[f"Train {test_metric}"] = -1 * results[metric.replace('test_', 'train_')]
                df[f"Test {test_metric}"] = -1 * results[metric]
            else:
                df[f"Train {test_metric}"] = results[metric.replace('test_', 'train_')]
                df[f"Test {test_metric}"] = results[metric]

    return df

def visualize_cv_results_grid(cv_results):
    """Visualizes cross-validation results in a 2x2 grid of line plots.

    Args:
        cv_results (pd.DataFrame): DataFrame containing train and test scores.
    """

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    for i, metric in enumerate(['MSE', 'RMSE', 'MAE', 'R2']):
        axes.flat[i].plot(cv_results['Train ' + metric], label='Train',lw=7)
        axes.flat[i].plot(cv_results['Test ' + metric], label='Test',lw=7)
        axes.flat[i].set_title(metric, fontsize=20)
        axes.flat[i].legend()
    plt.tight_layout()
    plt.show()


#---------  Pred v Actual Metrics  ----------#


def evaluate_regression(model_name, y_true, y_pred):
    """Calculates regression metrics and returns them as a DataFrame with model name."""

    metrics = {
        "Mean Squared Error (MSE)": mean_squared_error(y_true, y_pred),
        "Root Mean Squared Error (RMSE)": mean_squared_error(y_true, y_pred, squared=False),
        "Mean Absolute Error (MAE)": mean_absolute_error(y_true, y_pred),
        "R-squared": r2_score(y_true, y_pred)
    }

    results_df = pd.DataFrame(metrics.items(), columns=["Metric", "Score"])
    results_df["Model"] = model_name  # Add model name directly as a string column

    return results_df


def evaluate_classification(model_name, y_true, y_pred):
    """Calculates classification metrics and returns them as a DataFrame with model name."""

    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred)
    }

    results_df = pd.DataFrame(metrics.items(), columns=["Metric", "Score"])
    results_df["Model"] = model_name  # Add model name directly as a string column

    return results_df

#---------  Pred v Actual Plots  ----------#


def plot_combined_grid(y_true, y_pred, color="black"):
    """Creates a 1x2 grid of plots with scatter plot and residual plot, with optional color customization."""

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Scatter plot on the left
    axes[0].scatter(y_true, y_pred, s=20, color=color)  # Apply color
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color=color, linestyle='--', linewidth=1)  # Reference line
    axes[0].set_xlabel("Actual Values")
    axes[0].set_ylabel("Predicted Values")
    axes[0].set_title("Actual vs. Predicted")
    axes[0].grid(True)
    axes[0].axis('equal')  # Equal scaling

    # Residual plot on the right
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, s=20, color=color)  # Apply color
    axes[1].axhline(y=0, color=color, linestyle='--', linewidth=1)  # Zero-residual line
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residuals vs. Predicted")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_scatter_density(y_true, y_pred, title="Scatter Plot with Density Overlay",color='black'):
    """Creates a scatter plot with density overlay."""

    sns.jointplot(x=y_true, y=y_pred, kind="scatter", s=20, linewidth=0,color=color)
    plt.subplots_adjust(top=0.9)  # Adjust spacing for title
    plt.suptitle(title)
    plt.show()
    
#-------------   Compare Model Metrics ---------------#

def combine_model_metrics(model_results_list):
    """Combines a list of model metric results into a single DataFrame."""

    combined_df = pd.concat(model_results_list, ignore_index=True)

    # Set both "Model" and "Metric" as categorical
    combined_df["Model"] = combined_df["Model"].astype("category")
    combined_df["Metric"] = combined_df["Metric"].astype("category")

    # Rearrange columns
    combined_df = combined_df[["Model", "Metric", "Score"]]

    return combined_df



def plot_categorical_bar_grid_matplotlib(combined_df):
    """Creates a 2x2 grid of categorical bar plots with color-coded models and refined formatting."""

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))

    colors = plt.cm.tab20(range(len(combined_df["Model"].unique())))  # Generate custom color cycle

    for i, (metric, group_df) in enumerate(combined_df.groupby("Metric")):
        row, col = i // 2, i % 2

        axes[row, col].bar(group_df["Model"], group_df["Score"], color=colors)  # Use the custom colors
        axes[row, col].set_title(metric, fontsize=16)

        if group_df["Score"].max() - group_df["Score"].min() < 0.4:
            axes[row, col].set_ylim(group_df["Score"].min() - 0.05, group_df["Score"].max() + 0.05)

        for p in axes[row, col].patches:
            axes[row, col].annotate(
                f"{p.get_height():.2f}",  # Ensure value annotations are included
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=12,
                color="white",
            )

    plt.tight_layout()
    plt.show()

# -------------- CROSS VALIDATION -------------#

def save_cv_results_to_dataframe(results):
    """Saves cross-validation results to a DataFrame with positive scores.

    Args:
        results (dict): Dictionary of cross-validation results.

    Returns:
        pd.DataFrame: DataFrame containing train and test metric results.
    """

    # Create an empty DataFrame
    df = pd.DataFrame()

    # Iterate through metrics and create columns for train and test scores
    for metric in results:
        if metric.startswith('test_'):
            test_metric = metric.replace('test_', '')

            # Convert scores to positive for MSE, RMSE, and MAE
            if test_metric in ['MSE', 'RMSE', 'MAE']:
                df[f"Train {test_metric}"] = -1 * results[metric.replace('test_', 'train_')]
                df[f"Test {test_metric}"] = -1 * results[metric]
            else:
                df[f"Train {test_metric}"] = results[metric.replace('test_', 'train_')]
                df[f"Test {test_metric}"] = results[metric]

    return df

def visualize_cv_results_grid(cv_results):
    """Visualizes cross-validation results in a 2x2 grid of line plots.

    Args:
        cv_results (pd.DataFrame): DataFrame containing train and test scores.
    """

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    for i, metric in enumerate(['MSE', 'RMSE', 'MAE', 'R2']):
        axes.flat[i].plot(cv_results['Train ' + metric], label='Train',lw=7)
        axes.flat[i].plot(cv_results['Test ' + metric], label='Test',lw=7)
        axes.flat[i].set_title(metric, fontsize=20)
        axes.flat[i].legend()
    plt.tight_layout()
    plt.show()



#-------------- FEATURE IMPORTANCE ----------------#


def plot_top_feature_importances(fitted_pipeline, top_n=20):
    """
    Plots the top N feature importances from a fitted pipeline.
    The function is designed for tree-based models like Random Forest, Gradient Boosting

    Parameters:
    fitted_pipeline: The fitted pipeline containing a preprocessor and a model.
    top_n: Number of top features to display.
    """
    # Extract preprocessor and model from the pipeline
    fitted_preprocessor = fitted_pipeline.named_steps['preprocessor']
    trained_model = fitted_pipeline.named_steps['model']

    # Get feature names and clean them
    feature_names = fitted_preprocessor.get_feature_names_out()
    cleaned_feature_names = [name.split('__')[-1] for name in feature_names]

    # Extract feature importance
    feature_importance = trained_model.feature_importances_

    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': cleaned_feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', hue='Importance',
                palette='crest', legend=False,
                data=feature_importance_df.head(top_n))
    plt.title(f'Top {top_n} Feature Importances')
    plt.show()

# Example usage:
# plot_top_feature_importances(best_model, top_n=20)

