
#-----------------------------------------------------------#
#--------------    MACHINE LEARNING TOOLKIT    -------------#
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

