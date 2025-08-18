import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
import xgboost
from xgboost import XGBClassifier # Changed to XGBClassifier
import sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # New classification metrics
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.preprocessing import LabelEncoder
import cudf
import cupy as cp
import optuna

# --- Parse Configuration String ---
# Expected format: <validation><landuse><version> e.g. "Kt0726", "wf2031"

config_input = "KT0815"

# --- Run sweep for the entire nation ---
n_trials_for_national_study = 50 # Number of trials for the national study

# Normalize and validate input
config_input = config_input.strip().lower()
if len(config_input) != 6:
    raise ValueError("Invalid config format. Use 6 characters like 'kt0726' or 'wf2031'.")

# Parse components
validation_flag = config_input[0]
landuse_flag = config_input[1]
version_digits = config_input[2:]

# Determine validation strategy
if validation_flag == 'k':
    validation_strategy = 'kfold'
elif validation_flag == 'w':
    validation_strategy = 'walk'
else:
    raise ValueError("Invalid validation flag. Use 'K' for kfold or 'W' for walk_forward.")

# Determine land use flag
if landuse_flag == 't':
    USE_LANDUSE_FEATURES = True
    landuse_suffix = "lulc"
elif landuse_flag == 'f':
    USE_LANDUSE_FEATURES = False
    landuse_suffix = ""
else:
    raise ValueError("Land use flag must be 'T' or 'F'.")

# Validate version digits
if not version_digits.isdigit():
    raise ValueError("Version must be 4 digits.")
VERSION_STAMP = version_digits
version_suffix = f"{VERSION_STAMP}"

# --- Logging ---
print(f"Using validation strategy: {validation_strategy} in version {VERSION_STAMP}")
print(f"Land Use Features Included: {USE_LANDUSE_FEATURES}")
study_name = f"xgbC-nation-{validation_strategy}-{landuse_suffix}-{VERSION_STAMP}"
print(f"Name of study: {study_name}")

# Define project root based on notebook location (assuming this part is correct for your setup)
def find_project_root(current: Path, marker: str = ".git"):
    for parent in current.resolve().parents:
        if (parent / marker).exists():
            return parent
    return current.resolve() # fallback

PROJECT_ROOT = find_project_root(Path(__file__).parent)
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = REPORTS_DIR / "tables"

# Load the data once outside the train function for efficiency
df = pd.read_csv(PROCESSED_DIR / "INDONESIA" / "monthly_dengue_env_id_updated.csv")

# --- REGION REASSIGNMENT (Keep this for consistency) ---
df['Region_Group'] = df['Region'].replace({'Maluku Islands': 'Maluku-Papua', 'Papua': 'Maluku-Papua'})
print("--- DataFrame after Region_Group creation ---")
print(df['Region_Group'].value_counts())
print("-" * 50)

df['YearMonth'] = pd.to_datetime(df['YearMonth']) # Ensure YearMonth is datetime

# Define variable categories
env_vars = [
    'temperature_2m', 'temperature_2m_min', 'temperature_2m_max',
    'precipitation', 'potential_evaporation_sum', 'total_evaporation_sum',
    'evaporative_stress_index', 'aridity_index',
    'temperature_2m_ANOM', 'temperature_2m_min_ANOM', 'temperature_2m_max_ANOM',
    'potential_evaporation_sum_ANOM', 'total_evaporation_sum_ANOM', 'precipitation_ANOM'
]

land_use_vars = [
    'Class_70', 'Class_60', 'Class_50', 'Class_40', 'Class_95',
    'Class_30', 'Class_20', 'Class_10', 'Class_90', 'Class_80'
]

climate_vars = ['ANOM1+2', 'ANOM3', 'ANOM4', 'ANOM3.4', 'DMI', 'DMI_East']

# Initialize label encoder
le = LabelEncoder()

# Fit and transform the target variable
df['Risk_Category_encoded'] = le.fit_transform(df['Risk_Category'])

# If you want to see the mapping
category_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Category mapping:", category_mapping)

# Update target variable for model
target = 'Risk_Category_encoded'
# Changed target variable

# Sort data by time and region
df = df.sort_values(['YearMonth', 'ID_2'])

# Create lag features for environmental and climate variables
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            df[f'{var}_lag{lag}'] = df.groupby('ID_2')[var].shift(lag)

# Compile feature list
features = []
for var in env_vars + climate_vars:
    if var in df.columns:
        features.append(var)
if USE_LANDUSE_FEATURES:
    for var in land_use_vars:
        if var in df.columns:
            features.append(var)
for var_group in [env_vars, climate_vars]:
    for var in var_group:
        for lag in [1, 2, 3]:
            lagged_var = f'{var}_lag{lag}'
            if lagged_var in df.columns:
                features.append(lagged_var)

# Final feature list excluding metadata and target
actual_feature_columns = [
    col for col in features
    if col not in ['YearMonth', 'ID_2', 'Year', target]
]

print("\n--- Final list of features for the model ---")
print(actual_feature_columns)
print(f"Total features: {len(actual_feature_columns)}")
print("-" * 50)
print("--- Creating splits based on validation strategy ---")


# Data for hyperparameter tuning
df_train_val_national = df[df['YearMonth'].dt.year < 2023].copy().dropna(subset=actual_feature_columns + [target])
# Data for final, unseen test
df_test_national = df[df['YearMonth'].dt.year == 2023].copy().dropna(subset=actual_feature_columns + [target])

print(f"Shape of df_train_val for National: {df_train_val_national.shape}")
print(f"Shape of df_final_test for National: {df_test_national.shape}")

# Convert the full processed DataFrame to cudf and cupy once
X_gpu = cudf.DataFrame(df_train_val_national[actual_feature_columns])
y_gpu = cudf.DataFrame(df_train_val_national[[target]])

splits = []

if validation_strategy == 'kfold':
    n_splits = 5
    # KFold works with pandas indices
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=64)
    for train_index, test_index in kf.split(df_train_val_national):
        # Store the indices as cupy arrays
        splits.append((cp.array(train_index), cp.array(test_index)))

elif validation_strategy == 'walk':
    initial_train_months = 60
    test_window = 6
    unique_time_periods = df_train_val_national['YearMonth'].unique()
    n_time_periods = len(unique_time_periods)
    n_walks = (n_time_periods - initial_train_months) // test_window

    for i in range(n_walks):
        train_end_period_idx = initial_train_months + i * test_window
        test_start_period_idx = train_end_period_idx
        test_end_period_idx = test_start_period_idx + test_window

        if test_end_period_idx > n_time_periods:
            break

        train_end_time = unique_time_periods[train_end_period_idx - 1]
        test_start_time = unique_time_periods[test_start_period_idx]
        test_end_time = unique_time_periods[test_end_period_idx - 1]

        # Get pandas indices for the time periods
        train_indices_pd = df_train_val_national.loc[df_train_val_national['YearMonth'] <= train_end_time].index
        test_indices_pd = df_train_val_national.loc[
            (df_train_val_national['YearMonth'] >= test_start_time) & (df_train_val_national['YearMonth'] <= test_end_time)
        ].index
        
        # Convert indices to cupy arrays
        splits.append((cp.array(train_indices_pd.to_numpy()), cp.array(test_indices_pd.to_numpy())))

print(f"Pre-calculated {len(splits)} splits for {validation_strategy} validation.")

# def objective(trial, X_gpu, y_gpu, splits):
#     """
#     Objective function for Optuna to maximize accuracy for classification.
#     Uses multi:softprob so predictions are probabilities that are converted to class labels.
#     Assumes y_gpu contains integer-encoded class labels.
#     """
#     # Ensure y is a 1D Series
#     y_series = y_gpu.squeeze()
#     num_classes = int(y_series.nunique())

#     # --- Suggest Hyperparameters to Optuna ---
#     params = {
#         'objective': 'multi:softprob',  # output class probabilities
#         'tree_method': 'hist',
#         'device': 'cuda',
#         'random_state': 64,
#         'n_jobs': -1,
#         'num_class': num_classes,
#         'learning_rate': trial.suggest_float('learning_rate', 1e-6, 0.1, log=True),
#         'max_depth': trial.suggest_int('max_depth', 2, 5),
#         'subsample': trial.suggest_float('subsample', 0.1, 0.5),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
#         'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
#         'gamma': trial.suggest_float('gamma', 0.1, 10, log=True),
#         'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10, log=True),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1, 100, log=True),
#         'eval_metric': 'mlogloss'
#     }
#     num_boost_round = trial.suggest_int('n_estimators', 50, 7000)

#     all_preds = []
#     all_true = []

#     for train_index_gpu, test_index_gpu in splits:
#         X_train_fold_gpu = X_gpu.loc[train_index_gpu]
#         y_train_fold_gpu = y_gpu.loc[train_index_gpu]
#         X_test_fold_gpu = X_gpu.loc[test_index_gpu]
#         y_test_fold_gpu = y_gpu.loc[test_index_gpu]

#         if X_train_fold_gpu.empty or X_test_fold_gpu.empty:
#             continue

#         # Prepare DMatrices
#         dtrain = xgboost.DMatrix(X_train_fold_gpu, label=y_train_fold_gpu)
#         dtest = xgboost.DMatrix(X_test_fold_gpu, label=y_test_fold_gpu)

#         # Train
#         booster = xgboost.train(
#             params=params,
#             dtrain=dtrain,
#             num_boost_round=num_boost_round,
#             evals=[(dtest, 'test')],
#             verbose_eval=False
#         )

#         # Predict probabilities and convert to labels
#         predictions_proba = booster.predict(dtest)  # shape (n_samples, num_classes)
#         predictions_labels = np.argmax(predictions_proba, axis=1)

#         all_preds.append(predictions_labels)
#         all_true.append(y_test_fold_gpu.to_numpy().flatten())

#     if all_true:
#         accuracy = accuracy_score(
#             np.concatenate(all_true),
#             np.concatenate(all_preds)
#         )
#         print(f"Trial finished with Accuracy: {accuracy:.4f}")
#         return accuracy
#     else:
#         return 0.0



# if df_train_val_national.empty:
#     print(f"No training/validation data for national study. Cannot proceed.")
# else:
#     # --- Optuna Study Setup (offline-friendly) ---
#     study = optuna.create_study(
#         study_name=study_name,
#         direction='maximize',  # maximize accuracy
#         storage=f'sqlite:///{study_name}.db',
#         load_if_exists=True
#     )
    
#     print(f"Optuna study created/loaded: {study_name}.db")
#     print(f"Starting {n_trials_for_national_study} trials...")
    
#     # Run the optimization
#     study.optimize(
#         lambda trial: objective(
#             trial,
#             X_gpu=X_gpu,
#             y_gpu=y_gpu,
#             splits=splits
#         ),
#         n_trials=n_trials_for_national_study,
#         n_jobs=-1
#     )
#     print("\nNational study completed.")
    
#     # --- Retrieve and Log Best Hyperparameters for the Nation ---
#     print("\n--- Best Hyperparameters Found by Optuna ---")
#     best_params = study.best_params
#     best_accuracy = study.best_value   # This is accuracy now
#     print(f"Best Accuracy: {best_accuracy:.4f}")
#     print("Best hyperparameters:")
#     for key, value in best_params.items():
#         print(f"  {key}: {value}")

#     # Prepare data for CSV saving
#     all_best_hypers = [{
#         'Region': 'National',
#         'best_accuracy': best_accuracy,
#         **best_params
#     }]

#     # --- Final Aggregation and CSV Saving ---
#     if all_best_hypers:
#         best_hypers_df = pd.DataFrame(all_best_hypers)
#         best_hypers_csv_path = TABLES_DIR / f"{study_name}_params.csv"
#         best_hypers_df.to_csv(best_hypers_csv_path, index=False)
#         print(f"\nSaved best national hyperparameters to {best_hypers_csv_path}")
#     else:
#         print("\nNo best national hyperparameters found to save.")

# print("\n--- NATIONAL Study and Hyperparameter Retrieval Completed ---")



# --- Test with Final Model (This section remains largely the same) ---
print("\n--- Begin Final Model Training and Evaluation ---")
import pandas as pd
import cudf
import xgboost
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.backends.backend_pdf import PdfPages

# --- Read hyperparameters ---
best_hypers_csv_path = TABLES_DIR / f"{study_name}_params.csv"
hyperparams_df = pd.read_csv(best_hypers_csv_path)
print(f"Extracted hyperparameters for national model: {best_hypers_csv_path}")

params = hyperparams_df.iloc[0]
national_hyperparams = {
    'gamma': params['gamma'],
    'n_estimators': int(params['n_estimators']),
    'max_depth': int(params['max_depth']),
    'reg_alpha': params['reg_alpha'],
    'subsample': params['subsample'],
    'reg_lambda': params['reg_lambda'],
    'learning_rate': params['learning_rate'],
    'colsample_bytree': params['colsample_bytree'],
    'min_child_weight': int(params['min_child_weight']),
    'objective': 'multi:softprob',  # Use multi-class classification
    'num_class': 4
}
num_round = int(params['n_estimators'])

# --- Prepare data ---
X_train = cudf.DataFrame(df_train_val_national[actual_feature_columns])
y_train = cudf.DataFrame(df_train_val_national[[target]])
X_test = cudf.DataFrame(df_test_national[actual_feature_columns])
y_test = cudf.DataFrame(df_test_national[[target]])
X_full = cudf.DataFrame(df[actual_feature_columns])
y_full = cudf.DataFrame(df[[target]])

print(f"Shape of X_train (cudf): {X_train.shape}")
print(f"Shape of y_train (cudf): {y_train.shape}")
print(f"Shape of X_test (cudf): {X_test.shape}")
print(f"Shape of y_test (cudf): {y_test.shape}")
print(f"Shape of X_full (cudf): {X_full.shape}")
print(f"Shape of y_full (cudf): {y_full.shape}")

# Display copies for SHAP plotting
X_train_display = df_train_val_national[actual_feature_columns].copy()
X_test_display = df_test_national[actual_feature_columns].copy()

Dtrain = xgboost.DMatrix(X_train, label=y_train)
Dtest = xgboost.DMatrix(X_test, label=y_test)
Dfull = xgboost.DMatrix(X_full, label=y_full)

# --- Train model ---
model = xgboost.train(national_hyperparams, Dtrain, num_boost_round=num_round)
model.set_param({"device": "cuda"})
# --- Predict and compute classification metrics ---
y_pred_proba = model.predict(Dtest)  # shape: (n_samples, num_classes) for multi-class
y_pred_labels = np.argmax(y_pred_proba, axis=1) if national_hyperparams['num_class'] > 2 else (y_pred_proba > 0.5).astype(int)

# Accuracy
accuracy = accuracy_score(y_test.to_pandas(), y_pred_labels)

# Precision, Recall, F1 for each class
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test.to_pandas(), 
    y_pred_labels, 
    average=None  # get per-class metrics
)

# Classification report for easy reading
class_report = classification_report(y_test.to_pandas(), y_pred_labels)

# Store metrics in dictionary
national_summary_data = {
    "Accuracy": accuracy,
    "Precision_per_class": precision.tolist(),
    "Recall_per_class": recall.tolist(),
    "F1_per_class": f1.tolist()
}

print("--- Classification Report ---")
print(class_report)

# --- Prepare SHAP background data ---
X_test_pd = X_test.to_pandas()
X_train_pd = X_train.to_pandas()
background_data = X_train_pd.sample(100, random_state=42)

print("--- SHAP Debugging Information ---")
print("Columns in X_train (cudf):", X_train.columns.tolist())
print("Columns in X_test_pd (pandas):", X_test_pd.columns.tolist())

# --- Create SHAP explainer ---
explainer = shap.Explainer(model, background_data, feature_perturbation='interventional')

# For classification, SHAP will return an array of shape (n_samples, n_features, n_classes)
shap_values = explainer(X_test_pd, check_additivity=False)

pdf_path = FIGURES_DIR / f"{study_name}_shap_plots.pdf"
with PdfPages(pdf_path) as pdf:
    if national_hyperparams['num_class'] > 2:
        # Multi-class SHAP plots — loop over classes
        for class_idx in range(national_hyperparams['num_class']):
            print(f"Generating SHAP plots for class {class_idx}...")
            class_shap_values = shap_values[..., class_idx]

            # Beeswarm plot for this class
            shap.plots.beeswarm(class_shap_values, show=False)
            plt.title(f"SHAP Beeswarm - Class {class_idx}")
            pdf.savefig(bbox_inches="tight")
            plt.close()

            # Dependence plots for each feature
            for name in X_train.columns:
                shap.dependence_plot(name, class_shap_values.values, X_test_display, show=False)
                plt.title(f"SHAP Dependence - {name} - Class {class_idx}")
                pdf.savefig(bbox_inches="tight")
                plt.close()
    else:
        # Binary classification SHAP — works like regression
        shap.plots.beeswarm(shap_values, show=False)
        pdf.savefig(bbox_inches="tight")
        plt.close()

        for name in X_train.columns:
            shap.dependence_plot(name, shap_values.values, X_test_display, show=False)
            pdf.savefig(bbox_inches="tight")
            plt.close()

print(f"National SHAP plots saved to '{pdf_path}'")

# Save summary table
summary_df = pd.DataFrame([national_summary_data])
csv_filename = TABLES_DIR / f"{study_name}_results.csv"
summary_df.to_csv(csv_filename, index=False)
print(f"National summary table saved to '{csv_filename}'")