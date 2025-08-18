import pandas as pd
import numpy as np
import os
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import xgboost
from xgboost import XGBRegressor
import shap
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.inspection import PartialDependenceDisplay
import sklearn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import cudf
import cupy as cp
import optuna

# --- Parse Configuration String ---
# Expected format: <validation><landuse><version> e.g. "Kt0726", "wf2031"

config_input = "KF0816"

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

# Define project root based on notebook location (assuming this part is correct for your setup)
def find_project_root(current: Path, marker: str = ".git"):
    for parent in current.resolve().parents:
        if (parent / marker).exists():
            return parent
    return current.resolve() # fallback
    
def rmsle(y_true, y_pred):
    # Add 1 to avoid issues with log(0)
    return np.sqrt(np.mean(np.power(np.log1p(y_true) - np.log1p(y_pred), 2)))

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
# Create a list of regions to iterate over
regions_to_model = df['Region_Group'].unique()


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
target = 'Incidence_Rate'  

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

cudf_train_val_national = cudf.DataFrame(df_train_val_national)
cudf_test_national = cudf.DataFrame(df_test_national)

def objective(trial, X_gpu, y_gpu, splits):
    """
    Objective function for Optuna to minimize.
    It performs cross-validation or walk-forward validation and returns the
    overall RMSLE.
    """
    # --- Suggest Hyperparameters to Optuna ---
    params = {
        'objective': 'reg:tweedie',
        'tree_method': 'hist',
        'device': 'cuda',
        'random_state': 64,
        'n_jobs': -1,
        'eval_metric': 'rmsle',
        'learning_rate': trial.suggest_float('learning_rate', 1e-6, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 5),
        'subsample': trial.suggest_float('subsample', 0.1, 0.5),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 0.5),
        'min_child_weight': trial.suggest_int('min_child_weight', 10, 50),
        'gamma': trial.suggest_float('gamma', 0.1, 10, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 100, log=True),
    }
    num_boost_round = trial.suggest_int('n_estimators', 50, 7000)

    all_preds = []
    all_true = []

    for train_index_gpu, test_index_gpu in splits:
        X_train_fold_gpu = X_gpu.iloc[train_index_gpu]
        y_train_fold_gpu = y_gpu.iloc[train_index_gpu]
        X_test_fold_gpu = X_gpu.iloc[test_index_gpu]
        y_test_fold_gpu = y_gpu.iloc[test_index_gpu]

        if X_train_fold_gpu.empty or X_test_fold_gpu.empty:
            continue

        # Prepare DMatrices
        dtrain = xgboost.DMatrix(X_train_fold_gpu, label=y_train_fold_gpu)
        dtest = xgboost.DMatrix(X_test_fold_gpu, label=y_test_fold_gpu)

        # Train using xgboost.train
        booster = xgboost.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtest, 'test')],
            verbose_eval=False
        )

        # Predict
        predictions_gpu = booster.predict(dtest)

        all_preds.append(predictions_gpu)
        all_true.append(y_test_fold_gpu.to_numpy().flatten())

    if all_true:
        overall_rmsle = rmsle(
            np.concatenate(all_true),
            np.concatenate(all_preds)
        )
        print(f"Trial finished with RMSLE: {overall_rmsle:.4f}")
        return overall_rmsle
    else:
        return float('inf')



all_best_hypers = []

for region in regions_to_model:
    print(f"\n--- Starting modeling for Region: {region} ---")
    
    # Filter the data for the current region
    cudf_train_val_region = cudf_train_val_national[cudf_train_val_national['Region_Group'] == region].copy()
    columns_to_drop = [
    target, 
    'YearMonth', 
    'Region', 
    'Region_Group', 
    'Risk_Category', # Assuming Risk_Category is also not a feature
    'ID_2'
]
    X_cudf = cudf_train_val_region.drop(columns_to_drop, axis=1) # Replace 'target' with your actual target column name
    y_cudf = cudf_train_val_region[target]

    # Create the K-Fold splitter
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Generate the splits. These are numpy arrays of indices.
    splits = list(kf.split(X_cudf.to_pandas()))

    # Inside the region loop
    study_name = f"xgbR-{region}-RMSLE-{validation_strategy}-{landuse_suffix}-{VERSION_STAMP}"
    print(f"Name of study: {study_name}")
    
    # Update file paths
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=f'sqlite:///{study_name}.db',
        load_if_exists=True)
    
    print(f"Optuna study created/loaded: {study_name}.db")
    print(f"Starting {n_trials_for_national_study} trials...")
    
    # Run the optimization
    study.optimize(
        lambda trial: objective(
            trial,
            X_gpu=X_cudf,
            y_gpu=y_cudf,
            splits=splits
        ),
        n_trials=n_trials_for_national_study,
        n_jobs=-1
    )

    print("\nNational study completed.")
    
    # --- Retrieve and Log Best Hyperparameters for the Nation ---
    print("\n--- Best Hyperparameters Found by Optuna ---")
    best_params = study.best_params
    best_value = study.best_value
    print(f"Best RMSLE: {best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    # Prepare data for CSV saving
    current_region_params = {
    'Region': region,  # Use the current region variable
    'best_rmsle': best_value,
    **best_params
    }
    all_best_hypers.append(current_region_params)

if all_best_hypers:
    best_hypers_df = pd.DataFrame(all_best_hypers)
    # Use a consistent file name for the combined CSV
    final_hypers_csv_path = TABLES_DIR / f"all_regions_hyperparams_{validation_strategy}_{landuse_suffix}_{VERSION_STAMP}.csv"
    best_hypers_df.to_csv(final_hypers_csv_path, index=False)
    print(f"\nSaved all regional hyperparameters to {final_hypers_csv_path}")
else:
    print("\nNo regional hyperparameters found to save.")

print("\n--- NATIONAL Study and Hyperparameter Retrieval Completed ---")

# --- Test with Final Model (This section remains largely the same) ---
# print("\n--- Begin Final Model Training and Evaluation ---")

# import pandas as pd
# import cudf
# import xgboost
# import shap
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from matplotlib.backends.backend_pdf import PdfPages
# from sklearn.metrics import mean_squared_log_error

# # --- Read hyperparameters ---
# best_hypers_csv_path = TABLES_DIR / f"{study_name}_params.csv"
# hyperparams_df = pd.read_csv(best_hypers_csv_path)
# print(f"Extracted hyperparameters for national model: {best_hypers_csv_path}")

# params = hyperparams_df.iloc[0]
# national_hyperparams = {
#     'gamma': params['gamma'],
#     'n_estimators': int(params['n_estimators']),
#     'max_depth': int(params['max_depth']),
#     'reg_alpha': params['reg_alpha'],
#     'subsample': params['subsample'],
#     'reg_lambda': params['reg_lambda'],
#     'learning_rate': params['learning_rate'],
#     'colsample_bytree': params['colsample_bytree'],
#     'min_child_weight': int(params['min_child_weight']),
#     'objective': 'reg:tweedie'
# }
# num_round = int(params['n_estimators'])

# # --- Prepare data ---
# X_train = cudf.DataFrame(df_train_val_national[actual_feature_columns])
# y_train = cudf.DataFrame(df_train_val_national[[target]])
# X_test = cudf.DataFrame(df_test_national[actual_feature_columns])
# y_test = cudf.DataFrame(df_test_national[[target]])
# X_full = cudf.DataFrame(df[actual_feature_columns])
# y_full = cudf.DataFrame(df[[target]])

# print(f"Shape of X_train (cudf): {X_train.shape}")
# print(f"Shape of y_train (cudf): {y_train.shape}")
# print(f"Shape of X_test (cudf): {X_test.shape}")
# print(f"Shape of y_test (cudf): {y_test.shape}")
# print(f"Shape of X_full (cudf): {X_full.shape}")
# print(f"Shape of y_full (cudf): {y_full.shape}")

# # Display copies for SHAP plotting
# X_train_display = df_train_val_national[actual_feature_columns].copy()
# X_test_display = df_test_national[actual_feature_columns].copy()

# Dtrain = xgboost.DMatrix(X_train, label=y_train)
# Dtest = xgboost.DMatrix(X_test, label=y_test)
# Dfull = xgboost.DMatrix(X_full, label=y_full)

# # --- Train model ---
# model = xgboost.train(national_hyperparams, Dtrain, num_boost_round=num_round)
# model.set_param({"device": "cuda"})

# # --- Predict and compute performance metrics ---
# y_pred = model.predict(Dtest)  # returns a NumPy array
# y_test = y_test  # ensure this is a NumPy array too

# # If y_test is a cudf.Series or pandas.Series, convert it to NumPy first
# if hasattr(y_test, "to_numpy"):
#     y_test_np = y_test.to_numpy()
# else:
#     y_test_np = np.array(y_test)

# # Ensure non-negative values
# y_test_np = np.clip(y_test_np, 0, None)
# y_pred = np.clip(y_pred, 0, None)

# # Metrics
# rmsle = np.sqrt(mean_squared_log_error(y_test_np, y_pred))
# mae = mean_absolute_error(y_test_np, y_pred)
# r2 = r2_score(y_test_np, y_pred)

# national_summary_data = {
#     "RMSLE": rmsle,
#     "MAE": mae,
#     "R2": r2
# }

# ## Match format of training
# X_test_pd = X_test.to_pandas()
# X_train_pd = X_train.to_pandas()
# background_data = X_train_pd.sample(100, random_state=42)
# # --- Information to print for debugging ---
# print("--- SHAP Debugging Information ---")
# print("Columns in X_train (cudf):", X_train.columns.tolist())
# print("Columns in X_test_pd (pandas):", X_test_pd.columns.tolist())
# # --- Generate SHAP plots and save to one PDF ---
# explainer = shap.explainers.GPUTree(model, background_data, feature_perturbation='interventional')
# shap_values = explainer(X_test_pd, check_additivity=False)

# pdf_path = FIGURES_DIR / f"{study_name}_shap_plots.pdf"
# with PdfPages(pdf_path) as pdf:
#     # Beeswarm plot
#     shap.plots.beeswarm(shap_values, show=False)
#     pdf.savefig(bbox_inches="tight")
#     plt.close()

#     # Dependence plots for each feature
#     for name in X_train.columns:
#         shap.dependence_plot(name, shap_values.values, X_test_display, show=False)
#         pdf.savefig(bbox_inches="tight")
#         plt.close()

# print(f"National SHAP plots saved to '{pdf_path}'")

# # --- Save summary table ---
# summary_df = pd.DataFrame([national_summary_data])
# csv_filename = TABLES_DIR / f"{study_name}_results.csv"
# summary_df.to_csv(csv_filename, index=False, float_format="%.4f")
# print(f"National summary table saved to '{csv_filename}'")
# print("-" * 50)