import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from scipy.stats import spearmanr
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('Race1_Updated.csv')

# Handle empty values in 'LapTime_in_seconds'
# If both values are empty, delete such instances
df = df.dropna(subset=['LapTime_in_seconds'], how='all')

# Drop specified columns
columns_to_drop = ['Time', 'LapTime', 'Sector1SessionTime', 'Sector2SessionTime', 
                  'Sector3SessionTime', 'LapStartTime', 'LapStartDate']
df = df.drop(columns=columns_to_drop, errors='ignore')

# Print shape of dataset
print(f"Dataset shape: {df.shape}")

# Prepare data for modeling
# Separate features and target
X = df.drop(['LapTime_in_seconds', 'laptime_sum_sectortimes'], axis=1, errors='ignore')
y = df['LapTime_in_seconds']

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
encoder = OrdinalEncoder()
X[categorical_cols] = encoder.fit_transform(X[categorical_cols])

# Convert any remaining non-numeric columns
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean())

# Remove constant features or features with very low variance
variance = X.var()
X = X.loc[:, variance > 1e-10]
print(f"Dataset shape after removing low variance features: {X.shape}")

# 1. Feature selection using Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
rf_top5 = rf_importances.nlargest(5)
print("\nTop 5 features selected by Random Forest:")
print(rf_top5)

# 2. Feature selection using XGBoost
xgb_model = xgb.XGBRegressor(random_state=42)
xgb_model.fit(X, y)
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
xgb_top5 = xgb_importances.nlargest(5)
print("\nTop 5 features selected by XGBoost:")
print(xgb_top5)

# 3. Feature selection using Feature Agglomeration

# 4. Highly Variable Gene Selection (adapted for feature selection)
feature_variances = X.var()
hvgs_top5 = feature_variances.nlargest(5)
print("\nTop 5 features selected by Highly Variable Gene Selection:")
print(hvgs_top5)

# 5. Feature selection using Spearman's correlation
correlations = []
p_values = []

for column in X.columns:
    corr, p_val = spearmanr(X[column], y)
    correlations.append(abs(corr))  # Use absolute value for correlation strength
    p_values.append(p_val)

spearman_results = pd.DataFrame({
    'Feature': X.columns,
    'Correlation': correlations,
    'P-Value': p_values
})
spearman_top5 = spearman_results.sort_values('Correlation', ascending=False).head(5)
print("\nTop 5 features selected by Spearman's Correlation:")
print(spearman_top5)

# Function to perform cross-validation with both RMSE and R-squared
def perform_cv(X_selected, method_name, model_type='rf'):
    if model_type == 'rf':
        model = RandomForestRegressor(random_state=42)
    else:  # xgboost
        model = xgb.XGBRegressor(random_state=42)
    
    # Cross validation with multiple metrics
    cv_results = cross_validate(model, X_selected, y, cv=5, 
                              scoring={'neg_mean_squared_error': 'neg_mean_squared_error', 
                                       'r2': 'r2'})
    
    rmse_scores = np.sqrt(-cv_results['test_neg_mean_squared_error'])
    r2_scores = cv_results['test_r2']
    
    avg_rmse = rmse_scores.mean()
    avg_r2 = r2_scores.mean()
    
    print(f"{method_name} - Average RMSE: {avg_rmse:.4f}, Average R²: {avg_r2:.4f}")
    return avg_rmse, avg_r2

# Cross-validation for top 5 features
print("\n--- Cross-validation for Top 5 Features ---")

# Random Forest top 5
X_rf = X[rf_top5.index]
rf_cv_rmse, rf_cv_r2 = perform_cv(X_rf, "Random Forest", 'rf')

# XGBoost top 5
X_xgb = X[xgb_top5.index]
xgb_cv_rmse, xgb_cv_r2 = perform_cv(X_xgb, "XGBoost", 'xgb')

# Feature Agglomeration top 5

# HVGS top 5
X_hvgs = X[hvgs_top5.index]
hvgs_cv_rmse, hvgs_cv_r2 = perform_cv(X_hvgs, "HVGS", 'rf')

# Spearman top 5
X_spearman = X[spearman_top5['Feature'].values]
spearman_cv_rmse, spearman_cv_r2 = perform_cv(X_spearman, "Spearman", 'rf')

# Now remove the highest feature importance and reselect top 4
print("\n--- Removing highest importance feature and selecting top 4 ---")

# 1. Random Forest
most_important_rf = rf_importances.idxmax()
X_reduced_rf = X.drop(most_important_rf, axis=1)
rf_reduced = RandomForestRegressor(random_state=42)
rf_reduced.fit(X_reduced_rf, y)
rf_importances_reduced = pd.Series(rf_reduced.feature_importances_, index=X_reduced_rf.columns)
rf_top4 = rf_importances_reduced.nlargest(4)
print("\nTop 4 features from reduced dataset (Random Forest):")
print(rf_top4)
X_rf_reduced = X_reduced_rf[rf_top4.index]
rf_cv_rmse_reduced, rf_cv_r2_reduced = perform_cv(X_rf_reduced, "Random Forest (reduced)", 'rf')

# 2. XGBoost
most_important_xgb = xgb_importances.idxmax()
X_reduced_xgb = X.drop(most_important_xgb, axis=1)
xgb_reduced = xgb.XGBRegressor(random_state=42)
xgb_reduced.fit(X_reduced_xgb, y)
xgb_importances_reduced = pd.Series(xgb_reduced.feature_importances_, index=X_reduced_xgb.columns)
xgb_top4 = xgb_importances_reduced.nlargest(4)
print("\nTop 4 features from reduced dataset (XGBoost):")
print(xgb_top4)
X_xgb_reduced = X_reduced_xgb[xgb_top4.index]
xgb_cv_rmse_reduced, xgb_cv_r2_reduced = perform_cv(X_xgb_reduced, "XGBoost (reduced)", 'xgb')

# 3. Feature Agglomeration

# 4. HVGS
most_important_hvgs = hvgs_top5.idxmax()
X_reduced_hvgs = X.drop(most_important_hvgs, axis=1)
feature_variances_reduced = X_reduced_hvgs.var()
hvgs_top4 = feature_variances_reduced.nlargest(4)
print("\nTop 4 features from reduced dataset (HVGS):")
print(hvgs_top4)
X_hvgs_reduced = X_reduced_hvgs[hvgs_top4.index]
hvgs_cv_rmse_reduced, hvgs_cv_r2_reduced = perform_cv(X_hvgs_reduced, "HVGS (reduced)", 'rf')

# 5. Spearman
most_important_spearman = spearman_top5.iloc[0]['Feature']
X_reduced_spearman = X.drop(most_important_spearman, axis=1)
correlations_reduced = []
p_values_reduced = []
for column in X_reduced_spearman.columns:
    corr, p_val = spearmanr(X_reduced_spearman[column], y)
    correlations_reduced.append(abs(corr))
    p_values_reduced.append(p_val)
spearman_results_reduced = pd.DataFrame({
    'Feature': X_reduced_spearman.columns,
    'Correlation': correlations_reduced,
    'P-Value': p_values_reduced
})
spearman_top4 = spearman_results_reduced.sort_values('Correlation', ascending=False).head(4)
print("\nTop 4 features from reduced dataset (Spearman):")
print(spearman_top4)
X_spearman_reduced = X_reduced_spearman[spearman_top4['Feature'].values]
spearman_cv_rmse_reduced, spearman_cv_r2_reduced = perform_cv(X_spearman_reduced, "Spearman (reduced)", 'rf')

# Summary of results
print("\n--- Summary of Results ---")
print("Top 5 Features CV Results:")
print(f"Random Forest: RMSE={rf_cv_rmse:.4f}, R²={rf_cv_r2:.4f}")
print(f"XGBoost: RMSE={xgb_cv_rmse:.4f}, R²={xgb_cv_r2:.4f}")
#print(f"Feature Agglomeration: RMSE={fa_cv_rmse:.4f}, R²={fa_cv_r2:.4f}")
print(f"HVGS: RMSE={hvgs_cv_rmse:.4f}, R²={hvgs_cv_r2:.4f}")
print(f"Spearman: RMSE={spearman_cv_rmse:.4f}, R²={spearman_cv_r2:.4f}")

print("\nTop 4 Features (after removing highest importance) CV Results:")
print(f"Random Forest: RMSE={rf_cv_rmse_reduced:.4f}, R²={rf_cv_r2_reduced:.4f}")
print(f"XGBoost: RMSE={xgb_cv_rmse_reduced:.4f}, R²={xgb_cv_r2_reduced:.4f}")
#print(f"Feature Agglomeration: RMSE={fa_cv_rmse_reduced:.4f}, R²={fa_cv_r2_reduced:.4f}")
print(f"HVGS: RMSE={hvgs_cv_rmse_reduced:.4f}, R²={hvgs_cv_r2_reduced:.4f}")
print(f"Spearman: RMSE={spearman_cv_rmse_reduced:.4f}, R²={spearman_cv_r2_reduced:.4f}")
