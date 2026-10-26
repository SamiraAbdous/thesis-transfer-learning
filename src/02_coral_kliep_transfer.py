# Optional: Uncomment these lines to install any missing packages
# !pip install adapt
# !pip install protobuf
# !pip install tensorflow
# !pip install --upgrade keras

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

# Import the two adaptation methods
from adapt.instance_based import KLIEP
from adapt.feature_based import CORAL

# ------------------------------
# Load and Prepare the Data
# ------------------------------

# Update the file path as needed
file_path = "data/ind.xlsx"  # panel of tourism indicators, 2000-2020
df = pd.read_excel(file_path)

# Clean column names: remove extra spaces
df.columns = df.columns.str.strip().str.replace(" +", " ", regex=True)
print("Columns in dataset:", df.columns.tolist())

# ------------------------------
# Feature Engineering
# ------------------------------

# Define source countries and target country
source_countries = ["Egypt", "Morocco", "Jordan", "Tunisia"]
target_country = "Algeria"

# Create new features
df['receipts_per_tourist'] = df['Tourism receipts'] / df['Number of tourists']
df['expenditures_per_tourist'] = df['Tourism expenditures'] / df['Number of tourists']
df['tourism_balance'] = df['Tourism receipts'] - df['Tourism expenditures']

# Specify the country column, features, and target variable
country_column = "id"
features = ["Tourism expenditures", "Tourism receipts", 
            "receipts_per_tourist", "expenditures_per_tourist", "tourism_balance"]
target = "Number of tourists"

# ------------------------------
# Visualize Tourism Trends Over Time
# ------------------------------

plt.figure(figsize=(12, 8))
for country in source_countries + [target_country]:
    country_data = df[df[country_column] == country]
    plt.plot(country_data["year"], country_data[target], label=country)
plt.xlabel("Year")
plt.ylabel("Number of Tourists")
plt.title("Tourism Trends Across Countries (2000-2020)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------
# Experiment with KLIEP
# ------------------------------

results_kliep = {}

# Get target (Algeria) data
df_target = df[df[country_column] == target_country].copy()

for source in source_countries:
    print(f"\n[KLIEP] Training model using {source}'s data...")

    # Filter data for the source country
    df_source = df[df[country_column] == source]

    # Prepare source data
    X_source = df_source[features]
    y_source = df_source[target]

    # Prepare target data (Algeria)
    X_target = df_target[features]
    y_target = df_target[target]

    # Normalize data
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)

    # Split source data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_source_scaled, y_source, test_size=0.2, random_state=42)

    # Apply KLIEP: Fit the adapter (pass y_source as well)
    adapter = KLIEP()
    adapter.fit(X_source_scaled, y_source, Xt=X_target_scaled)
    weights = adapter.predict(X_source_scaled)

    # Train a weighted Random Forest model on the source data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_source_scaled, y_source, sample_weight=weights)

    # Evaluate on the source test set
    y_pred_test = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

    # Predict on target (Algeria)
    y_pred_algeria = model.predict(X_target_scaled)
    df_target[f"Pred_KLIEP_{source}"] = y_pred_algeria

    # Evaluate on target data
    target_r2 = r2_score(y_target, y_pred_algeria)
    target_mse = mean_squared_error(y_target, y_pred_algeria)
    target_mape = mean_absolute_percentage_error(y_target, y_pred_algeria)

    results_kliep[source] = {
        "Source_R2": test_r2,
        "Source_MSE": test_mse,
        "Source_MAPE": test_mape,
        "Target_R2": target_r2,
        "Target_MSE": target_mse,
        "Target_MAPE": target_mape
    }

    print(f"Source R²: {test_r2:.3f}, MSE: {test_mse:.0f}, MAPE: {test_mape:.2%}")
    print(f"Target R²: {target_r2:.3f}, MSE: {target_mse:.0f}, MAPE: {target_mape:.2%}")

# ------------------------------
# Experiment with CORAL
# ------------------------------

results_coral = {}

for source in source_countries:
    print(f"\n[CORAL] Training model using {source}'s data...")

    # Filter data for the source country
    df_source = df[df[country_column] == source]

    # Prepare source data
    X_source = df_source[features]
    y_source = df_source[target]

    # Prepare target data (Algeria)
    X_target = df_target[features]
    y_target = df_target[target]

    # Normalize data
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)

    # Split source data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_source_scaled, y_source, test_size=0.2, random_state=42)

    # Apply CORAL: Fit adapter to align source distribution with target distribution
    adapter = CORAL()
    X_source_coral = adapter.fit_transform(X_source_scaled, X_target_scaled)
    X_test_coral = adapter.transform(X_test)
    X_target_coral = adapter.transform(X_target_scaled)

    # Train a Random Forest model on the aligned source features
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_source_coral, y_source)

    # Evaluate on source test set (aligned)
    y_pred_test = model.predict(X_test_coral)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)

    # Predict on target (Algeria) using aligned features
    y_pred_algeria = model.predict(X_target_coral)
    df_target[f"Pred_CORAL_{source}"] = y_pred_algeria

    # Evaluate on target data
    target_r2 = r2_score(y_target, y_pred_algeria)
    target_mse = mean_squared_error(y_target, y_pred_algeria)
    target_mape = mean_absolute_percentage_error(y_target, y_pred_algeria)

    results_coral[source] = {
        "Source_R2": test_r2,
        "Source_MSE": test_mse,
        "Source_MAPE": test_mape,
        "Target_R2": target_r2,
        "Target_MSE": target_mse,
        "Target_MAPE": target_mape
    }

    print(f"Source R²: {test_r2:.3f}, MSE: {test_mse:.0f}, MAPE: {test_mape:.2%}")
    print(f"Target R²: {target_r2:.3f}, MSE: {target_mse:.0f}, MAPE: {target_mape:.2%}")

# ------------------------------
# Compare Performance for KLIEP and CORAL
# ------------------------------

# Create DataFrames for each method's target metrics
df_kliep = pd.DataFrame.from_dict(results_kliep, orient="index")[["Target_R2", "Target_MSE", "Target_MAPE"]]
df_coral = pd.DataFrame.from_dict(results_coral, orient="index")[["Target_R2", "Target_MSE", "Target_MAPE"]]

df_kliep["Method"] = "KLIEP"
df_coral["Method"] = "CORAL"

# Combine the two DataFrames
comparison_df = pd.concat([df_kliep, df_coral])
comparison_df.index.name = "Source Country"
comparison_df = comparison_df.reset_index()

print("\nPerformance Comparison on Target (Algeria):")
print(comparison_df)

# Plot comparison of Target R² for each method by source country
plt.figure(figsize=(10,6))
sns.barplot(data=comparison_df, x="Source Country", y="Target_R2", hue="Method")
plt.title("Comparison of Target R² (Algeria) by Method and Source Country")
plt.ylabel("Target R²")
plt.tight_layout()
plt.show()

# Plot comparison of Target MAPE for each method by source country
plt.figure(figsize=(10,6))
sns.barplot(data=comparison_df, x="Source Country", y="Target_MAPE", hue="Method")
plt.title("Comparison of Target MAPE by Method and Source Country")
plt.ylabel("Target MAPE")
plt.tight_layout()
plt.show()
