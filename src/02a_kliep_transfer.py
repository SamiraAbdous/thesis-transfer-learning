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

# Import the KLIEP adaptation method
from adapt.instance_based import KLIEP

# ------------------------------
# Load and Prepare the Data
# ------------------------------

# Load your dataset (update the file path as needed)
file_path = "data/ind.xlsx"  # panel of tourism indicators, 2000-2020
df = pd.read_excel(file_path)

# Clean column names: strip whitespace and replace multiple spaces with one
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

# Specify the country column, features, and target
country_column = "id"
features = ["Tourism expenditures", "Tourism receipts", 
            "receipts_per_tourist", "expenditures_per_tourist", "tourism_balance"]
target = "Number of tourists"

# ------------------------------
# Visualize Tourism Trends
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
# Domain Adaptation using KLIEP
# ------------------------------

# Initialize a dictionary to store results for each source country
results = {}

# Get target (Algeria) data
df_target = df[df[country_column] == target_country].copy()

# Loop through each source country
for source in source_countries:
    print(f"\nTraining KLIEP model using {source}'s data...")
    
    # Filter data for the source country
    df_source = df[df[country_column] == source]
    
    # Prepare training data
    X_source = df_source[features]
    y_source = df_source[target]
    
    # For target (Algeria)
    X_target = df_target[features]
    y_target = df_target[target]
    
    # Normalize the data
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    # Split source data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_source_scaled, y_source, test_size=0.2, random_state=42)
    
    # Apply KLIEP adaptation
    adapter = KLIEP()
    # Fit adapter with target data via Xt and source labels
    adapter.fit(X_source_scaled, y_source, Xt=X_target_scaled)
    weights = adapter.predict(X_source_scaled)
    
    # Train a weighted Random Forest model using the computed weights
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_source_scaled, y_source, sample_weight=weights)
    
    # Evaluate on source test data
    y_pred_test = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
    
    # Predict for Algeria (target)
    y_pred_algeria = model.predict(X_target_scaled)
    df_target[f"Pred_KLIEP_{source}"] = y_pred_algeria
    
    # Calculate error metrics on target data
    target_r2 = r2_score(y_target, y_pred_algeria)
    target_mse = mean_squared_error(y_target, y_pred_algeria)
    target_mape = mean_absolute_percentage_error(y_target, y_pred_algeria)
    
    # Store results for this source country
    results[source] = {
        "Source_R2": test_r2,
        "Source_MSE": test_mse,
        "Source_MAPE": test_mape,
        "Target_R2": target_r2,
        "Target_MSE": target_mse,
        "Target_MAPE": target_mape,
        "y_pred": y_pred_algeria,
        "model": model
    }
    
    print(f"Source R² Score: {test_r2:.3f}, MSE: {test_mse:.0f}, MAPE: {test_mape:.2%}")
    print(f"Target R² Score: {target_r2:.3f}, MSE: {target_mse:.0f}, MAPE: {target_mape:.2%}")

# ------------------------------
# Compare Performance Across Source Countries (KLIEP)
# ------------------------------

eval_df = pd.DataFrame.from_dict(
    {k: {m: v[m] for m in ["Target_R2", "Target_MSE", "Target_MAPE"]}
     for k, v in results.items()},
    orient="index"
)
eval_df = eval_df.sort_values(by="Target_R2", ascending=False)
print("\nKLIEP - Performance Comparison for Algeria:")
print(eval_df)

# Plot predictions for KLIEP
plt.figure(figsize=(12, 6))
plt.plot(df_target["year"], df_target[target], 
         label="Actual Tourists", marker="o", linestyle="-", linewidth=2)

for source in source_countries:
    pred_col = f"Pred_KLIEP_{source}"
    plt.plot(df_target["year"], df_target[pred_col], 
             label=f"{source} Model", marker="x", linestyle="--")

plt.xlabel("Year")
plt.ylabel("Number of Tourists")
plt.title("KLIEP - Actual vs Predicted Tourists for Algeria")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------------------
# Identify the Best Overall Model
# ------------------------------

best_country = max(results, key=lambda k: results[k]["Target_R2"])
best_r2 = results[best_country]["Target_R2"]
print(f"\n🏆 Best Overall KLIEP Model: using {best_country}'s data (Target R² = {best_r2:.3f})")

# ------------------------------
# Feature Importance from the Best Model
# ------------------------------

best_model = results[best_country]["model"]

if hasattr(best_model, "feature_importances_"):
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": best_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis")
    plt.title(f"Feature Importance (KLIEP model using {best_country}'s data)")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()

# ------------------------------
# Additional Visualizations
# ------------------------------

# Tourism expenditure vs receipts comparison
plt.figure(figsize=(12, 8))
for country in source_countries + [target_country]:
    country_data = df[df[country_column] == country]
    plt.scatter(country_data["Tourism expenditures"], 
                country_data["Tourism receipts"],
                label=country, alpha=0.7, s=50)
plt.xlabel("Tourism Expenditures")
plt.ylabel("Tourism Receipts")
plt.title("Tourism Expenditures vs Receipts by Country")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Calculate tourism ROI (Return on Investment)
df['tourism_roi'] = (df['Tourism receipts'] / df['Tourism expenditures']) * 100

# Plot tourism ROI trends
plt.figure(figsize=(12, 8))
for country in source_countries + [target_country]:
    country_data = df[df[country_column] == country]
    plt.plot(country_data["year"], country_data["tourism_roi"], 
             label=country, marker='o')
plt.axhline(y=100, color='r', linestyle='--', alpha=0.3, 
            label="Break-even (receipts = expenditures)")
plt.xlabel("Year")
plt.ylabel("Tourism ROI (Receipts/Expenditures × 100%)")
plt.title("Tourism ROI Comparison Across Countries")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
