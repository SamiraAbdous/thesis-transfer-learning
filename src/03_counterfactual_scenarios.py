import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# For demonstration, assume the following helper function is defined:
def forecast_features(df_target, features, start_year, num_years=10):
    """
    Compute a simple average annual growth rate for each feature from historical data 
    and extrapolate the value for each future year.
    """
    future_years = np.arange(start_year, start_year + num_years)
    forecasted_data = {}
    for feat in features:
        df_sorted = df_target.sort_values("year")
        first_val = df_sorted[feat].iloc[0]
        last_val = df_sorted[feat].iloc[-1]
        n = len(df_sorted) - 1
        growth_rate = (last_val / first_val) ** (1/n) - 1 if first_val > 0 and n > 0 else 0
        forecasted_vals = [last_val * ((1 + growth_rate) ** (i + 1)) for i in range(num_years)]
        forecasted_data[feat] = forecasted_vals
    future_features = pd.DataFrame(forecasted_data)
    future_features["year"] = future_years
    return future_features
# --- Assumed Predefined Variables ---
# df: your original DataFrame
# country_column: e.g., "id"
# target: e.g., "Number of tourists"
# features: e.g.,
#   ["Tourism expenditures", "Tourism receipts", "receipts_per_tourist",
#    "expenditures_per_tourist", "tourism_balance"]
# scaler: the fitted StandardScaler on your training features
# best_model: your trained model (for instance, one trained on Jordan's data)
# Filter Algeria's historical data
df_algeria = df[df[country_column] == "Algeria"]
# Forecast Algeria's future features (baseline scenario) for the next 10 years (assume current year = 2023)
future_features_baseline = forecast_features(df_algeria, features, start_year=2023, num_years=10)
print("Baseline Forecasted Future Features:")
print(future_features_baseline)
# Prepare to store forecasts from different scenarios
future_forecasts = []
# Baseline Scenario: Use Algeria's forecasted features as is
X_future_baseline = scaler.transform(future_features_baseline[features].values)
baseline_predictions = best_model.predict(X_future_baseline)
df_forecast_baseline = future_features_baseline[["year"]].copy()
df_forecast_baseline["Predicted Number of Tourists"] = baseline_predictions
df_forecast_baseline["Scenario"] = "Baseline"
future_forecasts.append(df_forecast_baseline)
# Define the source countries to simulate policy adoption
source_countries = ["Egypt", "Morocco", "Jordan", "Tunisia"]
# For each source country, compute multipliers and adjust Algeria's forecasted features accordingly.
for source in source_countries:
    # Filter the source country's historical data
    df_source = df[df[country_column] == source]
    # Compute multipliers for each feature: (source mean / Algeria mean)
    multipliers = {}
    for feat in features:
        source_mean = df_source[feat].mean()
        algeria_mean = df_algeria[feat].mean()
        multipliers[feat] = source_mean / algeria_mean if algeria_mean != 0 else 1
    print(f"Multipliers for {source}: {multipliers}")
    # Create a counterfactual future scenario: adjust the baseline future features by the computed multipliers
    future_features_policy = future_features_baseline.copy()
    for feat in features:
        future_features_policy[feat] = future_features_policy[feat] * multipliers[feat]
    # Transform the adjusted features using the same scaler
    X_future_policy = scaler.transform(future_features_policy[features].values)
    # Predict future tourist numbers using the best_model
    policy_predictions = best_model.predict(X_future_policy)
    # Store predictions in a DataFrame
    df_forecast_policy = future_features_policy[["year"]].copy()
    df_forecast_policy["Predicted Number of Tourists"] = policy_predictions
    df_forecast_policy["Scenario"] = f"Policy: {source}"
    future_forecasts.append(df_forecast_policy)
# Combine all forecast scenarios into one DataFrame for comparison
comparison_forecast = pd.concat(future_forecasts)
comparison_forecast.index.name = "Index"
comparison_forecast = comparison_forecast.reset_index(drop=True)
print("Comparison of Future Forecasts:")
print(comparison_forecast)
# Plot the comparison of future forecasts
plt.figure(figsize=(10, 6))
sns.lineplot(data=comparison_forecast, x="year", y="Predicted Number of Tourists", hue="Scenario", marker="o")
plt.xlabel("Year")
plt.ylabel("Predicted Number of Tourists")
plt.title("Counterfactual Forecasts for Algeria under Different Policy Scenarios")
plt.grid(True)
plt.tight_layout()
plt.show()
