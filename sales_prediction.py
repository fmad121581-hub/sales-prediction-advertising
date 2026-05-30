# =============================================================================
# Sales Prediction using Python
# Author: Fahim Ahmed | BUET Urban & Regional Planning
# Dataset: Advertising.csv (TV, Radio, Newspaper spend vs Sales)
# =============================================================================

# --- STEP 1: IMPORT LIBRARIES ---
# pandas: for loading and manipulating tabular data
# numpy: for numerical operations
# matplotlib & seaborn: for creating charts and visualizations
# sklearn: for building and evaluating the machine learning model

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set a clean visual style for all plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120


# =============================================================================
# STEP 2: LOAD AND EXPLORE THE DATA
# =============================================================================

# Load the CSV file into a DataFrame
# The first column is just a row index, so we drop it with index_col=0
df = pd.read_csv("Advertising.csv", index_col=0)

print("=" * 55)
print("         SALES PREDICTION — ADVERTISING DATASET")
print("=" * 55)

# Shape tells us (rows, columns)
print(f"\n📦 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Show the first 5 rows so we understand the structure
print("\n📋 First 5 Rows:")
print(df.head())

# .info() shows column names, data types, and if any values are missing
print("\n🔍 Dataset Info:")
print(df.info())

# .describe() gives statistical summary: mean, min, max, std, quartiles
print("\n📊 Statistical Summary:")
print(df.describe().round(2))

# Check for missing values — important before modeling
missing = df.isnull().sum()
print(f"\n✅ Missing Values:\n{missing}")
# Result: no missing values in this dataset


# =============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA) — VISUALIZATIONS
# =============================================================================

# --- Plot 1: Distribution of each variable ---
# This tells us if data is skewed or normally distributed
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Distribution of Advertising Spend & Sales", fontsize=14, fontweight='bold')

columns = ['TV', 'Radio', 'Newspaper', 'Sales']
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

for i, (col, color) in enumerate(zip(columns, colors)):
    # histplot draws a histogram with a KDE (smooth density curve) on top
    sns.histplot(df[col], ax=axes[i], color=color, kde=True, bins=20)
    axes[i].set_title(col)
    axes[i].set_xlabel("Value")

plt.tight_layout()
plt.savefig("plot1_distributions.png", bbox_inches='tight')
plt.close()
print("\n✅ Saved: plot1_distributions.png")


# --- Plot 2: Correlation Heatmap ---
# Correlation ranges from -1 to +1
# +1 = strong positive relationship, 0 = no relationship, -1 = inverse
fig, ax = plt.subplots(figsize=(7, 5))
corr_matrix = df.corr()  # compute pairwise correlations between all columns

sns.heatmap(
    corr_matrix,
    annot=True,          # show the number inside each cell
    fmt=".2f",           # 2 decimal places
    cmap="coolwarm",     # blue = negative, red = positive
    linewidths=0.5,
    ax=ax
)
ax.set_title("Correlation Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("plot2_correlation_heatmap.png", bbox_inches='tight')
plt.close()
print("✅ Saved: plot2_correlation_heatmap.png")


# --- Plot 3: Scatter plots — each ad channel vs Sales ---
# This visually shows how each feature relates to our target (Sales)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Advertising Spend vs Sales", fontsize=14, fontweight='bold')

features = ['TV', 'Radio', 'Newspaper']
colors = ['#4C72B0', '#DD8452', '#55A868']

for i, (feat, color) in enumerate(zip(features, colors)):
    axes[i].scatter(df[feat], df['Sales'], alpha=0.6, color=color, edgecolors='white', s=60)
    # Add a trend line using numpy's polyfit (degree 1 = straight line)
    m, b = np.polyfit(df[feat], df['Sales'], 1)  # m = slope, b = intercept
    x_line = np.linspace(df[feat].min(), df[feat].max(), 100)
    axes[i].plot(x_line, m * x_line + b, color='black', linewidth=1.5, linestyle='--', label='Trend')
    axes[i].set_xlabel(f"{feat} Spend ($000s)")
    axes[i].set_ylabel("Sales ($000s)")
    axes[i].set_title(f"{feat} vs Sales")
    axes[i].legend()

plt.tight_layout()
plt.savefig("plot3_scatter_plots.png", bbox_inches='tight')
plt.close()
print("✅ Saved: plot3_scatter_plots.png")


# =============================================================================
# STEP 4: PREPARE DATA FOR MODELING
# =============================================================================

# X = features (inputs to the model)
# y = target (what we want to predict)
X = df[['TV', 'Radio', 'Newspaper']]  # feature matrix
y = df['Sales']                        # target variable

# Split data: 80% for training, 20% for testing
# random_state=42 ensures the same split every time you run the script
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n📐 Training set size: {X_train.shape[0]} samples")
print(f"📐 Testing set size:  {X_test.shape[0]} samples")


# =============================================================================
# STEP 5: TRAIN THE LINEAR REGRESSION MODEL
# =============================================================================

# LinearRegression finds the best-fit line:
# Sales = b0 + b1*TV + b2*Radio + b3*Newspaper
model = LinearRegression()
model.fit(X_train, y_train)  # this is where the model "learns" from training data

# Display the learned coefficients
print("\n📈 Model Coefficients (what the model learned):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"   {feature:12s}: {coef:.4f}")
print(f"   {'Intercept':12s}: {model.intercept_:.4f}")

print("""
💡 How to read coefficients:
   TV = 0.046 means: for every $1000 increase in TV spend,
   Sales increase by ~$46 (all else held constant).
""")


# =============================================================================
# STEP 6: EVALUATE THE MODEL
# =============================================================================

# Use the trained model to predict Sales on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("=" * 40)
print("       MODEL EVALUATION RESULTS")
print("=" * 40)
print(f"  MAE  (Mean Absolute Error):  {mae:.3f}")
print(f"  RMSE (Root Mean Sq. Error):  {rmse:.3f}")
print(f"  R²   (R-squared Score):      {r2:.4f}")
print("=" * 40)

print(f"""
💡 What these numbers mean:
   MAE  = {mae:.2f} → On average, predictions are off by ~${mae:.2f}k in sales
   RMSE = {rmse:.2f} → Similar to MAE but penalizes large errors more
   R²   = {r2:.4f} → The model explains {r2*100:.1f}% of variance in Sales
          (1.0 = perfect, 0.0 = no better than guessing the mean)
""")


# --- Plot 4: Actual vs Predicted Sales ---
# A good model's points should cluster tightly along the diagonal line
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test, y_pred, alpha=0.7, color='#4C72B0', edgecolors='white', s=70)

# Draw the "perfect prediction" diagonal line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect Prediction')

ax.set_xlabel("Actual Sales ($000s)", fontsize=12)
ax.set_ylabel("Predicted Sales ($000s)", fontsize=12)
ax.set_title(f"Actual vs Predicted Sales\n(R² = {r2:.4f})", fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig("plot4_actual_vs_predicted.png", bbox_inches='tight')
plt.close()
print("✅ Saved: plot4_actual_vs_predicted.png")


# --- Plot 5: Residual Plot ---
# Residuals = actual - predicted
# A good model should have residuals randomly scattered around 0 (no pattern)
residuals = y_test - y_pred

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(y_pred, residuals, alpha=0.7, color='#DD8452', edgecolors='white', s=70)
ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5)  # zero line
ax.set_xlabel("Predicted Sales ($000s)", fontsize=12)
ax.set_ylabel("Residuals (Actual − Predicted)", fontsize=12)
ax.set_title("Residual Plot — Checking Model Assumptions", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("plot5_residuals.png", bbox_inches='tight')
plt.close()
print("✅ Saved: plot5_residuals.png")


# --- Plot 6: Feature Importance (Coefficient Bar Chart) ---
# This shows which advertising channel has the most impact on sales
fig, ax = plt.subplots(figsize=(7, 4))
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

colors_bar = ['#4C72B0' if c > 0 else '#C44E52' for c in coef_df['Coefficient']]
bars = ax.bar(coef_df['Feature'], coef_df['Coefficient'], color=colors_bar, edgecolor='white')

# Add value labels on top of each bar
for bar, val in zip(bars, coef_df['Coefficient']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_title("Feature Coefficients — Impact on Sales", fontsize=13, fontweight='bold')
ax.set_ylabel("Coefficient Value")
ax.set_xlabel("Advertising Channel")
plt.tight_layout()
plt.savefig("plot6_feature_importance.png", bbox_inches='tight')
plt.close()
print("✅ Saved: plot6_feature_importance.png")


# =============================================================================
# STEP 7: BUSINESS INSIGHTS & SAMPLE PREDICTION
# =============================================================================

print("\n" + "=" * 55)
print("          BUSINESS INSIGHTS")
print("=" * 55)

# Correlation values for insight summary
corr = df.corr()['Sales'].drop('Sales')
print(f"""
📺 TV Advertising:
   Correlation with Sales: {corr['TV']:.3f} (STRONGEST)
   → Highest return on investment. Priority channel.

📻 Radio Advertising:
   Correlation with Sales: {corr['Radio']:.3f} (MODERATE-HIGH)
   → Good supplementary channel. Worth investing in.

📰 Newspaper Advertising:
   Correlation with Sales: {corr['Newspaper']:.3f} (WEAKEST)
   → Lowest impact on sales. Consider reallocating budget.

✅ Recommendation: Maximize TV spend first,
   then Radio. Reduce Newspaper budget.
""")

# Demo prediction — show what the model predicts for a new scenario
print("=" * 55)
print("   SAMPLE PREDICTION (New Advertising Budget)")
print("=" * 55)

# Predict sales for a new campaign: TV=$200k, Radio=$40k, Newspaper=$20k
new_campaign = pd.DataFrame({
    'TV': [200],
    'Radio': [40],
    'Newspaper': [20]
})
predicted_sales = model.predict(new_campaign)[0]
print(f"""
   Budget:    TV = $200k | Radio = $40k | Newspaper = $20k
   ➡ Predicted Sales: ${predicted_sales:.2f}k
""")

print("=" * 55)
print("  Base model done. Now running model comparison...")
print("=" * 55)


# =============================================================================
# STEP 8: MODEL COMPARISON — Linear Regression vs Ridge vs Random Forest
# =============================================================================

# --- Why compare models? ---
# Linear Regression assumes a straight-line relationship between features and target.
# Ridge Regression is similar but adds a penalty (alpha) to shrink coefficients —
#   this helps when features might be correlated with each other (multicollinearity).
# Random Forest builds many decision trees and averages their predictions —
#   it can capture non-linear patterns that linear models miss.

# We'll train all 3 on the SAME train/test split so the comparison is fair.

# Define all three models in a dictionary
# alpha=1.0 is Ridge's regularization strength — a standard starting value
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression":  Ridge(alpha=1.0),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42)
    # n_estimators=100 means we build 100 decision trees and average their output
    # random_state=42 ensures reproducibility
}

# This dictionary will store the results for each model
results = {}

print("\n Training and evaluating all 3 models...\n")

for name, mdl in models.items():
    # Train the model on training data
    mdl.fit(X_train, y_train)

    # Predict on test data
    preds = mdl.predict(X_test)

    # Calculate metrics
    mae_val  = mean_absolute_error(y_test, preds)
    rmse_val = np.sqrt(mean_squared_error(y_test, preds))
    r2_val   = r2_score(y_test, preds)

    # Store results
    results[name] = {
        "MAE":  mae_val,
        "RMSE": rmse_val,
        "R2":   r2_val,
        "preds": preds
    }

    print(f"  {name}")
    print(f"    MAE:  {mae_val:.3f} | RMSE: {rmse_val:.3f} | R2: {r2_val:.4f}")
    print()

# Identify the best model by R2 score
best_name = max(results, key=lambda m: results[m]["R2"])
print(f"Best Model: {best_name} (R2 = {results[best_name]['R2']:.4f})")


# --- Plot 7: Model Comparison Bar Chart ---
# Side-by-side bars for MAE, RMSE, R2 across all 3 models
# This is the "money chart" for your LinkedIn post

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model Comparison: Linear Regression vs Ridge vs Random Forest",
             fontsize=13, fontweight='bold')

model_names  = list(results.keys())
short_names  = ["Linear\nRegression", "Ridge\nRegression", "Random\nForest"]
bar_colors   = ['#4C72B0', '#55A868', '#DD8452']

metrics       = ["MAE", "RMSE", "R2"]
metric_labels = ["MAE (lower is better)", "RMSE (lower is better)", "R2 Score (higher is better)"]

for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
    values = [results[m][metric] for m in model_names]
    bars = axes[i].bar(short_names, values, color=bar_colors, edgecolor='white', width=0.5)

    for bar, val in zip(bars, values):
        axes[i].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )

    axes[i].set_title(label, fontsize=11)
    axes[i].set_ylim(0, max(values) * 1.2)
    axes[i].set_ylabel(metric)

plt.tight_layout()
plt.savefig("plot7_model_comparison.png", bbox_inches='tight')
plt.close()
print("Saved: plot7_model_comparison.png")


# --- Plot 8: Actual vs Predicted for all 3 models (side by side) ---

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Actual vs Predicted Sales — All Models", fontsize=13, fontweight='bold')

for i, (name, short) in enumerate(zip(model_names, short_names)):
    preds  = results[name]["preds"]
    r2_val = results[name]["R2"]

    axes[i].scatter(y_test, preds, alpha=0.7, color=bar_colors[i], edgecolors='white', s=60)

    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

    axes[i].set_title(f"{short.replace(chr(10), ' ')}\nR2 = {r2_val:.4f}", fontsize=11)
    axes[i].set_xlabel("Actual Sales")
    axes[i].set_ylabel("Predicted Sales")

plt.tight_layout()
plt.savefig("plot8_all_models_actual_vs_pred.png", bbox_inches='tight')
plt.close()
print("Saved: plot8_all_models_actual_vs_pred.png")


# --- Final sample prediction using the best model ---
print("\n" + "=" * 55)
print(f"  FINAL PREDICTION using best model: {best_name}")
print("=" * 55)

best_model = models[best_name]
final_pred = best_model.predict(new_campaign)[0]
print(f"   Budget:    TV = $200k | Radio = $40k | Newspaper = $20k")
print(f"   Predicted Sales ({best_name}): ${final_pred:.2f}k")

print("\n" + "=" * 55)
print("  All 8 plots saved. Script complete!")
print("=" * 55)
