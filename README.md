# Sales Prediction from Advertising Spend

A regression analysis and machine learning project that predicts product sales based on advertising budgets across three channels: TV, Radio, and Newspaper. Three models are compared to find the best predictor, with actionable business insights on where to allocate marketing budget.

## Results

| Model | R² Score | MAE ($000s) | RMSE ($000s) |
|---|---|---|---|
| Linear Regression | ~0.90 | ~1.1 | ~1.5 |
| Ridge Regression | ~0.90 | ~1.1 | ~1.5 |
| Random Forest | ~0.97 | ~0.6 | ~0.8 |

**Random Forest is the best model**, capturing non-linear relationships between ad spend and sales that linear models miss.

## Key Business Findings

| Channel | Correlation with Sales | Recommendation |
|---|---|---|
| TV | 0.78 (Strongest) | Maximize budget here — highest ROI |
| Radio | 0.58 (Moderate) | Good supplementary channel |
| Newspaper | 0.23 (Weakest) | Consider reallocating this budget to TV/Radio |

**TV advertising has the highest return on investment.** A $1,000 increase in TV spend produces a much larger sales increase than the same investment in Newspaper advertising.

## Sample Prediction

For a campaign with TV=$200k, Radio=$40k, Newspaper=$20k:
- **Predicted Sales: ~$16–18k** (varies by model)

## Output Charts

| File | Description |
|---|---|
| `plot1_distributions.png` | Distribution of TV, Radio, Newspaper spend and Sales |
| `plot2_correlation_heatmap.png` | Correlation matrix between all variables |
| `plot3_scatter_channels.png` | Scatter plots: each channel vs Sales with trend lines |
| `plot4_actual_vs_predicted.png` | Actual vs Predicted Sales (Linear Regression) |
| `plot5_residuals.png` | Residual plot for checking model assumptions |
| `plot6_feature_importance.png` | Coefficient chart showing each channel's impact |
| `plot7_model_comparison.png` | Side-by-side model performance comparison |
| `plot8_all_models_actual_vs_pred.png` | Actual vs Predicted for all 3 models |

## Project Structure

```
sales-prediction-advertising/
│
├── sales_prediction.py    # Main script
├── Advertising.csv        # Dataset (200 campaigns)
├── plot1_distributions.png
├── plot2_correlation_heatmap.png
├── ... (8 plots total)
└── README.md
```

## Dataset

- **Source:** Classic Advertising dataset (ISLR textbook)
- **Rows:** 200 advertising campaigns
- **Columns:** TV spend ($000s), Radio spend ($000s), Newspaper spend ($000s), Sales ($000s)

## How to Run

```bash
# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn

# Place Advertising.csv in the same folder, then run:
python sales_prediction.py
```

## Tech Stack

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (LinearRegression, Ridge, RandomForestRegressor)

## Author

**Fahim Ahmed**  
2nd Year Student, Urban & Regional Planning  
Bangladesh University of Engineering and Technology (BUET)  
[LinkedIn](https://www.linkedin.com/in/fahim-ahmed-585b26357) | [GitHub](https://github.com/fmad121581-hub)
