# FinLLM — Development Progress

**Status:** Weeks 1–3 completed (market-data preparation, baseline models, and tree-based model evaluation)  
**Last updated:** November 26, 2026  
**Repository:** [remythomasvo/FinLLM](https://github.com/remythomasvo/FinLLM)

## Project overview

FinLLM is a financial machine-learning research project investigating whether historical market indicators, and eventually transformer-derived financial news sentiment, can help predict the **next trading day's direction** of a stock's closing price. The first research ticker is NVIDIA (`NVDA`). This is an experimental forecasting project, not a proven trading strategy or a claim of profitable performance.

## Completed — Week 1: Market data and feature engineering

- Created a public GitHub repository, `FinLLM`, with a README, Python `.gitignore`, and MIT license.
- Created and saved `notebooks/01_market_data.ipynb` using Google Colab.
- Downloaded historical daily NVIDIA data using `yfinance`, with a dataset spanning early 2016 through November 2026. Also downloaded SPY data during initial exploration; the completed first model uses NVDA-derived features only.
- Inspected the market data with `.shape`, `.info()`, `.describe()`, and `.isnull().sum()`; the original NVDA OHLCV columns had no missing entries in the downloaded rows.
- Engineered four input features using pandas:

| Feature | Column | Calculation | Interpretation |
| --- | --- | --- | --- |
| One-day return | `return_1day` | `Close.pct_change()` | Most recent daily price movement |
| Five-day return | `return_5day` | `Close.pct_change(5)` | Five-session momentum |
| 20-day volatility | `volatility_20d` | `return_1day.rolling(20).std()` | Rolling variability of daily returns |
| Relative volume | `rel_vol` | `Volume / Volume.rolling(20).mean()` | Today's volume versus its recent average |

- Constructed the next-day classification target: `1` when the next trading session's closing price is above the current close, and `0` otherwise. The last unknown target was treated as missing, not incorrectly labeled `0`.
- Removed rows with missing feature or target values to make a machine-learning-ready dataset.
- Obtained **2,708 labeled observations** with **four input features** after cleaning (from 2,729 downloaded price rows).
- Saved the processed dataset as `data/nvda_features.csv` and the Colab notebook to GitHub.

### Week 1 data preparation example

```python
features = ["return_1day", "return_5day", "volatility_20d", "rel_vol"]
model_data = nvda.dropna(subset=features + ["target"]).copy()
X = model_data[features]
y = model_data["target"].astype(int)
```

**Data target distribution before splitting:** 1,470 up observations (`1`) and 1,238 not-up observations (`0`).

## Completed — Week 2: Logistic regression baseline

- Created `02_logistic_regression.ipynb` in Google Colab and loaded the processed dataset from the GitHub-hosted CSV.
- Separated predictors (`X`) from the target (`y`).
- Sorted observations chronologically and split the data into approximately **70% training, 15% validation, and 15% testing**.
- Conservatively removed the last observation from the training and validation segments so each segment's final label would not depend on the next segment's closing price.
- Used `StandardScaler`: fitted on training features only, and used the fitted scaler to transform validation and test features. Verified the scaled training features had means near 0 and standard deviations of 1.
- Trained a `sklearn.linear_model.LogisticRegression(max_iter=1000)` model on the scaled training data.
- Generated class predictions and estimated probabilities on the validation set.
- Evaluated accuracy and compared against an always-up baseline.
- Inspected the validation confusion matrix and class precision, recall, and F1 scores.
- Documented model results in the notebook; the conversation indicates a GitHub commit was being prepared, but the final Week 2 commit itself was not independently verified.

### Data split

| Segment | Observations | Purpose |
| --- | ---: | --- |
| Training | 1,894 | Fit scaler and regression model |
| Validation | 405 | Evaluate and compare candidate models |
| Test | 407 | Reserved for final out-of-sample evaluation |

These counts include the two intentionally omitted boundary rows.

### Initial validation results

| Metric | Result |
| --- | ---: |
| Logistic regression accuracy | **55.31%** (224/405) |
| Always-up accuracy | **55.56%** (225/405) |
| Difference vs. always-up | **−0.25 percentage points** |
| Logistic regression macro-F1 | **~0.42** |

**Confusion matrix** (`rows = actual`, `columns = predicted`; class `0` = not up, class `1` = up):

| Actual / predicted | 0 | 1 |
| --- | ---: | ---: |
| 0 | 16 | 164 |
| 1 | 17 | 208 |

**Per-class metrics reported by scikit-learn:**

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Not up (`0`) | 0.48 | 0.09 | 0.15 | 180 |
| Up (`1`) | 0.56 | 0.92 | 0.70 | 225 |

### Findings and interpretation

- The logistic regression model **did not exceed the always-up baseline in overall validation accuracy**. It made one fewer correct prediction than always predicting an up day.
- The model predicted an up day for **372 of 405** validation observations (~91.9%), versus **225 actual up days** (~55.6%).
- Recall for not-up days was only about **9%** (16/180), demonstrating that the model struggled to distinguish negative/flat days.
- The four market indicators, together with this linear model and training setup, have **not yet demonstrated meaningful incremental predictive accuracy**. This is a useful baseline result rather than a project failure.
- No conclusion about performance on the untouched test set, statistical significance, or profitability has been established.

## Completed — Week 3: Tree-based model evaluation

- Trained and evaluated Random Forest and XGBoost using the same chronological training and validation split as the Week 2 baseline.
- Compared both models with logistic regression and the always-up baseline on the **405-row validation set**.
- Recorded accuracy, macro-F1, balanced accuracy, and confusion matrices for the tree-based models.
- Kept the **407-row final test set untouched**; no final test result is claimed.

### Validation results

| Model | Accuracy | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Always-up baseline | **55.56%** | — | — |
| Logistic regression | **55.31%** | — | — |
| Random Forest | **54.32%** | **0.4308** | **0.500** |
| XGBoost | **54.57%** | **0.4889** | **0.515** |

Confusion matrices use rows for actual classes and columns for predicted classes; class `0` means not up and class `1` means up.

**Random Forest:**

```text
[[ 20 160]
 [ 25 200]]
```

**XGBoost:**

```text
[[ 43 137]
 [ 47 178]]
```

### Findings and interpretation

- XGBoost has the strongest class-balanced results among the tested models: macro-F1 **0.4889** and balanced accuracy **0.515**. It correctly identifies 43 of 180 not-up days, compared with 20 of 180 for Random Forest, though it still misses most not-up days.
- This improved class balance does **not** improve overall accuracy over the always-up baseline: XGBoost scores **54.57%**, versus **55.56%** for always-up. Logistic regression (**55.31%**) and Random Forest (**54.32%**) are also below the baseline.
- The results suggest **limited predictive signal** from the current features and model setup. They do not establish useful predictive performance or profitability.
- Validation was used for model comparison only. The final test set remains reserved for a later out-of-sample evaluation.

## Repository structure

```text
FinLLM/
├── README.md
├── .gitignore
├── LICENSE
├── data/
│   ├── .gitkeep
│   └── nvda_features.csv
└── notebooks/
    ├── 01_market_data.ipynb
    └── 02_logistic_regression.ipynb  # created in Colab; verify GitHub commit
```

## Next steps — Week 4 and beyond

- [ ] Confirm that the Week 2 logistic-regression notebook is committed to `notebooks/` on GitHub.
- [ ] Use validation data for model selection; keep the **407-row test set untouched** until final evaluation.
- [ ] Add stronger market-context features, such as SPY returns, with careful historical alignment.
- [ ] Integrate **FinBERT** financial news sentiment, using historical headlines with verified publication timestamps.
- [ ] Evaluate market-only versus news-only versus combined models using controlled, leakage-resistant tests.
- [ ] Conduct walk-forward evaluation and document uncertainty, possible look-ahead bias, and limitations.
- [ ] Publish a research summary and, optionally, an interactive demo.

## Reproducibility and cautions

The current `nvda_features.csv` is a historical processed snapshot. Keep exact data-fetch dates, adjusted-price settings, dependencies, and train/validation/test boundaries documented as the project matures. Only use information observable at the prediction cutoff; next-day closing prices must **never** be model inputs. Scaling and all learned transformations must be fit on training data only. Historical accuracy does not itself imply a tradable edge, especially without transaction-cost and execution assumptions.
