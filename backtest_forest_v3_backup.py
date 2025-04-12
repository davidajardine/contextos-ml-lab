import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, learning_curve
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting backtest_forest_full_tuned.py")

# --- Data Loading from SQLite ---
logging.info("Connecting to SQLite database 'bybit_data.db'")
conn = sqlite3.connect('bybit_data.db')

logging.info("Loading tables: btcusdt_daily, btcusdt_3h, btcusdt_asia")
df_daily = pd.read_sql_query("SELECT * FROM btcusdt_daily", conn)
df_3h = pd.read_sql_query("SELECT * FROM btcusdt_3h", conn)
df_asia = pd.read_sql_query("SELECT * FROM btcusdt_asia", conn)
conn.close()
logging.info("Data loaded successfully.")

# Convert datetime columns to datetime objects
logging.info("Converting datetime columns")
df_daily['datetime'] = pd.to_datetime(df_daily['datetime'])
df_3h['datetime'] = pd.to_datetime(df_3h['datetime'])
df_asia['datetime'] = pd.to_datetime(df_asia['datetime'])

# Extract date for merging/comparison
df_daily['date'] = df_daily['datetime'].dt.date
df_3h['date'] = df_3h['datetime'].dt.date
df_asia['date'] = df_asia['datetime'].dt.date

# Precompute additional daily indicator: 5-day EMA on daily closes, shifted by 1 day.
df_daily.sort_values(by='date', inplace=True)
df_daily['ema_5'] = df_daily['close'].ewm(span=5, adjust=False).mean().shift(1)

# --- Feature Engineering from 3H candles ---
def extract_prev_day_candle_features(current_day):
    """
    For a given current day, use the previous day's last two 3H candles:
    P1: 18:00 - 20:59 and P2: 21:00 - 23:59.
    Compute features:
      - P1_range, P2_range: range = high - low
      - P1_body, P2_body: body = |close - open|
      - P2_P1_close_diff: difference in close between second and first candle.
      - P1_wick_ratio, P2_wick_ratio: (range - body) / range (normalized wick size)
    """
    prev_day = current_day - timedelta(days=1)
    candles = df_3h[(df_3h['date'] == prev_day.date()) & (df_3h['datetime'].dt.hour >= 18)]
    logging.info(f"Date {current_day.date()}: Found {len(candles)} previous-day candles for {prev_day.date()} with hour>=18.")
    
    if candles.empty or len(candles) < 2:
        return pd.Series({
            'P1_range': np.nan,
            'P2_range': np.nan,
            'P1_body': np.nan,
            'P2_body': np.nan,
            'P2_P1_close_diff': np.nan,
            'P1_wick_ratio': np.nan,
            'P2_wick_ratio': np.nan
        })
    
    candles = candles.sort_values(by='datetime')
    try:
        P1 = candles.iloc[0]
        P2 = candles.iloc[1]
    except IndexError:
        return pd.Series({
            'P1_range': np.nan,
            'P2_range': np.nan,
            'P1_body': np.nan,
            'P2_body': np.nan,
            'P2_P1_close_diff': np.nan,
            'P1_wick_ratio': np.nan,
            'P2_wick_ratio': np.nan
        })
    
    P1_range = P1['high'] - P1['low']
    P2_range = P2['high'] - P2['low']
    P1_body = abs(P1['close'] - P1['open'])
    P2_body = abs(P2['close'] - P2['open'])
    # Calculate normalized wick ratios; protect against division by zero.
    P1_wick_ratio = ((P1_range - P1_body) / P1_range) if P1_range != 0 else 0
    P2_wick_ratio = ((P2_range - P2_body) / P2_range) if P2_range != 0 else 0
    P2_P1_close_diff = P2['close'] - P1['close']
    
    return pd.Series({
        'P1_range': P1_range,
        'P2_range': P2_range,
        'P1_body': P1_body,
        'P2_body': P2_body,
        'P2_P1_close_diff': P2_P1_close_diff,
        'P1_wick_ratio': P1_wick_ratio,
        'P2_wick_ratio': P2_wick_ratio
    })

def extract_asia_feature(current_day):
    """
    For the given current day, get the Asian session midline from btcusdt_asia.
    """
    asia_row = df_asia[df_asia['date'] == current_day.date()]
    if asia_row.empty:
        return np.nan
    return asia_row['midline'].iloc[0]

# --- Building Full Feature Dataset ---
logging.info("Building full feature dataset with additional indicators")
features = []
# Loop through each daily row (skip days without prior data)
for idx, row in df_daily.iterrows():
    current_day = datetime.combine(row['date'], datetime.min.time())
    prev_candle_feats = extract_prev_day_candle_features(current_day)
    asia_midline = extract_asia_feature(current_day)
    
    prev_day = current_day - timedelta(days=1)
    prev_daily = df_daily[df_daily['date'] == prev_day.date()]
    if prev_daily.empty:
        continue
    prev_daily = prev_daily.iloc[0]
    prev_daily_range = prev_daily['high'] - prev_daily['low']
    prev_daily_pct_change = (prev_daily['close'] - prev_daily['open']) / prev_daily['open'] if prev_daily['open'] != 0 else 0
    prev_atr = prev_daily_range  # Proxy for ATR
    
    # 5-day EMA from previous day (already computed)
    prev_ema_5_series = df_daily[df_daily['date'] == prev_day.date()]['ema_5']
    ema_5_val = prev_ema_5_series.iloc[0] if not prev_ema_5_series.empty else np.nan
    
    ratio_asia_to_open = (asia_midline / row['open']) if row['open'] != 0 else np.nan
    diff_open_close = row['close'] - row['open']
    
    # Convert SND to binary (case-insensitive)
    target = 1 if str(row['SND']).strip().upper() == 'TRUE' else 0

    features.append({
        'date': row['date'],
        'P1_range': prev_candle_feats['P1_range'],
        'P2_range': prev_candle_feats['P2_range'],
        'P1_body': prev_candle_feats['P1_body'],
        'P2_body': prev_candle_feats['P2_body'],
        'P2_P1_close_diff': prev_candle_feats['P2_P1_close_diff'],
        'P1_wick_ratio': prev_candle_feats['P1_wick_ratio'],
        'P2_wick_ratio': prev_candle_feats['P2_wick_ratio'],
        'asia_midline': asia_midline,
        'daily_open': row['open'],
        'daily_close': row['close'],
        'prev_daily_range': prev_daily_range,
        'prev_daily_pct_change': prev_daily_pct_change,
        'prev_atr': prev_atr,
        'ema_5': ema_5_val,
        'ratio_asia_to_open': ratio_asia_to_open,
        'diff_open_close': diff_open_close,
        'target': target
    })

df_features = pd.DataFrame(features).dropna()
if df_features.empty:
    logging.error("No features were built. Exiting.")
    sys.exit("Error: Feature dataset is empty.")

logging.info(f"Feature dataset built with {len(df_features)} records.")

# --- Sort and Split Data (Time-Series based 70/30) ---
logging.info("Sorting and splitting data into training and testing sets")
df_features.sort_values(by='date', inplace=True)
n = len(df_features)
train_end = int(n * 0.7)
df_train = df_features.iloc[:train_end]
df_test = df_features.iloc[train_end:]

# Define full feature columns list (now 16 features)
feature_cols = [
    'P1_range', 'P2_range', 'P1_body', 'P2_body', 'P2_P1_close_diff',
    'P1_wick_ratio', 'P2_wick_ratio', 'asia_midline', 'daily_open', 'daily_close',
    'prev_daily_range', 'prev_daily_pct_change', 'prev_atr', 'ema_5',
    'ratio_asia_to_open', 'diff_open_close'
]
X_train = df_train[feature_cols]
y_train = df_train['target']
X_test = df_test[feature_cols]
y_test = df_test['target']

logging.info(f"Training set: {len(X_train)} records; Testing set: {len(X_test)} records")

# --- Hyperparameter Tuning using GridSearchCV with TimeSeriesSplit ---
logging.info("Starting hyperparameter tuning with GridSearchCV")
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [4, 6, 8],
    'min_samples_split': [8, 10, 12]
}
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

logging.info(f"Best parameters found: {grid_search.best_params_}")
logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_

# --- Model Evaluation ---
logging.info("Evaluating best model on training and test sets")
y_train_pred = best_rf.predict(X_train)
y_test_pred = best_rf.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
cm = confusion_matrix(y_test, y_test_pred)

logging.info("=== Model Performance ===")
logging.info(f"Train Accuracy: {train_acc:.4f}")
logging.info(f"Test Accuracy: {test_acc:.4f}")
logging.info(f"Confusion Matrix:\n{cm}")

print("=== Model Performance ===")
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Confusion Matrix:")
print(cm)

# --- Overfitting Checks: Permutation Importance ---
logging.info("Performing permutation importance test")
perm_importance = permutation_importance(best_rf, X_test, y_test, n_repeats=30, random_state=42)
print("\nPermutation Importances:")
for i in perm_importance.importances_mean.argsort()[::-1]:
    print(f"{feature_cols[i]}: {perm_importance.importances_mean[i]:.4f}")

# --- Learning Curve Visualization ---
logging.info("Generating learning curve")
train_sizes, train_scores, test_scores = learning_curve(
    best_rf, X_train, y_train, cv=tscv, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1, random_state=42
)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores_mean, 'o-', label='Training score')
plt.plot(train_sizes, test_scores_mean, 'o-', label='Cross-validation score')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

logging.info("Backtest complete. Exiting script.")
