#!/usr/bin/env python
"""
S&D Day Predictor – Full Production Pipeline with Additional Features and Interaction Terms
===========================================================================================

This script loads data from the local SQLite database (bybit_data.db) for a specified
date range, constructs an engineered feature set including additional features:
  - ATR_daily_7: Average daily range over the previous 7 days.
  - a1_in_P2_50_zone: Binary indicator if the 3H A1 candle is within P2's 50% zone.
  - DOP_cross_count: Count of 5-min candles in A1 that cross the Daily Open Price.
  - p2_vol_ratio: Ratio of P2 volume to P1 volume from the previous day's 3H data.
It then generates pairwise interaction features using PolynomialFeatures, scales them,
trains an ensemble stacking classifier with hyperparameter tuning via BayesSearchCV, evaluates
overall and per-year performance, and provides a simulation function to predict S&D for a given day
(using only data available by 03:00 UTC).

Author: [Your Name]
Date: [Current Date]
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Scikit-learn and related imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import TimeSeriesSplit, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from skopt import BayesSearchCV

# -----------------------
# Configure Logging
# -----------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -----------------------
# PARAMETERS
# -----------------------
# SQL date filters (adjust as needed)
SQL_START_DATE = '2024-02-01'
SQL_END_DATE   = '2025-02-01'

# Training window: train on data <= TRAIN_END_DATE; test on data > TRAIN_END_DATE.
TRAIN_END_DATE = '2024-08-01'

# Number of splits for internal CV in stacking
TS_CV_SPLITS = 5

# -----------------------
# Data Loading Functions
# -----------------------
def load_table(table_name, conn, start_date=None, end_date=None):
    """Load a table from SQLite with optional date filtering."""
    query = f"SELECT * FROM {table_name}"
    conditions = []
    if start_date:
        conditions.append(f"datetime >= '{start_date}'")
    if end_date:
        conditions.append(f"datetime <= '{end_date}'")
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    df = pd.read_sql_query(query, conn)
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

logging.info("Connecting to SQLite database 'bybit_data.db'...")
conn = sqlite3.connect("bybit_data.db")
logging.info(f"Loading tables within {SQL_START_DATE} to {SQL_END_DATE}...")
df_3h    = load_table("btcusdt_3h", conn, start_date=SQL_START_DATE, end_date=SQL_END_DATE)
df_daily = load_table("btcusdt_daily", conn, start_date=SQL_START_DATE, end_date=SQL_END_DATE)
df_1h    = load_table("btcusdt_1h", conn, start_date=SQL_START_DATE, end_date=SQL_END_DATE)
df_5min  = load_table("btcusdt_5min", conn, start_date=SQL_START_DATE, end_date=SQL_END_DATE)
conn.close()
logging.info("Data loading complete.")

# -----------------------
# Assign Candle Labels for 3H Data
# -----------------------
def assign_candle_names(df):
    """Assign candle labels based on UTC hour for 3H data."""
    def label_candle(dt):
        hour = dt.hour
        if hour == 0:
            return 'A1'
        elif hour == 3:
            return 'A2'
        elif hour == 6:
            return 'L1'
        elif hour == 9:
            return 'L2'
        elif hour == 12:
            return 'B1'
        elif hour == 15:
            return 'B2'
        elif hour == 18:
            return 'P1'
        elif hour == 21:
            return 'P2'
        else:
            return 'Unknown'
    df['candle'] = df['datetime'].apply(label_candle)
    return df

df_3h = assign_candle_names(df_3h)

# -----------------------
# Basic Feature Functions
# -----------------------
def get_candle_features(df):
    """Compute basic features for a candle."""
    df = df.copy()
    df['range'] = df['high'] - df['low']
    df['change'] = df['close'] - df['open']
    df['pct_change'] = df['change'] / df['open']
    return df[['open', 'high', 'low', 'close', 'volume', 'turnover', 'range', 'pct_change']]

def compute_5min_wick_metrics(df_5min_day):
    """Compute max upper and lower wick ratios from 5-min candles."""
    upper_ratios, lower_ratios = [], []
    for _, row in df_5min_day.iterrows():
        body = abs(row['close'] - row['open'])
        if body == 0:
            continue
        upper_wick = row['high'] - max(row['open'], row['close'])
        lower_wick = min(row['open'], row['close']) - row['low']
        upper_ratios.append(upper_wick / body)
        lower_ratios.append(lower_wick / body)
    max_upper = max(upper_ratios) if upper_ratios else np.nan
    max_lower = max(lower_ratios) if lower_ratios else np.nan
    return max_upper, max_lower

def compute_A1_return_std(df_5min_day):
    """Compute standard deviation of percentage returns for A1 5-min candles."""
    df = df_5min_day.sort_values('datetime').copy()
    df['pct_ret'] = df['close'].pct_change()
    return df['pct_ret'].std()

# -----------------------
# Build Training Set with Engineered Features (Including Additional Features)
# -----------------------
def build_training_set(df_3h, df_daily, df_1h, df_5min):
    """
    Build the training set with engineered features:
      - gap_pct: normalized gap between DOP and previous day's P2 midpoint.
      - p2_close_minus_p1_close, a1_open_minus_p1_close, a1_open_minus_p2_close.
      - max_upper_wick_ratio, max_lower_wick_ratio, a1_wick_asymmetry.
      - a1_total_volume, a1_total_turnover, a1_return_std.
      - sweep_magnitude, sweep_volume_ratio.
      - DOP, prev_daily_range, prev_daily_pct_change, prev_daily_volume.
      - ATR_daily_7: average daily range over previous 7 days.
      - a1_in_P2_50_zone: indicator if 3H A1 is within P2's 50% zone.
      - DOP_cross_count: count of 5-min A1 candles that cross DOP.
      - p2_vol_ratio: ratio of P2 volume to P1 volume.
    Returns X (features), y (target), and dates.
    """
    features, targets, dates = [], [], []
    df_daily = df_daily.sort_values('datetime').reset_index(drop=True)
    
    for idx, row in df_daily.iterrows():
        if idx == 0:
            continue
        date_today = row['datetime'].date()
        DOP = row['open']
        target = row['SND']  # SND is target
        
        # Previous day's daily data
        prev_day = date_today - timedelta(days=1)
        prev_daily = df_daily[df_daily['datetime'].dt.date == prev_day]
        if prev_daily.empty:
            continue
        prev_daily = prev_daily.iloc[0]
        prev_daily_range = prev_daily['high'] - prev_daily['low']
        prev_daily_vol = prev_daily['volume']
        
        # ATR_daily_7: Average daily range over previous 7 days
        if idx >= 7:
            recent_days = df_daily.iloc[idx-7:idx]
            ATR_daily_7 = recent_days.apply(lambda r: r['high'] - r['low'], axis=1).mean()
        else:
            ATR_daily_7 = prev_daily_range
        
        # 3H data for previous day: P1 and P2
        prev_day_3h = df_3h[df_3h['datetime'].dt.date == prev_day]
        p1_row = prev_day_3h[prev_day_3h['candle'] == 'P1']
        p2_row = prev_day_3h[prev_day_3h['candle'] == 'P2']
        if p1_row.empty or p2_row.empty:
            continue
        p1_row = p1_row.iloc[0]
        p2_row = p2_row.iloc[0]
        
        # gap_pct
        P2_mid = (p2_row['high'] + p2_row['low']) / 2.0
        P2_range = p2_row['high'] - p2_row['low']
        if P2_range == 0:
            continue
        gap_pct = abs(DOP - P2_mid) / P2_range
        p2_close_minus_p1_close = p2_row['close'] - p1_row['close']
        
        # Today's A1 candle from 3H data
        today_3h = df_3h[df_3h['datetime'].dt.date == date_today]
        a1_3h = today_3h[today_3h['candle'] == 'A1']
        if a1_3h.empty:
            continue
        a1_3h = a1_3h.iloc[0]
        a1_open_3h = a1_3h['open']
        a1_open_minus_p1_close = a1_open_3h - p1_row['close']
        a1_open_minus_p2_close = a1_open_3h - p2_row['close']
        
        # 5-min data for A1 period (00:00–02:59)
        a1_5min = df_5min[(df_5min['datetime'].dt.date == date_today) &
                          (df_5min['datetime'].dt.hour < 3)]
        if a1_5min.empty:
            continue
        max_upper_wick_ratio, max_lower_wick_ratio = compute_5min_wick_metrics(a1_5min)
        a1_total_volume = a1_5min['volume'].sum()
        a1_total_turnover = a1_5min['turnover'].sum()
        a1_return_std = compute_A1_return_std(a1_5min)
        
        a1_range = a1_3h['high'] - a1_3h['low']
        a1_upper_wick = a1_3h['high'] - max(a1_3h['open'], a1_3h['close'])
        a1_lower_wick = min(a1_3h['open'], a1_3h['close']) - a1_3h['low']
        a1_wick_asymmetry = (a1_upper_wick - a1_lower_wick) / a1_range if a1_range != 0 else 0
        
        # Liquidity Sweep Metrics
        ATR = prev_daily_range if prev_daily_range != 0 else 1
        if DOP > P2_mid:
            extreme_price = a1_5min['high'].max()
        else:
            extreme_price = a1_5min['low'].min()
        sweep_magnitude = abs(extreme_price - P2_mid) / ATR
        sweep_volume_ratio = a1_total_volume / (prev_daily_vol if prev_daily_vol != 0 else 1)
        
        # Additional Feature: a1_in_P2_50_zone
        P2_50_lower = P2_mid - 0.5 * P2_range
        P2_50_upper = P2_mid + 0.5 * P2_range
        a1_in_P2_50_zone = 1 if (a1_3h['high'] < P2_50_upper and a1_3h['low'] > P2_50_lower) else 0
        
        # Additional Feature: DOP_cross_count (from 5-min A1 candles)
        DOP_cross_count = 0
        for _, candle in a1_5min.iterrows():
            if (candle['open'] < DOP and candle['close'] > DOP) or (candle['open'] > DOP and candle['close'] < DOP):
                DOP_cross_count += 1
        
        # Additional Feature: p2_vol_ratio (ratio of P2 volume to P1 volume)
        p2_vol_ratio = p2_row['volume'] / (p1_row['volume'] if p1_row['volume'] != 0 else 1)
        
        feat = {
            'gap_pct': gap_pct,
            'p2_close_minus_p1_close': p2_close_minus_p1_close,
            'a1_open_minus_p1_close': a1_open_minus_p1_close,
            'a1_open_minus_p2_close': a1_open_minus_p2_close,
            'max_upper_wick_ratio': max_upper_wick_ratio,
            'max_lower_wick_ratio': max_lower_wick_ratio,
            'a1_wick_asymmetry': a1_wick_asymmetry,
            'a1_total_volume': a1_total_volume,
            'a1_total_turnover': a1_total_turnover,
            'a1_return_std': a1_return_std,
            'sweep_magnitude': sweep_magnitude,
            'sweep_volume_ratio': sweep_volume_ratio,
            'DOP': DOP,
            'prev_daily_range': prev_daily_range,
            'prev_daily_pct_change': ((prev_daily['close'] - prev_daily['open']) / prev_daily['open']) if prev_daily['open'] != 0 else 0,
            'prev_daily_volume': prev_daily_vol,
            'ATR_daily_7': ATR_daily_7,
            'a1_in_P2_50_zone': a1_in_P2_50_zone,
            'DOP_cross_count': DOP_cross_count,
            'p2_vol_ratio': p2_vol_ratio
        }
        features.append(feat)
        targets.append(target)
        dates.append(date_today)
    
    X = pd.DataFrame(features).dropna()
    y = np.array(targets[:len(X)])
    dates = pd.Series(dates, name='date').iloc[:len(X)]
    return X, y, dates

logging.info("Building training set from database...")
X, y, dates = build_training_set(df_3h, df_daily, df_1h, df_5min)
logging.info(f"Samples after build: {len(X)}")
logging.info(f"Features: {X.columns.tolist()}")

# -----------------------
# Training/Test Split
# -----------------------
TRAIN_END = datetime.strptime(TRAIN_END_DATE, '%Y-%m-%d').date()
train_mask = dates <= TRAIN_END
test_mask  = dates > TRAIN_END

X_train_full = X[train_mask]
y_train_full = y[train_mask]
dates_train = dates[train_mask]

X_test_full = X[test_mask]
y_test_full = y[test_mask]
dates_test = dates[test_mask]

logging.info(f"Training samples: {len(X_train_full)}, Test samples: {len(X_test_full)}")

# -----------------------
# Scale Features
# -----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
X_test_scaled = scaler.transform(X_test_full)

# -----------------------
# Use Full Feature Set (Disable RFECV)
# -----------------------
selected_features = X_train_full.columns.tolist()
logging.info(f"Using all features: {selected_features}")

X_train_selected = X_train_full[selected_features]
X_test_selected = X_test_full[selected_features]

# Reset test indices for safe indexing
X_test_selected = X_test_selected.reset_index(drop=True)
dates_test = dates_test.reset_index(drop=True)

X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# -----------------------
# Generate Interaction Features
# -----------------------
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_inter = poly.fit_transform(X_train_selected)
X_test_inter = poly.transform(X_test_selected)

scaler_poly = StandardScaler()
X_train_inter_scaled = scaler_poly.fit_transform(X_train_inter)
X_test_inter_scaled = scaler_poly.transform(X_test_inter)

# Combine original and interaction features
X_train_final = np.hstack((X_train_selected_scaled, X_train_inter_scaled))
X_test_final = np.hstack((X_test_selected_scaled, X_test_inter_scaled))

interaction_feature_names = poly.get_feature_names_out(selected_features)
combined_feature_names = list(selected_features) + list(interaction_feature_names)
logging.info(f"Total features after adding interactions: {len(combined_feature_names)}")

# -----------------------
# Ensemble Modeling with Stacking and Bayesian Hyperparameter Tuning
# -----------------------
logging.info("Training ensemble stacking classifier with interaction features...")

base_models = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)),
    ('lr', LogisticRegression(penalty='l2', solver='liblinear', random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=10, random_state=42))
]

meta_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

internal_cv = KFold(n_splits=TS_CV_SPLITS, shuffle=False)
stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=internal_cv, n_jobs=-1)

param_space = {
    'final_estimator__n_estimators': (100, 500),
    'final_estimator__max_depth': (10, 25)
}

logging.info("Starting BayesSearchCV for meta-model tuning...")
bayes_search = BayesSearchCV(
    estimator=stacking_clf,
    search_spaces=param_space,
    n_iter=20,
    cv=internal_cv,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train_final, y_train_full)

best_params = bayes_search.best_params_
best_score = bayes_search.best_score_
logging.info(f"Best parameters (BayesSearch): {best_params}")
logging.info(f"Best cross-validated training accuracy (BayesSearch): {best_score:.2f}")

best_model = bayes_search.best_estimator_
y_pred_tuned = best_model.predict(X_test_final)
tuned_accuracy = accuracy_score(y_test_full, y_pred_tuned)
logging.info(f"Tuned Model Test Accuracy (BayesSearch): {tuned_accuracy:.2f}")

# -----------------------
# Per-Year Evaluation
# -----------------------
test_dates = pd.to_datetime(dates_test)
test_years = test_dates.dt.year
yearly_results = {}
for year, idx_group in test_years.groupby(test_years):
    idx = idx_group.index
    X_year = X_test_final[idx]
    y_year = y_test_full[idx]
    y_pred_year = best_model.predict(X_year)
    acc_year = accuracy_score(y_year, y_pred_year)
    yearly_results[year] = acc_year
logging.info("Per-Year Test Accuracy:")
for year, acc in yearly_results.items():
    logging.info(f"  {year}: {acc:.2f}")

# -----------------------
# Simulation Function for a Specific Day (Fully Implemented)
# -----------------------
def simulate_day(model, scaler, scaler_poly, poly, selected_features, df_3h, df_daily, df_1h, df_5min, date_str):
    """
    Simulate prediction for a given day (YYYY-MM-DD) using data available by 03:00 UTC.
    This function recalculates all features (including interaction features) as in training.
    """
    date_today = datetime.strptime(date_str, '%Y-%m-%d').date()
    
    # Daily data for the target day
    today_daily = df_daily[df_daily['datetime'].dt.date == date_today]
    if today_daily.empty:
        logging.error(f"No daily data for {date_str}")
        return None
    DOP = today_daily.iloc[0]['open']
    
    # Previous day's daily data
    prev_day = date_today - timedelta(days=1)
    prev_daily = df_daily[df_daily['datetime'].dt.date == prev_day]
    if prev_daily.empty:
        logging.error(f"No previous daily data for {prev_day}")
        return None
    prev_daily = prev_daily.iloc[0]
    prev_daily_range = prev_daily['high'] - prev_daily['low']
    prev_daily_vol = prev_daily['volume']
    
    # ATR_daily_7 from previous 7 days
    daily_all = df_daily[df_daily['datetime'].dt.date < date_today].sort_values('datetime')
    if len(daily_all) >= 7:
        recent_days = daily_all.iloc[-7:]
        ATR_daily_7 = recent_days.apply(lambda r: r['high'] - r['low'], axis=1).mean()
    else:
        ATR_daily_7 = prev_daily_range
    
    # 3H data for previous day: P1 and P2
    prev_day_3h = df_3h[df_3h['datetime'].dt.date == prev_day]
    p1_row = prev_day_3h[prev_day_3h['candle'] == 'P1']
    p2_row = prev_day_3h[prev_day_3h['candle'] == 'P2']
    if p1_row.empty or p2_row.empty:
        logging.error(f"Missing 3H data for previous day {prev_day}")
        return None
    p1_row = p1_row.iloc[0]
    p2_row = p2_row.iloc[0]
    
    P2_mid = (p2_row['high'] + p2_row['low']) / 2.0
    P2_range = p2_row['high'] - p2_row['low']
    if P2_range == 0:
        return None
    gap_pct = abs(DOP - P2_mid) / P2_range
    p2_close_minus_p1_close = p2_row['close'] - p1_row['close']
    
    # Today's 3H data for A1
    today_3h = df_3h[df_3h['datetime'].dt.date == date_today]
    a1_3h = today_3h[today_3h['candle'] == 'A1']
    if a1_3h.empty:
        logging.error(f"Missing A1 3H data for {date_str}")
        return None
    a1_3h = a1_3h.iloc[0]
    a1_open_3h = a1_3h['open']
    a1_open_minus_p1_close = a1_open_3h - p1_row['close']
    a1_open_minus_p2_close = a1_open_3h - p2_row['close']
    
    # 5-min data for A1 period (00:00–02:59)
    a1_5min = df_5min[(df_5min['datetime'].dt.date == date_today) & (df_5min['datetime'].dt.hour < 3)]
    if a1_5min.empty:
        logging.error(f"Missing A1 5min data for {date_str}")
        return None
    max_upper_wick_ratio, max_lower_wick_ratio = compute_5min_wick_metrics(a1_5min)
    a1_total_volume = a1_5min['volume'].sum()
    a1_total_turnover = a1_5min['turnover'].sum()
    a1_return_std = compute_A1_return_std(a1_5min)
    
    a1_range = a1_3h['high'] - a1_3h['low']
    a1_upper_wick = a1_3h['high'] - max(a1_3h['open'], a1_3h['close'])
    a1_lower_wick = min(a1_3h['open'], a1_3h['close']) - a1_3h['low']
    a1_wick_asymmetry = (a1_upper_wick - a1_lower_wick) / a1_range if a1_range != 0 else 0
    
    ATR = prev_daily_range if prev_daily_range != 0 else 1
    if DOP > P2_mid:
        extreme_price = a1_5min['high'].max()
    else:
        extreme_price = a1_5min['low'].min()
    sweep_magnitude = abs(extreme_price - P2_mid) / ATR
    sweep_volume_ratio = a1_total_volume / (prev_daily_vol if prev_daily_vol != 0 else 1)
    
    # Additional Feature: a1_in_P2_50_zone
    P2_50_lower = P2_mid - 0.5 * P2_range
    P2_50_upper = P2_mid + 0.5 * P2_range
    a1_in_P2_50_zone = 1 if (a1_3h['high'] < P2_50_upper and a1_3h['low'] > P2_50_lower) else 0
    
    # Additional Feature: DOP_cross_count
    DOP_cross_count = 0
    for _, candle in a1_5min.iterrows():
        if (candle['open'] < DOP and candle['close'] > DOP) or (candle['open'] > DOP and candle['close'] < DOP):
            DOP_cross_count += 1
    
    # Additional Feature: p2_vol_ratio
    p2_vol_ratio = p2_row['volume'] / (p1_row['volume'] if p1_row['volume'] != 0 else 1)
    
    feat = {
        'gap_pct': gap_pct,
        'p2_close_minus_p1_close': p2_close_minus_p1_close,
        'a1_open_minus_p1_close': a1_open_minus_p1_close,
        'a1_open_minus_p2_close': a1_open_minus_p2_close,
        'max_upper_wick_ratio': max_upper_wick_ratio,
        'max_lower_wick_ratio': max_lower_wick_ratio,
        'a1_wick_asymmetry': a1_wick_asymmetry,
        'a1_total_volume': a1_total_volume,
        'a1_total_turnover': a1_total_turnover,
        'a1_return_std': a1_return_std,
        'sweep_magnitude': sweep_magnitude,
        'sweep_volume_ratio': sweep_volume_ratio,
        'DOP': DOP,
        'prev_daily_range': prev_daily_range,
        'prev_daily_pct_change': ((prev_daily['close'] - prev_daily['open']) / prev_daily['open']) if prev_daily['open'] != 0 else 0,
        'prev_daily_volume': prev_daily_vol,
        'ATR_daily_7': ATR_daily_7,
        'a1_in_P2_50_zone': a1_in_P2_50_zone,
        'DOP_cross_count': DOP_cross_count,
        'p2_vol_ratio': p2_vol_ratio
    }
    
    feat_df = pd.DataFrame([feat])
    feat_df = feat_df[selected_features]
    
    # Generate interaction features for this day's data
    X_poly = poly.transform(feat_df)
    X_poly_scaled = scaler_poly.transform(X_poly)
    
    # Scale original features (using the same scaler as in training)
    feat_scaled = scaler.transform(feat_df)
    
    # Combine original and interaction features
    X_combined = np.hstack((feat_scaled, X_poly_scaled))
    
    prediction = model.predict(X_combined)[0]
    logging.info(f"Prediction for {date_str} (by 03:00 UTC): S&D = {prediction}")
    return prediction

logging.info("Simulating prediction for a sample day...")
sample_day = df_daily['datetime'].dt.date.iloc[20].strftime('%Y-%m-%d')
simulate_day(best_model, scaler, scaler_poly, poly, selected_features, df_3h, df_daily, df_1h, df_5min, sample_day)
