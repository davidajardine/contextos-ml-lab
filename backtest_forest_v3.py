import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import matplotlib.pyplot as plt
import logging
import sys
from sklearn.base import clone
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import warnings

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Suppress warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#########################################
# Data Loading and Preprocessing
#########################################

def load_data():
    try:
        conn = sqlite3.connect('bybit_data.db')
        df_daily = pd.read_sql_query("SELECT * FROM btcusdt_daily", conn)
        df_3h = pd.read_sql_query("SELECT * FROM btcusdt_3h", conn)
        df_asia = pd.read_sql_query("SELECT * FROM btcusdt_asia", conn)
        return df_daily, df_3h, df_asia
    except Exception as e:
        logger.error(f"Failed to load data: {str(e)}")
        sys.exit(1)
    finally:
        conn.close()

def preprocess_data(df):
    # Convert 'datetime' column to datetime objects (assumed UTC) and sort
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    df.sort_values(by='datetime', inplace=True)
    df['date'] = df['datetime'].dt.date
    return df

#########################################
# Helper: Candle Labeling
#########################################

def get_candle_label(dt_obj):
    hour = dt_obj.hour
    if 0 <= hour < 3:
        return 'A1'
    elif 3 <= hour < 6:
        return 'A2'
    elif 6 <= hour < 9:
        return 'L1'
    elif 9 <= hour < 12:
        return 'L2'
    elif 12 <= hour < 15:
        return 'B1'
    elif 15 <= hour < 18:
        return 'B2'
    elif 18 <= hour < 21:
        return 'P1'
    elif 21 <= hour < 24:
        return 'P2'
    else:
        return None

#########################################
# Feature Engineering Functions
#########################################

def extract_prev_day_candle_features(df_3h, current_day):
    """
    For the given current day, extract previous day's P1 and P2 candle features.
    P1: from 18:00 to 20:59:59 of the previous day.
    P2: from 21:00 to 23:59:59 of the previous day.
    Also returns P2's open price.
    """
    prev_day = current_day - timedelta(days=1)
    P1_start = datetime.combine(prev_day, time(18, 0, 0))
    P1_end = datetime.combine(prev_day, time(21, 0, 0))
    P2_start = P1_end  # 21:00:00
    P2_end = datetime.combine(prev_day, time(23, 59, 59))
    
    candles_P1 = df_3h[(df_3h['datetime'] >= P1_start) & (df_3h['datetime'] < P1_end)].copy()
    candles_P2 = df_3h[(df_3h['datetime'] >= P2_start) & (df_3h['datetime'] <= P2_end)].copy()
    
    if candles_P1.empty or candles_P2.empty:
        return pd.Series({
            'P1_range': np.nan,
            'P2_range': np.nan,
            'P1_body': np.nan,
            'P2_body': np.nan,
            'P2_P1_close_diff': np.nan,
            'P1_wick_ratio': np.nan,
            'P2_wick_ratio': np.nan,
            'P2_open': np.nan
        })
    
    def select_candle(candles, start, end):
        midpoint = start + (end - start) / 2
        candles['diff'] = abs(candles['datetime'] - midpoint)
        return candles.sort_values('diff').iloc[0]
    
    P1 = select_candle(candles_P1, P1_start, P1_end)
    P2 = select_candle(candles_P2, P2_start, P2_end)
    
    P1_range = P1['high'] - P1['low']
    P2_range = P2['high'] - P2['low']
    P1_body = abs(P1['close'] - P1['open'])
    P2_body = abs(P2['close'] - P2['open'])
    P1_wick_ratio = ((P1_range - P1_body) / P1_range) if P1_range != 0 else 0
    P2_wick_ratio = ((P2_range - P2_body) / P2_range) if P2_range != 0 else 0
    close_diff = P2['close'] - P1['close']
    P2_open = P2['open']
    
    return pd.Series({
        'P1_range': P1_range,
        'P2_range': P2_range,
        'P1_body': P1_body,
        'P2_body': P2_body,
        'P2_P1_close_diff': close_diff,
        'P1_wick_ratio': P1_wick_ratio,
        'P2_wick_ratio': P2_wick_ratio,
        'P2_open': P2_open
    })

def extract_A1_features(df_3h, current_day, P2_features):
    """
    Extract A1 candle features for the current day (00:00 to 02:59:59)
    and compute the deviation of A1's close from the 50% midpoint of P2.
    The 50% midpoint is: P2_mid = P2_open + 0.5 * P2_range.
    Deviation is normalized by (0.25 * P2_range).
    """
    A1_start = datetime.combine(current_day, time(0, 0, 0))
    A1_end = datetime.combine(current_day, time(3, 0, 0))
    candles_A1 = df_3h[(df_3h['datetime'] >= A1_start) & (df_3h['datetime'] < A1_end)].copy()
    
    if candles_A1.empty:
        return pd.Series({
            'A1_range': np.nan,
            'A1_body': np.nan,
            'A1_close': np.nan,
            'A1_deviation_from_P2_mid_pct': np.nan
        })
    
    def select_candle(candles, start, end):
        midpoint = start + (end - start) / 2
        candles['diff'] = abs(candles['datetime'] - midpoint)
        return candles.sort_values('diff').iloc[0]
    
    A1 = select_candle(candles_A1, A1_start, A1_end)
    A1_range = A1['high'] - A1['low']
    A1_body = abs(A1['close'] - A1['open'])
    A1_close = A1['close']
    
    P2_range = P2_features.get('P2_range', np.nan)
    P2_open = P2_features.get('P2_open', np.nan)
    if np.isnan(P2_range) or np.isnan(P2_open) or P2_range == 0:
        deviation = np.nan
    else:
        P2_mid = P2_open + 0.5 * P2_range
        deviation = (A1_close - P2_mid) / (0.25 * P2_range)
    
    return pd.Series({
        'A1_range': A1_range,
        'A1_body': A1_body,
        'A1_close': A1_close,
        'A1_deviation_from_P2_mid_pct': deviation
    })

def extract_asia_feature(df_asia, current_day):
    """
    Extract the Asia session midline for the given current day.
    """
    asia_row = df_asia[df_asia['date'] == current_day]
    if asia_row.empty:
        return np.nan
    return asia_row['midline'].iloc[0]

def build_and_split_feature_dataset(df_daily, df_3h, df_asia):
    features = []
    # Preprocess data
    df_daily = preprocess_data(df_daily)
    df_3h = preprocess_data(df_3h)
    df_asia = preprocess_data(df_asia)
    
    # Compute 5-day EMA on daily closes, shifted by one day
    df_daily['ema_5'] = df_daily['close'].ewm(span=5, adjust=False).mean().shift(1)
    
    # Iterate over each row in daily data
    for idx, row in df_daily.iterrows():
        current_day = row['date']
        prev_day = current_day - timedelta(days=1)
        
        # Extract previous day's 3H candle features (P1 & P2)
        prev_candle_feats = extract_prev_day_candle_features(df_3h, current_day)
        asia_midline = extract_asia_feature(df_asia, current_day)
        
        # Additional previous day's daily data
        prev_daily_row = df_daily[df_daily['date'] == prev_day]
        if prev_daily_row.empty:
            logger.warning(f"Missing previous day's daily data for {current_day}")
            continue
        prev_daily = prev_daily_row.iloc[0]
        prev_daily_range = prev_daily['high'] - prev_daily['low']
        prev_daily_pct_change = (prev_daily['close'] - prev_daily['open']) / prev_daily['open'] if prev_daily['open'] else np.nan
        ema_5_val = prev_daily_row['ema_5'].iloc[0] if 'ema_5' in prev_daily_row.columns else np.nan
        
        # Extract A1 candle features from current day
        A1_feats = extract_A1_features(df_3h, current_day, prev_candle_feats)
        
        # Calculate additional ratios and differences
        ratio_asia_to_open = (asia_midline / row['open']) if row['open'] else np.nan
        diff_open_close = row['close'] - row['open']
        
        # Convert SND column to binary target
        target = 1 if str(row.get('SND', '')).strip().upper() == 'TRUE' else 0
        
        features.append({
            'date': current_day,
            'P1_range': prev_candle_feats.get('P1_range', np.nan),
            'P2_range': prev_candle_feats.get('P2_range', np.nan),
            'P1_body': prev_candle_feats.get('P1_body', np.nan),
            'P2_body': prev_candle_feats.get('P2_body', np.nan),
            'P2_P1_close_diff': prev_candle_feats.get('P2_P1_close_diff', np.nan),
            'P1_wick_ratio': prev_candle_feats.get('P1_wick_ratio', np.nan),
            'P2_wick_ratio': prev_candle_feats.get('P2_wick_ratio', np.nan),
            'asia_midline': asia_midline,
            'daily_open': row['open'],
            'daily_close': row['close'],
            'prev_daily_range': prev_daily_range,
            'prev_daily_pct_change': prev_daily_pct_change,
            'prev_atr': prev_daily_range,
            'ema_5': ema_5_val,
            'ratio_asia_to_open': ratio_asia_to_open,
            'diff_open_close': diff_open_close,
            'A1_range': A1_feats.get('A1_range', np.nan),
            'A1_body': A1_feats.get('A1_body', np.nan),
            'A1_close': A1_feats.get('A1_close', np.nan),
            'A1_deviation_from_P2_mid_pct': A1_feats.get('A1_deviation_from_P2_mid_pct', np.nan),
            'target': target
        })
    
    df_features = pd.DataFrame(features).dropna()
    df_features.sort_values(by='date', inplace=True)
    logger.info(f"Feature dataset built with {len(df_features)} records before filtering.")
    
    # Filter dataset to desired period: 1 Feb 2024 to 1 Feb 2025
    start_date = pd.to_datetime('2024-02-01').date()
    end_date = pd.to_datetime('2025-02-01').date()
    df_features = df_features[(df_features['date'] >= start_date) & (df_features['date'] <= end_date)]
    logger.info(f"Feature dataset size after filtering: {len(df_features)} records.")
    
    # Perform 70/30 time-series split
    n = len(df_features)
    train_end = int(n * 0.7)
    df_train = df_features.iloc[:train_end]
    df_test = df_features.iloc[train_end:]
    logger.info(f"Split into {len(df_train)} training and {len(df_test)} testing records.")
    
    return df_train, df_test

#########################################
# Model Building and Evaluation
#########################################

def main():
    logger.info("Loading and preprocessing data...")
    df_daily, df_3h, df_asia = load_data()
    
    logger.info("Building and splitting feature dataset...")
    df_train, df_test = build_and_split_feature_dataset(df_daily, df_3h, df_asia)
    
    feature_cols = [
        'P1_range', 'P2_range', 'P1_body', 'P2_body', 'P2_P1_close_diff',
        'P1_wick_ratio', 'P2_wick_ratio', 'asia_midline', 'daily_open',
        'daily_close', 'prev_daily_range', 'prev_daily_pct_change',
        'prev_atr', 'ema_5', 'ratio_asia_to_open', 'diff_open_close',
        'A1_range', 'A1_body', 'A1_close', 'A1_deviation_from_P2_mid_pct'
    ]
    
    X_train = df_train[feature_cols]
    y_train = df_train['target']
    X_test = df_test[feature_cols]
    y_test = df_test['target']
    
    # Hyperparameter tuning for Random Forest
    tscv = TimeSeriesSplit(n_splits=5)
    try:
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid={
                'n_estimators': [100, 150],
                'max_depth': [4, 6],
                'min_samples_split': [8, 10]
            },
            cv=tscv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_rf = grid_search.best_estimator_
        logger.info(f"Best RF params: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
    except Exception as eg:
        logger.error(f"GridSearchCV failed: {str(eg)}")
        sys.exit(1)
    
    # Define base estimators for stacking ensemble
    base_estimators = [
        ('rf', best_rf),
        ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                                  random_state=42, verbosity=0, n_jobs=-1)),
        ('lgbm', lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1))
    ]
    
    # Generate out-of-fold predictions and collect corresponding targets
    oof_predictions = []
    oof_targets = []
    tscv_cv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv_cv.split(X_train):
        X_fold_train, X_fold_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_fold_train = y_train.iloc[train_idx]
        y_fold_test = y_train.iloc[test_idx]
        fold_preds = []
        for name, estimator in base_estimators:
            model = clone(estimator)
            model.fit(X_fold_train, y_fold_train)
            fold_preds.append(model.predict(X_fold_test))
        oof_predictions.append(np.column_stack(fold_preds))
        oof_targets.append(y_fold_test.values)
    oof_predictions = np.vstack(oof_predictions)
    oof_targets = np.concatenate(oof_targets)
    
    final_model = LogisticRegression()
    final_model.fit(oof_predictions, oof_targets)
    
    # For final test predictions, train each base estimator on the entire training set and stack predictions
    test_preds = []
    for name, est in base_estimators:
        model = clone(est)
        model.fit(X_train, y_train)
        test_preds.append(model.predict(X_test))
    stacked_test_preds = np.column_stack(test_preds)
    y_pred = final_model.predict(stacked_test_preds)
    
    train_acc = accuracy_score(oof_targets, final_model.predict(oof_predictions))
    test_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info("=== Ensemble Model Performance ===")
    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Test Accuracy: {test_acc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    perm_importance = permutation_importance(
        best_rf,
        X_train,
        y_train,
        n_repeats=10,
        random_state=42
    )
    logger.info("\n=== Feature Importances ===")
    for i in perm_importance.importances_mean.argsort()[::-1]:
        logger.info(f"{feature_cols[i]}: {perm_importance.importances_mean[i]:.4f}")
    
    # Optional: Learning curve visualization
    train_sizes, train_scores, cv_scores = learning_curve(
        best_rf, X_train, y_train, cv=tscv, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1, random_state=42
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    cv_scores_mean = np.mean(cv_scores, axis=1)
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_scores_mean, 'o-', label='Training Score')
    plt.plot(train_sizes, cv_scores_mean, 'o-', label='CV Score')
    plt.title('Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
