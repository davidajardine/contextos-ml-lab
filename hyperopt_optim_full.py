#!/usr/bin/env python3
"""
Optimized S&D Day Identification & Trading Strategy Script

This script:
  - Loads data from "bybit_data.db" and filters by TEST_START_DATE/TEST_END_DATE.
  - Clears the error log table at the start.
  - Computes baselines from aggregated_set data once.
  - Precomputes raw (unweighted) features for each trading day and caches them.
  - Uses hyperopt to optimize hyperparameters that act as multipliers on these raw features.
  - Uses joblib for parallel processing of the weighted score computation.
  - Logs daily results to the error log table and prints an error summary.

Ensure that your database "bybit_data.db" contains the expected tables.
"""

import sqlite3
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.metrics import accuracy_score, confusion_matrix
import logging
from datetime import datetime, time
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import matplotlib.pyplot as plt

# ===================== USER CONFIGURATION =====================
TEST_START_DATE = "2024-11-01"   # Adjust as needed
TEST_END_DATE   = "2024-11-30"
MAX_EVALS       = 1500         # Number of hyperopt iterations

DATA_DB = 'bybit_data.db'
LOG_DB  = 'error_logs.db'

# Thresholds and warm start defaults
VOLUME_SPIKE_THRESHOLD = 1.2
OI_THRESHOLD           = 1.1
DEFAULT_WICK_PCT       = 0.575   # Default baseline for daily wick percentage
DEFAULT_BODY_MEDIAN    = 0.5     # Default baseline for daily body ratio
DEFAULT_RET_TO_OPEN    = 0.002   # Default baseline for return-to-open
DOJI_THRESHOLD         = 0.1     # A1 candle is considered doji if its body < 10% of its range

# ===================== PARAMETER RANGES (Hyperparameters) =====================
PARAM_RANGES = {
    'bias': (-20, 20, 1),
    'p2_bias': (-50, 75, 1),
    'p2_wick_strength': (-50, 75, 1),
    'liquidity_sweep': (-50, 60, 1),
    'wick_size': (-75, 50, 1),
    'volume_spike': (-80, 80, 1),
    'return_to_open': (-30, 60, 1),
    'daily_wick_pct': (-100, 20, 1),
    'daily_body_median': (-50, 75, 1),
    'p2_50_breakout': (-50, 50, 1),
    'p2_50_reentry': (-50, 50, 1),
    'dop_crosses': (-50, 50, 1),
    'oi_spike': (-50, 50, 1),
    'ema_spread_weight': (-50, 50, 1),
    'atr_weight': (-50, 50, 1),
    'funding_rate_weight': (-50, 75, 1),
    'a1_raid': (-50, 50, 1),
    'zone_escape': (-50, 50, 1),
    'a1_doji_adjust': (-10, 10, 1)
}

# ===================== GLOBAL CACHED VARIABLES =====================
df_agg = None
df_daily = None
df_5min = None
df_a1 = None
df_15 = None
df_1h = None
raw_features_df = None

# Baselines (computed once)
baseline_wick_pct = DEFAULT_WICK_PCT
baseline_body_median = DEFAULT_BODY_MEDIAN
baseline_ret_to_open = DEFAULT_RET_TO_OPEN

# ===================== LOGGING CONFIGURATION =====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== DATA LOADING FUNCTIONS =====================
def load_data(db_path=DATA_DB):
    """Load aggregated_set, daily flag, and 5‑min data; filter by TEST_START_DATE/TEST_END_DATE."""
    conn = sqlite3.connect(db_path)
    df_agg = pd.read_sql("SELECT * FROM aggregated_set", conn)
    df_daily = pd.read_sql("SELECT datetime as trading_day, SND FROM btcusdt_daily", conn)
    df_5min = pd.read_sql("SELECT datetime, open, high, low, close FROM btcusdt_5min", conn)
    conn.close()
    
    df_daily['trading_day'] = pd.to_datetime(df_daily['trading_day']).dt.date
    df_agg['trading_day'] = pd.to_datetime(df_agg['trading_day']).dt.date
    df_5min['datetime'] = pd.to_datetime(df_5min['datetime'])
    df_5min['trading_day'] = df_5min['datetime'].dt.date
    
    start_date = pd.to_datetime(TEST_START_DATE).date()
    end_date = pd.to_datetime(TEST_END_DATE).date()
    df_agg = df_agg[(df_agg['trading_day'] >= start_date) & (df_agg['trading_day'] <= end_date)]
    df_daily = df_daily[(df_daily['trading_day'] >= start_date) & (df_daily['trading_day'] <= end_date)]
    df_5min = df_5min[(df_5min['trading_day'] >= start_date) & (df_5min['trading_day'] <= end_date)]
    return df_agg, df_daily, df_5min

def load_custom_3h(db_path=DATA_DB):
    """Load agg_custom_3h data; derive trading_day and session; return A1 candles."""
    conn = sqlite3.connect(db_path)
    df_custom = pd.read_sql("SELECT * FROM agg_custom_3h", conn)
    conn.close()
    df_custom['datetime'] = pd.to_datetime(df_custom['datetime'])
    df_custom['trading_day'] = pd.to_datetime(df_custom['trading_day_start']).dt.date
    
    def get_session(dt):
        t = dt.time()
        if time(0,0) <= t < time(3,0):
            return 'A1'
        elif time(3,0) <= t < time(6,0):
            return 'A2'
        elif time(6,0) <= t < time(9,0):
            return 'L1'
        elif time(9,0) <= t < time(12,0):
            return 'L2'
        elif time(12,0) <= t < time(15,0):
            return 'B1'
        elif time(15,0) <= t < time(18,0):
            return 'B2'
        elif time(18,0) <= t < time(21,0):
            return 'P1'
        elif time(21,0) <= t <= time(23,59,59):
            return 'P2'
        else:
            return 'Other'
    
    df_custom['session'] = df_custom['datetime'].apply(get_session)
    df_a1 = df_custom[df_custom['session'] == 'A1'].copy()
    return df_a1

def load_custom_15min(db_path=DATA_DB):
    """Load agg_custom_15min data (with open interest) and derive trading_day."""
    conn = sqlite3.connect(db_path)
    df_15 = pd.read_sql("SELECT * FROM agg_custom_15min", conn)
    conn.close()
    df_15['datetime'] = pd.to_datetime(df_15['datetime'])
    df_15['trading_day'] = pd.to_datetime(df_15['trading_day_start']).dt.date
    return df_15

def load_custom_1h(db_path=DATA_DB):
    """Load agg_custom_1h data (including funding_rate) and derive trading_day."""
    conn = sqlite3.connect(db_path)
    df_1h = pd.read_sql("SELECT * FROM agg_custom_1h", conn)
    conn.close()
    df_1h['datetime'] = pd.to_datetime(df_1h['datetime'])
    df_1h['trading_day'] = pd.to_datetime(df_1h['trading_day_start']).dt.date
    return df_1h

# ===================== BASELINE COMPUTATION =====================
def compute_baselines(df_agg):
    """Compute baselines for daily_wick_pct, daily_body_median, and return-to-open using df_agg."""
    daily_wick_pcts = []
    daily_body_pcts = []
    ret_to_open_values = []
    for _, row in df_agg.iterrows():
        day_high  = row.get('high', 0)
        day_low   = row.get('low', 0)
        day_open  = row.get('open', 0)
        day_close = row.get('close', 0)
        if day_high - day_low > 0 and day_open:
            daily_range = day_high - day_low
            wick_pct = (daily_range - abs(day_close - day_open)) / daily_range
            body_pct = abs(day_open - day_close) / daily_range
            ret_val  = (day_close - day_open) / day_open
            daily_wick_pcts.append(wick_pct)
            daily_body_pcts.append(body_pct)
            ret_to_open_values.append(ret_val)
    base_wick = np.mean(daily_wick_pcts) if daily_wick_pcts else DEFAULT_WICK_PCT
    base_body = np.mean(daily_body_pcts) if daily_body_pcts else DEFAULT_BODY_MEDIAN
    base_ret  = np.mean(ret_to_open_values) if ret_to_open_values else DEFAULT_RET_TO_OPEN
    return base_wick, base_body, base_ret

# ===================== HELPER FUNCTION: Count DOP Crosses =====================
def count_dop_crosses(df, dop):
    """Vectorized count of DOP crosses (excluding the first candle)."""
    crosses = ((df['open'] - dop) * (df['close'] - dop) < 0) | ((df['high'] >= dop) & (df['low'] <= dop))
    return crosses.iloc[1:].sum()

# ===================== RAW FEATURE COMPUTATION =====================
def compute_raw_features(df_agg, df_5min, df_a1, df_15, df_1h):
    """
    Compute raw (unweighted) features for each trading day and return a DataFrame.
    These raw features are independent of the hyperparameters.
    """
    records = []
    # Pre-filter 5min data for A1 session (00:00 - 03:00) and group by trading_day
    df_5min_a1 = df_5min[(df_5min['datetime'].dt.time >= time(0,0)) &
                         (df_5min['datetime'].dt.time < time(3,0))]
    grouped_5min_a1 = df_5min_a1.groupby('trading_day')
    
    for _, row in df_agg.iterrows():
        day = row['trading_day']
        rec = {}
        # P2 raw bias ratio
        if row.get('high_P2') and row.get('low_P2'):
            mid_p2 = (row['high_P2'] + row['low_P2']) / 2
            rec['p2_bias_raw'] = (row.get('close_P2', 0) - mid_p2) / mid_p2 if mid_p2 != 0 else 0
        else:
            rec['p2_bias_raw'] = 0
        
        # Volume spike raw
        rec['volume_spike_raw'] = (row.get('volume_P2', 0) / row.get('volume_P1', 1)) - VOLUME_SPIKE_THRESHOLD
        
        # P2 wick strength raw
        p2_open = row.get('open_P2', 0)
        p2_close = row.get('close_P2', 0)
        p2_high = row.get('high_P2', 0)
        p2_low = row.get('low_P2', 0)
        p2_body = abs(p2_open - p2_close)
        p2_range = p2_high - p2_low if (p2_high - p2_low) != 0 else 1
        upper_wick = p2_high - max(p2_open, p2_close)
        lower_wick = min(p2_open, p2_close) - p2_low
        rec['p2_wick_strength_raw'] = (upper_wick + lower_wick) / p2_body if p2_body != 0 else 0
        
        # Liquidity sweep raw
        rec['liquidity_sweep_raw'] = (p2_high - p2_low) / p2_open if p2_open != 0 else 0
        
        # Wick size raw
        rec['wick_size_raw'] = ((p2_high - p2_low) - p2_body) / (p2_high - p2_low) if (p2_high - p2_low) != 0 else 0
        
        # Return-to-open raw
        rec['return_to_open_raw'] = (row.get('close', 0) - row.get('open', 0)) / row.get('open', 1)
        
        # Daily wick percentage raw
        day_high = row.get('high', 0)
        day_low = row.get('low', 0)
        daily_range = day_high - day_low if (day_high - day_low) != 0 else 1
        rec['daily_wick_pct_raw'] = (daily_range - abs(row.get('close', 0) - row.get('open', 0))) / daily_range
        
        # Daily body median raw
        rec['daily_body_median_raw'] = abs(row.get('open', 0) - row.get('close', 0)) / daily_range
        
        # EMA spread raw
        rec['ema_spread_raw'] = row.get('ema_short', 0) - row.get('ema_long', 0)
        
        # ATR raw
        rec['atr_raw'] = row.get('atr', 0)
        
        # Funding rate raw (from 1h data)
        df_1h_day = df_1h[df_1h['trading_day'] == day]
        df_1h_recent = df_1h_day[df_1h_day['datetime'].dt.hour < 3]
        rec['funding_rate'] = df_1h_recent.sort_values(by='datetime', ascending=False).iloc[0]['funding_rate'] if not df_1h_recent.empty else 0
        
        # Open Interest raw from 15min (P2)
        df_15_day = df_15[df_15['trading_day'] == day]
        df_15_p2 = df_15_day[df_15_day['datetime'].dt.hour >= 21]
        oi_p2 = df_15_p2.sort_values(by='datetime', ascending=False).iloc[0].get('open_interest', 0) if not df_15_p2.empty else row.get('open_interest_P2', 0)
        rec['oi_p2_raw'] = oi_p2
        rec['oi_spike_raw'] = (oi_p2 / row.get('open_interest_P1', 1)) - OI_THRESHOLD
        
        # DOP cross count raw from 5min A1 data
        if day in grouped_5min_a1.groups:
            df_5min_day = grouped_5min_a1.get_group(day).sort_values(by='datetime')
            dop = df_5min_day.iloc[0]['open']
            rec['dop_crosses_raw'] = count_dop_crosses(df_5min_day, dop)
        else:
            rec['dop_crosses_raw'] = 0
        
        # A1 raw features from df_a1 (agg_custom_3h A1 session)
        df_a1_day = df_a1[df_a1['trading_day'] == day]
        if df_a1_day.empty:
            a1_open = a1_high = a1_low = a1_close = 0
        else:
            a1_row = df_a1_day.iloc[0]
            a1_open = a1_row.get('open', 0)
            a1_high = a1_row.get('high', 0)
            a1_low  = a1_row.get('low', 0)
            a1_close = a1_row.get('close', 0)
        rec['a1_open'] = a1_open
        rec['a1_high'] = a1_high
        rec['a1_low'] = a1_low
        rec['a1_close'] = a1_close
        a1_body = abs(a1_open - a1_close)
        a1_range = a1_high - a1_low if (a1_high - a1_low) != 0 else 1
        if a1_close < a1_open:
            a1_upper_wick = a1_high - a1_open
            a1_lower_wick = a1_open - a1_low
        else:
            a1_upper_wick = a1_high - a1_close
            a1_lower_wick = a1_close - a1_low
        rec['a1_upper_wick'] = a1_upper_wick
        rec['a1_lower_wick'] = a1_lower_wick
        rec['a1_U_wick_pct'] = ((a1_high - a1_open) / a1_range) if a1_close < a1_open else ((a1_high - a1_close) / a1_range)
        rec['a1_L_wick_pct'] = (min(a1_open, a1_close) - a1_low) / a1_range
        rec['a1_total_wick_pct'] = rec['a1_U_wick_pct'] + rec['a1_L_wick_pct']
        rec['a1_body_pct'] = a1_body / a1_range
        rec['a1_doji_flag'] = 1 if a1_body < DOJI_THRESHOLD * a1_range else 0
        
        # A1 Raid raw
        raid_upper = max(0, a1_high - p2_high) / p2_range
        raid_lower = max(0, p2_low - a1_low) / p2_range
        rec['a1_raid_raw'] = raid_upper - raid_lower
        
        # Zone escape raw
        mid_p2 = (p2_high + p2_low) / 2 if p2_high and p2_low else 0
        half_zone = 0.25 * p2_range
        if a1_close > (mid_p2 + half_zone):
            rec['zone_escape_raw'] = (a1_close - (mid_p2 + half_zone)) / p2_range
        elif a1_close < (mid_p2 - half_zone):
            rec['zone_escape_raw'] = ((mid_p2 - half_zone) - a1_close) / p2_range
        else:
            rec['zone_escape_raw'] = 0
        
        # p2_50_breakout and p2_50_reentry raw
        if a1_close > (mid_p2 + half_zone):
            rec['p2_50_breakout_raw'] = (a1_close - (mid_p2 + half_zone)) / p2_range
        else:
            rec['p2_50_breakout_raw'] = 0
        if a1_close < (mid_p2 - half_zone):
            rec['p2_50_reentry_raw'] = ((mid_p2 - half_zone) - a1_close) / p2_range
        else:
            rec['p2_50_reentry_raw'] = 0
        
        rec['trading_day'] = day
        records.append(rec)
    
    raw_features_df = pd.DataFrame(records).set_index('trading_day')
    return raw_features_df

# ===================== SQL LOGGING FUNCTIONS =====================
def init_error_log_table(db_path=LOG_DB):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("DELETE FROM daily_error_logs")  # Clear table at start
    c.execute("""
        CREATE TABLE IF NOT EXISTS daily_error_logs (
            trading_day TEXT PRIMARY KEY,
            prediction INTEGER,
            actual INTEGER,
            final_score REAL,
            bias REAL,
            p2_bias_factor REAL,
            volume_spike_factor REAL,
            oi_spike_factor REAL,
            return_to_open_factor REAL,
            daily_wick_pct_factor REAL,
            p2_wick_strength_factor REAL,
            liquidity_sweep_factor REAL,
            wick_size_factor REAL,
            daily_body_median_factor REAL,
            p2_50_breakout_factor REAL,
            p2_50_reentry_factor REAL,
            dop_crosses_factor REAL,
            ema_spread_factor REAL,
            atr_factor REAL,
            funding_rate_factor REAL,
            funding_rate REAL,
            oi_p2_15min REAL,
            a1_raid_factor REAL,
            zone_escape_factor REAL,
            a1_doji_factor REAL,
            a1_open REAL,
            a1_high REAL,
            a1_low REAL,
            a1_close REAL,
            a1_upper_wick REAL,
            a1_lower_wick REAL,
            a1_U_wick_pct REAL,
            a1_L_wick_pct REAL,
            a1_total_wick_pct REAL,
            a1_body_pct REAL
        )
    """)
    conn.commit()
    conn.close()

def log_daily_record(metrics, trading_day, actual, db_path=LOG_DB):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO daily_error_logs (
            trading_day, prediction, actual, final_score, bias, p2_bias_factor, volume_spike_factor,
            oi_spike_factor, return_to_open_factor, daily_wick_pct_factor, p2_wick_strength_factor,
            liquidity_sweep_factor, wick_size_factor, daily_body_median_factor, p2_50_breakout_factor,
            p2_50_reentry_factor, dop_crosses_factor, ema_spread_factor, atr_factor, funding_rate_factor,
            funding_rate, oi_p2_15min, a1_raid_factor, zone_escape_factor, a1_doji_factor,
            a1_open, a1_high, a1_low, a1_close, a1_upper_wick, a1_lower_wick, a1_U_wick_pct,
            a1_L_wick_pct, a1_total_wick_pct, a1_body_pct
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        str(trading_day),
        metrics['prediction'],
        actual,
        metrics['final_score'],
        metrics['bias'],
        metrics['p2_bias_factor'],
        metrics['volume_spike_factor'],
        metrics['oi_spike_factor'],
        metrics['return_to_open_factor'],
        metrics['daily_wick_pct_factor'],
        metrics['p2_wick_strength_factor'],
        metrics['liquidity_sweep_factor'],
        metrics['wick_size_factor'],
        metrics['daily_body_median_factor'],
        metrics['p2_50_breakout_factor'],
        metrics['p2_50_reentry_factor'],
        metrics['dop_crosses_factor'],
        metrics['ema_spread_factor'],
        metrics['atr_factor'],
        metrics['funding_rate_factor'],
        metrics['funding_rate'],
        metrics['oi_p2_15min'],
        metrics['a1_raid_factor'],
        metrics['zone_escape_factor'],
        metrics['a1_doji_factor'],
        metrics['a1_open'],
        metrics['a1_high'],
        metrics['a1_low'],
        metrics['a1_close'],
        metrics['a1_upper_wick'],
        metrics['a1_lower_wick'],
        metrics['a1_U_wick_pct'],
        metrics['a1_L_wick_pct'],
        metrics['a1_total_wick_pct'],
        metrics['a1_body_pct']
    ))
    conn.commit()
    conn.close()

def print_error_summary():
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("""
        SELECT COALESCE(AVG(return_to_open_factor),0), COALESCE(AVG(daily_wick_pct_factor),0), COUNT(*) 
        FROM daily_error_logs 
        WHERE prediction = 0 AND actual = 1
    """)
    fn_summary = c.fetchone()
    c.execute("""
        SELECT COALESCE(AVG(return_to_open_factor),0), COALESCE(AVG(daily_wick_pct_factor),0), COUNT(*) 
        FROM daily_error_logs 
        WHERE prediction = 1 AND actual = 0
    """)
    fp_summary = c.fetchone()
    conn.close()
    logger.info("=== Error Summary ===")
    logger.info(f"False Negatives: Count={fn_summary[2]}, Avg Return-to-Open Factor={fn_summary[0]:.5f}, Avg Daily Wick % Factor={fn_summary[1]:.5f}")
    logger.info(f"False Positives: Count={fp_summary[2]}, Avg Return-to-Open Factor={fp_summary[0]:.5f}, Avg Daily Wick % Factor={fp_summary[1]:.5f}")
    logger.info("=" * 40)

# ===================== WEIGHTED SCORE COMPUTATION =====================
def compute_weighted_score(raw, params):
    """
    Compute the final weighted score using raw features and hyperparameters.
    Returns (score, prediction) where prediction = 1 if score >= 0, else 0.
    """
    score = ( params['bias'] +
              params['p2_bias'] * raw['p2_bias_raw'] +
              params['volume_spike'] * raw['volume_spike_raw'] +
              params['p2_wick_strength'] * raw['p2_wick_strength_raw'] +
              params['liquidity_sweep'] * raw['liquidity_sweep_raw'] +
              params['wick_size'] * raw['wick_size_raw'] +
              params['return_to_open'] * (raw['return_to_open_raw'] - baseline_ret_to_open) +
              params['daily_wick_pct'] * (raw['daily_wick_pct_raw'] - baseline_wick_pct) +
              params['daily_body_median'] * (raw['daily_body_median_raw'] - baseline_body_median) +
              params['oi_spike'] * raw['oi_spike_raw'] +
              params['ema_spread_weight'] * raw['ema_spread_raw'] +
              params['atr_weight'] * raw['atr_raw'] +
              params['dop_crosses'] * raw['dop_crosses_raw'] +
              params['funding_rate_weight'] * raw['funding_rate'] +
              params['a1_raid'] * (raw['a1_raid_raw'] * (raw['a1_doji_flag'] * params['a1_doji_adjust'] if raw['a1_doji_flag'] else 1)) +
              params['zone_escape'] * raw['zone_escape_raw'] +
              params['p2_50_breakout'] * raw['p2_50_breakout_raw'] +
              params['p2_50_reentry'] * raw['p2_50_reentry_raw']
            )
    return score, 1 if score >= 0 else 0

# ===================== PARALLEL WEIGHTED SCORE COMPUTATION (Joblib) =====================
def compute_weighted_for_row_joblib(day, raw_features, params):
    return day, *compute_weighted_score(raw_features, params)

# ===================== OBJECTIVE FUNCTION (Using Cached Raw Features & Joblib) =====================
def objective(params):
    global raw_features_df, df_daily
    results = Parallel(n_jobs=cpu_count())(
        delayed(compute_weighted_for_row_joblib)(day, raw_features_df.loc[day], params)
        for day in raw_features_df.index
    )
    predictions = []
    actuals = []
    for day, score, pred in results:
        predictions.append(pred)
        actual_row = df_daily[df_daily['trading_day'] == day]
        actual = 1 if (not actual_row.empty and actual_row['SND'].iloc[0] == 'TRUE') else 0
        actuals.append(actual)
    cm = confusion_matrix(actuals, predictions)
    if cm.size == 1:
        if list(set(actuals)) == [0]:
            tn, fp, fn, tp = cm[0,0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0,0]
    else:
        tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(actuals, predictions)
    logger.info(f"Trial accuracy: {accuracy:.2%}, FP: {fp}, FN: {fn}")
    return {'loss': 100 - (accuracy * 100), 'status': STATUS_OK}

# ===================== LOG FINAL DAILY RESULTS FUNCTION =====================
def log_final_daily_results(best_params):
    for day in raw_features_df.index:
        raw = raw_features_df.loc[day]
        score, pred = compute_weighted_score(raw, best_params)
        actual_row = df_daily[df_daily['trading_day'] == day]
        actual = 1 if (not actual_row.empty and actual_row['SND'].iloc[0] == 'TRUE') else 0
        weighted = {
            'final_score': score,
            'prediction': pred,
            'bias': best_params['bias'],
            'p2_bias_factor': best_params['p2_bias'] * raw['p2_bias_raw'],
            'volume_spike_factor': best_params['volume_spike'] * raw['volume_spike_raw'],
            'oi_spike_factor': best_params['oi_spike'] * raw['oi_spike_raw'],
            'return_to_open_factor': best_params['return_to_open'] * (raw['return_to_open_raw'] - baseline_ret_to_open),
            'daily_wick_pct_factor': best_params['daily_wick_pct'] * (raw['daily_wick_pct_raw'] - baseline_wick_pct),
            'p2_wick_strength_factor': best_params['p2_wick_strength'] * raw['p2_wick_strength_raw'],
            'liquidity_sweep_factor': best_params['liquidity_sweep'] * raw['liquidity_sweep_raw'],
            'wick_size_factor': best_params['wick_size'] * raw['wick_size_raw'],
            'daily_body_median_factor': best_params['daily_body_median'] * (raw['daily_body_median_raw'] - baseline_body_median),
            'p2_50_breakout_factor': best_params['p2_50_breakout'] * raw['p2_50_breakout_raw'],
            'p2_50_reentry_factor': best_params['p2_50_reentry'] * raw['p2_50_reentry_raw'],
            'dop_crosses_factor': best_params['dop_crosses'] * raw['dop_crosses_raw'],
            'ema_spread_factor': best_params['ema_spread_weight'] * raw['ema_spread_raw'],
            'atr_factor': best_params['atr_weight'] * raw['atr_raw'],
            'funding_rate_factor': best_params['funding_rate_weight'] * raw['funding_rate'],
            'funding_rate': raw['funding_rate'],
            'oi_p2_15min': raw['oi_p2_raw'],
            'a1_raid_factor': best_params['a1_raid'] * raw['a1_raid_raw'],
            'zone_escape_factor': best_params['zone_escape'] * raw['zone_escape_raw'],
            'a1_doji_factor': best_params['a1_doji_adjust'] if raw['a1_doji_flag'] else 1.0,
            'a1_open': raw['a1_open'],
            'a1_high': raw['a1_high'],
            'a1_low': raw['a1_low'],
            'a1_close': raw['a1_close'],
            'a1_upper_wick': raw['a1_upper_wick'],
            'a1_lower_wick': raw['a1_lower_wick'],
            'a1_U_wick_pct': raw['a1_U_wick_pct'],
            'a1_L_wick_pct': raw['a1_L_wick_pct'],
            'a1_total_wick_pct': raw['a1_total_wick_pct'],
            'a1_body_pct': raw['a1_body_pct']
        }
        log_daily_record(weighted, day, actual)

# ===================== MAIN RUN =====================
if __name__ == '__main__':
    # Load and cache data once
    df_agg, df_daily, df_5min = load_data()
    df_a1 = load_custom_3h()
    df_15 = load_custom_15min()
    df_1h = load_custom_1h()
    
    # Compute baselines once using df_agg
    baseline_wick_pct, baseline_body_median, baseline_ret_to_open = compute_baselines(df_agg)
    logger.info(f"Computed baselines: daily_wick_pct={baseline_wick_pct:.4f}, daily_body_median={baseline_body_median:.4f}, return_to_open={baseline_ret_to_open:.4f}")
    
    # Precompute raw features for each trading day and cache them
    raw_features_df = compute_raw_features(df_agg, df_5min, df_a1, df_15, df_1h)
    logger.info(f"Computed raw features for {len(raw_features_df)} trading days.")
    
    # Clear error log table at the start
    conn = sqlite3.connect(LOG_DB)
    c = conn.cursor()
    c.execute("DELETE FROM daily_error_logs")
    conn.commit()
    conn.close()
    
    # Run hyperparameter optimization using cached raw features and joblib
    trials = Trials()
    best = fmin(fn=objective,
                space={key: hp.quniform(key, *val) for key, val in PARAM_RANGES.items()},
                algo=tpe.suggest,
                max_evals=MAX_EVALS,
                trials=trials)
    best_accuracy = 100 - trials.best_trial['result']['loss']
    logger.info("=== Final Optimization Results ===")
    logger.info(f"Best Overall Accuracy: {best_accuracy:.2f}%")
    logger.info(f"Best Parameters: {best}")
    
    log_final_daily_results(best)
    print_error_summary()
