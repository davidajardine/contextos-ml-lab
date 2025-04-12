import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_table(db_path, table_name):
    """Load data from a given table and sort by date ascending."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name} ORDER BY date ASC;", conn)
    conn.close()
    # Ensure that date column is of type datetime.date
    df['date'] = pd.to_datetime(df['date']).dt.date
    return df

def save_predictions(db_path, predictions_df, table_name):
    """Save predictions DataFrame into a table."""
    conn = sqlite3.connect(db_path)
    predictions_df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    logger.info(f"Predictions successfully saved to database table '{table_name}'.")

def main():
    db_path = 'bybit_data.db'
    
    # Load full available dataset from the 'training_set' table.
    # (Note: In your case, this table currently covers 2024-02-01 to 2025-02-01.)
    logger.info("Loading full dataset from training_set table...")
    full_df = load_table(db_path, 'training_set')
    logger.info(f"Full dataset loaded: {len(full_df)} records.")
    
    # If you wanted to split by date but your full dataset only covers one year,
    # you can instead do an 80/20 temporal split.
    # Determine the split date using the earliest and latest dates.
    all_dates = sorted(full_df['date'].unique())
    split_index = int(len(all_dates) * 0.8)
    train_dates = all_dates[:split_index]
    test_dates = all_dates[split_index:]
    
    df_train = full_df[full_df['date'].isin(train_dates)]
    df_test = full_df[full_df['date'].isin(test_dates)]
    
    logger.info(f"Training set: {len(df_train)} records loaded.")
    logger.info(f"Testing set: {len(df_test)} records loaded.")
    
    # Define feature columns (all columns except 'date' and 'target')
    feature_cols = [col for col in full_df.columns if col not in ['date', 'target']]
    
    X_train = df_train[feature_cols].values
    y_train = df_train['target'].astype(int).values
    X_test = df_test[feature_cols].values
    y_test = df_test['target'].astype(int).values
    
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predict on the testing set
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Evaluate predictions
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    
    logger.info(f"Accuracy: {acc * 100:.2f}%")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{cr}")
    
    # Save predictions with date, probability, predicted flag, and actual outcome
    predictions_df = df_test[['date']].copy()
    predictions_df['prediction'] = y_pred_proba
    predictions_df['predicted_flag'] = y_pred
    predictions_df['actual_SND'] = y_test
    
    save_predictions(db_path, predictions_df, 'predictions_logistic_split')
    
if __name__ == "__main__":
    main()
