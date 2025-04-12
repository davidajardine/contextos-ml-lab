import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def load_data(db_path, table_name, start_date, end_date):
    """
    Load data from a given table with a date filter (dates in 'YYYY-MM-DD' format).
    """
    conn = sqlite3.connect(db_path)
    query = f"""
    SELECT * FROM {table_name}
    WHERE date BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY date ASC;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    # Ensure that the "date" column is converted to a date object
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
    
    # Define date ranges for the full dataset, training, and testing.
    # We assume that the "training_set" table spans from 2023-02-01 to 2025-02-01.
    full_start, full_end = '2023-02-01', '2025-02-01'
    # Training: 2023-02-01 to 2024-02-01; Testing: 2024-02-01 to 2025-02-01
    train_start, train_end = '2023-02-01', '2024-02-01'
    test_start, test_end = '2024-02-01', '2025-02-01'
    
    # Load the full dataset from the training_set table.
    logger.info("Loading complete data from training_set table...")
    full_df = load_data(db_path, 'training_set', full_start, full_end)
    logger.info(f"Full dataset loaded: {len(full_df)} records.")
    
    # Split into training and testing sets based on the date.
    df_train = full_df[(full_df['date'] >= pd.to_datetime(train_start).date()) &
                       (full_df['date'] < pd.to_datetime(train_end).date())]
    df_test = full_df[(full_df['date'] >= pd.to_datetime(test_start).date()) &
                      (full_df['date'] <= pd.to_datetime(test_end).date())]
    
    logger.info(f"Training set: {len(df_train)} records loaded.")
    logger.info(f"Testing set: {len(df_test)} records loaded.")
    
    if len(df_train) < 2:
        raise ValueError("Training set must contain samples from at least 2 classes. Check your date range and data generation.")
    
    # Separate features and outcome.
    # Drop the 'date' column; the outcome is in the 'target' column.
    feature_cols = [col for col in df_train.columns if col not in ['date', 'target']]
    X_train = df_train[feature_cols].values
    y_train = df_train['target'].astype(int).values  # Ensure target is 0/1 integers
    X_test = df_test[feature_cols].values
    y_test = df_test['target'].astype(int).values
    
    # Normalize the feature columns using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a simple logistic regression model
    logger.info("Training Logistic Regression model...")
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Predict on the test set
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability for class "1"
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Evaluate the predictions
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f"Accuracy: {acc*100:.2f}%")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Classification Report:\n{cr}")
    
    # Prepare a DataFrame with predictions for saving:
    # Include the date, predicted probability, predicted label, and the actual S&D outcome.
    predictions_df = df_test[['date']].copy()
    predictions_df['prediction'] = y_pred_proba
    predictions_df['predicted_flag'] = y_pred
    predictions_df['actual_SND'] = y_test  # actual outcome (0 or 1)
    
    # Save predictions into a new table
    save_predictions(db_path, predictions_df, 'predictions_logistic_split')
    
if __name__ == "__main__":
    main()
