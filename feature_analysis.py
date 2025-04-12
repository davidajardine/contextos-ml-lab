import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    # 1. Load the training_set from your SQLite database.
    db_path = 'bybit_data.db'
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM training_set", conn)
    conn.close()

    # 2. Convert the 'date' column to datetime and create a numeric target column.
    df['date'] = pd.to_datetime(df['date'])
    df['target_numeric'] = df['target'].apply(
        lambda x: 1 if str(x).strip().upper() == "TRUE" or float(x) >= 0.5 else 0
    )

    # 3. Identify numeric columns to analyze, excluding 'date', 'target', and 'SND'
    exclude_cols = {'date', 'target', 'SND'}
    numeric_cols = [
        col for col in df.columns
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
    ]

    # 4. Plot distributions for each numeric feature by target.
    print("=== Plotting Feature Distributions by Target ===")
    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(data=df, x=col, hue='target_numeric', bins=30, kde=True, palette="viridis")
        plt.title(f"Distribution of {col} by Target")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.legend(title="Target", labels=["0 (non-S&D)", "1 (S&D)"])
        plt.tight_layout()
        # Save the figure as a PNG and then close it.
        plt.savefig(f"{col}_distribution.png")
        plt.close()

    # 5. Compute correlation of each numeric feature with target_numeric.
    print("\n=== Correlation of Numeric Features with target_numeric ===")
    # Build a correlation matrix for numeric_cols + 'target_numeric'
    corr_matrix = df[numeric_cols + ['target_numeric']].corr()
    
    # Extract the correlation series for 'target_numeric' using .loc and squeeze it to ensure 1D.
    if 'target_numeric' in corr_matrix.columns:
        corrs = corr_matrix.loc[:, 'target_numeric'].squeeze()
        # Drop the target itself if present.
        if 'target_numeric' in corrs.index:
            corrs = corrs.drop('target_numeric')
        
        # Convert the series to a numpy array.
        corr_values = corrs.to_numpy()
        corr_index = np.array(corrs.index)
        
        # Sort indices by absolute correlation (largest to smallest)
        sorted_indices = np.argsort(np.abs(corr_values))
        corrs_sorted = pd.Series(corr_values[sorted_indices][::-1],
                                 index=corr_index[sorted_indices][::-1])
        print(corrs_sorted)
    else:
        print("No 'target_numeric' column found in the correlation matrix.")

if __name__ == "__main__":
    main()
