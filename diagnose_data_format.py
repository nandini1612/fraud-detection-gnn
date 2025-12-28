"""
Diagnostic script to understand the exact data format
Run this first: python diagnose_data_format.py
"""

import pandas as pd
from pathlib import Path

# Path to your data
data_dir = Path("data/raw/elliptic")
features_path = data_dir / "elliptic_txs_features.csv"
classes_path = data_dir / "elliptic_txs_classes.csv"
edges_path = data_dir / "elliptic_txs_edgelist.csv"

print("=" * 80)
print("DIAGNOSING DATA FORMAT")
print("=" * 80)

# Check features file
print("\n📊 FEATURES FILE:")
print(f"Path: {features_path}")

# Read first 5 rows with headers
df_with_headers = pd.read_csv(features_path, nrows=5)
print(f"\nWith headers - Shape: {df_with_headers.shape}")
print(
    f"Columns: {list(df_with_headers.columns[:5])} ... {list(df_with_headers.columns[-3:])}"
)
print("\nFirst 3 rows:")
print(df_with_headers.iloc[:3, :5])

# Read first 5 rows without headers
df_no_headers = pd.read_csv(features_path, header=None, nrows=5)
print(f"\nWithout headers - Shape: {df_no_headers.shape}")
print(
    f"Columns: {list(df_no_headers.columns[:5])} ... {list(df_no_headers.columns[-3:])}"
)
print("\nFirst 3 rows:")
print(df_no_headers.iloc[:3, :5])

# Check if first column looks like txId
first_col_with_header = df_with_headers.iloc[:, 0]
first_col_no_header = df_no_headers.iloc[:, 0]

print(f"\n🔍 FIRST COLUMN ANALYSIS:")
print(f"With header: {first_col_with_header.tolist()[:3]}")
print(f"Without header: {first_col_no_header.tolist()[:3]}")

# Check last column (should be time_step)
last_col_with_header = df_with_headers.iloc[:, -1]
last_col_no_header = df_no_headers.iloc[:, -1]

print(f"\n🔍 LAST COLUMN ANALYSIS (should be time_step):")
print(f"With header: {last_col_with_header.tolist()[:3]}")
print(f"Without header: {last_col_no_header.tolist()[:3]}")

# Check for 'txId' in columns
print(f"\n🔍 CHECKING FOR 'txId' COLUMN:")
print(f"'txId' in columns (with headers): {'txId' in df_with_headers.columns}")
print(
    f"'Time step' in columns (with headers): {'Time step' in df_with_headers.columns}"
)

# Full column info
print(f"\n📋 FULL COLUMN LIST (with headers):")
print(df_with_headers.columns.tolist())

# Check classes file
print("\n" + "=" * 80)
print("📊 CLASSES FILE:")
df_classes = pd.read_csv(classes_path, nrows=5)
print(f"Shape: {df_classes.shape}")
print(f"Columns: {df_classes.columns.tolist()}")
print("\nFirst 5 rows:")
print(df_classes.head())

# Check unique classes
df_classes_full = pd.read_csv(classes_path)
print(f"\nUnique class values: {df_classes_full.iloc[:, 1].unique()}")
print(f"Class distribution:\n{df_classes_full.iloc[:, 1].value_counts()}")

# Check edges file
print("\n" + "=" * 80)
print("📊 EDGES FILE:")
df_edges = pd.read_csv(edges_path, nrows=5)
print(f"Shape: {df_edges.shape}")
print(f"Columns: {df_edges.columns.tolist()}")
print("\nFirst 5 rows:")
print(df_edges.head())

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
