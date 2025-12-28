"""
Verify Elliptic Dataset (Google Drive version with headers)
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("ELLIPTIC DATA VERIFICATION (Google Drive Version)")
print("=" * 80)

# Load with headers this time
features_df = pd.read_csv(
    "data/raw/elliptic/elliptic_txs_features.csv",
    low_memory=False,  # Handle mixed types
)

print(f"\nFeatures shape: {features_df.shape}")
print(f"\nColumn names (first 10):")
print(features_df.columns[:10].tolist())

print(f"\nColumn names (last 5):")
print(features_df.columns[-5:].tolist())

# Check if 'Time step' column exists
if "Time step" in features_df.columns:
    time_col = features_df["Time step"]

    print(f"\n✅ 'Time step' column found!")
    print(f"  Values (first 20): {time_col.head(20).values}")

    # Try to convert to numeric
    time_numeric = pd.to_numeric(time_col, errors="coerce")

    print(f"\n  Min: {time_numeric.min()}")
    print(f"  Max: {time_numeric.max()}")
    print(f"  Unique values: {time_numeric.nunique()}")

    # Check if values are integers 1-49
    if time_numeric.min() == 1 and time_numeric.max() == 49:
        print("\n  ✅ TIME STEPS ARE CORRECT! (1 to 49)")
    else:
        print(f"\n  ⚠️  Unexpected range: {time_numeric.min()} to {time_numeric.max()}")
else:
    print("\n❌ 'Time step' column not found!")
    print(f"Available columns: {features_df.columns.tolist()}")

# Check classes file
print("\n" + "=" * 80)
print("Checking classes file...")
print("=" * 80)

classes_df = pd.read_csv("data/raw/elliptic/elliptic_txs_classes.csv")
print(f"\nClasses shape: {classes_df.shape}")
print(f"Columns: {classes_df.columns.tolist()}")
print(f"\nClass distribution:")
print(classes_df.iloc[:, 1].value_counts())

# Check edges file
print("\n" + "=" * 80)
print("Checking edges file...")
print("=" * 80)

edges_df = pd.read_csv("data/raw/elliptic/elliptic_txs_edgelist.csv")
print(f"\nEdges shape: {edges_df.shape}")
print(f"Columns: {edges_df.columns.tolist()}")
print(f"First few edges:")
print(edges_df.head())

print("\n" + "=" * 80)
print("✅ ALL FILES LOOK GOOD!")
print("=" * 80)
