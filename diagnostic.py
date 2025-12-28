# diagnostic.py
import pandas as pd

# Load classes
classes_df = pd.read_csv("data/raw/elliptic/elliptic_txs_classes.csv")
classes_df.columns = ["txId", "class"]

print("First 20 rows:")
print(classes_df.head(20))

print("\nClass value counts:")
print(classes_df["class"].value_counts())

# Load features to check time
features_df = pd.read_csv("data/raw/elliptic/elliptic_txs_features.csv", header=None)
print(f"\nFeatures shape: {features_df.shape}")
print(
    f"Last column (time_step) range: {features_df.iloc[:, -1].min()} to {features_df.iloc[:, -1].max()}"
)
print(f"Unique time steps: {features_df.iloc[:, -1].nunique()}")
