import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_config
from src.data.download import EllipticDataLoader


class GraphBuilder:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.scaler = StandardScaler()  # Zero mean, unit variance
        self.node_id_to_idx = None  # Maps txId → index (0 to N-1)
        self.num_features = None

    def build(self, save_path: Optional[Path] = None) -> Data:
        print("=" * 80)
        print("BUILDING PYTORCH GEOMETRIC GRAPH")
        print("=" * 80)

        # Load data
        features_df, edges_df, classes_df = self._load_data()

        # Process features
        x, node_id_to_idx, time_steps = self._build_node_features(features_df)
        self.node_id_to_idx = node_id_to_idx

        # Build edges
        edge_index = self._build_edge_index(edges_df, node_id_to_idx)

        # Create labels
        y = self._build_labels(classes_df, node_id_to_idx)

        # Create temporal masks
        train_mask, val_mask, test_mask = self._build_temporal_masks(time_steps, y)

        # Combine into PyTorch Geometric Data object
        data = Data(
            x=x,  # Node features [num_nodes, num_features]
            edge_index=edge_index,  # Edge connectivity [2, num_edges]
            y=y,  # Node labels [num_nodes]
            train_mask=train_mask,  # Boolean mask [num_nodes]
            val_mask=val_mask,  # Boolean mask [num_nodes]
            test_mask=test_mask,  # Boolean mask [num_nodes]
        )

        # Validate
        self._validate_graph(data)

        # Save
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, save_path)
            print(f"\n✓ Graph saved to: {save_path}")
            print(f"   File size: {save_path.stat().st_size / 1e6:.2f} MB")

        return data

    def _load_data(self):
        print("\n📂 STEP 1: Loading data...")
        loader = EllipticDataLoader(self.config.data.raw_data_dir / "elliptic")
        features_df, edges_df, classes_df = loader.load()

        print(f"   ✓ Features: {features_df.shape}")
        print(f"   ✓ Edges: {edges_df.shape}")
        print(f"   ✓ Classes: {classes_df.shape}")

        return features_df, edges_df, classes_df

    def _build_node_features(self, features_df: pd.DataFrame):
        print("\n🔢 STEP 2: Processing node features...")

        # Check format by looking for 'txId' column
        if "txId" in features_df.columns:
            print("   ✓ Format: Google Drive (with headers)")

            # Extract components from named columns
            account_ids = features_df["txId"].values
            time_steps = features_df["Time step"].values.astype(np.int32)

            # Feature columns (exclude txId and Time step)
            # This gives us 182 features: 93 Local + 72 Aggregate + 17 named features
            feature_cols = [
                col for col in features_df.columns if col not in ["txId", "Time step"]
            ]

        else:
            print("   ✓ Format: Kaggle (no headers)")

            # Assign column names manually
            # Format: [txId, feature_1, feature_2, ..., feature_N, time_step]
            features_df.columns = (
                ["txId"]
                + [f"feature_{i}" for i in range(1, features_df.shape[1] - 1)]
                + ["time_step"]
            )

            account_ids = features_df["txId"].values
            time_steps = features_df.iloc[:, -1].values.astype(np.int32)

            feature_cols = [
                col for col in features_df.columns if col.startswith("feature_")
            ]

        # Create node ID mapping: txId → continuous index (0, 1, 2, ...)
        node_id_to_idx = {txId: idx for idx, txId in enumerate(account_ids)}

        print(f"   ✓ Unique nodes: {len(node_id_to_idx):,}")
        print(f"   ✓ Time range: {time_steps.min()} to {time_steps.max()}")
        print(f"   ✓ Unique time steps: {len(np.unique(time_steps))}")
        print(f"   ✓ Feature columns: {len(feature_cols)}")

        # Sanity check: Time steps should range from 1 to ~49
        if time_steps.max() < 10:
            raise ValueError(
                f"Time steps invalid! Max={time_steps.max()}, expected ~49"
            )

        # Extract features as numpy array
        features_np = features_df[feature_cols].values
        print(f"   ✓ Feature matrix: {features_np.shape}")

        # Convert to numeric (handle any string values that might exist)
        # errors='coerce' converts non-numeric to NaN
        features_numeric = (
            pd.DataFrame(features_np).apply(pd.to_numeric, errors="coerce").values
        )
        
        num_missing = np.isnan(features_numeric).sum()
        if num_missing > 0:
            print(f"   ⚠️  {num_missing} missing values → imputing with means")
            from sklearn.impute import SimpleImputer

            imputer = SimpleImputer(strategy="mean")
            features_numeric = imputer.fit_transform(features_numeric)

        # Normalize: Transform to zero mean, unit variance
        # Formula: (x - mean) / std
        # Result: Most values fall between -3 and +3
        features_normalized = self.scaler.fit_transform(features_numeric)

        # Convert to PyTorch tensor
        x = torch.tensor(features_normalized, dtype=torch.float)

        print(f"   ✓ Normalized: mean={x.mean():.6f}, std={x.std():.6f}")

        self.num_features = x.shape[1]

        return x, node_id_to_idx, time_steps

    def _build_edge_index(self, edges_df: pd.DataFrame, node_id_to_idx: Dict):
        print("\n🔗 STEP 3: Building edges...")

        # Get column names (might be 'txId1','txId2' or numeric indices 0,1)
        if "txId1" not in edges_df.columns:
            edges_df.columns = ["txId1", "txId2"]

        print(f"   ✓ Total edges: {len(edges_df):,}")

        # Filter to valid edges (both nodes must exist in node_id_to_idx)
        # WHY? Some transactions might be filtered out in preprocessing
        valid_mask = edges_df["txId1"].isin(node_id_to_idx) & edges_df["txId2"].isin(
            node_id_to_idx
        )
        valid_edges = edges_df[valid_mask]

        print(f"   ✓ Valid edges: {len(valid_edges):,}")

        # Map transaction IDs to continuous indices
        # Example: txId 123456 → index 0, txId 789012 → index 1
        source = valid_edges["txId1"].map(node_id_to_idx).values
        target = valid_edges["txId2"].map(node_id_to_idx).values

        # Create COO format: [2, num_edges]
        # Row 0: source nodes
        # Row 1: target nodes
        edge_array = np.array([source, target])
        edge_index = torch.tensor(edge_array, dtype=torch.long)

        print(f"   ✓ Edge index: {edge_index.shape}")

        return edge_index

    def _build_labels(self, classes_df: pd.DataFrame, node_id_to_idx: Dict):
        print("\n🏷️  STEP 4: Creating labels...")

        if "class" not in classes_df.columns:
            classes_df.columns = ["txId", "class"]

        # Print unique classes for debugging
        unique_classes = classes_df["class"].unique()
        print(f"   ℹ️  Unique class values: {unique_classes}")

        # Count each class
        class_counts = classes_df["class"].value_counts()
        print(f"   ℹ️  Class distribution: {dict(class_counts)}")

        # Detect version based on class values
        has_string_unknown = "unknown" in unique_classes
        has_numeric_3 = 3 in unique_classes or "3" in unique_classes

        if has_string_unknown:
            # KAGGLE FORMAT
            print("   ✓ Detected: Kaggle version (string 'unknown')")
            # Kaggle: '1'=licit, '2'=illicit
            class_mapping = {
                "unknown": -1,
                "1": 0,  # Licit
                "2": 1,  # Illicit
            }
        elif has_numeric_3:
            # GOOGLE DRIVE FORMAT
            print("   ✓ Detected: Google Drive version (numeric 3)")

            # Determine which class is smaller (should be fraud)
            # Fraud is always the minority (~2-10% of labeled data)
            class_1_count = class_counts.get(1, 0) + class_counts.get("1", 0)
            class_2_count = class_counts.get(2, 0) + class_counts.get("2", 0)

            if class_1_count < class_2_count:
                # Class 1 is minority → Class 1 = Fraud
                print("   ✓ Class 1 is smaller → Class 1 = Illicit (fraud)")
                print("   ✓ Class 2 is larger → Class 2 = Licit (legitimate)")
                # Google Drive: REVERSED from Kaggle!
                class_mapping = {
                    3: -1,  # Unknown
                    "3": -1,
                    1: 1,  # Illicit (fraud) - REVERSED!
                    "1": 1,
                    2: 0,  # Licit (legitimate) - REVERSED!
                    "2": 0,
                }
            else:
                # Class 2 is minority → Class 2 = Fraud
                print("   ✓ Standard mapping (Class 1 = Licit)")
                class_mapping = {
                    3: -1,
                    "3": -1,
                    1: 0,  # Licit
                    "1": 0,
                    2: 1,  # Illicit
                    "2": 1,
                }
        else:
            raise ValueError(f"Unknown label format! Unique classes: {unique_classes}")

        print(f"   ℹ️  Using mapping: {class_mapping}")

        # Map labels: Create pandas Series with txId as index
        label_map = classes_df.set_index("txId")["class"].map(class_mapping)

        # Initialize all labels to -1 (unknown)
        labels = np.full(len(node_id_to_idx), -1, dtype=np.int64)

        # Assign labels for nodes that have them
        for txId, idx in node_id_to_idx.items():
            if txId in label_map.index:
                mapped = label_map[txId]
                if pd.notna(mapped):  # Handle NaN from unmapped values
                    labels[idx] = int(mapped)

        # Convert to PyTorch tensor
        y = torch.tensor(labels, dtype=torch.long)

        # Statistics
        num_unknown = (y == -1).sum().item()
        num_licit = (y == 0).sum().item()
        num_illicit = (y == 1).sum().item()

        print(f"   ✓ Label distribution:")
        print(
            f"     - Unknown (-1): {num_unknown:,} ({num_unknown / len(y) * 100:.1f}%)"
        )
        print(f"     - Licit (0): {num_licit:,} ({num_licit / len(y) * 100:.1f}%)")
        print(
            f"     - Illicit (1): {num_illicit:,} ({num_illicit / len(y) * 100:.1f}%)"
        )

        if num_licit + num_illicit > 0:
            fraud_ratio = num_illicit / (num_licit + num_illicit)
            print(f"   ✓ Fraud ratio: {fraud_ratio * 100:.2f}%")

            # Sanity check: Fraud should be ~10%
            # If it's way off, labels might be wrong!
            if 0.05 < fraud_ratio < 0.15:
                print(f"   ✅ Fraud ratio looks correct (~10%)")
            else:
                print(
                    f"   ⚠️  Unexpected fraud ratio! Expected ~10%, got {fraud_ratio * 100:.1f}%"
                )

        return y

    def _build_temporal_masks(self, time_steps: np.ndarray, y: torch.Tensor):
        print("\n📅 STEP 5: Creating temporal masks...")

        min_step = time_steps.min()
        max_step = time_steps.max()
        total_range = max_step - min_step + 1

        print(f"   ✓ Time range: {min_step} to {max_step} ({total_range} steps)")

        # 70/15/15 split based on time steps
        train_end = min_step + int(total_range * 0.7)
        val_end = min_step + int(total_range * 0.85)

        print(f"   ✓ Train: {min_step}-{train_end}")
        print(f"   ✓ Val: {train_end + 1}-{val_end}")
        print(f"   ✓ Test: {val_end + 1}-{max_step}")

        # Create time-based boolean masks
        train_time = time_steps <= train_end
        val_time = (time_steps > train_end) & (time_steps <= val_end)
        test_time = time_steps > val_end

        # Filter to labeled nodes only (exclude unknown=-1)
        labeled = (y != -1).numpy()

        # Combine: Must be in time range AND labeled
        train_mask = torch.tensor(train_time & labeled, dtype=torch.bool)
        val_mask = torch.tensor(val_time & labeled, dtype=torch.bool)
        test_mask = torch.tensor(test_time & labeled, dtype=torch.bool)

        # Stats
        print(f"\n   📊 Splits:")
        print(
            f"     Train: {train_mask.sum():,} ({((y == 1) & train_mask).sum()} fraud)"
        )
        print(f"     Val: {val_mask.sum():,} ({((y == 1) & val_mask).sum()} fraud)")
        print(f"     Test: {test_mask.sum():,} ({((y == 1) & test_mask).sum()} fraud)")

        # Sanity check: All splits should have data
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            raise ValueError("Empty split!")

        return train_mask, val_mask, test_mask

    def _validate_graph(self, data: Data):
        print("\n🔍 STEP 6: Validating...")

        assert not torch.isnan(data.x).any(), "NaN in features!"
        assert not torch.isinf(data.x).any(), "Inf in features!"
        assert data.edge_index.min() >= 0, "Invalid edges!"
        assert data.edge_index.max() < data.num_nodes, "Out of bounds!"
        assert data.y.min() >= -1 and data.y.max() <= 1, "Invalid labels!"
        assert not (data.train_mask & data.test_mask).any(), "Split overlap!"

        print("   ✅ All checks passed!")

        print("\n" + "=" * 80)
        print("✅ GRAPH READY")
        print("=" * 80)
        print(f"Nodes: {data.num_nodes:,}")
        print(f"Edges: {data.num_edges:,}")
        print(f"Features: {data.num_node_features}")
        print(f"Train: {data.train_mask.sum():,}")
        print(f"Val: {data.val_mask.sum():,}")
        print(f"Test: {data.test_mask.sum():,}")
        print("=" * 80)


def main():
    from src.utils.config import get_config

    config = get_config()
    builder = GraphBuilder(config)

    save_path = config.data.processed_data_dir / "fraud_graph.pt"
    data = builder.build(save_path=save_path)

    print("\n✅ Success! Run: python test_graph.py")


if __name__ == "__main__":
    main()
