import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm


class EllipticDataLoader:
    # Official URLs (from Kaggle/GitHub)
    URLS = {
        "features": "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set/download",
        # Backup: GitHub release if Kaggle requires authentication
        "github_backup": "https://github.com/elliptic-co/elliptic-dataset/archive/refs/heads/main.zip",
    }

    def __init__(self, data_dir: Path = Path("data/raw/elliptic")):
        """
        Initialize data loader

        Args:
            data_dir: Directory to store raw data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths (standard Elliptic structure)
        self.features_path = self.data_dir / "elliptic_txs_features.csv"
        self.edges_path = self.data_dir / "elliptic_txs_edgelist.csv"
        self.classes_path = self.data_dir / "elliptic_txs_classes.csv"

    def download(self, force: bool = False):
        if self._is_downloaded() and not force:
            print(f"✓ Data already exists at {self.data_dir}")
            return

        print("=" * 70)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 70)
        print("\nThe Elliptic dataset requires Kaggle authentication.")
        print("\nSTEPS:")
        print("1. Go to: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
        print("2. Click 'Download' (requires Kaggle account)")
        print("3. Extract the ZIP file")
        print("4. Place these files in:", self.data_dir)
        print("   - elliptic_txs_features.csv")
        print("   - elliptic_txs_edgelist.csv")
        print("   - elliptic_txs_classes.csv")
        print("\n" + "=" * 70)

        # Alternative: Provide synthetic data generator for immediate start
        print("\nALTERNATIVE: Generate synthetic data for immediate experimentation")
        print("Run: python src/data/synthetic_generator.py")

    def _is_downloaded(self) -> bool:
        """Check if all required files exist"""
        return (
            self.features_path.exists()
            and self.edges_path.exists()
            and self.classes_path.exists()
        )

    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not self._is_downloaded():
            self.download()
            if not self._is_downloaded():
                raise FileNotFoundError(
                    f"Dataset files not found in {self.data_dir}. "
                    "Please download manually (see instructions above)."
                )

        print("Loading Elliptic dataset...")

        features = pd.read_csv(self.features_path)  # Let pandas infer headers
        print(f"✓ Features: {features.shape} (nodes x features)")

        # Load edges (has headers: txId1, txId2)
        edges = pd.read_csv(self.edges_path)
        print(f"✓ Edges: {edges.shape}")

        # Load classes (has headers: txId, class)
        classes = pd.read_csv(self.classes_path)
        print(f"✓ Classes: {classes.shape}")

        # VALIDATION: Verify we loaded the correct format
        # This helps catch issues early if someone uses a different dataset version
        if "txId" not in features.columns:
            raise ValueError(
                f"Expected 'txId' column in features file. "
                f"Got columns: {features.columns.tolist()[:5]}... "
                f"Did you download from Google Drive? Kaggle version has no headers."
            )

        if "Time step" not in features.columns:
            raise ValueError(
                f"Expected 'Time step' column in features file. "
                f"Got columns: {features.columns.tolist()[:5]}... "
                f"This might be the Kaggle version (no headers)."
            )

        # Additional validation for edges and classes
        if "txId1" not in edges.columns or "txId2" not in edges.columns:
            raise ValueError(
                f"Expected 'txId1' and 'txId2' columns in edges file. "
                f"Got columns: {edges.columns.tolist()}"
            )

        if "class" not in classes.columns:
            raise ValueError(
                f"Expected 'class' column in classes file. "
                f"Got columns: {classes.columns.tolist()}"
            )

        return features, edges, classes

    def get_statistics(self) -> Dict:
        features, edges, classes = self.load()

        # Class distribution (Google Drive uses numeric: 3=unknown, 2=licit, 1=illicit)
        class_counts = classes["class"].value_counts()

        # Graph structure
        num_nodes = len(features)
        num_edges = len(edges)

        # Temporal span (from 'Time step' column)
        time_steps = features["Time step"].nunique()

        # UPDATED: Handle Google Drive numeric format (3, 2, 1)
        stats = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_features": features.shape[1] - 2,  # Exclude txId and Time step
            "num_time_steps": time_steps,
            "class_distribution": {
                "unknown": class_counts.get(3, 0),  # Changed from "unknown" string
                "licit": class_counts.get(2, 0),  # Changed from "1" string
                "illicit": class_counts.get(1, 0),  # Changed from "2" string
            },
            "fraud_ratio": class_counts.get(1, 0) / len(classes),  # 1 = illicit
            "labeled_ratio": (class_counts.get(2, 0) + class_counts.get(1, 0))
            / len(classes),
        }

        return stats


# ============================================================================
# USAGE EXAMPLE
# ============================================================================


def main():
    """
    Demo script to download and inspect data

    RUN THIS FIRST:
    ```bash
    python src/data/download.py
    ```
    """
    from src.utils.config import get_config

    config = get_config()
    loader = EllipticDataLoader(config.data.raw_data_dir / "elliptic")

    # Download (or show instructions)
    loader.download()

    # If data exists, show statistics
    if loader._is_downloaded():
        stats = loader.get_statistics()

        print("\n" + "=" * 70)
        print("ELLIPTIC DATASET STATISTICS")
        print("=" * 70)
        print(f"Nodes (transactions): {stats['num_nodes']:,}")
        print(f"Edges (flows): {stats['num_edges']:,}")
        print(f"Features per node: {stats['num_features']}")
        print(f"Time steps: {stats['num_time_steps']}")
        print(f"\nClass Distribution:")
        print(f"  Unknown: {stats['class_distribution']['unknown']:,}")
        print(f"  Licit:   {stats['class_distribution']['licit']:,}")
        print(f"  Illicit: {stats['class_distribution']['illicit']:,}")
        print(f"\nFraud ratio: {stats['fraud_ratio']:.2%}")
        print(f"Labeled ratio: {stats['labeled_ratio']:.2%}")
        print("=" * 70)


if __name__ == "__main__":
    main()
