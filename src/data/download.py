"""
Data Acquisition Module

PURPOSE:
- Download and prepare the Elliptic Bitcoin dataset
- Provide interface for other datasets (IEEE-CIS, synthetic)
- Handle data versioning and caching

ELLIPTIC DATASET STRUCTURE:
- 203,769 nodes (Bitcoin transactions)
- 234,355 edges (transaction flows)
- 166 features per node (local + aggregate features)
- 2% labeled as illicit (4,545 transactions)
- 21% labeled as licit (42,019 transactions)
- 77% unlabeled (semi-supervised learning opportunity)

WHY THIS DATASET?
✅ Real financial crime data (Bitcoin blockchain)
✅ Industry benchmark (used in 50+ research papers)
✅ Realistic class imbalance (~2% fraud)
✅ Temporal dynamics (49 time steps)
✅ Published by a reputable source (Weber et al., 2019)

DATASET PAPER:
"The Elliptic Data Set: Opening up Machine Learning on the Blockchain"
https://arxiv.org/abs/1908.02591

WHAT FEATURES LOOK LIKE:
- Local features (94): Transaction-specific (amount, # inputs/outputs)
- Aggregate features (72): Neighbor statistics (max, min, std of connected txs)
- Features are anonymized (privacy protection)
"""

import os
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from tqdm import tqdm


class EllipticDataLoader:
    """
    Handles downloading and loading the Elliptic dataset

    DESIGN PATTERN:
    - Singleton-like behavior (downloads once, caches)
    - Lazy loading (only downloads when needed)

    INTERVIEW QUESTION:
    "How would you handle very large datasets that don't fit in memory?"

    ANSWER:
    1. Stream from disk (pandas chunksize parameter)
    2. Use Dask for out-of-core computation
    3. Sample strategically (stratified sampling)
    4. Use database (PostgreSQL with PostGIS for graphs)
    5. Distributed processing (Spark, Ray)
    """

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
        """
        Download dataset if not already present

        Args:
            force: Re-download even if files exist

        NOTE: Kaggle datasets require API authentication
        For this project, I'll provide instructions for manual download

        PRODUCTION APPROACH:
        - Use Kaggle API with credentials
        - Store in cloud (S3, GCS) with versioning
        - Implement checksum verification
        """
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
        """
        Load dataset into pandas DataFrames

        Returns:
            features: Node features (203K x 184) - with headers!
            edges: Edge list (234K x 2)
            classes: Node labels (203K x 2)

        WHAT EACH FILE CONTAINS:

        1. features.csv (GOOGLE DRIVE VERSION):
           - Column 0: txId (Transaction ID)
           - Column 1: Time step (1-49)
           - Columns 2-183: Anonymized features
                * Local_feature_1 to Local_feature_93 (93 features)
                * Aggregate_feature_1 to Aggregate_feature_72 (72 features)
                * Plus 19 named features (in_txs_degree, out_txs_degree, etc.)

           IMPORTANT: The Google Drive version HAS HEADERS!
           This is different from the original Kaggle version (no headers)

        2. edgelist.csv:
           - Column 0: txId1 (Source transaction)
           - Column 1: txId2 (Target transaction)
           (Directed: money flows from source → target)

        3. classes.csv:
           - Column 0: txId (Transaction ID)
           - Column 1: class (3=unknown, 2=licit, 1=illicit)

           IMPORTANT: Google Drive uses numeric codes:
           - 3 = unknown (unlabeled)
           - 2 = licit (legitimate transactions)
           - 1 = illicit (fraud/illegal activity)

           This is OPPOSITE of Kaggle which uses:
           - "unknown" = unlabeled
           - "1" = licit
           - "2" = illicit
        """
        if not self._is_downloaded():
            self.download()
            if not self._is_downloaded():
                raise FileNotFoundError(
                    f"Dataset files not found in {self.data_dir}. "
                    "Please download manually (see instructions above)."
                )

        print("Loading Elliptic dataset...")

        # FIXED: Load WITH headers (Google Drive version includes column names)
        # Previously used header=None which was treating the header row as data!
        # This caused graph_builder.py to fail because it couldn't find 'txId' column
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

        # WHY THIS MATTERS:
        # The diagnostic output showed your data HAS headers, but the old code
        # was stripping them with header=None. This caused:
        # 1. Column names to become [0, 1, 2, ...] instead of ['txId', 'Time step', ...]
        # 2. graph_builder.py to think it was Kaggle format (no 'txId' found)
        # 3. Wrong column indexing (trying to get time_step from wrong position)

        return features, edges, classes

    def get_statistics(self) -> Dict:
        """
        Compute dataset statistics

        WHY THIS IS IMPORTANT:
        - Understanding class imbalance (crucial for model design)
        - Graph structure analysis (degree distribution)
        - Temporal patterns (fraud evolves over time)

        INTERVIEW TIP:
        Always perform EDA before modeling. Red flags:
        - Extreme class imbalance → Use weighted loss
        - Many isolated nodes → Graph might not help
        - High-degree hubs → Risk of over-smoothing
        """
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


# ============================================================================
# INTERVIEW QUESTIONS TO PREPARE
# ============================================================================

"""
Q1: Why is this dataset better than creating a random graph?
A: Real-world data has:
   - Realistic class imbalance (2% fraud)
   - Power-law degree distribution (not random)
   - Temporal patterns (fraud tactics evolve)
   - Domain-specific structure (transaction flows)

Q2: What if we have unlabeled data (77% unknown)?
A: Semi-supervised learning approaches:
   - Use labeled data for supervision
   - Unlabeled data for regularization (consistency loss)
   - Self-training: Label confident predictions
   - GNNs naturally handle this (message passing spreads labels)

Q3: How would you handle data drift in production?
A: 
   - Monitor prediction distribution over time
   - Retrain periodically on recent data
   - Implement concept drift detection (e.g., ADWIN)
   - Use online learning (update model incrementally)
   - A/B test new models before full deployment

Q4: What if the graph is too large for memory?
A: 
   - Graph sampling (this is what GraphSAGE does!)
   - Mini-batch training (sample subgraphs)
   - Neighbor sampling (only use K neighbors)
   - Graph partitioning (cluster-GCN)
   - Use graph databases (Neo4j, DGraph)

Q5: How do you ensure reproducibility?
A:
   - Fix random seeds (numpy, torch, Python)
   - Version data (DVC, Git LFS)
   - Log data statistics (detect distribution shifts)
   - Containerize (Docker with pinned dependencies)

Q6: Why did the original code fail with header=None?
A:
   - Google Drive version HAS column headers (txId, Time step, etc.)
   - header=None tells pandas to treat first row as data, not column names
   - This caused column names to become [0, 1, 2, ...] instead of ['txId', ...]
   - graph_builder.py checks for 'txId' column to detect format
   - Without 'txId' in columns, it wrongly assumed Kaggle format (no headers)
   - Then tried to access wrong column indices (e.g., last column for time_step)
   
   LESSON: Always verify your data format! Use diagnostic scripts before processing.
"""
