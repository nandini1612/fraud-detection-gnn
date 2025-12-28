from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:

    # File paths
    raw_data_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_data_dir: Path = PROJECT_ROOT / "data" / "processed"

    # Dataset choice
    dataset_name: str = "elliptic"  # Options: "elliptic", "synthetic", "ieee-cis"

    # Graph construction
    edge_threshold: float = 0.0  # Min transaction amount to create edge
    directed: bool = True  # Money flows have direction (A→B ≠ B→A)

    # Feature engineering
    use_node_features: bool = True
    use_edge_features: bool = True
    use_temporal_features: bool = True  # Time-based patterns

    # Train/val/test split
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Fraud ring simulation (for synthetic data)
    num_fraud_rings: int = 50
    ring_size_range: tuple = (3, 10)  # Min/max accounts per ring

    def __post_init__(self):

        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, (
            "Split ratios must sum to 1.0"
        )
        assert 0 < self.train_ratio < 1, "Train ratio must be in (0, 1)"


@dataclass(frozen=True)
class GraphSAGEConfig:
    # Architecture
    input_dim: int = 166  # Will be set dynamically based on features
    hidden_channels: int = 64
    output_dim: int = 2  # Binary classification (fraud/legitimate)
    num_layers: int = 2

    # Aggregation
    aggregator_type: str = "mean"  # Options: "mean", "max", "lstm", "pool"

    # Regularization
    dropout: float = 0.5

    # Neighbor sampling (KEY GRAPHSAGE FEATURE)
    num_neighbors: List[int] = field(default_factory=lambda: [10, 5])

@dataclass(frozen=True)
class TrainingConfig:

    # Optimization
    optimizer: str = "adam"  # Options: "adam", "adamw", "sgd"
    learning_rate: float = 0.001
    weight_decay: float = 5e-4  # L2 regularization

    # Training loop
    num_epochs: int = 100
    batch_size: int = 512  # Number of target nodes per batch
    early_stopping_patience: int = 15  # Stop if no improvement

    # Loss function
    loss_function: str = "weighted_ce"  # Handle class imbalance
    fraud_class_weight: float = 10.0  # Fraud is ~2% of data

    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "step"  # Options: "step", "cosine", "plateau"
    scheduler_step_size: int = 20
    scheduler_gamma: float = 0.5

    # Hardware
    device: str = "cuda"  # Will auto-detect GPU availability
    num_workers: int = 4  # DataLoader parallel workers

    # Reproducibility
    seed: int = 42  # Fix random seed for reproducibility


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Experiment tracking and logging
    """

    # MLflow
    experiment_name: str = "fraud-detection-gnn"
    tracking_uri: str = "file:./mlruns"  # Local tracking

    # Logging
    log_interval: int = 10  # Log every N batches
    save_model_interval: int = 10  # Save checkpoint every N epochs

    # Checkpointing
    checkpoint_dir: Path = Path("checkpoints")
    save_best_only: bool = True


# ============================================================================
# MAIN CONFIG: Combines all sub-configs
# ============================================================================


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: GraphSAGEConfig = field(default_factory=GraphSAGEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# ============================================================================
# FACTORY FUNCTION: Returns config instance
# ============================================================================


def get_config() -> Config:
    return Config()
