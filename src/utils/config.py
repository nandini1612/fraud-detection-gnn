"""
Configuration Management System

WHY THIS FILE EXISTS:
- Centralizes all hyperparameters (avoid magic numbers in code)
- Enables reproducibility (fixed seed, versioned configs)
- Simplifies hyperparameter tuning (change one file, not scattered code)

DESIGN PATTERN:
- Uses dataclasses (Python 3.7+) for type-safe configuration
- Frozen classes prevent accidental mutation during runtime
- Hierarchical structure (data config, model config, training config)

ALTERNATIVES:
1. YAML/JSON files → More common in production (external config)
2. Hydra framework → Advanced (Facebook's config system)
3. Pydantic → Better validation + auto-documentation

PRO TIP FOR INTERVIEWS:
"I use dataclasses for configs because they provide type hints, default values,
and immutability (frozen=True), which prevents bugs from accidental mutation."
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    """
    Data-related configuration

    WHAT THIS CONTROLS:
    - Dataset paths (where to load/save data)
    - Graph construction parameters (how to build the network)
    - Feature engineering choices

    frozen=True: Immutable after creation (prevents bugs)
    """

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
        """
        Validation after initialization

        WHY THIS IS IMPORTANT:
        - Catches config errors early (before training starts)
        - Better than silent failures or cryptic errors later
        """
        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0, (
            "Split ratios must sum to 1.0"
        )
        assert 0 < self.train_ratio < 1, "Train ratio must be in (0, 1)"


@dataclass(frozen=True)
class GraphSAGEConfig:
    """
    GraphSAGE model architecture configuration

    KEY DECISIONS EXPLAINED:

    1. hidden_channels: Embedding dimension at each layer
       - Too small (16): Underfitting, can't capture complex patterns
       - Too large (512): Overfitting, slow training
       - Sweet spot: 64-128 for medium graphs (10K-100K nodes)

    2. num_layers: Depth of message passing
       - 1 layer: Only direct neighbors (1-hop)
       - 2 layers: Friends-of-friends (2-hop) ← MOST COMMON
       - 3+ layers: Risk of over-smoothing (all nodes become similar)

       RULE: K layers = K-hop neighborhood aggregation

    3. dropout: Regularization during training
       - Randomly drops neurons to prevent co-adaptation
       - 0.5 is standard; 0.3-0.7 range is typical

    4. aggregator_type: How to combine neighbor features
       - "mean": Average (smooth, stable)
       - "max": Takes strongest signal (good for anomalies)
       - "lstm": Learns aggregation (more expressive, slower)
    """

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
    # Layer 1: Sample 10 neighbors, Layer 2: Sample 5 from each
    # Total nodes per target: 1 + 10 + 50 = 61 nodes

    # INTERVIEW QUESTION:
    # "Why sample neighbors instead of using all?"
    # ANSWER: "Computational efficiency and regularization. Without sampling,
    # we'd need to process exponentially growing neighborhoods (10→100→1000).
    # Sampling provides O(K^L) complexity where K=samples, L=layers."


@dataclass(frozen=True)
class TrainingConfig:
    """
    Training loop configuration

    DESIGN DECISIONS:

    1. loss_function:
       - "cross_entropy": Standard for classification
       - "focal_loss": Handles class imbalance (fraud is rare)
       - "weighted_ce": Assign higher weight to fraud class

    2. optimizer: Adam vs SGD vs AdamW
       - Adam: Adaptive learning rate, good default
       - SGD: Slower but better generalization (sometimes)
       - AdamW: Adam + weight decay (current best practice)

    3. scheduler: Learning rate decay strategy
       - "step": Decay every N epochs
       - "cosine": Smooth decay following cosine curve
       - "plateau": Decay when validation stops improving
    """

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
    """
    Master configuration object

    USAGE IN CODE:
    ```python
    from src.utils.config import get_config

    config = get_config()
    model = GraphSAGE(
        input_dim=config.model.input_dim,
        hidden_channels=config.model.hidden_channels
    )
    ```
    """

    data: DataConfig = field(default_factory=DataConfig)
    model: GraphSAGEConfig = field(default_factory=GraphSAGEConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


# ============================================================================
# FACTORY FUNCTION: Returns config instance
# ============================================================================


def get_config() -> Config:
    """
    Factory function to create config

    WHY USE A FUNCTION?
    - Single point of access (easy to modify)
    - Can add logic (e.g., load from YAML if exists)
    - Better for testing (can inject custom configs)

    PRODUCTION EVOLUTION:
    In real systems, this would:
    1. Load base config from YAML
    2. Override with environment variables
    3. Override with command-line args
    4. Validate and return

    Example:
    ```python
    config = get_config()
    config = override_from_env(config)
    config = override_from_args(config, sys.argv)
    ```
    """
    return Config()


# ============================================================================
# INTERVIEW TALKING POINTS
# ============================================================================

"""
KEY CONCEPTS TO EXPLAIN:

1. **Why not just use global variables?**
   - Type safety: dataclasses enforce types
   - Immutability: frozen=True prevents bugs
   - Validation: __post_init__ catches errors early
   - Testability: Easy to create test configs

2. **What happens in production?**
   - Configs stored in version control (Git)
   - Different configs for dev/staging/prod
   - Loaded from external files (YAML/JSON)
   - Secrets (API keys) from environment variables

3. **Hyperparameter tuning?**
   - This config is the "default"
   - Use Optuna/Ray Tune to search config space
   - Track all experiments in MLflow
   
4. **How does this help reproducibility?**
   - Fixed seed (training.seed = 42)
   - Version config with code (Git commit)
   - Log config to MLflow with each run
   - Anyone can reproduce by loading same config

COMMON MISTAKES STUDENTS MAKE:
❌ Hardcoding hyperparameters in model code
❌ Using different configs for train/test
❌ Not validating config values
❌ Forgetting to set random seeds

✅ THIS APPROACH: Single source of truth, validated, versioned
"""
