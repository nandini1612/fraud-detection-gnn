"""
GraphSAGE: Graph Sample and Aggregate

PURPOSE:
- Our main model! Advanced GNN for fraud detection
- Uses SAGEConv layers (better than GCN)
- Full-batch training (simplified for this project)

NOTE: This version uses full-batch training instead of neighbor sampling
because pyg-lib/torch-sparse have Windows compatibility issues.
For production at scale, use Linux with proper sampling libraries.

WHY GRAPHSAGE ARCHITECTURE?
✅ Better aggregation than GCN (mean aggregation)
✅ Deeper network (128 → 64 vs 64 for GCN)
✅ More parameters (better capacity)
✅ SAGEConv designed for sampling (even if we can't use it here)

EXPECTED RESULTS:
- Test F1: 0.15-0.30 (hopefully better than baselines!)
- Should show if graph structure + better architecture helps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

from src.utils.config import get_config


class GraphSAGE(nn.Module):
    """
    GraphSAGE Model

    ARCHITECTURE:
    Input (182 features)
      → SAGE Layer 1 (182 → 128) + ReLU + Dropout
      → SAGE Layer 2 (128 → 64) + ReLU + Dropout
      → Linear (64 → 2 classes)
      → Log Softmax

    IMPROVEMENTS OVER SIMPLE GCN:
    1. Larger hidden dimensions (128, 64 vs 64)
    2. SAGEConv aggregation (better than GCNConv)
    3. More capacity to learn complex patterns
    """

    def __init__(
        self,
        num_features,
        hidden_dim1=128,
        hidden_dim2=64,
        num_classes=2,
        dropout=0.5,
        aggr="mean",
    ):
        """
        Initialize GraphSAGE

        Args:
            num_features: Input feature dimension (182)
            hidden_dim1: First hidden layer (128)
            hidden_dim2: Second hidden layer (64)
            num_classes: Output classes (2)
            dropout: Dropout probability (0.5)
            aggr: Aggregation method ('mean')
        """
        super(GraphSAGE, self).__init__()

        # Layer 1: Input → Hidden1 (with mean aggregation)
        self.conv1 = SAGEConv(num_features, hidden_dim1, aggr=aggr)

        # Layer 2: Hidden1 → Hidden2
        self.conv2 = SAGEConv(hidden_dim1, hidden_dim2, aggr=aggr)

        # Output layer: Hidden2 → Classes
        self.lin = nn.Linear(hidden_dim2, num_classes)

        self.dropout = dropout

    def forward(self, x, edge_index):
        """Forward pass"""
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Output layer
        x = self.lin(x)

        return F.log_softmax(x, dim=1)


class GraphSAGETrainer:
    """
    Training and evaluation wrapper for GraphSAGE

    Uses full-batch training (processes entire graph at once)
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def train(self, data_path: Path = None, epochs=200, lr=0.01, patience=20):
        """
        Train GraphSAGE model

        Args:
            data_path: Path to processed graph
            epochs: Maximum training epochs (200)
            lr: Learning rate (0.01)
            patience: Early stopping patience (20 epochs)
        """
        print("=" * 80)
        print("TRAINING GRAPHSAGE")
        print("=" * 80)

        # Load data
        if data_path is None:
            data_path = self.config.data.processed_data_dir / "fraud_graph.pt"

        print(f"\n📂 Loading data from {data_path}...")
        data = torch.load(data_path)

        print(f"   ✓ Loaded: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
        print(f"   ✓ Device: {self.device}")

        # Move data to device
        data = data.to(self.device)

        # Count fraud in each split
        train_fraud = (data.y[data.train_mask] == 1).sum().item()
        val_fraud = (data.y[data.val_mask] == 1).sum().item()
        test_fraud = (data.y[data.test_mask] == 1).sum().item()

        print(f"\n📊 Dataset splits:")
        print(f"   Train: {data.train_mask.sum():,} samples ({train_fraud:,} fraud)")
        print(f"   Val:   {data.val_mask.sum():,} samples ({val_fraud:,} fraud)")
        print(f"   Test:  {data.test_mask.sum():,} samples ({test_fraud:,} fraud)")

        print(f"\n🔄 Training mode: Full-batch (processing entire graph)")
        print(
            f"   Note: For large-scale production, use neighbor sampling with pyg-lib"
        )

        # Calculate class weights
        train_labels = data.y[data.train_mask]
        num_legitimate = (train_labels == 0).sum().item()
        num_fraud = (train_labels == 1).sum().item()

        weight_legitimate = len(train_labels) / (2 * num_legitimate)
        weight_fraud = len(train_labels) / (2 * num_fraud)
        class_weights = torch.tensor([weight_legitimate, weight_fraud]).to(self.device)

        print(
            f"\n⚖️  Class weights: Legitimate={weight_legitimate:.4f}, Fraud={weight_fraud:.4f}"
        )

        # Initialize model
        print(f"\n🔧 Initializing GraphSAGE...")
        print(f"   • Architecture: {data.num_node_features} → 128 → 64 → 2")
        print(f"   • Aggregation: Mean")
        print(f"   • Dropout: 0.5")
        print(f"   • Optimizer: Adam (lr={lr})")

        self.model = GraphSAGE(
            num_features=data.num_node_features,
            hidden_dim1=128,
            hidden_dim2=64,
            num_classes=2,
            dropout=0.5,
            aggr="mean",
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.NLLLoss(weight=class_weights)

        # Training loop
        print(f"\n🚀 Training for up to {epochs} epochs...")
        print(f"   Early stopping patience: {patience} epochs")

        best_val_f1 = 0
        patience_counter = 0
        train_losses = []
        val_f1_scores = []

        start_time = time.time()

        for epoch in tqdm(range(epochs), desc="Training"):
            # Train
            self.model.train()
            optimizer.zero_grad()

            # Forward pass on entire graph
            out = self.model(data.x, data.edge_index)

            # Compute loss only on training nodes
            loss = criterion(out[data.train_mask], data.y[data.train_mask])

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Evaluate on validation set
            self.model.eval()
            with torch.no_grad():
                out = self.model(data.x, data.edge_index)
                pred = out.argmax(dim=1)

                val_pred = pred[data.val_mask].cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()

                val_f1 = f1_score(val_true, val_pred, average="binary", zero_division=0)
                val_f1_scores.append(val_f1)

            # Early stopping check
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            # Print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(
                    f"\n   Epoch {epoch + 1:3d}/{epochs}: Loss={loss.item():.4f}, Val F1={val_f1:.4f}, Time={elapsed:.1f}s"
                )

            # Early stopping
            if patience_counter >= patience:
                print(
                    f"\n   ⚠️  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)"
                )
                break

        # Load best model
        self.model.load_state_dict(best_model_state)

        total_time = time.time() - start_time
        print(f"\n   ✓ Training complete!")
        print(f"   ✓ Best Val F1: {best_val_f1:.4f}")
        print(f"   ✓ Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")

        # Full evaluation on all splits
        print(f"\n📈 Final Evaluation...")

        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            proba = torch.exp(out)[:, 1]

        self.results["train"] = self._evaluate_predictions(
            data.y[data.train_mask].cpu().numpy(),
            pred[data.train_mask].cpu().numpy(),
            proba[data.train_mask].cpu().numpy(),
            "Train",
        )

        self.results["val"] = self._evaluate_predictions(
            data.y[data.val_mask].cpu().numpy(),
            pred[data.val_mask].cpu().numpy(),
            proba[data.val_mask].cpu().numpy(),
            "Validation",
        )

        self.results["test"] = self._evaluate_predictions(
            data.y[data.test_mask].cpu().numpy(),
            pred[data.test_mask].cpu().numpy(),
            proba[data.test_mask].cpu().numpy(),
            "Test",
        )

        # Print summary
        self._print_summary()

        # Plot results
        self._plot_results()

        # Plot training curves
        self._plot_training_curves(train_losses, val_f1_scores)

        # Save model
        self._save_model()

        return self.results

    def _evaluate_predictions(self, y_true, y_pred, y_proba, split_name):
        """Full evaluation with all metrics"""
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        auc = roc_auc_score(y_true, y_proba)
        cm = confusion_matrix(y_true, y_pred)

        results = {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "auc": auc,
            "confusion_matrix": cm,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        return results

    def _print_summary(self):
        """Print results summary"""
        print("\n" + "=" * 80)
        print("RESULTS SUMMARY")
        print("=" * 80)

        for split_name in ["train", "val", "test"]:
            results = self.results[split_name]
            print(f"\n{split_name.upper()} SET:")
            print(f"   Precision: {results['precision']:.4f}")
            print(f"   Recall:    {results['recall']:.4f}")
            print(f"   F1-Score:  {results['f1']:.4f}")
            print(f"   AUC-ROC:   {results['auc']:.4f}")

            # Confusion matrix
            tn, fp, fn, tp = results["confusion_matrix"].ravel()
            print(f"\n   Confusion Matrix:")
            print(f"   ┌─────────────┬──────────┬──────────┐")
            print(f"   │             │ Pred Neg │ Pred Pos │")
            print(f"   ├─────────────┼──────────┼──────────┤")
            print(f"   │ Actual Neg  │ {tn:8d} │ {fp:8d} │")
            print(f"   │ Actual Pos  │ {fn:8d} │ {tp:8d} │")
            print(f"   └─────────────┴──────────┴──────────┘")

        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print("\n📊 Test Set F1 Scores:")
        print("   • Logistic Regression: 0.098")
        print("   • XGBoost:            0.035")
        print("   • Simple GCN:         0.045")
        print(f"   • GraphSAGE:          {self.results['test']['f1']:.3f}")

        # Determine if we improved
        best_baseline = max(0.098, 0.035, 0.045)
        improvement = (self.results["test"]["f1"] - best_baseline) / best_baseline * 100

        if self.results["test"]["f1"] > best_baseline:
            print(f"\n✅ GraphSAGE beats best baseline by {improvement:.1f}%!")
        else:
            print(f"\n⚠️  GraphSAGE didn't beat best baseline ({improvement:.1f}%)")

        print("=" * 80)

    def _plot_results(self):
        """Create visualizations"""
        results = self.results["test"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Confusion Matrix
        cm = results["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="RdPu",
            ax=axes[0],
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
        )
        axes[0].set_title("Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("Predicted Label")

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(results["y_true"], results["y_proba"])
        axes[1].plot(
            fpr, tpr, linewidth=2, label=f"AUC = {results['auc']:.4f}", color="darkred"
        )
        axes[1].plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title("ROC Curve (Test Set)", fontsize=14, fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(
            results["y_true"], results["y_proba"]
        )
        axes[2].plot(recall, precision, linewidth=2, color="darkred")
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].set_title(
            "Precision-Recall Curve (Test Set)", fontsize=14, fontweight="bold"
        )
        axes[2].grid(alpha=0.3)

        # Add baseline
        fraud_rate = (results["y_true"] == 1).mean()
        axes[2].axhline(
            fraud_rate,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Random (P={fraud_rate:.3f})",
        )
        axes[2].legend()

        plt.tight_layout()

        # Save
        output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "graphsage_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Plots saved to: {save_path}")

        plt.show()

    def _plot_training_curves(self, train_losses, val_f1_scores):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        axes[0].plot(train_losses, linewidth=2, color="darkred")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Training Loss Curve", fontsize=14, fontweight="bold")
        axes[0].grid(alpha=0.3)

        # F1 curve
        axes[1].plot(val_f1_scores, linewidth=2, color="darkorange")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation F1 Score")
        axes[1].set_title("Validation F1 Curve", fontsize=14, fontweight="bold")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        # Save
        output_dir = Path("outputs")
        save_path = output_dir / "graphsage_training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Training curves saved to: {save_path}")

        plt.show()

    def _save_model(self):
        """Save trained model"""
        output_dir = Path("outputs/models")
        output_dir.mkdir(parents=True, exist_ok=True)

        save_path = output_dir / "graphsage_best.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "results": self.results,
            },
            save_path,
        )

        print(f"\n✓ Model saved to: {save_path}")


def main():
    """Train GraphSAGE model"""
    config = get_config()
    trainer = GraphSAGETrainer(config)

    # Train
    results = trainer.train(epochs=200, lr=0.01, patience=20)

    print("\n✅ GraphSAGE training complete!")
    print("\n💡 Next steps:")
    print("   • Analyze results vs baselines")
    print("   • Create final comparison table")
    print("   • Prepare interview talking points")


if __name__ == "__main__":
    main()

# ============================================================================
# INTERVIEW QUESTIONS TO PREPARE
# ============================================================================

"""
Q1: Why GraphSAGE instead of GCN?
A:
   - Scalability: Samples neighbors (O(K) vs O(N))
   - No over-smoothing: Fixed K neighbors per layer
   - Inductive: Works on new nodes without retraining
   - Mini-batch training: Faster, more stable
   - Industry proven: Used in production at scale

Q2: How does neighbor sampling work?
A:
   - Layer 1: Sample 10 random neighbors (1-hop)
   - Layer 2: Sample 5 neighbors of those 10 (2-hop)
   - Total: 1 + 10 + 50 = 61 nodes per target node
   - Much smaller than full graph (200K nodes!)
   - Balances information gain vs computation

Q3: What if GraphSAGE also fails on test set?
A:
   - Confirms temporal drift is the core problem
   - No architecture solves distribution shift
   - Production solution: Continuous retraining
   - Or: Add temporal features (time embeddings)
   - Or: Ensemble with LogReg (best baseline)

Q4: How would you deploy GraphSAGE in production?
A:
   - Pre-compute embeddings offline (batch job)
   - Store in vector DB (Pinecone, Milvus)
   - Real-time: Lookup embedding + simple classifier
   - Retrain weekly on recent data
   - Monitor for performance degradation
   - Human-in-loop for high-risk decisions

Q5: Can you combine GraphSAGE with XGBoost?
A:
   - Yes! Use GraphSAGE to generate node embeddings
   - Feed embeddings + original features to XGBoost
   - Best of both: Graph structure + tree ensembles
   - Often beats pure GNN in practice
   - More interpretable than pure GNN

Q6: What are GraphSAGE's limitations?
A:
   - Still needs retraining for new graph structure
   - Sampling introduces variance (different runs differ)
   - Harder to debug than simple models
   - Requires GPU for reasonable speed
   - Over-parameterized for small graphs
"""
