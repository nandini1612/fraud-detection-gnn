import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils.config import get_config


class GCN(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=2, dropout=0.5):
        super(GCN, self).__init__()

        # Layer 1: Input → Hidden
        self.conv1 = GCNConv(num_features, hidden_dim)

        # Layer 2: Hidden → Output
        self.conv2 = GCNConv(hidden_dim, num_classes)

        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)

        # Output (log probabilities for numerical stability)
        return F.log_softmax(x, dim=1)


class GCNBaseline:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}

    def train(self, data_path: Path = None, epochs=200, lr=0.01, patience=20):
        print("=" * 80)
        print("TRAINING GCN BASELINE")
        print("=" * 80)

        # Load data
        if data_path is None:
            data_path = self.config.data.processed_data_dir / "fraud_graph.pt"

        print(f"\n📂 Loading data from {data_path}...")
        data = torch.load(data_path)
        data = data.to(self.device)

        print(f"   ✓ Loaded: {data.num_nodes:,} nodes, {data.num_edges:,} edges")
        print(f"   ✓ Device: {self.device}")

        # Count fraud in each split
        train_fraud = (data.y[data.train_mask] == 1).sum().item()
        val_fraud = (data.y[data.val_mask] == 1).sum().item()
        test_fraud = (data.y[data.test_mask] == 1).sum().item()

        print(f"\n📊 Dataset splits:")
        print(f"   Train: {data.train_mask.sum():,} samples ({train_fraud:,} fraud)")
        print(f"   Val:   {data.val_mask.sum():,} samples ({val_fraud:,} fraud)")
        print(f"   Test:  {data.test_mask.sum():,} samples ({test_fraud:,} fraud)")

        # Calculate class weights for weighted loss
        # Give more weight to minority class (fraud)
        train_labels = data.y[data.train_mask]
        num_legitimate = (train_labels == 0).sum().item()
        num_fraud = (train_labels == 1).sum().item()

        # Weight = total / (num_classes * class_count)
        weight_legitimate = len(train_labels) / (2 * num_legitimate)
        weight_fraud = len(train_labels) / (2 * num_fraud)
        class_weights = torch.tensor([weight_legitimate, weight_fraud]).to(self.device)

        print(
            f"\n⚖️  Class weights: Legitimate={weight_legitimate:.4f}, Fraud={weight_fraud:.4f}"
        )

        # Initialize model
        print(f"\n🔧 Initializing GCN...")
        print(f"   • Architecture: {data.num_node_features} → 64 → 2")
        print(f"   • Dropout: 0.5")
        print(f"   • Optimizer: Adam (lr={lr})")

        self.model = GCN(
            num_features=data.num_node_features,
            hidden_dim=64,
            num_classes=2,
            dropout=0.5,
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        criterion = nn.NLLLoss(
            weight=class_weights
        )  # Negative Log Likelihood with weights

        # Training loop with early stopping
        print(f"\n🚀 Training for up to {epochs} epochs...")
        print(f"   Early stopping patience: {patience} epochs")

        best_val_f1 = 0
        patience_counter = 0
        train_losses = []
        val_f1_scores = []

        for epoch in tqdm(range(epochs), desc="Training"):
            # Train
            self.model.train()
            optimizer.zero_grad()

            out = self.model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

            # Evaluate on validation set
            self.model.eval()
            with torch.no_grad():
                out = self.model(data.x, data.edge_index)
                pred = out.argmax(dim=1)

                # Calculate validation F1
                val_pred = pred[data.val_mask].cpu().numpy()
                val_true = data.y[data.val_mask].cpu().numpy()

                from sklearn.metrics import f1_score

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
                print(
                    f"\n   Epoch {epoch + 1}/{epochs}: Loss={loss.item():.4f}, Val F1={val_f1:.4f}"
                )

            # Early stopping
            if patience_counter >= patience:
                print(
                    f"\n   ⚠️  Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)"
                )
                break

        # Load best model
        self.model.load_state_dict(best_model_state)
        print(f"\n   ✓ Training complete! Best Val F1: {best_val_f1:.4f}")

        # Evaluate on all splits
        print(f"\n📈 Evaluating...")
        self.results["train"] = self._evaluate(data, data.train_mask, "Train")
        self.results["val"] = self._evaluate(data, data.val_mask, "Validation")
        self.results["test"] = self._evaluate(data, data.test_mask, "Test")

        # Print summary
        self._print_summary()

        # Plot results
        self._plot_results(data)

        # Plot training curves
        self._plot_training_curves(train_losses, val_f1_scores)

        return self.results

    def _evaluate(self, data, mask, split_name):
        """Evaluate model on a dataset split"""
        self.model.eval()

        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            proba = torch.exp(out)[:, 1]  # Probability of fraud class

            # Get predictions for this split
            y_true = data.y[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
            y_proba = proba[mask].cpu().numpy()

        # Metrics
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

    def _plot_results(self, data):
        """Create visualizations"""
        results = self.results["test"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Confusion Matrix
        cm = results["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Purples",
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
            fpr, tpr, linewidth=2, label=f"AUC = {results['auc']:.4f}", color="purple"
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
        axes[2].plot(recall, precision, linewidth=2, color="purple")
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
        save_path = output_dir / "baseline_gcn_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Plots saved to: {save_path}")

        plt.show()

    def _plot_training_curves(self, train_losses, val_f1_scores):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curve
        axes[0].plot(train_losses, linewidth=2, color="purple")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Training Loss Curve", fontsize=14, fontweight="bold")
        axes[0].grid(alpha=0.3)

        # F1 curve
        axes[1].plot(val_f1_scores, linewidth=2, color="orange")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Validation F1 Score")
        axes[1].set_title("Validation F1 Curve", fontsize=14, fontweight="bold")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()

        # Save
        output_dir = Path("outputs")
        save_path = output_dir / "gcn_training_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Training curves saved to: {save_path}")

        plt.show()


def main():
    config = get_config()
    baseline = GCNBaseline(config)

    # Train and evaluate
    results = baseline.train(epochs=200, lr=0.01, patience=20)

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print("\n📊 Test Set F1 Scores:")
    print("   • Logistic Regression: 0.098")
    print("   • XGBoost:            0.035")
    print(f"   • GCN:                {results['test']['f1']:.3f}")

    if results["test"]["f1"] > 0.15:
        print("\nGCN beats baselines! Graph structure helps!")
    else:
        print("\nGCN didn't improve much. Graph might not be very informative.")

    print("\nNext: Train GraphSAGE for better performance")
    print("   Command: python -m src.models.train_graphsage")


if __name__ == "__main__":
    main()
