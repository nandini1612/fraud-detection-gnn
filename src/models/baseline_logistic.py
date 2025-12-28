import torch
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

from src.utils.config import get_config


class LogisticRegressionBaseline:
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.results = {}

    def train(self, data_path: Path = None):
        print("=" * 80)
        print("TRAINING LOGISTIC REGRESSION BASELINE")
        print("=" * 80)

        # Load data
        if data_path is None:
            data_path = self.config.data.processed_data_dir / "fraud_graph.pt"

        print(f"\n📂 Loading data from {data_path}...")
        data = torch.load(data_path)

        print(
            f" Loaded: {data.num_nodes:,} nodes, {data.num_node_features} features"
        )

        # Extract numpy arrays
        X = data.x.numpy()  # [num_nodes, num_features]
        y = data.y.numpy()  # [num_nodes]

        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()

        # Get train data (only labeled nodes)
        X_train = X[train_mask]
        y_train = y[train_mask]

        X_val = X[val_mask]
        y_val = y[val_mask]

        X_test = X[test_mask]
        y_test = y[test_mask]

        print(f"\n📊 Dataset splits:")
        print(f"   Train: {len(y_train):,} samples ({(y_train == 1).sum():,} fraud)")
        print(f"   Val:   {len(y_val):,} samples ({(y_val == 1).sum():,} fraud)")
        print(f"   Test:  {len(y_test):,} samples ({(y_test == 1).sum():,} fraud)")

        # Train model
        print(f"\n🔧 Training Logistic Regression...")
        print(f"   • Solver: lbfgs (efficient for high-dim data)")
        print(f"   • Max iterations: 1000")
        print(f"   • Class weight: balanced (handles imbalance)")

        self.model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",  # Critical for imbalanced data!
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
        )

        self.model.fit(X_train, y_train)
        print(f"   ✓ Training complete!")

        # Evaluate on all splits
        print(f"\n📈 Evaluating...")

        self.results["train"] = self._evaluate(X_train, y_train, "Train")
        self.results["val"] = self._evaluate(X_val, y_val, "Validation")
        self.results["test"] = self._evaluate(X_test, y_test, "Test")

        # Print summary
        self._print_summary()

        # Plot results
        self._plot_results(X_test, y_test)

        return self.results

    def _evaluate(self, X, y, split_name):
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]  # Probability of fraud

        # Classification report (precision, recall, F1)
        report = classification_report(y, y_pred, output_dict=True)

        # AUC-ROC
        auc = roc_auc_score(y, y_proba)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)

        results = {
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1": report["1"]["f1-score"],
            "auc": auc,
            "confusion_matrix": cm,
            "y_true": y,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        return results

    def _print_summary(self):
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

    def _plot_results(self, X_test, y_test):
        results = self.results["test"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Confusion Matrix
        cm = results["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[0],
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
        )
        axes[0].set_title("Confusion Matrix (Test Set)", fontsize=14, fontweight="bold")
        axes[0].set_ylabel("True Label")
        axes[0].set_xlabel("Predicted Label")

        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(results["y_true"], results["y_proba"])
        axes[1].plot(fpr, tpr, linewidth=2, label=f"AUC = {results['auc']:.4f}")
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
        axes[2].plot(recall, precision, linewidth=2)
        axes[2].set_xlabel("Recall")
        axes[2].set_ylabel("Precision")
        axes[2].set_title(
            "Precision-Recall Curve (Test Set)", fontsize=14, fontweight="bold"
        )
        axes[2].grid(alpha=0.3)

        # Add baseline (random classifier)
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
        save_path = output_dir / "baseline_logistic_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Plots saved to: {save_path}")

        plt.show()

    def get_feature_importance(self, top_k=20):
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Get coefficients
        coefs = self.model.coef_[0]

        # Sort by absolute value (magnitude)
        indices = np.argsort(np.abs(coefs))[::-1][:top_k]

        print(f"\n📊 Top {top_k} Most Important Features:")
        print(f"{'Feature':<15} {'Coefficient':>12} {'Effect':>10}")
        print("-" * 40)

        for i, idx in enumerate(indices, 1):
            coef = coefs[idx]
            effect = "↑ Fraud" if coef > 0 else "↓ Fraud"
            print(f"{i:2d}. Feature {idx:<3d}  {coef:12.6f}  {effect:>10}")


def main():
    config = get_config()
    baseline = LogisticRegressionBaseline(config)

    # Train and evaluate
    results = baseline.train()

    # Show feature importance
    baseline.get_feature_importance(top_k=20)

    print("\nBaseline complete!")
    print("Next: Run XGBoost baseline for comparison")
    print("  Command: python -m src.models.baseline_xgboost")


if __name__ == "__main__":
    main()
