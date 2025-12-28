"""
Baseline Model 2: XGBoost

PURPOSE:
- More powerful baseline than Logistic Regression
- Can capture non-linear patterns and feature interactions
- Still ignores graph structure (uses only node features)

WHY XGBOOST?
✅ State-of-the-art for tabular data
✅ Handles non-linear relationships
✅ Built-in feature importance
✅ Robust to overfitting (regularization)
✅ Industry standard (Kaggle winner, used at Uber/Airbnb)

EXPECTED IMPROVEMENT OVER LOGISTIC REGRESSION:
- Better at capturing complex patterns
- Should improve F1 by 5-15%
- But still won't use graph structure!

INTERVIEW QUESTION: "When would you use XGBoost vs GNN?"
ANSWER:
- XGBoost: When graph structure doesn't matter (isolated fraud)
- GNN: When relationships matter (fraud rings, money laundering chains)
- Try both! Sometimes XGBoost + graph features beats pure GNN
"""

import torch
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import get_config


class XGBoostBaseline:
    """
    XGBoost baseline for fraud detection

    ADVANTAGES OVER LOGISTIC REGRESSION:
    - Non-linear decision boundaries (trees!)
    - Automatic feature interactions (no manual engineering)
    - Built-in regularization (prevents overfitting)
    - Handles missing values naturally

    DISADVANTAGES:
    - Still ignores graph structure
    - Harder to interpret than logistic regression
    - Slower to train (but still fast compared to GNNs)
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self.model = None
        self.results = {}

    def train(self, data_path: Path = None):
        """
        Train XGBoost model

        HYPERPARAMETERS EXPLAINED:
        - max_depth=6: How deep trees can grow (prevents overfitting)
        - learning_rate=0.1: Step size for gradient descent (smaller = more robust)
        - n_estimators=200: Number of trees (more = better, but slower)
        - subsample=0.8: Train each tree on 80% of data (prevents overfitting)
        - colsample_bytree=0.8: Use 80% of features per tree (faster, more robust)
        - scale_pos_weight: Handle class imbalance (fraud is minority)
        - eval_metric='aucpr': Optimize for area under PR curve (better for imbalanced data)

        WHY THESE VALUES?
        - Typical defaults for fraud detection
        - Balanced between performance and training time
        - Can be tuned later with hyperparameter search
        """
        print("=" * 80)
        print("TRAINING XGBOOST BASELINE")
        print("=" * 80)

        # Load data
        if data_path is None:
            data_path = self.config.data.processed_data_dir / "fraud_graph.pt"

        print(f"\n📂 Loading data from {data_path}...")
        data = torch.load(data_path)

        print(
            f"   ✓ Loaded: {data.num_nodes:,} nodes, {data.num_node_features} features"
        )

        # Extract numpy arrays
        X = data.x.numpy()
        y = data.y.numpy()

        train_mask = data.train_mask.numpy()
        val_mask = data.val_mask.numpy()
        test_mask = data.test_mask.numpy()

        # Get train/val/test data
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

        # Calculate scale_pos_weight for class imbalance
        # Formula: (# negative samples) / (# positive samples)
        # This makes fraud examples more important during training
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(
            f"\n⚖️  Class imbalance ratio: {scale_pos_weight:.2f}:1 (legitimate:fraud)"
        )

        # Train model
        print(f"\n🔧 Training XGBoost...")
        print(f"   • Trees: 200")
        print(f"   • Max depth: 6")
        print(f"   • Learning rate: 0.1")
        print(f"   • Subsample: 0.8")
        print(f"   • Scale pos weight: {scale_pos_weight:.2f}")

        self.model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",  # Area under PR curve (good for imbalanced data)
            random_state=42,
            n_jobs=-1,  # Use all CPU cores
            tree_method="hist",  # Faster histogram-based algorithm
        )

        # Train with validation set for early stopping
        # This prevents overfitting by stopping when validation performance plateaus
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,  # Set to True to see training progress
        )

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
        """Evaluate model on a dataset split"""
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]

        # Metrics
        report = classification_report(y, y_pred, output_dict=True)
        auc = roc_auc_score(y, y_proba)
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

    def _plot_results(self, X_test, y_test):
        """Create visualizations"""
        results = self.results["test"]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Confusion Matrix
        cm = results["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Greens",
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
            fpr, tpr, linewidth=2, label=f"AUC = {results['auc']:.4f}", color="green"
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
        axes[2].plot(recall, precision, linewidth=2, color="green")
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
        save_path = output_dir / "baseline_xgboost_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"\n✓ Plots saved to: {save_path}")

        plt.show()

    def get_feature_importance(self, top_k=20):
        """
        Get most important features

        XGBOOST FEATURE IMPORTANCE:
        - Gain: Total improvement in loss from splits on this feature
        - Higher gain = more important feature
        - Shows which features the model relies on most

        BUSINESS VALUE:
        - Focus feature engineering on important features
        - Remove unimportant features (faster inference)
        - Explain model to stakeholders
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Get feature importance (gain-based)
        importance = self.model.feature_importances_

        # Sort by importance
        indices = np.argsort(importance)[::-1][:top_k]

        print(f"\n📊 Top {top_k} Most Important Features:")
        print(f"{'Rank':<6} {'Feature':<15} {'Importance':>12}")
        print("-" * 40)

        for i, idx in enumerate(indices, 1):
            imp = importance[idx]
            print(f"{i:4d}.  Feature {idx:<7d}  {imp:12.6f}")

        # Plot feature importance
        fig, ax = plt.subplots(figsize=(10, 8))

        top_features = indices[:top_k]
        top_importance = importance[top_features]

        ax.barh(range(len(top_features)), top_importance, color="green", alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([f"Feature {i}" for i in top_features])
        ax.invert_yaxis()
        ax.set_xlabel("Importance (Gain)", fontsize=12)
        ax.set_title(
            f"Top {top_k} Most Important Features (XGBoost)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()

        # Save
        output_dir = Path("outputs")
        save_path = output_dir / "xgboost_feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Feature importance plot saved to: {save_path}")

        plt.show()


def main():
    """
    Run XGBoost baseline

    USAGE:
    python -m src.models.baseline_xgboost

    EXPECTED OUTPUT:
    - Should beat Logistic Regression by 5-15% F1
    - But still won't use graph structure!
    - Shows what's achievable with better tabular model
    """
    config = get_config()
    baseline = XGBoostBaseline(config)

    # Train and evaluate
    results = baseline.train()

    # Show feature importance
    baseline.get_feature_importance(top_k=20)

    print("\n" + "=" * 80)
    print("COMPARISON: LOGISTIC REGRESSION vs XGBOOST")
    print("=" * 80)
    print("\n💡 Key Takeaways:")
    print("   • XGBoost usually improves F1 by 5-15% over Logistic Regression")
    print("   • Both models ignore graph structure (only use node features)")
    print(
        "   • If XGBoost doesn't improve much, features might not be very informative"
    )
    print("   • Next: Try GNN to leverage graph structure!")
    print("\n✅ Baseline complete!")
    print("💡 Next: Train simple GCN to see benefit of graph structure")
    print("   Command: python -m src.models.baseline_gcn")


if __name__ == "__main__":
    main()


# ============================================================================
# INTERVIEW QUESTIONS TO PREPARE
# ============================================================================

"""
Q1: Why does XGBoost usually beat Logistic Regression?
A:
   - Non-linear decision boundaries (trees vs linear)
   - Automatic feature interactions (no manual engineering)
   - Better handles complex patterns
   - More expressive model class

Q2: When would Logistic Regression be better than XGBoost?
A:
   - Need interpretability (linear weights easier to explain)
   - Very small dataset (XGBoost might overfit)
   - Real-time inference (LR is faster)
   - Linearly separable data (no need for complexity)

Q3: What if XGBoost doesn't improve over Logistic Regression?
A:
   - Features might already be linear
   - Or features aren't very informative
   - Or class imbalance is too severe
   - Try: Better features, graph structure (GNN), more data

Q4: How do you tune XGBoost hyperparameters?
A:
   - Grid search (exhaustive but slow)
   - Random search (faster, often good enough)
   - Bayesian optimization (smart, efficient)
   - Cross-validation for each setting
   - Monitor train vs val to avoid overfitting

Q5: Can you combine XGBoost with graph structure?
A:
   - Yes! Extract graph features (degree, clustering, PageRank)
   - Add as additional features to XGBoost
   - Or: Use GNN embeddings as features for XGBoost
   - Sometimes beats pure GNN (best of both worlds)

Q6: Why use aucpr instead of accuracy?
A:
   - Fraud is ~10% of data (imbalanced)
   - Accuracy misleading (90% accuracy by predicting "all legitimate")
   - AUCPR focuses on precision-recall tradeoff
   - Better for imbalanced classification
"""
