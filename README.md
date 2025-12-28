# Bitcoin Fraud Detection under Temporal Concept Drift

This repository presents a focused experimental study on fraud detection in Bitcoin transaction networks, with an emphasis on **temporal generalization** rather than peak offline performance.

The project evaluates linear models, tree-based ensembles, and graph neural networks under strict chronological splits. While all models achieved strong training performance, evaluation on future data revealed a counterintuitive result: **the simplest model generalized best**, while complex models failed sharply.

The findings show that **temporal concept drift—not model architecture—is the dominant challenge in fraud detection**.

---

## Project Motivation

Fraud detection is often framed as a modeling problem: use more expressive architectures, add graph structure, or engineer richer features. However, real-world fraud systems operate in non-stationary environments where adversaries continuously adapt.

This project was designed to examine that gap.

Rather than optimizing for a single metric, the goal was to understand:
- why high-performing models fail under realistic evaluation
- whether model complexity helps or hurts under temporal shift
- what actually matters when deploying fraud detection systems in production

The result is a vertical slice of a fraud detection pipeline that prioritizes **failure analysis and system-level insight** over leaderboard performance.

---

## Dataset

- **Elliptic Bitcoin Transaction Dataset**
- ~203K transaction nodes
- ~234K directed edges
- 49 discrete temporal snapshots
- Node-level transaction features
- Labels: licit / illicit / unknown

The dataset’s explicit temporal structure makes it well-suited for studying **distribution shift and evolving fraud behavior**.

---

## Experimental Design

### Temporal Data Splits

All experiments were conducted using **strict chronological splits** to avoid information leakage:

- **Training:** Timesteps 1–35  
- **Validation:** Timesteps 36–42  
- **Test:** Timesteps 43–49  

This setup reflects how models are trained and deployed in real systems—on historical data, then evaluated on future behavior.

---

## Methods

### Modeling Approach

A progression of models was evaluated to isolate the effect of model capacity and graph structure:

1. Logistic Regression (linear baseline)
2. XGBoost (tree-based ensemble)
3. Graph Neural Networks (GCN, GraphSAGE)

All models were trained on the same feature space and evaluated using identical temporal splits. The intent was not to tune each model to its maximum potential, but to observe **how different levels of complexity behave under temporal drift**.

### Evaluation Metrics

Given class imbalance and the cost of missed fraud, evaluation focused on:
- F1 score
- Recall on illicit transactions
- Generalization gap between training and test periods

Random splits were intentionally avoided, as they obscure temporal failure modes.

---

## Results

| Model | Train F1 | Test F1 | Test Recall |
|------|----------|---------|-------------|
| Logistic Regression | 0.73 | **0.098** | **46.7%** |
| XGBoost | 0.98 | 0.035 | 1.8% |
| GCN | 0.63 | 0.045 | 5.3% |
| GraphSAGE | 0.96 | 0.021 | 1.8% |

Despite lower training performance, **Logistic Regression outperformed complex GNNs by ~4× on test F1**, demonstrating substantially better temporal generalization.

---

## Analysis & Key Insights

### The Models Didn’t Fail — They Revealed the Real Problem

All evaluated models trained successfully, achieving **90%+ F1** on historical data. However, test performance collapsed to **2–10% F1** across architectures.

This is not a bug or implementation issue—it is **temporal concept drift**.

Fraud tactics changed within weeks, rendering learned patterns obsolete. The resulting **~86% performance drop from train to test** provides strong evidence that the core challenge is **data distribution shift**, not insufficient model expressiveness.

---

### Why Simpler Models Won

High-capacity models (XGBoost, GNNs) memorized transient fraud signatures that did not persist into the future. Logistic Regression, constrained by its simplicity, avoided overfitting to short-lived patterns and retained significantly higher recall.

In effect, the simplest model generalized better because it was *less capable of memorization*.

This counterintuitive outcome underscores an important production lesson: **more powerful models can fail faster under non-stationarity**.

---

### Limits of Graph Structure Under Drift

While transaction graphs encode rich relational information, the topology itself evolved over time. Static graph assumptions limited the effectiveness of GNNs, as neighborhood structure and interaction patterns shifted alongside fraud strategies.

Graph learning is most effective when relationships are stable—an assumption violated in adversarial, fast-moving domains.

---

## Figures and Visualisations
1. Data Distribution
[class_distribution.png]
Left: Labeled data distribution (excluding unknown class), showing the imbalance between licit and illicit transactions.

Right: Full dataset class distribution, highlighting a significant proportion of unknown data.

Key Insight: The dataset is highly imbalanced, which directly affects model evaluation and emphasizes the need for careful metric selection (F1-score over accuracy).

2. Graph Structure Analysis
[degree_distribution.png]
Left: In-degree distribution (log-log scale).

Right: Out-degree distribution (log-log scale).

Key Insight: Both distributions exhibit heavy-tailed behavior, indicating a few nodes have very high connectivity while most nodes have low connectivity—common in real-world transactional graphs.

3. Feature Distributions
[feature_distributions.png]
Shows selected feature distributions (feature_1, feature_2, feature_10, feature_50) for licit vs. fraudulent transactions.

Key Insight: Certain features demonstrate strong separation between licit and fraudulent classes, which can be exploited by simpler models (e.g., logistic regression).

4. Temporal Fraud Analysis
[temporal_fraud_ratio.png]
Plot of fraud ratio over time steps.

Key Insight: The fraud ratio changes drastically over time, confirming temporal concept drift, which explains why models trained on historical data fail to generalize.

5. Baseline GNN Model Performance
[baseline_logreg.png]

[baseline_xgboost.png]

[baseline_gcn.png]

Confusion matrices, ROC curves, and precision-recall curves for baseline GCN and GraphSAGE models.

Key Insight: All GNNs train perfectly on historical data (90%+ F1), but test performance drops drastically (2–10% F1), confirming that temporal concept drift dominates performance over model complexity.

6. Feature Importance
[xgboost_imp_features.png]
Bar chart of feature importance from XGBoost.

Key Insight: Certain features dominate the prediction signal, reinforcing that simpler models leveraging key features can outperform complex GNNs on evolving fraud patterns.

7. Training and Validation Curves
[training_validation_baseline_gcn.png]
Left: Training loss curve.

Right: Validation F1 curve over epochs for baseline GCN.

Key Insight: Training converges smoothly, but validation performance remains low, highlighting that model overfitting to historical data occurs due to concept drift rather than inadequate training.

8. GraphSAGE Test Performance
[graphsage.png]
left: GraphSAGE confusion matrix on the test set.
Middle: ROC curve showing model’s discriminative ability.
Right: Precision-recall curve highlighting performance on the positive (fraudulent) class.

Key Insight: Despite GraphSAGE capturing graph structures, test performance is still very low due to severe temporal concept drift. Precision-recall curve shows that the model struggles to detect fraudulent transactions, reinforcing that historical graph patterns do not generalize well.

9. Training vs. Validation Performance Comparison
[train_val_graphsage.png]
Left: Training loss (or F1) over epochs.
Right: Validation loss (or F1) over epochs.

Key Insight: Training performance improves steadily, but validation performance stagnates or decreases, indicating overfitting. This comparison confirms that even well-optimized GNNs fail to generalize on evolving data, emphasizing the importance of temporal evaluation.

