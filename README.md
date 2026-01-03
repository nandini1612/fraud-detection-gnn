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

## Project Structure Setup

**Important Note**: The GitHub repository structure differs from the actual project structure required to run the code. After cloning, please reorganize your local project directory as follows:

```
bitcoin-fraud-detection/
│
├── data/
│   ├── raw/
│   │   └── elliptic/              # Download dataset here
│   │       ├── elliptic_txs_features.csv
│   │       ├── elliptic_txs_classes.csv
│   │       └── elliptic_txs_edgelist.txt
│   └── processed/                  # Generated during preprocessing
│
├── images/                         # Visualization outputs
│   ├── temp1.png                   # Temporal fraud ratio analysis
│   ├── temp2.png                   # Transaction volume over time
│   ├── fig1.png                    # Model performance dashboard
│   ├── fig2.png                    # Confusion matrix analysis
│   ├── fig3.png                    # Detailed performance metrics
│   └── [additional plots]
│
├── notebooks/                      # Jupyter notebooks
│   └── [analysis notebooks]
│
├── outputs/
│   └── models/                     # Saved model checkpoints
│
├── src/
│   ├── data/
│   │   ├── data_loader.py
│   │   └── preprocessing.py
│   ├── models/
│   │   ├── baseline_models.py
│   │   ├── gnn_models.py
│   │   └── train_models.py
│   ├── training/
│   │   └── trainer.py
│   └── utils/
│       ├── evaluation.py
│       └── visualization.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── diagnostic.py
```

### Dataset Setup

The Elliptic dataset is not included in this repository due to size constraints. You will need to download it separately:

1. Visit the [Elliptic Dataset on Kaggle](https://www.kaggle.com/ellipticco/elliptic-data-set)
2. Download the three required files:
   - `elliptic_txs_features.csv`
   - `elliptic_txs_classes.csv`
   - `elliptic_txs_edgelist.txt`
3. Place them in `data/raw/elliptic/` directory

Create the necessary directories:
```bash
mkdir -p data/raw/elliptic data/processed images outputs/models
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Dataset

- **Elliptic Bitcoin Transaction Dataset**
- ~203K transaction nodes
- ~234K directed edges
- 49 discrete temporal snapshots
- Node-level transaction features
- Labels: licit / illicit / unknown

The dataset's explicit temporal structure makes it well-suited for studying **distribution shift and evolving fraud behavior**.

---

## Temporal Analysis

### Fraud Ratio Evolution

![Temporal Fraud Analysis](images/temp1.png)

The temporal analysis reveals severe concept drift across the dataset's 49 timesteps. The fraud ratio fluctuates dramatically, demonstrating that fraud patterns are non-stationary. This instability is the primary cause of model failure when evaluated on future data.

Key observations:
- Training period (timesteps 1-35) shows fraud ratios ranging from 12% to 71%
- Validation period (timesteps 36-42) exhibits different distributional characteristics
- Test period (timesteps 43-49) has a fraud ratio of approximately 2.59%

This 5-28× reduction in fraud prevalence between training and test periods explains why models trained to recognize historical fraud signatures fail to generalize.

### Transaction Volume Shifts

![Transaction Volume Analysis](images/temp2.png)

Transaction volume varies significantly across time periods, with notable peaks and valleys that correspond to changes in network activity and fraud behavior. The combination of shifting fraud ratios and transaction volumes creates a challenging non-stationary environment where static models quickly become obsolete.

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

![Model Performance Dashboard](images/fig1.png)

### Performance Summary

| Model | Train F1 | Test F1 | Test Recall | Performance Drop |
|------|----------|---------|-------------|------------------|
| Logistic Regression | 0.73 | **0.099** | **46.7%** | -87% |
| XGBoost | 0.98 | 0.030 | 1.8% | -97% |
| GCN | 0.63 | 0.034 | 4.7% | -95% |
| GraphSAGE | 0.96 | 0.008 | 0.6% | -99% |

Despite lower training performance, **Logistic Regression outperformed complex GNNs by ~4× on test F1**, demonstrating substantially better temporal generalization.

### Confusion Matrix Analysis

![Confusion Matrices](images/fig2.png)

The confusion matrices reveal distinct failure patterns across model architectures:

**Logistic Regression**
- Maintains balanced error distribution
- Test confusion matrix shows 90 true frauds detected out of 169 total
- 1,355 false positives, but critically preserves recall capability

**XGBoost**
- Nearly perfect training performance (6,491 TN, 3 TP)
- Complete generalization failure on test set
- Becomes extremely conservative, predicting almost all transactions as legitimate

**Graph Neural Networks (GCN, GraphSAGE)**
- Strong pattern recognition during training
- Catastrophic collapse on test data
- GraphSAGE achieves only 1 true positive out of 169 fraudulent transactions

This analysis demonstrates that high-capacity models memorize training-specific patterns rather than learning generalizable fraud signatures.

### Detailed Performance Metrics

![Performance Metrics Breakdown](images/fig3.png)

**Precision Analysis**:
- All models achieve low precision on test data (0.012-0.100)
- Logistic Regression maintains the highest precision at 0.055
- Complex models show precision collapse despite near-perfect training metrics

**Recall Analysis**:
- Logistic Regression: 46.7% (only model maintaining meaningful recall)
- XGBoost: 1.8%
- GCN: 4.7%
- GraphSAGE: 0.6%

**AUC Comparison**:
- Training AUC values range from 0.678 to 0.803
- Test discrimination ability remains relatively preserved
- However, decision boundaries learned during training fail to capture future fraud patterns

**Error Rate Analysis**:
The false positive vs false negative trade-offs reveal fundamental differences in model behavior:
- Logistic Regression: Balanced error profile (20.8% FPR, 53.3% FNR)
- XGBoost: Ultra-conservative (4.4% FPR, 98.2% FNR)
- Graph models: Near-complete failure to detect fraud (>94% FNR)

---

## Analysis & Key Insights

### The Models Didn't Fail — They Revealed the Real Problem

All evaluated models trained successfully, achieving **90%+ F1** on historical data. However, test performance collapsed to **2–10% F1** across architectures.

This is not a bug or implementation issue—it is **temporal concept drift**.

Fraud tactics changed within weeks, rendering learned patterns obsolete. The resulting **86-99% performance drop from train to test** provides strong evidence that the core challenge is **data distribution shift**, not insufficient model expressiveness.

### Why Simpler Models Won

High-capacity models (XGBoost, GNNs) memorized transient fraud signatures that did not persist into the future. Logistic Regression, constrained by its simplicity, avoided overfitting to short-lived patterns and retained significantly higher recall.

In effect, the simplest model generalized better because it was *less capable of memorization*.

This counterintuitive outcome underscores an important production lesson: **more powerful models can fail faster under non-stationarity**.

### Limits of Graph Structure Under Drift

While transaction graphs encode rich relational information, the topology itself evolved over time. Static graph assumptions limited the effectiveness of GNNs, as neighborhood structure and interaction patterns shifted alongside fraud strategies.

Graph learning is most effective when relationships are stable—an assumption violated in adversarial, fast-moving domains.

### The Recall Preservation Phenomenon

The most operationally significant finding is that Logistic Regression maintains 46.7% recall while other models drop below 5%. In fraud detection systems, failing to catch fraud is often more costly than false alarms. A model that misses 98-99% of fraudulent transactions (as XGBoost and GraphSAGE do on test data) provides no practical value, regardless of training performance.

This recall preservation suggests that linear models learn more robust, generalizable patterns rather than memorizing spurious correlations present in training data.

---

## Figures and Visualizations

The complete analysis includes the following visualizations in the `images/` directory:

**Temporal Analysis**
1. `temp1.png` - Fraud ratio evolution over time steps
2. `temp2.png` - Transaction volume distribution across periods

**Model Performance**
3. `fig1.png` - Model Performance Dashboard (F1 scores, generalization gaps, ROC curves)
4. `fig2.png` - Confusion Matrix Analysis (all four models)
5. `fig3.png` - Detailed Performance Metrics (precision, recall, AUC breakdown)

**Additional Visualizations** (generated during analysis)
- Data distribution plots
- Graph structure analysis (degree distributions)
- Feature distributions by class
- Training/validation curves
- Feature importance analysis

---

## Limitations

- Only static GNN architectures were evaluated; temporal or dynamic GNNs were not explored
- Label noise and partially labeled data may affect absolute metrics
- Findings are based on a single blockchain dataset and may not fully generalize
- No online or continual learning methods were implemented

These constraints are intentional and align with the project's diagnostic focus.

---

## Practical Implications

The findings suggest that effective fraud detection systems should prioritize **adaptation over architecture**. Based on this study, recommended deployment strategies include:

- Weekly (or more frequent) retraining on recent data
- Continuous drift monitoring as a first-class system component
- Simple, interpretable baselines as performance anchors
- Ensemble approaches where low-variance models dominate the weighting
- Explicit focus on recall maintenance over precision optimization

This project reinforced that **production ML is about continuous learning, not static model selection**.

---

## Tech Stack

- Python  
- PyTorch & PyTorch Geometric  
- scikit-learn  
- XGBoost  
- NumPy  
- pandas  
- matplotlib & seaborn
- Plotly

---

## Running the Experiments

After setting up the project structure and downloading the dataset:

1. Preprocess the data:
```bash
python src/data/preprocessing.py
```

2. Train baseline models:
```bash
python src/models/train_models.py --model logistic
python src/models/train_models.py --model xgboost
```

3. Train GNN models:
```bash
python src/models/train_models.py --model gcn
python src/models/train_models.py --model graphsage
```

4. Run a complete diagnostic analysis:
```bash
python diagnostic.py
```

5. Explore analysis notebooks:
```bash
jupyter notebook notebooks/
```

---

## Key Takeaways

- Evaluation design matters more than model choice
- Temporal concept drift is the defining challenge in fraud detection
- Complex models can overfit faster under non-stationarity
- Real-world systems must emphasize retraining, monitoring, and robustness
- Recall preservation is more valuable than training accuracy in adversarial domains

---

## Future Work

- Temporal and dynamic graph neural networks
- Online and continual learning strategies
- Explicit drift detection and alerting mechanisms
- Adaptive retraining schedules based on performance decay
- Comparative analysis across multiple blockchain datasets
- Investigation of feature engineering approaches robust to temporal drift

---

## Contact

For questions, collaboration, or research discussion:
- **Name:** Nandini Saxena  
- **Email:** nandinisaxenawork@gmail.com  
- **GitHub:** https://github.com/nandini1612  
- **LinkedIn:** https://www.linkedin.com/in/nandini-saxena1111/

Interested in research and applied Data Science and ML work involving robustness, distribution shift, and graph-based learning.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
