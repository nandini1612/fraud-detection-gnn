import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
import torch
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# PROFESSIONAL COLOR PALETTE (Inspired by modern data viz)
# ============================================================================

MODEL_COLORS = {
    "Logistic Regression": {
        "primary": "#4A90E2",  # Professional blue
        "light": "#A8D0F0",
        "dark": "#2E5C8A",
    },
    "XGBoost": {
        "primary": "#F5A623",  # Warm orange
        "light": "#FFD89B",
        "dark": "#C67D00",
    },
    "Simple GCN": {
        "primary": "#E85D75",  # Coral red
        "light": "#FFB3C1",
        "dark": "#B8344A",
    },
    "GraphSAGE": {
        "primary": "#50C9CE",  # Teal
        "light": "#9FE5E7",
        "dark": "#2A9499",
    },
}

COLORS = {
    "bg": "#FAFBFC",
    "card": "#FFFFFF",
    "text": "#2D3748",
    "text_light": "#718096",
    "border": "#E2E8F0",
    "grid": "#EDF2F7",
    "accent": "#4299E1",
    "warning": "#F56565",
    "success": "#48BB78",
}

# Professional matplotlib style
plt.style.use("default")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "600",
        "axes.titlepad": 15,
        "axes.labelsize": 10,
        "axes.labelweight": "500",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["card"],
        "axes.edgecolor": COLORS["border"],
        "axes.linewidth": 1.5,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "grid.alpha": 0.5,
    }
)

# ============================================================================
# LOAD DATA
# ============================================================================


def get_predictions_from_models(data):
    """Get predictions from all trained models"""
    predictions = {}
    test_mask = data.test_mask
    y_test_true = data.y[test_mask].numpy()

    # Logistic Regression
    try:
        from sklearn.linear_model import LogisticRegression

        X_train = data.x[data.train_mask].numpy()
        y_train = data.y[data.train_mask].numpy()
        X_test = data.x[test_mask].numpy()

        model = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train)

        predictions["Logistic Regression"] = {
            "y_true": y_test_true,
            "y_pred": model.predict(X_test),
            "y_proba": model.predict_proba(X_test)[:, 1],
        }
        print("Loaded Logistic Regression")
    except Exception as e:
        print(f"Logistic Regression failed: {e}")

    # XGBoost
    try:
        import xgboost as xgb

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False)

        predictions["XGBoost"] = {
            "y_true": y_test_true,
            "y_pred": model.predict(X_test),
            "y_proba": model.predict_proba(X_test)[:, 1],
        }
        print("Loaded XGBoost")
    except Exception as e:
        print(f"XGBoost failed: {e}")

    # Simple GCN
    try:
        import torch.nn as nn
        import torch.nn.functional as F
        from torch_geometric.nn import GCNConv

        class GCN(nn.Module):
            def __init__(self, num_features, hidden_dim=64, num_classes=2, dropout=0.5):
                super(GCN, self).__init__()
                self.conv1 = GCNConv(num_features, hidden_dim)
                self.conv2 = GCNConv(hidden_dim, num_classes)
                self.dropout = dropout

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.conv2(x, edge_index)
                return F.log_softmax(x, dim=1)

        device = torch.device("cpu")
        model = GCN(num_features=data.num_node_features).to(device)

        gcn_path = Path("outputs/models/gcn_best.pt")
        if gcn_path.exists():
            checkpoint = torch.load(gcn_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                proba = torch.exp(out)[:, 1]

            predictions["Simple GCN"] = {
                "y_true": y_test_true,
                "y_pred": pred[test_mask].cpu().numpy(),
                "y_proba": proba[test_mask].cpu().numpy(),
            }
            print("Loaded Simple GCN")
    except Exception as e:
        print(f"Simple GCN failed: {e}")

    # GraphSAGE
    try:
        from torch_geometric.nn import SAGEConv

        class GraphSAGE(nn.Module):
            def __init__(
                self,
                num_features,
                hidden_dim1=128,
                hidden_dim2=64,
                num_classes=2,
                dropout=0.5,
                aggr="mean",
            ):
                super(GraphSAGE, self).__init__()
                self.conv1 = SAGEConv(num_features, hidden_dim1, aggr=aggr)
                self.conv2 = SAGEConv(hidden_dim1, hidden_dim2, aggr=aggr)
                self.lin = nn.Linear(hidden_dim2, num_classes)
                self.dropout = dropout

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.conv2(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.lin(x)
                return F.log_softmax(x, dim=1)

        model = GraphSAGE(num_features=data.num_node_features).to(device)

        sage_path = Path("outputs/models/graphsage_best.pt")
        if sage_path.exists():
            checkpoint = torch.load(sage_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred = out.argmax(dim=1)
                proba = torch.exp(out)[:, 1]

            predictions["GraphSAGE"] = {
                "y_true": y_test_true,
                "y_pred": pred[test_mask].cpu().numpy(),
                "y_proba": proba[test_mask].cpu().numpy(),
            }
            print("Loaded GraphSAGE")
    except Exception as e:
        print(f"GraphSAGE failed: {e}")

    return predictions


# Load data
print("=" * 80)
print("LOADING MODEL RESULTS")
print("=" * 80)
data_path = Path("data/processed/fraud_graph.pt")
data = torch.load(data_path)
predictions = get_predictions_from_models(data)

# Compute metrics
models = {}
for model_name, pred_data in predictions.items():
    y_true = pred_data["y_true"]
    y_pred = pred_data["y_pred"]
    y_proba = pred_data["y_proba"]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    models[model_name] = {
        "test_f1": report["1"]["f1-score"],
        "test_precision": report["1"]["precision"],
        "test_recall": report["1"]["recall"],
        "test_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_proba": y_proba,
        "train_f1": {
            "Logistic Regression": 0.7315,
            "XGBoost": 0.9790,
            "Simple GCN": 0.6270,
            "GraphSAGE": 0.9583,
        }.get(model_name, 0.5),
    }

model_names = list(models.keys())
print(f"\nProcessed {len(models)} models\n")

# ============================================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================================

fig1 = plt.figure(figsize=(20, 11), facecolor=COLORS["bg"])

# Main title with subtitle - LEFT ALIGNED
fig1.text(
    0.06,
    0.97,
    "Model Performance Dashboard",
    ha="left",
    fontsize=18,
    fontweight="600",
    color=COLORS["text"],
)
fig1.text(
    0.06,
    0.945,
    "Fraud Detection System Evaluation",
    ha="left",
    fontsize=11,
    fontweight="400",
    color=COLORS["text_light"],
)

gs = fig1.add_gridspec(
    3, 3, hspace=0.55, wspace=0.35, left=0.06, right=0.96, top=0.86, bottom=0.06
)

# ---------- 1. F1 Score Comparison ----------
ax1 = fig1.add_subplot(gs[0, :2])
ax1.set_facecolor(COLORS["card"])

test_f1_scores = [models[m]["test_f1"] for m in model_names]
x_pos = np.arange(len(model_names))

bars = ax1.bar(
    x_pos,
    test_f1_scores,
    width=0.6,
    color=[MODEL_COLORS[m]["primary"] for m in model_names],
    edgecolor=[MODEL_COLORS[m]["dark"] for m in model_names],
    linewidth=2,
    alpha=0.9,
)

for bar, score, name in zip(bars, test_f1_scores, model_names):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.005,
        f"{score:.3f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
        color=MODEL_COLORS[name]["dark"],
    )

ax1.set_ylabel("F1 Score", fontweight="500")
ax1.set_title("Test F1 Score by Model", fontweight="600", pad=12, loc="left")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_names, fontsize=10)
ax1.set_ylim(0, max(test_f1_scores) * 1.18)
ax1.grid(axis="y", alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ---------- 2. Key Metrics Summary ----------
ax2 = fig1.add_subplot(gs[0, 2])
ax2.set_facecolor("#F7FAFC")  # Subtle background instead of border
ax2.axis("off")

best_model = model_names[test_f1_scores.index(max(test_f1_scores))]
best_f1 = max(test_f1_scores)

summary_y = 0.88
ax2.text(
    0.5,
    summary_y,
    "Top Performer",
    ha="center",
    fontsize=12,
    fontweight="600",
    color=COLORS["success"],
)
summary_y -= 0.18
ax2.text(
    0.5,
    summary_y,
    best_model,
    ha="center",
    fontsize=12,
    fontweight="600",
    color=MODEL_COLORS[best_model]["primary"],
)
summary_y -= 0.15
ax2.text(
    0.5,
    summary_y,
    f"F1: {best_f1:.3f}",
    ha="center",
    fontsize=11,
    fontweight="500",
    color=COLORS["text"],
)

summary_y -= 0.22
ax2.text(
    0.5,
    summary_y,
    "Metrics",
    ha="center",
    fontsize=11,
    fontweight="600",
    color=COLORS["text"],
)
summary_y -= 0.16
metrics_text = f"Precision: {models[best_model]['test_precision']:.3f}\n"
metrics_text += f"Recall: {models[best_model]['test_recall']:.3f}\n"
metrics_text += f"AUC: {models[best_model]['test_auc']:.3f}"
ax2.text(
    0.5,
    summary_y,
    metrics_text,
    ha="center",
    va="top",
    fontsize=10,
    color=COLORS["text_light"],
    linespacing=2.0,
)

# ---------- 3. Train vs Test Comparison ----------
ax3 = fig1.add_subplot(gs[1, :])
ax3.set_facecolor(COLORS["card"])

train_scores = [models[m]["train_f1"] for m in model_names]
x = np.arange(len(model_names))
width = 0.35

for i, name in enumerate(model_names):
    # Train bar
    ax3.bar(
        i - width / 2,
        train_scores[i],
        width * 0.9,
        color=MODEL_COLORS[name]["light"],
        edgecolor=MODEL_COLORS[name]["primary"],
        linewidth=1.5,
        label="Train" if i == 0 else "",
        alpha=0.8,
    )

    # Test bar
    ax3.bar(
        i + width / 2,
        test_f1_scores[i],
        width * 0.9,
        color=MODEL_COLORS[name]["primary"],
        edgecolor=MODEL_COLORS[name]["dark"],
        linewidth=1.5,
        label="Test" if i == 0 else "",
        alpha=0.9,
    )

    # Drift indicator - IMPROVED POSITIONING
    drift_pct = ((train_scores[i] - test_f1_scores[i]) / train_scores[i]) * 100

    if drift_pct > 5:  # Only show significant drift
        # Position label above the train bar to avoid overlap
        label_y = train_scores[i] + (max(train_scores) * 0.06)

        # Draw arrow
        ax3.annotate(
            "",
            xy=(i + width / 2, test_f1_scores[i] + 0.01),
            xytext=(i - width / 2, train_scores[i] - 0.01),
            arrowprops=dict(
                arrowstyle="->",
                lw=2.5,
                color=COLORS["warning"],
                alpha=0.7,
                shrinkA=0,
                shrinkB=0,
            ),
        )

        # Position label above train bar
        ax3.text(
            i,
            label_y,
            f"−{drift_pct:.0f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="700",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=COLORS["warning"],
                edgecolor=COLORS["warning"],
                linewidth=1.5,
                alpha=0.95,
            ),
        )

ax3.set_ylabel("F1 Score", fontweight="500")
ax3.set_title(
    "Generalization Analysis: Train vs Test Performance",
    fontweight="600",
    pad=12,
    loc="left",
)
ax3.set_xticks(x)
ax3.set_xticklabels(model_names, fontsize=10)
ax3.set_ylim(0, max(train_scores) * 1.12)
ax3.legend(loc="upper right", framealpha=0.95, fontsize=10)
ax3.grid(axis="y", alpha=0.3)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ---------- 4. ROC Curves ----------
ax4 = fig1.add_subplot(gs[2, 0])
ax4.set_facecolor(COLORS["card"])

for name in model_names:
    fpr, tpr, _ = roc_curve(models[name]["y_true"], models[name]["y_proba"])
    auc = models[name]["test_auc"]
    ax4.plot(
        fpr,
        tpr,
        linewidth=2.5,
        color=MODEL_COLORS[name]["primary"],
        label=f"{name} (AUC={auc:.3f})",
        alpha=0.9,
    )

ax4.plot(
    [0, 1],
    [0, 1],
    "--",
    linewidth=1.5,
    color=COLORS["text_light"],
    alpha=0.5,
    label="Random",
)

ax4.set_xlabel("False Positive Rate", fontweight="500")
ax4.set_ylabel("True Positive Rate", fontweight="500")
ax4.set_title("ROC Curves", fontweight="600", pad=12, loc="left")
ax4.legend(loc="lower right", fontsize=9, framealpha=0.95)
ax4.grid(alpha=0.3)
ax4.set_xlim(-0.02, 1.02)
ax4.set_ylim(-0.02, 1.02)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# ---------- 5. Precision-Recall Curves ----------
ax5 = fig1.add_subplot(gs[2, 1])
ax5.set_facecolor(COLORS["card"])

for name in model_names:
    precision, recall, _ = precision_recall_curve(
        models[name]["y_true"], models[name]["y_proba"]
    )
    ax5.plot(
        recall,
        precision,
        linewidth=2.5,
        color=MODEL_COLORS[name]["primary"],
        label=name,
        alpha=0.9,
    )

baseline = (models[model_names[0]]["y_true"] == 1).mean()
ax5.axhline(
    baseline,
    color=COLORS["warning"],
    linestyle="--",
    linewidth=1.5,
    alpha=0.6,
    label=f"Baseline ({baseline:.3f})",
)

ax5.set_xlabel("Recall", fontweight="500")
ax5.set_ylabel("Precision", fontweight="500")
ax5.set_title("Precision-Recall Curves", fontweight="600", pad=12, loc="left")
ax5.legend(loc="upper right", fontsize=9, framealpha=0.95)
ax5.grid(alpha=0.3)
ax5.spines["top"].set_visible(False)
ax5.spines["right"].set_visible(False)

# ---------- 6. Recommendations ----------
ax6 = fig1.add_subplot(gs[2, 2])
ax6.set_facecolor("#F7FAFC")  # Subtle background
ax6.axis("off")

rec_y = 0.88
ax6.text(
    0.5,
    rec_y,
    "Recommendations",
    ha="center",
    fontsize=13,
    fontweight="600",
    color=COLORS["accent"],
)

rec_y -= 0.18

# Section 1: Deploy
ax6.text(
    0.5,
    rec_y,
    "Deploy:",
    ha="center",
    fontsize=10,
    fontweight="700",
    color=COLORS["text"],
)
rec_y -= 0.13
ax6.text(
    0.5,
    rec_y,
    best_model,
    ha="center",
    fontsize=11,
    fontweight="600",
    color=MODEL_COLORS[best_model]["primary"],
)
rec_y -= 0.11
ax6.text(
    0.5,
    rec_y,
    f"Best F1 ({best_f1:.3f})",
    ha="center",
    fontsize=9,
    fontweight="400",
    color=COLORS["text_light"],
)

rec_y -= 0.16

# Section 2: Monitor
ax6.text(
    0.5,
    rec_y,
    "Monitor:",
    ha="center",
    fontsize=10,
    fontweight="700",
    color=COLORS["text"],
)
rec_y -= 0.13
ax6.text(
    0.5,
    rec_y,
    "Temporal Drift",
    ha="center",
    fontsize=10,
    fontweight="500",
    color=COLORS["warning"],
)
rec_y -= 0.11
ax6.text(
    0.5,
    rec_y,
    "Retrain Weekly",
    ha="center",
    fontsize=9,
    fontweight="400",
    color=COLORS["text_light"],
)

rec_y -= 0.16

# Section 3: Focus
ax6.text(
    0.5,
    rec_y,
    "Focus:",
    ha="center",
    fontsize=10,
    fontweight="700",
    color=COLORS["text"],
)
rec_y -= 0.13
ax6.text(
    0.5,
    rec_y,
    "High Recall",
    ha="center",
    fontsize=10,
    fontweight="500",
    color=COLORS["success"],
)

output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(
    output_dir / "page1_executive_summary.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=COLORS["bg"],
)
print("Saved Page 1: Executive Summary")

# ============================================================================
# PAGE 2: CONFUSION MATRICES
# ============================================================================

fig2 = plt.figure(figsize=(20, 11), facecolor=COLORS["bg"])

# LEFT ALIGNED TITLES
fig2.text(
    0.08,
    0.97,
    "Confusion Matrix Analysis",
    ha="left",
    fontsize=18,
    fontweight="600",
    color=COLORS["text"],
)
fig2.text(
    0.08,
    0.945,
    "Classification Performance Breakdown",
    ha="left",
    fontsize=11,
    fontweight="400",
    color=COLORS["text_light"],
)

gs2 = fig2.add_gridspec(
    2, 2, hspace=0.45, wspace=0.40, left=0.08, right=0.88, top=0.89, bottom=0.08
)

for idx, name in enumerate(model_names):
    ax = fig2.add_subplot(gs2[idx // 2, idx % 2])
    ax.set_facecolor(COLORS["card"])

    cm = models[name]["confusion_matrix"]

    # Custom colormap
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "custom",
        ["#FFFFFF", MODEL_COLORS[name]["light"], MODEL_COLORS[name]["primary"]],
        N=256,
    )

    sns.heatmap(
        cm,
        annot=False,
        cmap=cmap,
        square=True,
        ax=ax,
        cbar=True,
        linewidths=4,
        linecolor="white",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        cbar_kws={"label": "Count", "shrink": 0.75},
    )

    # Add values with proper contrast
    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            color = "white" if value > cm.max() * 0.6 else MODEL_COLORS[name]["dark"]
            ax.text(
                j + 0.5,
                i + 0.5,
                f"{value:,}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="600",
                color=color,
            )

    ax.set_title(
        name, fontweight="600", fontsize=14, color=MODEL_COLORS[name]["primary"], pad=15
    )
    ax.set_xlabel("Predicted Label", fontweight="500", fontsize=11)
    ax.set_ylabel("True Label", fontweight="500", fontsize=11)

    # METRICS ON LEFT SIDE (VERTICAL LAYOUT)
    f1 = models[name]["test_f1"]
    prec = models[name]["test_precision"]
    rec = models[name]["test_recall"]

    metrics_y_positions = [0.75, 0.50, 0.25]
    metrics_labels = [f"F1: {f1:.3f}", f"Precision: {prec:.3f}", f"Recall: {rec:.3f}"]

    for y_pos, label in zip(metrics_y_positions, metrics_labels):
        ax.text(
            -0.35,
            y_pos,
            label,
            ha="right",
            va="center",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="500",
            color=MODEL_COLORS[name]["dark"],
            rotation=0,
        )

plt.savefig(
    output_dir / "page2_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=COLORS["bg"],
)
print("Saved Page 2: Confusion Matrices")

# ============================================================================
# PAGE 3: DETAILED METRICS
# ============================================================================

fig3 = plt.figure(figsize=(20, 11), facecolor=COLORS["bg"])

# LEFT ALIGNED TITLES
fig3.text(
    0.06,
    0.97,
    "Detailed Performance Metrics",
    ha="left",
    fontsize=18,
    fontweight="600",
    color=COLORS["text"],
)
fig3.text(
    0.06,
    0.945,
    "Comprehensive Model Evaluation",
    ha="left",
    fontsize=11,
    fontweight="400",
    color=COLORS["text_light"],
)

gs3 = fig3.add_gridspec(
    2, 3, hspace=0.45, wspace=0.35, left=0.11, right=0.96, top=0.89, bottom=0.08
)

# ---------- Precision Comparison ----------
ax1 = fig3.add_subplot(gs3[0, 0])
ax1.set_facecolor(COLORS["card"])

precisions = [models[m]["test_precision"] for m in model_names]
y_pos = np.arange(len(model_names))

bars = ax1.barh(
    y_pos,
    precisions,
    height=0.6,
    color=[MODEL_COLORS[m]["primary"] for m in model_names],
    edgecolor=[MODEL_COLORS[m]["dark"] for m in model_names],
    linewidth=2,
    alpha=0.9,
)

for bar, prec in zip(bars, precisions):
    width = bar.get_width()
    ax1.text(
        width + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{prec:.3f}",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="600",
    )

ax1.set_xlabel("Precision Score", fontweight="500")
ax1.set_title("Precision by Model", fontweight="600", pad=12, loc="left")
ax1.set_yticks(y_pos)
ax1.set_yticklabels(model_names, fontsize=10)
ax1.set_xlim(0, max(precisions) * 1.2)
ax1.grid(axis="x", alpha=0.3)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# ---------- Recall Comparison ----------
ax2 = fig3.add_subplot(gs3[0, 1])
ax2.set_facecolor(COLORS["card"])

recalls = [models[m]["test_recall"] for m in model_names]

bars = ax2.barh(
    y_pos,
    recalls,
    height=0.6,
    color=[MODEL_COLORS[m]["primary"] for m in model_names],
    edgecolor=[MODEL_COLORS[m]["dark"] for m in model_names],
    linewidth=2,
    alpha=0.9,
)

for bar, rec in zip(bars, recalls):
    width = bar.get_width()
    ax2.text(
        width + 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{rec:.3f}",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="600",
    )

ax2.set_xlabel("Recall Score", fontweight="500")
ax2.set_title("Recall by Model", fontweight="600", pad=12, loc="left")
ax2.set_yticks(y_pos)
ax2.set_yticklabels(model_names, fontsize=10)
ax2.set_xlim(0, max(recalls) * 1.35)
ax2.grid(axis="x", alpha=0.3)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# ---------- AUC Comparison ----------
ax3 = fig3.add_subplot(gs3[0, 2])
ax3.set_facecolor(COLORS["card"])

aucs = [models[m]["test_auc"] for m in model_names]

bars = ax3.barh(
    y_pos,
    aucs,
    height=0.6,
    color=[MODEL_COLORS[m]["primary"] for m in model_names],
    edgecolor=[MODEL_COLORS[m]["dark"] for m in model_names],
    linewidth=2,
    alpha=0.9,
)

for bar, auc in zip(bars, aucs):
    width = bar.get_width()
    ax3.text(
        width + 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{auc:.3f}",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="600",
    )

ax3.set_xlabel("AUC Score", fontweight="500")
ax3.set_title("AUC by Model", fontweight="600", pad=12, loc="left")
ax3.set_yticks(y_pos)
ax3.set_yticklabels(model_names, fontsize=10)
ax3.set_xlim(0, 1.15)
ax3.grid(axis="x", alpha=0.3)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

# ---------- Error Analysis ----------
ax4 = fig3.add_subplot(gs3[1, :2])
ax4.set_facecolor(COLORS["card"])

fp_rates = []
fn_rates = []

for name in model_names:
    cm = models[name]["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
    fp_rates.append(fp_rate)
    fn_rates.append(fn_rate)

x = np.arange(len(model_names))
width = 0.35

bars1 = ax4.bar(
    x - width / 2,
    fp_rates,
    width * 0.9,
    label="False Positive Rate",
    color="#F59E0B",
    alpha=0.85,
    edgecolor="#D97706",
    linewidth=2,
)

bars2 = ax4.bar(
    x + width / 2,
    fn_rates,
    width * 0.9,
    label="False Negative Rate",
    color="#EF4444",
    alpha=0.85,
    edgecolor="#DC2626",
    linewidth=2,
)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0.03:
            ax4.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="600",
            )

ax4.set_ylabel("Error Rate", fontweight="500")
ax4.set_title(
    "Error Rate Analysis: False Positives vs False Negatives",
    fontweight="600",
    pad=12,
    loc="left",
)
ax4.set_xticks(x)
ax4.set_xticklabels(model_names, fontsize=10)
ax4.legend(loc="upper right", fontsize=10, framealpha=0.95)
ax4.grid(axis="y", alpha=0.3)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)

# ---------- Performance Summary Table ----------
ax5 = fig3.add_subplot(gs3[1, 2])
ax5.set_facecolor("#F7FAFC")  # Subtle background
ax5.axis("off")

summary_y = 0.90
ax5.text(
    0.5,
    summary_y,
    "Model Comparison",
    ha="center",
    fontsize=12,
    fontweight="600",
    color=COLORS["text"],
)

summary_y -= 0.13
table_header = f"{'Model':<16} {'F1':>6} {'AUC':>6}"
ax5.text(
    0.5,
    summary_y,
    table_header,
    ha="center",
    va="top",
    fontsize=9,
    fontweight="600",
    color=COLORS["text"],
    family="monospace",
)

summary_y -= 0.09
ax5.text(
    0.5,
    summary_y,
    "─" * 32,
    ha="center",
    va="top",
    fontsize=9,
    color=COLORS["text_light"],
    family="monospace",
)

summary_y -= 0.09
for name in sorted(model_names, key=lambda m: models[m]["test_f1"], reverse=True):
    short_name = name[:15].ljust(16)
    f1 = models[name]["test_f1"]
    auc = models[name]["test_auc"]
    row = f"{short_name} {f1:>6.3f} {auc:>6.3f}"
    ax5.text(
        0.5,
        summary_y,
        row,
        ha="center",
        va="top",
        fontsize=9,
        fontweight="400",
        color=MODEL_COLORS[name]["primary"],
        family="monospace",
    )
    summary_y -= 0.09

summary_y -= 0.06
ax5.text(
    0.5,
    summary_y,
    "─" * 32,
    ha="center",
    va="top",
    fontsize=9,
    color=COLORS["text_light"],
    family="monospace",
)

summary_y -= 0.14
best_f1 = max(model_names, key=lambda m: models[m]["test_f1"])
ax5.text(
    0.5,
    summary_y,
    f"Best Overall:",
    ha="center",
    fontsize=10,
    fontweight="600",
    color=COLORS["success"],
)

summary_y -= 0.10
ax5.text(
    0.5,
    summary_y,
    best_f1,
    ha="center",
    fontsize=9,
    fontweight="600",
    color=MODEL_COLORS[best_f1]["primary"],
)

plt.savefig(
    output_dir / "page3_detailed_metrics.png",
    dpi=300,
    bbox_inches="tight",
    facecolor=COLORS["bg"],
)
print("Saved Page 3: Detailed Metrics")

print("\n" + "=" * 80)
print("PROFESSIONAL DASHBOARD COMPLETE")
print("=" * 80)
print(f"Output: {output_dir.absolute()}")
print(f"\nGenerated Files:")
print(f"  1. page1_executive_summary.png  - Overview & key insights")
print(f"  2. page2_confusion_matrices.png - Classification breakdown")
print(f"  3. page3_detailed_metrics.png   - Comprehensive analysis")
print("=" * 80)

plt.show()
