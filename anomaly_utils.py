"""Helper functions for ASGN2703: Isolation Forest anomaly detection on industrial sensor data."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import cross_val_score, StratifiedKFold

RANDOM_STATE = 42
SENSORS = ["Temperature", "Pressure", "Vibration", "Humidity", "FlowRate", "PowerConsumption", "RuntimeHours"]
FEATURES = SENSORS + ["DaysSinceService"]
BLUE, RED = "#4C72B0", "#c44e52"


# ----------------------------------------------------------------- preprocessing
def impute_missing(df):
    """Fill missing values; the FaultFlag label is deliberately left untouched."""
    df = df.copy()
    known = df["SensorID"].notna()
    assert (df.loc[known, "SensorID"] == df.index[known] + 1).all(), "SensorID is not sequential"
    df["SensorID"] = df.index + 1                                        # rebuild IDs from row position

    df["LastServiceDate"] = pd.to_datetime(df["LastServiceDate"], format="%d-%m-%Y", errors="coerce")
    assert df["LastServiceDate"].dropna().is_monotonic_increasing, "dates are not ordered"
    df["LastServiceDate"] = df["LastServiceDate"].ffill().bfill()        # rows are date-ordered

    df[SENSORS] = df[SENSORS].fillna(df[SENSORS].median())               # median is robust to outliers
    df["has_label"] = df["FaultFlag"].notna()
    return df


def add_service_age(df):
    """Convert LastServiceDate into DaysSinceService (relative to the latest date)."""
    df = df.copy()
    df["DaysSinceService"] = (df["LastServiceDate"].max() - df["LastServiceDate"]).dt.days
    return df


def count_iqr_outliers(series):
    q1, q3 = series.quantile([.25, .75])
    iqr = q3 - q1
    return int(((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum())


def winsorise(df, columns, lower=0.005, upper=0.995):
    """Clip each column to its [lower, upper] percentile range. Returns (clipped df, #values clipped)."""
    df, n_clipped = df.copy(), 0
    for col in columns:
        lo, hi = df[col].quantile([lower, upper])
        n_clipped += int(((df[col] < lo) | (df[col] > hi)).sum())
        df[col] = df[col].clip(lo, hi)
    return df, n_clipped


def standardise(df, columns):
    """Z-score the given columns. Returns (numpy array, DataFrame)."""
    values = StandardScaler().fit_transform(df[columns])
    return values, pd.DataFrame(values, columns=columns, index=df.index)


# ------------------------------------------------------------------------- model
def fit_isolation_forest(X, contamination=0.05, n_estimators=300):
    return IsolationForest(n_estimators=n_estimators, contamination=contamination,
                           random_state=RANDOM_STATE, n_jobs=-1).fit(X)


def predict_anomalies(model, X):
    """Return (binary flag 1 = anomaly, anomaly score where higher = more anomalous)."""
    return (model.predict(X) == -1).astype(int), -model.score_samples(X)


# --------------------------------------------------------------------- evaluation
def evaluate(y_true, y_pred, y_score):
    return dict(precision=precision_score(y_true, y_pred), recall=recall_score(y_true, y_pred),
                f1=f1_score(y_true, y_pred), roc_auc=roc_auc_score(y_true, y_score))


def contamination_sweep(X, label_mask, y_true, levels):
    """Re-fit the model at several contamination levels and score each against the labels."""
    rows = []
    for level in levels:
        model = fit_isolation_forest(X, contamination=min(level, 0.5))
        pred = predict_anomalies(model, X)[0][label_mask]
        rows.append(dict(contamination=round(level, 3), precision=precision_score(y_true, pred),
                         recall=recall_score(y_true, pred), f1=f1_score(y_true, pred), flagged=int(pred.sum())))
    return pd.DataFrame(rows).round(3)


def supervised_baseline_auc(X, y):
    """5-fold CV ROC-AUC of a Random Forest - shows whether the label is learnable at all."""
    scores = cross_val_score(RandomForestClassifier(300, random_state=RANDOM_STATE, n_jobs=-1), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


# ----------------------------------------------------------------------- plotting
def _save(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_missing(missing_pct, path):
    ax = missing_pct.plot.bar(figsize=(8, 3.5), color=BLUE)
    ax.set_ylabel("% missing"); ax.set_title("Missing values per column")
    _save(path)


def plot_distributions(df, path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, col in zip(axes.ravel(), FEATURES):
        sns.histplot(df[col], bins=40, kde=True, ax=ax, color=BLUE); ax.set_title(col)
    fig.suptitle("Feature distributions (after cleaning)", fontsize=14)
    _save(path)


def plot_correlation(corr, path):
    plt.figure(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True)
    plt.title("Correlation matrix (incl. FaultFlag)")
    _save(path)


def plot_evaluation(y_true, y_pred, y_score, auc, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=["Normal", "Fault"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Confusion matrix (contamination = 5%)")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    axes[1].plot(fpr, tpr, lw=2, label=f"Isolation Forest (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
    axes[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    axes[1].legend()
    _save(path)


def plot_timeseries(df, path, columns=("Temperature", "Vibration", "PowerConsumption", "Pressure")):
    ts = df.sort_values("SensorID")
    fig, axes = plt.subplots(len(columns), 1, figsize=(14, 12), sharex=True)
    for ax, col in zip(axes, columns):
        ax.plot(ts["SensorID"], ts[col], color="#9aa5b1", lw=0.5, label="raw reading")
        ax.plot(ts["SensorID"], ts[col].rolling(25, center=True).median(), color="#1f4e79", lw=1.5,
                label="rolling median (25)")
        flagged = ts[ts["iso_pred"] == 1]
        ax.scatter(flagged["SensorID"], flagged[col], color="red", s=14, zorder=3, label="anomaly")
        ax.set_ylabel(col)
    axes[0].legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("SensorID (record order / time)")
    fig.suptitle("Sensor time-series with Isolation Forest anomalies", fontsize=14)
    _save(path)


def plot_anomaly_rate(df, path, window=250):
    ts = df.sort_values("SensorID")
    rate = ts.groupby((ts["SensorID"] - 1) // window)["iso_pred"].mean()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(rate.index * window + window / 2, rate * 100, width=window * 0.88, color=RED)
    ax.axhline(5, color="k", ls="--", label="expected 5%")
    ax.set(xlabel="SensorID", ylabel="% flagged anomalous", title=f"Anomaly rate per {window}-record window")
    ax.legend()
    _save(path)
    return rate


def plot_scatter_pairs(df, path, pairs=(("Temperature", "Vibration"), ("Pressure", "PowerConsumption"),
                                         ("FlowRate", "RuntimeHours"))):
    normal, anomal = df[df.iso_pred == 0], df[df.iso_pred == 1]
    fig, axes = plt.subplots(1, len(pairs), figsize=(17, 5))
    for ax, (x, y) in zip(axes, pairs):
        ax.scatter(normal[x], normal[y], s=8, alpha=.35, color=BLUE, label="normal")
        ax.scatter(anomal[x], anomal[y], s=22, color="red", edgecolor="k", linewidth=.3, label="anomaly")
        ax.set(xlabel=x, ylabel=y, title=f"{x} vs {y}")
    axes[0].legend()
    _save(path)


def plot_pca(X, df, path):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Z = pca.fit_transform(X)
    flagged = df["iso_pred"].values == 1
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    sc = axes[0].scatter(Z[:, 0], Z[:, 1], c=df["iso_score"], cmap="viridis", s=10)
    plt.colorbar(sc, ax=axes[0], label="anomaly score (higher = more anomalous)")
    axes[0].set_title("PCA projection - coloured by anomaly score")
    axes[1].scatter(Z[~flagged, 0], Z[~flagged, 1], s=8, alpha=.35, color=BLUE, label="normal")
    axes[1].scatter(Z[flagged, 0], Z[flagged, 1], s=22, color="red", edgecolor="k", linewidth=.3, label="anomaly")
    axes[1].set_title("PCA projection - normal vs anomalous"); axes[1].legend()
    for ax in axes:
        ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    _save(path)


def plot_scores(df, labelled, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.histplot(df["iso_score"], bins=50, ax=axes[0], color=BLUE)
    axes[0].axvline(df.loc[df.iso_pred == 1, "iso_score"].min(), color="red", ls="--", label="anomaly threshold")
    axes[0].legend(); axes[0].set_title("Isolation Forest anomaly score distribution")
    sns.kdeplot(data=labelled, x="iso_score", hue=labelled["FaultFlag"].map({0: "Normal", 1: "Fault"}),
                ax=axes[1], common_norm=False, fill=True)
    axes[1].set_title("Anomaly score by labelled FaultFlag")
    _save(path)


def feature_profile(X_df, flags):
    """Mean |z| per feature for anomalies vs normal rows (signed z-scores cancel: anomalies sit at both extremes)."""
    absz = X_df.abs()
    prof = pd.DataFrame({"anomalies": absz[flags == 1].mean(), "normal": absz[flags == 0].mean()})
    prof["ratio"] = prof["anomalies"] / prof["normal"]
    return prof.sort_values("ratio")


def plot_feature_profile(prof, path):
    y, h = np.arange(len(prof)), 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(y - h / 2, prof["normal"], h, color=BLUE, label="normal")
    ax.barh(y + h / 2, prof["anomalies"], h, color=RED, label="anomaly")
    ax.set_yticks(y); ax.set_yticklabels(prof.index)
    ax.set(xlabel="mean |z-score| (distance from fleet average)",
           title="How far from 'typical' are flagged readings, per feature?")
    ax.legend()
    _save(path)
