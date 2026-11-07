"""
Water Quality and Potability Prediction
Phase 2 (continued): Exploratory Data Analysis (EDA) plots

Run from the terminal (with (venv) active):
    python eda_plots.py

All charts are saved as PNG files in the 'figures' folder.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save images without opening pop-up windows
import matplotlib.pyplot as plt
import seaborn as sns

# ----- Project settings -----
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "water_potability.csv"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)   # create the folder if it does not exist

TARGET = "Potability"
FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]

sns.set_theme(style="whitegrid")


def save_figure(fig, file_name):
    """Tidy the layout, save the figure as a PNG and close it."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / file_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: figures/{file_name}")


def plot_class_distribution(df):
    """Bar chart showing how many samples belong to each class."""
    counts = df[TARGET].value_counts().sort_index()
    total = len(df)
    labels = ["Not Potable (0)", "Potable (1)"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=["#d9534f", "#5cb85c"])
    for bar, value in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value}\n({value / total * 100:.1f}%)",
            ha="center", va="bottom", fontsize=11,
        )
    ax.set_title("Target Class Distribution (Potability)", fontsize=14)
    ax.set_ylabel("Number of samples")
    ax.set_ylim(0, counts.max() * 1.15)
    save_figure(fig, "01_class_distribution.png")


def plot_missing_values(df):
    """Horizontal bar chart of missing values per column."""
    missing_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(missing_pct.index, missing_pct.values, color="#f0ad4e")
    for i, value in enumerate(missing_pct.values):
        ax.text(value + 0.3, i, f"{value:.1f}%", va="center")
    ax.set_title("Percentage of Missing Values per Column", fontsize=14)
    ax.set_xlabel("Missing values (%)")
    ax.set_xlim(0, missing_pct.max() * 1.15)
    save_figure(fig, "02_missing_values.png")


def plot_feature_distributions(df):
    """3x3 grid of histograms, one per feature, split by class."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for ax, col in zip(axes.flatten(), FEATURES):
        sns.histplot(
            data=df, x=col, hue=TARGET, kde=True,
            stat="density", common_norm=False, element="step", ax=ax,
        )
        ax.set_title(col)
    fig.suptitle("Feature Distributions by Potability Class", fontsize=16)
    save_figure(fig, "03_feature_distributions.png")


def plot_boxplots(df):
    """3x3 grid of boxplots to spot outliers in each feature."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, col in zip(axes.flatten(), FEATURES):
        sns.boxplot(x=df[col].dropna(), ax=ax, color="#5bc0de")
        ax.set_title(col)
    fig.suptitle("Boxplots for Outlier Detection", fontsize=16)
    save_figure(fig, "04_boxplots.png")


def plot_correlation_heatmap(df):
    """Heatmap of correlations between all features and the target."""
    corr = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title("Correlation Heatmap (features and Potability)", fontsize=14)
    save_figure(fig, "05_correlation_heatmap.png")

    # Print correlations with the target, strongest first
    print("\nCorrelation of each feature with Potability:")
    target_corr = corr[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
    print(target_corr.round(3).to_string())


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows and {df.shape[1]} columns.\n")

    plot_class_distribution(df)
    plot_missing_values(df)
    plot_feature_distributions(df)
    plot_boxplots(df)
    plot_correlation_heatmap(df)

    print("\nAll EDA plots saved in the 'figures' folder.")


if __name__ == "__main__":
    main()