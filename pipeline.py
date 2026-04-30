"""
Data preprocessing pipeline for the NYC Airbnb Open Dataset (2019).

Runs end-to-end: load → inspect → clean → encode → scale → visualise → report.
All charts are saved to ./plots/.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend so no display is required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

warnings.filterwarnings("ignore")

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# DataLoader
# ──────────────────────────────────────────────────────────────────────────────

class DataLoader:
    """Loads the dataset and provides first-look diagnostics."""

    def __init__(self, filepath: str = "AB_NYC_2019.csv") -> None:
        """
        Args:
            filepath: Path to the raw CSV file.
        """
        self.filepath = filepath
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Read the CSV into a DataFrame and return it."""
        self.df = pd.read_csv(self.filepath)
        return self.df

    def inspect(self) -> None:
        """Print shape, dtypes, sample rows and per-column missing value counts."""
        if self.df is None:
            raise RuntimeError("Call load() before inspect().")
        df = self.df
        print("=" * 60)
        print("DATASET OVERVIEW")
        print("=" * 60)
        print(f"Shape         : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Memory usage  : {df.memory_usage(deep=True).sum() / 1_048_576:.2f} MB")
        print()
        print("── Data types ─────────────────────────────────────────────")
        print(df.dtypes.to_string())
        print()
        print("── Sample rows (3) ────────────────────────────────────────")
        print(df.sample(3, random_state=42).to_string())
        print()
        print("── Missing values per column ───────────────────────────────")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        summary = pd.DataFrame({"missing": missing, "pct": missing_pct})
        print(summary[summary["missing"] > 0].to_string())
        print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# DataCleaner
# ──────────────────────────────────────────────────────────────────────────────

class DataCleaner:
    """Cleans the DataFrame in-place, logging every transformation."""

    def __init__(self, df: pd.DataFrame) -> None:
        """
        Args:
            df: Raw DataFrame produced by DataLoader.
        """
        self.df = df.copy()
        self._report: dict = {
            "rows_before": len(df),
            "rows_after": None,
            "duplicates_removed": 0,
            "missing_filled": {},
            "missing_dropped_rows": 0,
            "outliers_capped": {},
            "columns_encoded": [],
            "columns_scaled": [],
        }

    # ── missing values ────────────────────────────────────────────────────────

    def handle_missing_values(self) -> "DataCleaner":
        """
        Fill or drop missing values depending on column type and missingness rate.

        Strategy per column:
        - name / host_name  : fill with 'Unknown' (categorical, low impact)
        - last_review       : fill with 'Unknown' (date string, not used in modelling)
        - reviews_per_month : fill with 0 (listing has no reviews yet)
        - Any column > 60%  : drop (not applicable here but included for robustness)
        Remaining numeric NaNs: fill with column median.
        """
        df = self.df
        before = df.isnull().sum().sum()

        # String columns
        for col in ["name", "host_name"]:
            if col in df.columns:
                n = df[col].isnull().sum()
                df[col].fillna("Unknown", inplace=True)
                self._report["missing_filled"][col] = n

        if "last_review" in df.columns:
            n = df["last_review"].isnull().sum()
            df["last_review"].fillna("Unknown", inplace=True)
            self._report["missing_filled"]["last_review"] = n

        if "reviews_per_month" in df.columns:
            n = df["reviews_per_month"].isnull().sum()
            df["reviews_per_month"].fillna(0, inplace=True)
            self._report["missing_filled"]["reviews_per_month"] = n

        # Drop columns that are more than 60 % missing
        thresh = int(0.4 * len(df))
        high_missing = [c for c in df.columns if df[c].isnull().sum() > len(df) - thresh]
        if high_missing:
            df.drop(columns=high_missing, inplace=True)

        # Remaining numeric NaNs → median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            n = df[col].isnull().sum()
            if n > 0:
                df[col].fillna(df[col].median(), inplace=True)
                self._report["missing_filled"][col] = n

        # Drop any rows that still have nulls
        rows_before_drop = len(df)
        df.dropna(inplace=True)
        self._report["missing_dropped_rows"] = rows_before_drop - len(df)

        after = df.isnull().sum().sum()
        print(f"[handle_missing_values] {before:,} → {after:,} missing cells")
        self.df = df
        return self

    # ── duplicates ────────────────────────────────────────────────────────────

    def remove_duplicates(self) -> "DataCleaner":
        """Detect and remove fully duplicate rows; logs count removed."""
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        removed = before - len(self.df)
        self._report["duplicates_removed"] = removed
        print(f"[remove_duplicates] {removed:,} duplicate rows removed")
        return self

    # ── outliers ──────────────────────────────────────────────────────────────

    def fix_outliers(self) -> "DataCleaner":
        """
        Cap outliers in *price* and *minimum_nights* using the IQR method.

        Values below Q1 - 1.5·IQR are raised to that floor.
        Values above Q3 + 1.5·IQR are lowered to that ceiling.
        Capping is preferred over dropping so no information is lost.
        """
        for col in ["price", "minimum_nights"]:
            if col not in self.df.columns:
                continue
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            n_low = (self.df[col] < lower).sum()
            n_high = (self.df[col] > upper).sum()
            self.df[col] = self.df[col].clip(lower=lower, upper=upper)
            self._report["outliers_capped"][col] = {
                "lower_bound": round(lower, 2),
                "upper_bound": round(upper, 2),
                "capped_low": int(n_low),
                "capped_high": int(n_high),
            }
            print(
                f"[fix_outliers] {col}: {n_low + n_high:,} values capped "
                f"(floor={lower:.2f}, ceiling={upper:.2f})"
            )
        return self

    # ── encoding ──────────────────────────────────────────────────────────────

    def encode_categoricals(self) -> "DataCleaner":
        """Label-encode *neighbourhood_group* and *room_type* in-place."""
        le = LabelEncoder()
        for col in ["neighbourhood_group", "room_type"]:
            if col in self.df.columns:
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self._report["columns_encoded"].append(col)
                print(f"[encode_categoricals] {col} encoded ({self.df[col].nunique()} classes)")
        return self

    # ── scaling ───────────────────────────────────────────────────────────────

    def scale_numerics(self) -> "DataCleaner":
        """Apply MinMaxScaler to *price*, *minimum_nights*, and *number_of_reviews*."""
        scaler = MinMaxScaler()
        cols = [c for c in ["price", "minimum_nights", "number_of_reviews"] if c in self.df.columns]
        self.df[cols] = scaler.fit_transform(self.df[cols])
        self._report["columns_scaled"] = cols
        print(f"[scale_numerics] MinMaxScaler applied to: {cols}")
        return self

    # ── finalise ──────────────────────────────────────────────────────────────

    def get_clean_df(self) -> pd.DataFrame:
        """Return the cleaned DataFrame."""
        self._report["rows_after"] = len(self.df)
        return self.df

    def get_report(self) -> dict:
        """Return the internal cleaning report dictionary."""
        self._report["rows_after"] = len(self.df)
        return self._report


# ──────────────────────────────────────────────────────────────────────────────
# DataVisualizer
# ──────────────────────────────────────────────────────────────────────────────

class DataVisualizer:
    """Generates and saves all diagnostic charts to PLOTS_DIR."""

    def __init__(self, raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
        """
        Args:
            raw_df  : DataFrame as loaded (before any cleaning).
            clean_df: DataFrame after the full cleaning pipeline.
        """
        self.raw = raw_df
        self.clean = clean_df
        sns.set_theme(style="whitegrid", palette="muted")

    # ── 1 ─────────────────────────────────────────────────────────────────────

    def missing_heatmap(self) -> None:
        """Seaborn heatmap of missing values across all columns before cleaning."""
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(
            self.raw.isnull(),
            yticklabels=False,
            cbar=False,
            cmap="viridis",
            ax=ax,
        )
        ax.set_title("Missing Value Map — Raw Data", fontsize=14, fontweight="bold")
        ax.set_xlabel("Columns")
        plt.tight_layout()
        path = PLOTS_DIR / "01_missing_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[DataVisualizer] Saved {path}")

    # ── 2 ─────────────────────────────────────────────────────────────────────

    def price_distribution_before_after(self) -> None:
        """Side-by-side histograms of price before and after outlier capping + scaling."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(
            self.raw["price"].dropna().clip(upper=1_000),
            bins=60,
            color="#4878d0",
            edgecolor="white",
            linewidth=0.4,
        )
        axes[0].set_title("Price Distribution — Before Cleaning", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Price (USD, clipped at $1,000 for display)")
        axes[0].set_ylabel("Count")

        axes[1].hist(
            self.clean["price"],
            bins=60,
            color="#6acc65",
            edgecolor="white",
            linewidth=0.4,
        )
        axes[1].set_title("Price Distribution — After Cleaning", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Price (MinMax scaled 0–1)")
        axes[1].set_ylabel("Count")

        plt.tight_layout()
        path = PLOTS_DIR / "02_price_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[DataVisualizer] Saved {path}")

    # ── 3 ─────────────────────────────────────────────────────────────────────

    def outlier_boxplots(self) -> None:
        """Boxplots for price and minimum_nights before and after IQR capping."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        for i, col in enumerate(["price", "minimum_nights"]):
            # before
            data_before = self.raw[col].dropna()
            axes[i][0].boxplot(data_before, vert=True, patch_artist=True,
                               boxprops=dict(facecolor="#4878d0", alpha=0.7),
                               medianprops=dict(color="red", linewidth=2))
            axes[i][0].set_title(f"{col} — Before", fontsize=11, fontweight="bold")
            axes[i][0].set_ylabel(col)

            # after (still scaled, so we show relative distribution)
            data_after = self.clean[col]
            axes[i][1].boxplot(data_after, vert=True, patch_artist=True,
                               boxprops=dict(facecolor="#6acc65", alpha=0.7),
                               medianprops=dict(color="red", linewidth=2))
            axes[i][1].set_title(f"{col} — After (scaled)", fontsize=11, fontweight="bold")
            axes[i][1].set_ylabel("Scaled value (0–1)")

        fig.suptitle("Outlier Boxplots: Before vs After IQR Capping", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = PLOTS_DIR / "03_outlier_boxplots.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[DataVisualizer] Saved {path}")

    # ── 4 ─────────────────────────────────────────────────────────────────────

    def neighbourhood_breakdown(self) -> None:
        """Horizontal bar chart of listing counts by neighbourhood_group (raw data)."""
        counts = self.raw["neighbourhood_group"].value_counts().sort_values()
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette("muted", len(counts))
        bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="white")

        for bar, val in zip(bars, counts.values):
            ax.text(val + 50, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=10)

        ax.set_title("Listings by Neighbourhood Group", fontsize=14, fontweight="bold")
        ax.set_xlabel("Number of Listings")
        ax.set_xlim(0, counts.max() * 1.18)
        plt.tight_layout()
        path = PLOTS_DIR / "04_neighbourhood_breakdown.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[DataVisualizer] Saved {path}")

    # ── 5 ─────────────────────────────────────────────────────────────────────

    def correlation_heatmap(self) -> None:
        """Heatmap of Pearson correlations between numeric columns in the clean dataset."""
        num_df = self.clean.select_dtypes(include=[np.number])
        corr = num_df.corr()

        fig, ax = plt.subplots(figsize=(12, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Correlation Heatmap — Cleaned Dataset", fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = PLOTS_DIR / "05_correlation_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[DataVisualizer] Saved {path}")

    def generate_all(self) -> None:
        """Run every visualisation method in sequence."""
        self.missing_heatmap()
        self.price_distribution_before_after()
        self.outlier_boxplots()
        self.neighbourhood_breakdown()
        self.correlation_heatmap()


# ──────────────────────────────────────────────────────────────────────────────
# PipelineReport
# ──────────────────────────────────────────────────────────────────────────────

class PipelineReport:
    """Prints a formatted summary of every transformation applied by the pipeline."""

    def __init__(self, report: dict) -> None:
        """
        Args:
            report: The dict returned by DataCleaner.get_report().
        """
        self.report = report

    def print_summary(self) -> None:
        """Print the full pipeline summary to stdout."""
        r = self.report
        rows_before = r.get("rows_before", "?")
        rows_after = r.get("rows_after", "?")
        rows_removed = (rows_before - rows_after) if isinstance(rows_before, int) else "?"

        print()
        print("╔" + "═" * 58 + "╗")
        print("║" + " PIPELINE SUMMARY ".center(58) + "║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  Rows before cleaning        : {rows_before:>10,}              ║")
        print(f"║  Rows after  cleaning        : {rows_after:>10,}              ║")
        print(f"║  Rows removed (net)          : {rows_removed:>10,}              ║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  Duplicate rows dropped      : {r['duplicates_removed']:>10,}              ║")
        print(f"║  Rows dropped (missing)      : {r['missing_dropped_rows']:>10,}              ║")
        print("╠" + "═" * 58 + "╣")
        print("║  Missing values filled:                                  ║")
        for col, n in r.get("missing_filled", {}).items():
            print(f"║    • {col:<28} {n:>6,} cells filled         ║")
        print("╠" + "═" * 58 + "╣")
        print("║  Outliers capped (IQR method):                           ║")
        for col, info in r.get("outliers_capped", {}).items():
            total = info["capped_low"] + info["capped_high"]
            print(f"║    • {col:<28} {total:>6,} values capped        ║")
            print(f"║      floor={info['lower_bound']:.2f}  ceiling={info['upper_bound']:.2f}".ljust(60) + "║")
        print("╠" + "═" * 58 + "╣")
        encoded = ", ".join(r.get("columns_encoded", [])) or "none"
        scaled  = ", ".join(r.get("columns_scaled", [])) or "none"
        print(f"║  Encoded columns : {encoded:<38} ║")
        print(f"║  Scaled  columns : {scaled:<38} ║")
        print("╠" + "═" * 58 + "╣")
        print(f"║  Charts saved to ./plots/    : 5 files                  ║")
        print("╚" + "═" * 58 + "╝")
        print()


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(filepath: str = "AB_NYC_2019.csv") -> tuple[pd.DataFrame, dict]:
    """
    Execute the full preprocessing pipeline and return the clean DataFrame
    together with the cleaning report.

    Args:
        filepath: Path to the raw CSV.

    Returns:
        Tuple of (clean_df, report_dict).
    """
    print("\n▶  Loading data …")
    loader = DataLoader(filepath)
    raw_df = loader.load()
    loader.inspect()

    raw_snapshot = raw_df.copy()

    print("\n▶  Cleaning data …")
    cleaner = DataCleaner(raw_df)
    (
        cleaner
        .handle_missing_values()
        .remove_duplicates()
        .fix_outliers()
        .encode_categoricals()
        .scale_numerics()
    )
    clean_df = cleaner.get_clean_df()
    report   = cleaner.get_report()

    print("\n▶  Generating visualisations …")
    viz = DataVisualizer(raw_snapshot, clean_df)
    viz.generate_all()

    print("\n▶  Pipeline report:")
    PipelineReport(report).print_summary()

    return clean_df, report


if __name__ == "__main__":
    run_pipeline()
