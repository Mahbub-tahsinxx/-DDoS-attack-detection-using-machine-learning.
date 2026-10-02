"""
ddos_pipeline.py
----------------
Comparing Detection Speed vs. Accuracy for DDoS Attacks
Using Time-Window-Based Traffic Analysis

Pipeline stages:
    1. Load flow records (synthetic now, CICDDoS2019 later)
    2. Clean / normalise column names
    3. Aggregate flows into fixed time windows (1s, 5s, 10s)
    4. Compute per-window features
    5. Label each window
    6. Train Random Forest + Logistic Regression per window size
    7. Evaluate accuracy metrics AND detection latency
    8. Emit comparison tables and plots

Design decision (important for the report):
    The train/test split is CHRONOLOGICAL, not random. Adjacent windows are
    temporally correlated, so a random split would leak information across the
    boundary and inflate the scores.

Usage:
    python ddos_pipeline.py --input data/synthetic_ddos.csv --outdir results
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

WINDOW_SIZES = [1, 2, 5, 10, 20]   # seconds — keep in sync with run_experiment.py's default
ATTACK_WINDOW_THRESHOLD = 0.5      # fraction of attack flows to label window as attack
TRAIN_FRACTION = 0.70              # chronological split point


# ------------------------------------------------------------------ #
# 1-2. LOADING AND CLEANING
# ------------------------------------------------------------------ #
def load_and_clean(path):
    """
    Load flow records and normalise the columns we need.

    CICDDoS2019 CSVs have leading spaces in many column names
    (e.g. ' Source IP'), so we strip them defensively. The COLUMN_MAP
    below is the single place to adjust if the real CSV uses variants.
    """
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    required = [
        "Timestamp", "Source IP",
        "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "SYN Flag Count", "Label",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Timestamp"])

    # Replace infinities that CICFlowMeter sometimes emits, then drop bad rows
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[c for c in required if c in numeric_cols])

    # Binary label: 1 = attack, 0 = benign
    df["is_attack"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)

    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


# ------------------------------------------------------------------ #
# 3-5. WINDOWING, FEATURE EXTRACTION, LABELLING
# ------------------------------------------------------------------ #
def build_windows(df, window_seconds):
    """
    Aggregate flow records into fixed time windows and compute per-window features.

    Features (deliberately minimal and cheap to compute in real time):
        packet_count    - total packets observed in the window
        byte_count      - total bytes observed in the window
        unique_src_ips  - distinct source addresses (spoofing indicator)
        syn_ratio       - SYN flags / total packets (SYN-flood indicator)
        flow_count      - number of flows in the window (arrival rate proxy)
    """
    d = df.set_index("Timestamp")
    rule = f"{window_seconds}s"

    total_pkts = d["Total Fwd Packets"] + d["Total Backward Packets"]
    total_bytes = (d["Total Length of Fwd Packets"]
                   + d["Total Length of Bwd Packets"])

    tmp = pd.DataFrame({
        "pkts": total_pkts,
        "bytes": total_bytes,
        "syn": d["SYN Flag Count"],
        "src": d["Source IP"],
        "is_attack": d["is_attack"],
    })

    g = tmp.resample(rule)

    win = pd.DataFrame({
        "packet_count":   g["pkts"].sum(),
        "byte_count":     g["bytes"].sum(),
        "unique_src_ips": g["src"].nunique(),
        "syn_count":      g["syn"].sum(),
        "flow_count":     g["pkts"].count(),
        "attack_frac":    g["is_attack"].mean(),
    })

    # Drop empty windows (no traffic observed at all)
    win = win[win["flow_count"] > 0].copy()

    # SYN ratio, guarding against divide-by-zero
    win["syn_ratio"] = win["syn_count"] / win["packet_count"].replace(0, np.nan)
    win["syn_ratio"] = win["syn_ratio"].fillna(0.0)

    # Window label: attack if attack flows dominate the window
    win["label"] = (win["attack_frac"] >= ATTACK_WINDOW_THRESHOLD).astype(int)

    win = win.drop(columns=["syn_count", "attack_frac"])
    win = win.reset_index().rename(columns={"Timestamp": "window_start"})
    return win


# ------------------------------------------------------------------ #
# 6. TRAINING
# ------------------------------------------------------------------ #
FEATURES = ["packet_count", "byte_count", "unique_src_ips", "syn_ratio", "flow_count"]


def chronological_split(win, train_fraction=TRAIN_FRACTION):
    """Split by time, not at random — adjacent windows are correlated."""
    cut = int(len(win) * train_fraction)
    return win.iloc[:cut].copy(), win.iloc[cut:].copy()


def get_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
    }


# ------------------------------------------------------------------ #
# 7. EVALUATION — INCLUDING DETECTION LATENCY
# ------------------------------------------------------------------ #
def detection_latency(test_df, y_pred, window_seconds):
    """
    Detection latency proxy.

    For each contiguous run of attack windows in the ground truth, find the first
    window the model flags as attack. Latency is how far into the attack that is,
    measured in seconds:

        latency = (index_of_first_detection - index_of_attack_start) * window_seconds

    A model that flags the very first attack window has latency 0s for that
    episode, but note that in a live system the decision could still only be made
    after the window had elapsed — so the floor on real-world latency is the
    window size itself. We report both.
    """
    y_true = test_df["label"].values
    latencies = []

    i = 0
    n = len(y_true)
    while i < n:
        if y_true[i] == 1:
            start = i
            while i < n and y_true[i] == 1:
                i += 1
            end = i  # exclusive
            # first predicted detection inside this attack episode
            detected_at = None
            for j in range(start, end):
                if y_pred[j] == 1:
                    detected_at = j
                    break
            if detected_at is not None:
                latencies.append((detected_at - start) * window_seconds)
            else:
                latencies.append(np.nan)  # episode never detected
        else:
            i += 1

    if len(latencies) == 0:
        return np.nan, np.nan, 0, 0

    arr = np.array(latencies, dtype=float)
    detected = np.sum(~np.isnan(arr))
    mean_lat = np.nanmean(arr) if detected > 0 else np.nan
    # Effective latency includes the unavoidable wait for the window to fill
    effective = mean_lat + window_seconds if detected > 0 else np.nan
    return mean_lat, effective, int(detected), len(latencies)


def evaluate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "false_alarm_rate": fpr,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


# ------------------------------------------------------------------ #
# 8. PLOTTING
# ------------------------------------------------------------------ #
def make_plots(results, outdir):
    res = pd.DataFrame(results)
    os.makedirs(outdir, exist_ok=True)

    # --- Accuracy / F1 vs window size ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in res["model"].unique():
        sub = res[res["model"] == model].sort_values("window_s")
        ax.plot(sub["window_s"], sub["f1"], marker="o", label=f"{model} — F1")
        ax.plot(sub["window_s"], sub["accuracy"], marker="s", linestyle="--",
                alpha=0.6, label=f"{model} — Accuracy")
    ax.set_xlabel("Window size (seconds)")
    ax.set_ylabel("Score")
    ax.set_title("Detection Accuracy vs. Window Size")
    ax.set_xticks(sorted(res["window_s"].unique()))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_vs_window.png"), dpi=150)
    plt.close(fig)

    # --- Detection latency vs window size ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in res["model"].unique():
        sub = res[res["model"] == model].sort_values("window_s")
        ax.plot(sub["window_s"], sub["effective_latency_s"], marker="o", label=model)
    ax.set_xlabel("Window size (seconds)")
    ax.set_ylabel("Effective detection latency (s)")
    ax.set_title("Detection Latency vs. Window Size")
    ax.set_xticks(sorted(res["window_s"].unique()))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "latency_vs_window.png"), dpi=150)
    plt.close(fig)

    # --- The trade-off: latency vs F1 ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in res["model"].unique():
        sub = res[res["model"] == model].sort_values("window_s")
        ax.plot(sub["effective_latency_s"], sub["f1"], marker="o", label=model)
        for _, r in sub.iterrows():
            ax.annotate(f"{int(r['window_s'])}s",
                        (r["effective_latency_s"], r["f1"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Effective detection latency (s)  →  slower")
    ax.set_ylabel("F1 score  →  more accurate")
    ax.set_title("The Speed / Accuracy Trade-off")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "tradeoff_curve.png"), dpi=150)
    plt.close(fig)

    # --- False alarm rate vs window size ---
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in res["model"].unique():
        sub = res[res["model"] == model].sort_values("window_s")
        ax.plot(sub["window_s"], sub["false_alarm_rate"], marker="o", label=model)
    ax.set_xlabel("Window size (seconds)")
    ax.set_ylabel("False alarm rate")
    ax.set_title("False Alarm Rate vs. Window Size")
    ax.set_xticks(sorted(res["window_s"].unique()))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "false_alarms_vs_window.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/synthetic_ddos.csv")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--windows", type=int, nargs="+", default=WINDOW_SIZES)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 72)
    print("DDoS Detection: Speed vs. Accuracy across Time Window Sizes")
    print("=" * 72)

    print(f"\n[1] Loading flows from {args.input} ...")
    df = load_and_clean(args.input)
    print(f"    {len(df):,} flow records "
          f"({df['is_attack'].sum():,} attack / {(1-df['is_attack']).sum():,} benign)")
    print(f"    Time span: {df['Timestamp'].min()} -> {df['Timestamp'].max()}")

    results = []
    window_tables = {}

    for w in args.windows:
        print(f"\n[2] Window size = {w}s")
        win = build_windows(df, w)
        window_tables[w] = win
        print(f"    {len(win):,} windows  "
              f"({win['label'].sum():,} attack / {(1-win['label']).sum():,} benign)")

        train, test = chronological_split(win)
        X_tr, y_tr = train[FEATURES], train["label"]
        X_te, y_te = test[FEATURES], test["label"]

        if y_tr.nunique() < 2 or y_te.nunique() < 2:
            print("    ! Split does not contain both classes — skipping.")
            continue

        for name, model in get_models().items():
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)

            m = evaluate(y_te.values, y_pred)
            raw_lat, eff_lat, det, total = detection_latency(test, y_pred, w)

            results.append({
                "window_s": w,
                "model": name,
                **{k: m[k] for k in
                   ["accuracy", "precision", "recall", "f1", "false_alarm_rate"]},
                "raw_latency_s": raw_lat,
                "effective_latency_s": eff_lat,
                "episodes_detected": f"{det}/{total}",
                "n_windows": len(win),
            })

            print(f"    {name:<20} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}  "
                  f"FAR={m['false_alarm_rate']:.4f}  "
                  f"latency={eff_lat:.1f}s" if not np.isnan(eff_lat)
                  else f"    {name:<20} acc={m['accuracy']:.4f}  f1={m['f1']:.4f}")

    res = pd.DataFrame(results)

    # Save tables
    res_path = os.path.join(args.outdir, "comparison_results.csv")
    res.to_csv(res_path, index=False)

    for w, win in window_tables.items():
        win.to_csv(os.path.join(args.outdir, f"windows_{w}s.csv"), index=False)

    print("\n" + "=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)
    show = res[["window_s", "model", "accuracy", "precision", "recall", "f1",
                "false_alarm_rate", "effective_latency_s", "n_windows"]]
    print(show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\n[3] Generating plots in {args.outdir}/ ...")
    make_plots(results, args.outdir)
    print("    accuracy_vs_window.png")
    print("    latency_vs_window.png")
    print("    tradeoff_curve.png")
    print("    false_alarms_vs_window.png")
    print(f"\n[4] Tables written to {res_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
