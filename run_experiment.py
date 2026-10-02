"""
run_experiment.py
-----------------
Runs the full window-size comparison and reports mean +/- standard deviation
for every metric, in one of two modes:

  SYNTHETIC MODE (default, no --input given):
      Repeats the experiment across several independent synthetic traffic
      realisations (different random seeds). A single run on a single
      capture can produce a perfect score by luck, so this checks stability.

  REAL-DATA MODE (--input path/to/real.csv):
      There is only one real capture, so instead of varying the random seed,
      this repeats the experiment across several different CHRONOLOGICAL
      train/test split points (50/50, 60/40, 70/30, 80/20). This matters
      because larger windows produce far fewer total windows (a 20s window
      can leave under 100 windows total), so a single 70/30 split can be
      dominated by a handful of test cases. Repeating across split points
      shows whether a result is a stable pattern or noise from one split.

Both modes produce the same output format (raw_runs.csv, aggregated_results.csv,
and the same set of plots), so results from synthetic and real runs can be
compared directly using identical downstream analysis.

Usage:
    # synthetic mode
    python run_experiment.py --runs 10 --outdir results_synthetic

    # real-data mode
    python run_experiment.py --input data/real_UDPLag.csv --outdir results_real
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_synthetic_data import generate
from ddos_pipeline import (
    load_and_clean, build_windows, chronological_split, get_models,
    evaluate, detection_latency, FEATURES,
)

warnings.filterwarnings("ignore")

REAL_SPLIT_FRACTIONS = [0.5, 0.6, 0.7, 0.8]


# ------------------------------------------------------------------ #
# One evaluation at a single (window_size, split) configuration
# ------------------------------------------------------------------ #
def evaluate_one_split(win, window_s, train, test, run_label, run_kind):
    """Train both models on `train`, evaluate on `test`, return metric rows."""
    out = []
    if train["label"].nunique() < 2 or test["label"].nunique() < 2:
        return out
    if len(test) < 5:
        return out

    X_tr, y_tr = train[FEATURES], train["label"]
    X_te, y_te = test[FEATURES], test["label"]

    for name, model in get_models().items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        m = evaluate(y_te.values, pred)
        _, eff, det, tot = detection_latency(test, pred, window_s)
        out.append({
            "run_kind": run_kind,       # "seed" or "split"
            "run_label": run_label,     # seed number or split fraction
            "window_s": window_s,
            "model": name,
            "n_test": len(test),
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "false_alarm_rate": m["false_alarm_rate"],
            "effective_latency_s": eff,
            "detect_rate": det / tot if tot else np.nan,
        })
    return out


# ------------------------------------------------------------------ #
# SYNTHETIC MODE
# ------------------------------------------------------------------ #
def run_synthetic(seed, windows, duration):
    df = generate(duration_seconds=duration, seed=seed)
    df["is_attack"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
    out = []
    for w in windows:
        win = build_windows(df, w)
        train, test = chronological_split(win)
        out.extend(evaluate_one_split(win, w, train, test, run_label=seed, run_kind="seed"))
    return out


# ------------------------------------------------------------------ #
# REAL-DATA MODE
# ------------------------------------------------------------------ #
def run_real(input_path, windows):
    df = load_and_clean(input_path)
    print(f"  {len(df):,} flows, {df['is_attack'].sum():,} attack / "
          f"{(1 - df['is_attack']).sum():,} benign")
    print(f"  span: {df['Timestamp'].min()} -> {df['Timestamp'].max()}")

    out = []
    for w in windows:
        win = build_windows(df, w)
        print(f"  window={w}s -> {len(win)} windows "
              f"({win['label'].sum()} attack / {(win['label'] == 0).sum()} benign)")
        for frac in REAL_SPLIT_FRACTIONS:
            cut = int(len(win) * frac)
            train, test = win.iloc[:cut], win.iloc[cut:]
            out.extend(evaluate_one_split(
                win, w, train, test, run_label=frac, run_kind="split"
            ))
    return out


# ------------------------------------------------------------------ #
# PLOTTING (shared by both modes)
# ------------------------------------------------------------------ #
def plot_with_errorbars(agg, outdir, title_suffix=""):
    os.makedirs(outdir, exist_ok=True)

    metrics = [
        ("f1", "F1 score", "f1_vs_window.png", f"Detection Accuracy (F1) vs. Window Size{title_suffix}"),
        ("false_alarm_rate", "False alarm rate", "far_vs_window.png",
         f"False Alarm Rate vs. Window Size{title_suffix}"),
        ("effective_latency_s", "Effective detection latency (s)",
         "latency_vs_window.png", f"Detection Latency vs. Window Size{title_suffix}"),
    ]
    for col, ylab, fname, title in metrics:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for model in agg["model"].unique():
            sub = agg[agg["model"] == model].sort_values("window_s")
            ax.errorbar(sub["window_s"], sub[f"{col}_mean"],
                        yerr=sub[f"{col}_std"], marker="o", capsize=4, label=model)
        ax.set_xlabel("Window size (seconds)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.set_xticks(sorted(agg["window_s"].unique()))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close(fig)

    # Trade-off curve
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for model in agg["model"].unique():
        sub = agg[agg["model"] == model].sort_values("window_s")
        ax.errorbar(sub["effective_latency_s_mean"], sub["f1_mean"],
                    xerr=sub["effective_latency_s_std"], yerr=sub["f1_std"],
                    marker="o", capsize=4, label=model)
        for _, r in sub.iterrows():
            ax.annotate(f"{int(r['window_s'])}s",
                        (r["effective_latency_s_mean"], r["f1_mean"]),
                        textcoords="offset points", xytext=(7, 5), fontsize=9)
    ax.set_xlabel("Effective detection latency (s)   →   slower")
    ax.set_ylabel("F1 score   →   more accurate")
    ax.set_title(f"The Speed / Accuracy Trade-off{title_suffix}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "tradeoff_curve.png"), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ #
# MAIN
# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None,
                    help="Path to a real CICDDoS2019 CSV. If omitted, runs in "
                         "synthetic mode instead.")
    ap.add_argument("--runs", type=int, default=10,
                    help="[synthetic mode] number of random seeds to average over")
    ap.add_argument("--duration", type=int, default=900,
                    help="[synthetic mode] seconds of traffic to simulate per run")
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    real_mode = args.input is not None

    print("=" * 74)
    if real_mode:
        print(f"REAL-DATA MODE — input: {args.input}")
        print(f"Evaluating across {len(REAL_SPLIT_FRACTIONS)} chronological "
              f"split points: {REAL_SPLIT_FRACTIONS}")
    else:
        print(f"SYNTHETIC MODE — {args.runs} random traffic realisations")
    print(f"Window sizes: {args.windows}s")
    print("=" * 74)

    if real_mode:
        rows = run_real(args.input, args.windows)
    else:
        rows = []
        for i in range(args.runs):
            seed = 100 + i
            rows.extend(run_synthetic(seed, args.windows, args.duration))
            print(f"  run {i + 1}/{args.runs} (seed {seed}) complete")

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(args.outdir, "raw_runs.csv"), index=False)

    if raw.empty:
        print("\nNo valid train/test splits produced results at any window "
              "size — usually means too few examples of one class. Try a "
              "narrower --windows list, or check class balance with "
              "ddos_pipeline.py first.")
        return

    metric_cols = ["accuracy", "precision", "recall", "f1",
                   "false_alarm_rate", "effective_latency_s", "detect_rate"]
    agg = (raw.groupby(["window_s", "model"])[metric_cols]
              .agg(["mean", "std", "count"]).reset_index())
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg.to_csv(os.path.join(args.outdir, "aggregated_results.csv"), index=False)

    n_runs_label = (f"{len(REAL_SPLIT_FRACTIONS)} chronological splits" if real_mode
                     else f"{args.runs} runs")
    print("\n" + "=" * 74)
    print(f"AGGREGATED RESULTS  (mean ± std over {n_runs_label})")
    print("=" * 74)
    for model in agg["model"].unique():
        print(f"\n{model}")
        print(f"  {'Win':>4} {'n':>3} {'F1':>16} {'FalseAlarm':>16} {'Latency(s)':>16}")
        sub = agg[agg["model"] == model].sort_values("window_s")
        for _, r in sub.iterrows():
            n = int(r["f1_count"])
            f1_std = r["f1_std"] if n > 1 else 0.0
            far_std = r["false_alarm_rate_std"] if n > 1 else 0.0
            lat_std = r["effective_latency_s_std"] if n > 1 else 0.0
            print(f"  {int(r['window_s']):>3}s {n:>3} "
                  f"{r['f1_mean']:>8.4f}±{f1_std:<6.4f} "
                  f"{r['false_alarm_rate_mean']:>8.4f}±{far_std:<6.4f} "
                  f"{r['effective_latency_s_mean']:>8.2f}±{lat_std:<6.2f}")

    title_suffix = f" — {os.path.basename(args.input)}" if real_mode else " (synthetic)"
    plot_with_errorbars(agg, args.outdir, title_suffix=title_suffix)
    print(f"\nPlots and tables written to {args.outdir}/")


if __name__ == "__main__":
    main()
