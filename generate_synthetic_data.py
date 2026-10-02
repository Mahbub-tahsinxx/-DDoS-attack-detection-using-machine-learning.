"""
generate_synthetic_data.py
--------------------------
Generates a synthetic flow-level dataset mimicking the schema and statistical
behaviour of CICDDoS2019, so the detection pipeline can be developed and
demonstrated before the real dataset is available.

IMPORTANT: This is SYNTHETIC data. Results from it demonstrate that the pipeline
works end-to-end; they are NOT empirical findings about real DDoS traffic.
Replace with the real CICDDoS2019 CSV for actual results.

Realism features (these matter — without them the classes separate perfectly
and the speed/accuracy trade-off is invisible):
  * Benign FLASH CROWDS  - legitimate traffic spikes that superficially resemble
                           volumetric attacks (the main source of false alarms)
  * RAMPED attack onset  - attacks build over several seconds rather than
                           switching on instantly, so early attack windows are
                           genuinely ambiguous
  * LOW-RATE attack      - a stealthy episode with far weaker statistical
                           signature, much harder to detect in short windows
  * Per-second jitter    - arrival rates fluctuate, adding noise

Schema produced (subset of real CICDDoS2019 columns used downstream):
    Timestamp, Source IP, Destination IP,
    Total Fwd Packets, Total Backward Packets,
    Total Length of Fwd Packets, Total Length of Bwd Packets,
    SYN Flag Count, ACK Flag Count, Flow Duration, Label

Usage:
    python generate_synthetic_data.py --out data/synthetic_ddos.csv
"""

import argparse
import os
import numpy as np
import pandas as pd


VICTIM_IP = "192.168.50.10"


def random_ip(rng, n, pool=None):
    if pool is not None:
        return rng.choice(pool, size=n)
    octets = rng.integers(1, 255, size=(n, 4))
    return np.array([".".join(map(str, row)) for row in octets])


def benign_block(rng, n, t_offsets, pool, burst=False):
    """
    Benign flows. During a flash crowd (burst=True) the volume rises sharply and
    connections are shorter, which is exactly what makes flash crowds look like
    an attack to a volume-only detector.
    """
    if burst:
        fwd = rng.lognormal(mean=1.3, sigma=0.8, size=n).astype(int) + 1
        syn = rng.choice([1, 2, 3], size=n, p=[0.55, 0.32, 0.13])
        dur = rng.lognormal(mean=12.2, sigma=1.2, size=n)
    else:
        fwd = rng.lognormal(mean=2.0, sigma=0.9, size=n).astype(int) + 1
        syn = rng.choice([0, 1, 2], size=n, p=[0.25, 0.65, 0.10])
        dur = rng.lognormal(mean=14.0, sigma=1.3, size=n)

    bwd = (fwd * rng.uniform(0.5, 1.5, size=n)).astype(int)
    fwd_b = fwd * rng.integers(200, 1400, size=n)
    bwd_b = bwd * rng.integers(200, 1400, size=n)
    ack = np.clip(fwd + bwd - syn, 0, None)

    return pd.DataFrame({
        "t": t_offsets,
        "Source IP": random_ip(rng, n, pool=pool),
        "Total Fwd Packets": fwd,
        "Total Backward Packets": bwd,
        "Total Length of Fwd Packets": fwd_b,
        "Total Length of Bwd Packets": bwd_b,
        "SYN Flag Count": syn,
        "ACK Flag Count": ack,
        "Flow Duration": dur,
        "Label": "BENIGN",
    })


def attack_block(rng, n, t_offsets, intensity=1.0, label="Syn"):
    """
    Attack flows. `intensity` in (0, 1] blends the attack signature toward benign:
    intensity 1.0 = full-strength flood, 0.3 = stealthy low-rate attack whose
    per-flow characteristics partially overlap normal traffic.
    """
    # Full-strength: tiny 1-3 packet flows, nearly all SYN
    p_small = 0.75 * intensity + 0.25 * (1 - intensity)
    fwd = rng.choice([1, 2, 3, 5], size=n,
                     p=[p_small, 0.18, 0.05, max(0.02, 1 - p_small - 0.23)])
    bwd = rng.choice([0, 1], size=n, p=[0.85 * intensity + 0.15,
                                        1 - (0.85 * intensity + 0.15)])
    fwd_b = fwd * rng.integers(40, 120 + int(400 * (1 - intensity)), size=n)
    bwd_b = bwd * rng.integers(40, 120, size=n)

    # SYN ratio degrades as intensity falls
    syn = np.where(rng.random(n) < (0.95 * intensity + 0.30 * (1 - intensity)),
                   fwd, rng.choice([0, 1], size=n))
    ack = rng.choice([0, 1], size=n, p=[0.9, 0.1])
    dur = rng.lognormal(mean=9.0 + 3.0 * (1 - intensity), sigma=0.8, size=n)

    # Spoofing: full-strength attacks use fully random sources; low-rate attacks
    # reuse a smaller bot pool, so source diversity is a weaker signal
    if intensity > 0.6:
        src = random_ip(rng, n)
    else:
        pool = random_ip(rng, max(30, int(300 * intensity)))
        src = random_ip(rng, n, pool=pool)

    return pd.DataFrame({
        "t": t_offsets,
        "Source IP": src,
        "Total Fwd Packets": fwd,
        "Total Backward Packets": bwd,
        "Total Length of Fwd Packets": fwd_b,
        "Total Length of Bwd Packets": bwd_b,
        "SYN Flag Count": syn,
        "ACK Flag Count": ack,
        "Flow Duration": dur,
        "Label": label,
    })


def generate(duration_seconds=900, seed=42):
    rng = np.random.default_rng(seed)
    base_time = pd.Timestamp("2019-03-11 09:00:00")
    benign_pool = random_ip(rng, 220)

    # Attack episodes: (start, end, peak_intensity, ramp_seconds, label)
    episodes = [
        (180, 300, 1.00, 12, "Syn"),        # strong flood, ramps in over 12s
        (450, 520, 0.35, 20, "Syn-lowrate"),# stealthy low-rate attack
        (640, 760, 0.85, 8,  "UDP"),        # second strong flood
    ]

    # Benign flash crowds (legit spikes that mimic attacks): (start, end, multiplier)
    flash_crowds = [
        (90, 130, 7.0),
        (380, 410, 9.0),
        (580, 610, 6.0),
        (800, 840, 8.0),
    ]

    # NOTE: peak attack rate is deliberately set close to the flash-crowd peak
    # (45 * 8 = 360 flows/s). If the attack were an order of magnitude louder,
    # raw volume alone would separate the classes perfectly and the study would
    # have nothing to measure. Keeping them overlapping forces the classifier to
    # rely on SYN ratio and source diversity, which is the realistic case.
    base_benign_rate = 45      # flows/sec
    peak_attack_rate = 260     # flows/sec at intensity 1.0

    parts = []

    # ---- build second by second so rates can vary smoothly ----
    for sec in range(duration_seconds):
        # --- benign rate for this second ---
        rate = base_benign_rate * rng.uniform(0.75, 1.25)  # jitter
        burst = False
        for (fs, fe, mult) in flash_crowds:
            if fs <= sec < fe:
                # smooth rise and fall of the flash crowd
                pos = (sec - fs) / max(1, (fe - fs))
                shape = np.sin(np.pi * pos)
                rate *= 1 + (mult - 1) * shape
                burst = True
        n_b = max(1, int(rate))
        parts.append(benign_block(
            rng, n_b, rng.uniform(sec, sec + 1, size=n_b), benign_pool, burst=burst
        ))

        # --- attack rate for this second ---
        for (a_s, a_e, peak, ramp, lab) in episodes:
            if a_s <= sec < a_e:
                # ramp up at the start, ramp down at the end
                since = sec - a_s
                until = a_e - sec
                f_up = min(1.0, since / ramp) if ramp > 0 else 1.0
                f_dn = min(1.0, until / max(1, ramp // 2))
                inten = peak * min(f_up, f_dn)
                inten = max(inten, 0.0)
                if inten <= 0.02:
                    continue
                n_a = int(peak_attack_rate * inten * rng.uniform(0.8, 1.2))
                if n_a < 1:
                    continue
                parts.append(attack_block(
                    rng, n_a, rng.uniform(sec, sec + 1, size=n_a),
                    intensity=max(0.15, inten), label=lab
                ))

    df = pd.concat(parts, ignore_index=True)
    df["Timestamp"] = base_time + pd.to_timedelta(df["t"], unit="s")
    df["Destination IP"] = VICTIM_IP
    df = df.drop(columns=["t"])
    df = df[[
        "Timestamp", "Source IP", "Destination IP",
        "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "SYN Flag Count", "ACK Flag Count", "Flow Duration", "Label",
    ]]
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_ddos.csv")
    ap.add_argument("--duration", type=int, default=900)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df = generate(duration_seconds=args.duration, seed=args.seed)
    df.to_csv(args.out, index=False)

    n_atk = (df["Label"] != "BENIGN").sum()
    print(f"Wrote {len(df):,} synthetic flow records to {args.out}")
    print(f"  Benign flows : {(df['Label'] == 'BENIGN').sum():,}")
    print(f"  Attack flows : {n_atk:,}")
    print(f"  Attack types : {sorted(df.loc[df['Label'] != 'BENIGN', 'Label'].unique())}")
    print(f"  Time span    : {df['Timestamp'].min()}  ->  {df['Timestamp'].max()}")


if __name__ == "__main__":
    main()
