import argparse
import json
import os

import numpy as np

from main_1 import run_or_resume_incremental


def build_close_freq_beta(scales):
    s1, s2 = float(scales[0]), float(scales[1])
    # Two close frequencies: 2.0*pi and 2.2*pi
    return [
        lambda t: s1 * np.sin(2.0 * np.pi * t),
        lambda t: s2 * np.sin(2.2 * np.pi * t + 0.25),
    ]


def parse_list_float(s):
    return [float(x) for x in str(s).split(",") if x.strip()]


def stage_rmse_from_history(history):
    out = {}
    for h in history:
        if "stage" in h and "train_rmse" in h:
            out[int(h["stage"])] = float(h["train_rmse"])
    return out


def main():
    parser = argparse.ArgumentParser(description="Small-dim RMSE trend validation for incremental VCM.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/validate_small_dim")
    parser.add_argument("--t-final", type=int, default=8)
    parser.add_argument("--n-per-segment", type=int, default=250)
    parser.add_argument("--P", type=int, default=20)
    parser.add_argument("--signal-idx", default="1,7")
    parser.add_argument("--beta-scales", default="1.0,0.9")
    parser.add_argument("--sigma", type=float, default=0.15)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--knot-step", type=float, default=0.1)
    parser.add_argument("--seed-cv", type=int, default=2025)
    parser.add_argument("--use-1se", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--save-checkpoint-data", action="store_true")
    parser.add_argument("--out-json", default="checkpoints/validate_small_dim/rmse_summary.json")
    args = parser.parse_args()

    signal_idx = [int(x) for x in args.signal_idx.split(",") if x.strip()]
    if len(signal_idx) != 2:
        raise ValueError("This validator expects exactly 2 active variables in --signal-idx.")
    beta_scales = parse_list_float(args.beta_scales)
    if len(beta_scales) != 2:
        raise ValueError("This validator expects exactly 2 scales in --beta-scales.")

    beta_funcs = build_close_freq_beta(beta_scales)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    per_seed = []
    for i in range(int(args.n_seeds)):
        seed_data = int(args.seed_start + i)
        seed_dir = os.path.join(args.checkpoint_dir, f"seed{seed_data}")
        _, history = run_or_resume_incremental(
            checkpoint_dir=seed_dir,
            t_final=float(args.t_final),
            n_per_segment=int(args.n_per_segment),
            P=int(args.P),
            signal_idx=signal_idx,
            beta_funcs=beta_funcs,
            sigma=float(args.sigma),
            k=int(args.k),
            knot_step=float(args.knot_step),
            seed_data=seed_data,
            seed_cv=int(args.seed_cv),
            use_1se=bool(args.use_1se),
            save_checkpoint_data=bool(args.save_checkpoint_data),
            debug=False,
        )
        rmse_by_stage = stage_rmse_from_history(history)
        per_seed.append({"seed_data": seed_data, "rmse_by_stage": rmse_by_stage})

    stages = list(range(1, int(args.t_final) + 1))
    agg = {}
    for s in stages:
        vals = [d["rmse_by_stage"][s] for d in per_seed if s in d["rmse_by_stage"]]
        agg[s] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }

    mean_curve = [agg[s]["mean"] for s in stages]
    diffs = [mean_curve[i] - mean_curve[i - 1] for i in range(1, len(mean_curve))]
    non_decreasing = bool(all(d >= -1e-10 for d in diffs))
    increases = int(sum(1 for d in diffs if d > 0))

    print("=== RMSE by seed/stage ===")
    for rec in per_seed:
        print(f"seed={rec['seed_data']} rmse={rec['rmse_by_stage']}")

    print("\n=== Mean RMSE curve ===")
    for s in stages:
        print(f"stage={s} mean={agg[s]['mean']:.6f} std={agg[s]['std']:.6f} n={agg[s]['n']}")

    print("\n=== Trend check ===")
    print(f"mean_rmse_non_decreasing={non_decreasing}")
    print(f"num_increase_steps={increases} / {max(0, len(stages)-1)}")
    print(f"stage_diffs={diffs}")

    summary = {
        "config": {
            "t_final": int(args.t_final),
            "n_per_segment": int(args.n_per_segment),
            "P": int(args.P),
            "signal_idx": signal_idx,
            "beta_scales": beta_scales,
            "sigma": float(args.sigma),
            "k": int(args.k),
            "knot_step": float(args.knot_step),
            "n_seeds": int(args.n_seeds),
            "seed_start": int(args.seed_start),
            "seed_cv": int(args.seed_cv),
            "use_1se": bool(args.use_1se),
        },
        "per_seed": per_seed,
        "aggregate": agg,
        "trend": {
            "mean_rmse_non_decreasing": non_decreasing,
            "num_increase_steps": increases,
            "stage_diffs": diffs,
        },
    }
    out_dir = os.path.dirname(args.out_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary: {args.out_json}")


if __name__ == "__main__":
    main()
