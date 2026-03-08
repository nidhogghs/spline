import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from main_1 import IncrementalVCMTrainer, bspline_design_matrix, true_beta_funcs_default



def _parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_float_list(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def _find_latest_checkpoint(checkpoint_dir: str):
    best = None
    for name in os.listdir(checkpoint_dir):
        if not (name.startswith("ckpt_t") and name.endswith(".json")):
            continue
        token = name[len("ckpt_t") : -len(".json")]
        if not token.isdigit():
            continue
        t_val = int(token)
        npz = os.path.join(checkpoint_dir, f"ckpt_t{t_val}.npz")
        if os.path.exists(npz):
            if best is None or t_val > best:
                best = t_val
    if best is None:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    return best


def main():
    parser = argparse.ArgumentParser(description="Visualize main_1 reconstruction and save metrics record.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--signal-idx", default="1,2,3,4,5")
    parser.add_argument("--beta-scales", default="1,1,1,1,1")
    parser.add_argument("--grid", type=int, default=400)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-record", required=True)
    args = parser.parse_args()

    ckpt_dir = os.path.normpath(args.checkpoint_dir)
    t_latest = _find_latest_checkpoint(ckpt_dir)
    ckpt_prefix = os.path.join(ckpt_dir, f"ckpt_t{t_latest}")

    trainer = IncrementalVCMTrainer.load_checkpoint(ckpt_prefix)
    t_grid = np.linspace(0.0, float(trainer.t_end), int(args.grid))
    B = bspline_design_matrix(t_grid, trainer.knots, trainer.k)
    beta_hat = np.column_stack([B @ trainer.coef_blocks[p] for p in range(trainer.P)])


    signal_idx = _parse_int_list(args.signal_idx)
    beta_funcs = true_beta_funcs_default(_parse_float_list(args.beta_scales))

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_record) or ".", exist_ok=True)

    n_plots = len(signal_idx)
    cols = 2 if n_plots > 1 else 1
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    per_signal_rmse = {}
    for i, p in enumerate(signal_idx):
        ax = axes[i]
        idx = i % len(beta_funcs)
        true_vals = beta_funcs[idx](t_grid)
        pred_vals = beta_hat[:, p]

        rmse = float(np.sqrt(np.mean((pred_vals - true_vals) ** 2)))
        per_signal_rmse[str(p)] = rmse

        ax.plot(t_grid, pred_vals, label="estimated beta")
        ax.plot(t_grid, true_vals, "--", label="true beta")
        ax.set_title(f"p={p}, rmse={rmse:.4f}")
        ax.grid(alpha=0.3)
        ax.legend()

    for k in range(n_plots, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    fig.savefig(args.out_png, dpi=160)

    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "checkpoint_dir": ckpt_dir,
        "checkpoint_prefix": ckpt_prefix,
        "t_end": float(trainer.t_end),
        "P": int(trainer.P),
        "k": int(trainer.k),
        "knot_step": float(trainer.knot_step),
        "signal_idx": signal_idx,
        "per_signal_beta_rmse": per_signal_rmse,
        "mean_signal_beta_rmse": float(np.mean(list(per_signal_rmse.values()))),
        "plot_path": os.path.normpath(args.out_png),
    }

    with open(args.out_record, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    print(f"[ok] plot saved: {args.out_png}")
    print(f"[ok] record saved: {args.out_record}")


if __name__ == "__main__":
    main()
