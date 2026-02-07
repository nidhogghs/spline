import argparse
import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from main import IncrementalVCMTrainer, true_beta_funcs_default


DEFAULT_CHECKPOINT_ROOT = "checkpoints"


def _resolve_checkpoint_base(checkpoint_dir):
    ckpt_dir = str(checkpoint_dir).strip()
    if not ckpt_dir:
        return ckpt_dir
    norm_dir = os.path.normpath(ckpt_dir)
    if os.path.isabs(ckpt_dir):
        return ckpt_dir
    if norm_dir == DEFAULT_CHECKPOINT_ROOT:
        return ckpt_dir
    if norm_dir.startswith(DEFAULT_CHECKPOINT_ROOT + os.sep):
        return ckpt_dir
    return os.path.join(DEFAULT_CHECKPOINT_ROOT, ckpt_dir)


def _parse_list_int(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def _parse_list_float(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _build_checkpoint_dir(tag: str, checkpoint_dir: str, seed: int) -> str:
    if checkpoint_dir:
        base = _resolve_checkpoint_base(checkpoint_dir)
    else:
        default_name = f"checkpoints_vcm_{tag}" if tag else "checkpoints_vcm"
        base = os.path.join(DEFAULT_CHECKPOINT_ROOT, default_name)
    return f"{base}_seed{seed}"


def _ckpt_path(ckpt_dir: str, t_end: float) -> str:
    return os.path.join(ckpt_dir, f"ckpt_t{int(round(t_end))}")


def main():
    parser = argparse.ArgumentParser(description="Plot average beta(t) across seeds.")
    parser.add_argument("--n-seeds", type=int, required=True, help="Number of seeds to average.")
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed.")
    parser.add_argument("--tag", default="default", help="Experiment tag (used if checkpoint-dir not set).")
    parser.add_argument("--checkpoint-dir", default="", help="Override base checkpoint directory.")
    parser.add_argument("--t-final", type=float, default=3.0, help="Final time endpoint.")
    parser.add_argument("--signal-idx", default="1,2,3,4,5", help="Comma-separated active indices.")
    parser.add_argument("--beta-scales", default="1,1,1,1,1", help="Comma-separated beta scales.")
    parser.add_argument("--grid", type=int, default=400, help="Grid size for plotting.")
    parser.add_argument("--out", default="avg_beta.png", help="Output image path.")
    parser.add_argument("--plot-all", action="store_true",
                        help="Plot all P covariates in separate subplots (may be large).")
    parser.add_argument("--P", type=int, default=100, help="Number of covariates (needed if --plot-all).")

    args = parser.parse_args()

    signal_idx = _parse_list_int(args.signal_idx)
    beta_scales = _parse_list_float(args.beta_scales)
    beta_funcs = true_beta_funcs_default(scales=beta_scales)

    seeds = [args.seed_start + i for i in range(args.n_seeds)]
    t_grid = np.linspace(0.0, float(args.t_final), int(args.grid))

    beta_list = []
    loaded = []
    for seed in seeds:
        ckpt_dir = _build_checkpoint_dir(args.tag, args.checkpoint_dir, seed)
        ckpt = _ckpt_path(ckpt_dir, args.t_final)
        if not (os.path.exists(ckpt + ".json") and os.path.exists(ckpt + ".npz")):
            print(f"[skip] missing checkpoint for seed={seed}: {ckpt}")
            continue
        trainer = IncrementalVCMTrainer.load_checkpoint(ckpt)
        beta_hat = trainer.eval_beta(t_grid)  # (grid, P)
        beta_list.append(beta_hat)
        loaded.append(seed)

    if not beta_list:
        raise SystemExit("No checkpoints loaded. Check --tag/--checkpoint-dir/--t-final.")

    beta_mean = np.mean(np.stack(beta_list, axis=0), axis=0)  # (grid, P)

    if args.plot_all:
        P = int(args.P)
        cols = 4
        rows = int(np.ceil(P / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows), sharex=True)
        axes = np.atleast_1d(axes).ravel()
        for p in range(P):
            ax = axes[p]
            ax.plot(t_grid, beta_mean[:, p], label="mean beta")
            if p in signal_idx:
                idx = signal_idx.index(p) % len(beta_funcs)
                true_vals = beta_funcs[idx](t_grid)
                ax.plot(t_grid, true_vals, linestyle="--", label="true beta")
            ax.set_title(f"p={p}")
            ax.grid(alpha=0.3)
        for k in range(P, len(axes)):
            axes[k].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right")
        fig.tight_layout()
        fig.savefig(args.out, dpi=160)
        print(f"[ok] saved: {args.out} (loaded seeds: {loaded})")
        return

    n_plots = len(signal_idx)
    cols = 2 if n_plots > 1 else 1
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for i, p in enumerate(signal_idx):
        ax = axes[i]
        ax.plot(t_grid, beta_mean[:, p], label="mean beta")
        idx = i % len(beta_funcs)
        true_vals = beta_funcs[idx](t_grid)
        ax.plot(t_grid, true_vals, linestyle="--", label="true beta")
        ax.set_title(f"p={p}")
        ax.grid(alpha=0.3)
        ax.legend()
    for k in range(n_plots, len(axes)):
        axes[k].axis("off")

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"[ok] saved: {args.out} (loaded seeds: {loaded})")


if __name__ == "__main__":
    main()
