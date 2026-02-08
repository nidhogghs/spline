import argparse
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main as alg_main
import main_1 as alg_main1
import main_2 as alg_main2


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _plot_rmse_compare(summary, out_path):
    agg = summary["aggregate"]
    stages = [int(s) for s in agg["stages"]]
    main_mean = [agg["main_mean"].get(str(s), agg["main_mean"].get(s)) for s in stages]
    main1_mean = [agg["main1_mean"].get(str(s), agg["main1_mean"].get(s)) for s in stages]
    main2_mean = [agg["main2_mean"].get(str(s), agg["main2_mean"].get(s)) for s in stages]

    plt.figure(figsize=(9, 5))
    plt.plot(stages, main_mean, label="main.py", linewidth=2)
    plt.plot(stages, main1_mean, label="main_1.py", linewidth=2)
    plt.plot(stages, main2_mean, label="main_2.py", linewidth=2)
    plt.xlabel("Stage")
    plt.ylabel("Mean Train RMSE (across seeds)")
    plt.title("RMSE Comparison: main vs main_1 vs main_2")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _load_trainer(algo_name, ckpt_prefix):
    if algo_name == "main":
        return alg_main.IncrementalVCMTrainer.load_checkpoint(ckpt_prefix)
    if algo_name == "main1":
        return alg_main1.IncrementalVCMTrainer.load_checkpoint(ckpt_prefix, debug=False)
    if algo_name == "main2":
        return alg_main2.IncrementalVCMTrainer.load_checkpoint(ckpt_prefix, debug=False)
    raise ValueError(f"Unknown algo_name: {algo_name}")


def _eval_beta_any(algo_name, trainer, t_grid):
    if hasattr(trainer, "eval_beta"):
        return trainer.eval_beta(t_grid)
    if algo_name == "main":
        B = alg_main.bspline_design_matrix(t_grid, trainer.knots, trainer.k)
    elif algo_name == "main1":
        B = alg_main1.bspline_design_matrix(t_grid, trainer.knots, trainer.k)
    elif algo_name == "main2":
        B = alg_main2.bspline_design_matrix(t_grid, trainer.knots, trainer.k)
    else:
        raise ValueError(f"Unknown algo_name: {algo_name}")
    return np.column_stack([B @ trainer.coef_blocks[p] for p in range(len(trainer.coef_blocks))])


def _collect_seed_checkpoints(summary, algo_name, run_root, t_final):
    ok_seeds = [int(r["seed"]) for r in summary["results"] if r.get("ok", False)]
    prefixes = []
    for seed in ok_seeds:
        prefix = os.path.join(run_root, algo_name, f"seed{seed}", f"ckpt_t{int(t_final)}")
        if os.path.exists(prefix + ".json") and os.path.exists(prefix + ".npz"):
            prefixes.append(prefix)
    return prefixes


def _plot_coef_interval_for_algo(
    algo_name,
    summary,
    run_root,
    t_final,
    t_start,
    t_end,
    signal_idx,
    beta_scales,
    out_path,
):
    prefixes = _collect_seed_checkpoints(summary, algo_name, run_root, t_final)
    if not prefixes:
        raise FileNotFoundError(f"No checkpoints for {algo_name} at t={t_final}")

    t_grid = np.linspace(float(t_start), float(t_end), 400)
    beta_list = []
    for pfx in prefixes:
        tr = _load_trainer(algo_name, pfx)
        beta_list.append(_eval_beta_any(algo_name, tr, t_grid))
    beta_arr = np.stack(beta_list, axis=0)  # (n_seed, grid, P)
    beta_mean = beta_arr.mean(axis=0)
    beta_std = beta_arr.std(axis=0, ddof=1) if beta_arr.shape[0] > 1 else np.zeros_like(beta_mean)

    true_funcs = alg_main.true_beta_funcs_default(scales=beta_scales)

    n_plots = len(signal_idx)
    cols = 2 if n_plots > 1 else 1
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for i, p in enumerate(signal_idx):
        ax = axes[i]
        est = beta_mean[:, p]
        ax.plot(t_grid, est, label="estimated mean", linewidth=2)
        if beta_arr.shape[0] > 1:
            band = 1.96 * beta_std[:, p] / np.sqrt(beta_arr.shape[0])
            ax.fill_between(t_grid, est - band, est + band, alpha=0.2, label="~95% mean band")

        true_idx = i % len(true_funcs)
        truth = true_funcs[true_idx](t_grid)
        ax.plot(t_grid, truth, "--", label="true beta", linewidth=1.8)
        ax.set_title(f"{algo_name}: p={p}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{algo_name} coefficient fit on [{t_start}, {t_end}] (n_seeds={len(prefixes)})")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize benchmark results for main/main_1/main_2")
    parser.add_argument("--config", required=True, help="benchmark config path")
    parser.add_argument("--summary-json", default="", help="optional summary path override")
    parser.add_argument("--coef-interval", default="90,100", help="coefficient plot interval a,b")
    args = parser.parse_args()

    cfg = _load_json(args.config)
    exp = cfg["experiment"]
    model = cfg["model"]
    data = cfg["data"]

    run_root = exp["run_root"]
    summary_json = args.summary_json or exp.get("out_json", os.path.join(run_root, "benchmark_summary.json"))
    summary = _load_json(summary_json)

    os.makedirs(run_root, exist_ok=True)
    out_rmse = os.path.join(run_root, "rmse_compare_three.png")
    _plot_rmse_compare(summary, out_rmse)

    ab = [float(x) for x in args.coef_interval.split(",")]
    if len(ab) != 2:
        raise ValueError("--coef-interval must be 'a,b'")
    t_start, t_end = ab

    signal_idx = [int(x) for x in model["signal_idx"]]
    beta_scales = [float(x) for x in model.get("beta_scales", [1, 1, 1, 1, 1])]
    t_final = int(data["t_final"])
    if t_end > t_final:
        t_end = float(t_final)
    if t_start < 0:
        t_start = 0.0
    if t_start >= t_end:
        t_start = max(0.0, t_end - 10.0)

    out_main = os.path.join(run_root, f"coef_fit_main_{int(t_start)}_{int(t_end)}.png")
    out_main1 = os.path.join(run_root, f"coef_fit_main1_{int(t_start)}_{int(t_end)}.png")
    out_main2 = os.path.join(run_root, f"coef_fit_main2_{int(t_start)}_{int(t_end)}.png")

    for algo_name, outp in [
        ("main", out_main),
        ("main1", out_main1),
        ("main2", out_main2),
    ]:
        try:
            _plot_coef_interval_for_algo(
                algo_name, summary, run_root, t_final, t_start, t_end, signal_idx, beta_scales, outp
            )
            print(f"[saved] {outp}")
        except FileNotFoundError as e:
            print(f"[warn] skip {algo_name} coef plot: {e}")

    print(f"[saved] {out_rmse}")


if __name__ == "__main__":
    main()
