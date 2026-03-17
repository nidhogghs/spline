import argparse
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import main_1 as alg_main1
import main_2 as alg_main2


_CKPT_RE = re.compile(r"^ckpt_t(\d+)\.json$")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parse_interval(text):
    parts = [x.strip() for x in str(text).split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError("interval must be 'a,b'")
    a, b = float(parts[0]), float(parts[1])
    if a >= b:
        raise ValueError("interval must satisfy a < b")
    return a, b


def _list_ckpt_stages(result_dir):
    out = []
    if not os.path.isdir(result_dir):
        return out
    for name in os.listdir(result_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        st = int(m.group(1))
        base = os.path.join(result_dir, f"ckpt_t{st}")
        if os.path.exists(base + ".json") and os.path.exists(base + ".npz"):
            out.append(st)
    return sorted(set(out))


def _find_config_by_result_dir(result_dir):
    cfg_dir = "configs"
    if not os.path.isdir(cfg_dir):
        return None

    run_name = os.path.basename(os.path.normpath(result_dir))
    candidates = []
    for name in os.listdir(cfg_dir):
        if not name.lower().endswith(".json"):
            continue
        path = os.path.join(cfg_dir, name)
        try:
            cfg = _load_json(path)
        except Exception:
            continue
        exp = cfg.get("experiment", {}) if isinstance(cfg, dict) else {}
        ckpt_dir = str(exp.get("checkpoint_dir", "")).strip()
        if ckpt_dir and os.path.normpath(ckpt_dir) == os.path.normpath(run_name):
            candidates.append(path)

    if not candidates:
        return None
    candidates.sort()
    return candidates[0]


def _detect_algo(result_dir, algo_arg):
    if algo_arg in ("main1", "main2"):
        return algo_arg

    stages = _list_ckpt_stages(result_dir)
    if not stages:
        raise FileNotFoundError(f"no checkpoints found in: {result_dir}")

    latest = stages[-1]
    meta = _load_json(os.path.join(result_dir, f"ckpt_t{latest}.json"))
    if "local_window_units" in meta or "local_support_margin" in meta:
        return "main2"
    return "main1"


def _load_trainer(algo, ckpt_prefix):
    if algo == "main1":
        return alg_main1.IncrementalVCMTrainer.load_checkpoint(ckpt_prefix, debug=False)
    if algo == "main2":
        return alg_main2.IncrementalVCMTrainer.load_checkpoint(ckpt_prefix, debug=False)
    raise ValueError(f"unsupported algo: {algo}")


def _build_beta_funcs(cfg):
    model = cfg.get("model", {})
    beta_cfg = model.get("beta", None)
    if beta_cfg is None:
        scales = model.get("beta_scales", [1, 1, 1, 1, 1])
        return alg_main1.true_beta_funcs_default(scales=scales)
    return alg_main1.build_beta_functions_from_config(beta_cfg)


def _plot_rmse(history_path, out_path):
    if not os.path.exists(history_path):
        return False, "history file not found"

    history = _load_json(history_path)
    rows = []
    for h in history:
        if isinstance(h, dict) and ("stage" in h) and ("train_rmse" in h):
            rows.append((int(h["stage"]), float(h["train_rmse"])))
    rows.sort(key=lambda x: x[0])
    if not rows:
        return False, "no train_rmse in history"

    stages = [x[0] for x in rows]
    rmses = [x[1] for x in rows]

    plt.figure(figsize=(8, 4.8))
    plt.plot(stages, rmses, marker="o", linewidth=2)
    plt.xlabel("Stage")
    plt.ylabel("Train RMSE")
    plt.title("RMSE by Stage")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return True, "ok"


def _plot_coef(algo, cfg, result_dir, ckpt_stage, coef_interval, out_path):
    stages = _list_ckpt_stages(result_dir)
    if not stages:
        raise FileNotFoundError(f"no checkpoints found in: {result_dir}")

    if ckpt_stage <= 0:
        stage_use = stages[-1]
    else:
        if ckpt_stage not in stages:
            raise FileNotFoundError(f"checkpoint stage {ckpt_stage} not found in {result_dir}")
        stage_use = ckpt_stage

    a, b = coef_interval
    b = min(float(b), float(stage_use))
    if a >= b:
        raise ValueError(f"invalid interval after clipping: [{a}, {b}] with stage={stage_use}")

    signal_idx = [int(x) for x in cfg["model"]["signal_idx"]]
    beta_funcs = _build_beta_funcs(cfg)

    ckpt_prefix = os.path.join(result_dir, f"ckpt_t{stage_use}")
    trainer = _load_trainer(algo, ckpt_prefix)

    t_grid = np.linspace(a, b, 400)
    if algo == "main1":
        B = alg_main1.bspline_design_matrix(t_grid, trainer.knots, trainer.k)
    else:
        B = alg_main2.bspline_design_matrix(t_grid, trainer.knots, trainer.k)
    beta_hat = np.column_stack([B @ trainer.coef_blocks[p] for p in range(len(trainer.coef_blocks))])

    n_plots = len(signal_idx)
    cols = 2 if n_plots > 1 else 1
    rows = int(np.ceil(n_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.8 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for i, p in enumerate(signal_idx):
        ax = axes[i]
        ax.plot(t_grid, beta_hat[:, p], label="estimated", linewidth=2)
        truth = beta_funcs[i % len(beta_funcs)](t_grid)
        ax.plot(t_grid, truth, "--", label="true", linewidth=1.8)
        ax.set_title(f"p={p}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(n_plots, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{algo} coefficient fit (ckpt_t={stage_use}, interval=[{a},{b}])")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return stage_use


def main():
    parser = argparse.ArgumentParser(description="Visualize one single-algorithm experiment output.")
    parser.add_argument("--result-dir", required=True, help="Path to run checkpoint directory.")
    parser.add_argument("--config", default="", help="Optional config path. If empty, try auto-detect from configs/.")
    parser.add_argument("--algo", choices=["auto", "main1", "main2"], default="auto")
    parser.add_argument("--ckpt-stage", type=int, default=0, help="0 means latest checkpoint.")
    parser.add_argument("--coef-interval", default="", help="Coefficient plot interval a,b. Empty means [0,t_final].")
    parser.add_argument("--history-json", default="", help="Optional history json path override.")
    parser.add_argument("--out-dir", default="", help="Output directory. Default: result-dir.")
    args = parser.parse_args()

    result_dir = os.path.normpath(args.result_dir)
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(f"result dir not found: {result_dir}")

    cfg_path = args.config.strip()
    if not cfg_path:
        cfg_path = _find_config_by_result_dir(result_dir)
        if cfg_path is None:
            raise FileNotFoundError("cannot auto-detect config; please pass --config")
    cfg = _load_json(cfg_path)

    algo = _detect_algo(result_dir, args.algo)

    t_final = float(cfg.get("data", {}).get("t_final", 0.0))
    if args.coef_interval.strip():
        coef_interval = _parse_interval(args.coef_interval)
    else:
        if t_final <= 0:
            raise ValueError("t_final is invalid in config; pass --coef-interval explicitly")
        coef_interval = (0.0, float(t_final))

    out_dir = os.path.normpath(args.out_dir.strip()) if args.out_dir.strip() else result_dir
    os.makedirs(out_dir, exist_ok=True)

    history_path = args.history_json.strip()
    if not history_path:
        history_path = str(cfg.get("experiment", {}).get("history_json", "")).strip()
    if history_path and not os.path.isabs(history_path):
        history_path = os.path.normpath(history_path)

    out_rmse = os.path.join(out_dir, f"rmse_{algo}.png")
    out_coef = os.path.join(out_dir, f"coef_fit_{algo}.png")

    ok_rmse, msg = _plot_rmse(history_path, out_rmse) if history_path else (False, "history path empty")
    stage_use = _plot_coef(algo, cfg, result_dir, int(args.ckpt_stage), coef_interval, out_coef)

    print(f"[config] {cfg_path}")
    print(f"[algo] {algo}")
    print(f"[ckpt_stage] {stage_use}")
    if ok_rmse:
        print(f"[saved] {out_rmse}")
    else:
        print(f"[warn] rmse plot skipped: {msg}")
    print(f"[saved] {out_coef}")


if __name__ == "__main__":
    main()
