"""
多种子平均可视化：从日志中提取各 seed 的逐 stage 训练数据，
绘制均值±标准差曲线和 beta 函数拟合对比。

用法：
    # 5 种子平均，区间 90~100
    python visualize_multiseed.py \
        --config configs/longcycle_shift_beta2.json \
        --log-dir logs/longcycle_shift \
        --n-seeds 5 \
        --t-range 90,100

    # 全范围
    python visualize_multiseed.py \
        --config configs/longcycle_shift_beta2.json \
        --log-dir logs/longcycle_shift \
        --n-seeds 5
"""
import argparse
import ast
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from vcm import (
    IncrementalVCMTrainer,
    bspline_design_matrix,
    build_beta_functions_from_config,
    true_beta_funcs_default,
    _find_latest_checkpoint,
    _load_config,
    _get_cfg,
    _parse_int_list,
    _parse_float_list,
    _resolve_checkpoint_base,
)


# ─────────────────── 日志解析 ───────────────────

def parse_seed_log(log_path):
    """
    从单个 seed 的日志文件中解析逐 stage 记录。

    返回 list[dict]，每条 dict 包含:
      stage, t_end, train_rmse_cn, elapsed_sec, lambda_best, n_active_vars, ...
    """
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                rec = ast.literal_eval(line)
                if isinstance(rec, dict) and "stage" in rec:
                    records.append(rec)
            except Exception:
                continue
    return records


def parse_rmse_by_stage_from_log(log_path):
    """
    从日志末尾的 rmse_by_stage= {...} 行解析 dict{stage: rmse}。
    """
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("rmse_by_stage="):
                try:
                    return ast.literal_eval(line.split("=", 1)[1].strip())
                except Exception:
                    return {}
    return {}


def load_all_seeds(log_dir, n_seeds, seed_start=0):
    """加载所有 seed 的日志数据。"""
    all_records = []
    all_rmse_maps = []
    for i in range(seed_start, seed_start + n_seeds):
        log_path = os.path.join(log_dir, f"seed{i}.log")
        if not os.path.exists(log_path):
            print(f"[WARN] {log_path} not found, skipping", file=sys.stderr)
            continue
        recs = parse_seed_log(log_path)
        rmse_map = parse_rmse_by_stage_from_log(log_path)
        all_records.append(recs)
        all_rmse_maps.append(rmse_map)
        print(f"  seed{i}: {len(recs)} stage records, rmse_map has {len(rmse_map)} entries")
    return all_records, all_rmse_maps


# ─────────────────── 数据聚合 ───────────────────

def aggregate_metric(all_records, key, stages=None):
    """
    对所有 seed 的某个指标做逐 stage 聚合。

    返回 (stages_arr, mean_arr, std_arr)
    """
    # 收集每个 seed 在每个 stage 的值
    seed_data = []
    for recs in all_records:
        stage_val = {}
        for r in recs:
            s = int(r["stage"])
            v = r.get(key)
            if v is not None:
                stage_val[s] = float(v)
        seed_data.append(stage_val)

    # 找到所有 stage（取所有 seed 的交集）
    if not seed_data:
        return np.array([]), np.array([]), np.array([])

    all_stages = set(seed_data[0].keys())
    for sd in seed_data[1:]:
        all_stages &= sd.keys()
    all_stages = sorted(all_stages)

    if stages is not None:
        all_stages = [s for s in all_stages if stages[0] <= s <= stages[1]]

    if not all_stages:
        return np.array([]), np.array([]), np.array([])

    stages_arr = np.array(all_stages)
    values = np.array([[sd[s] for s in all_stages] for sd in seed_data])  # (n_seeds, n_stages)
    mean_arr = values.mean(axis=0)
    std_arr = values.std(axis=0)

    return stages_arr, mean_arr, std_arr


def aggregate_rmse_from_maps(all_rmse_maps, stages=None):
    """从 rmse_by_stage 字典做聚合。"""
    if not all_rmse_maps:
        return np.array([]), np.array([]), np.array([])

    all_stages = set(all_rmse_maps[0].keys())
    for rm in all_rmse_maps[1:]:
        all_stages &= rm.keys()
    all_stages = sorted(all_stages)

    if stages is not None:
        all_stages = [s for s in all_stages if stages[0] <= s <= stages[1]]

    if not all_stages:
        return np.array([]), np.array([]), np.array([])

    stages_arr = np.array(all_stages)
    values = np.array([[rm[s] for s in all_stages] for rm in all_rmse_maps])
    mean_arr = values.mean(axis=0)
    std_arr = values.std(axis=0)

    return stages_arr, mean_arr, std_arr


# ─────────────────── 绘图函数 ───────────────────

def plot_metric_band(ax, stages, mean, std, label, color, ylabel="", title=""):
    """绘制均值±标准差的带状图"""
    ax.plot(stages, mean, "-", color=color, linewidth=2, label=f"{label} (mean)")
    ax.fill_between(stages, mean - std, mean + std, alpha=0.2, color=color, label=f"±1 std")
    ax.set_xlabel("Stage", fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 标注末尾均值
    if len(mean) > 0:
        ax.annotate(f"{mean[-1]:.4f}±{std[-1]:.4f}",
                    (stages[-1], mean[-1]),
                    textcoords="offset points", xytext=(-80, 10),
                    fontsize=9, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))


def plot_all_seeds_lines(ax, all_records, key, stages_range=None, title="", ylabel=""):
    """绘制每个 seed 的独立曲线（半透明），加上均值粗线"""
    colors_seed = ["#90CAF9", "#EF9A9A", "#A5D6A7", "#FFCC80", "#CE93D8"]

    all_stage_vals = []
    for idx, recs in enumerate(all_records):
        stage_val = {}
        for r in recs:
            s = int(r["stage"])
            v = r.get(key)
            if v is not None:
                stage_val[s] = float(v)
        all_stage_vals.append(stage_val)

        stages_s = sorted(stage_val.keys())
        if stages_range:
            stages_s = [s for s in stages_s if stages_range[0] <= s <= stages_range[1]]
        if stages_s:
            vals = [stage_val[s] for s in stages_s]
            c = colors_seed[idx % len(colors_seed)]
            ax.plot(stages_s, vals, "-", color=c, alpha=0.4, linewidth=1, label=f"seed{idx}")

    # 均值
    if all_stage_vals:
        common_stages = set(all_stage_vals[0].keys())
        for sv in all_stage_vals[1:]:
            common_stages &= sv.keys()
        common_stages = sorted(common_stages)
        if stages_range:
            common_stages = [s for s in common_stages if stages_range[0] <= s <= stages_range[1]]
        if common_stages:
            mean_vals = np.mean([[sv[s] for s in common_stages] for sv in all_stage_vals], axis=0)
            ax.plot(common_stages, mean_vals, "-", color="#D32F2F", linewidth=2.5, label="mean")

    ax.set_xlabel("Stage", fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)


def plot_beta_multiseed_avg(trainers, beta_funcs, signal_idx, ax_list, 
                             t_range=None, n_grid=500):
    """
    绘制多 seed 的 beta 估计的均值±标准差 vs 真实值。
    
    trainers: list of IncrementalVCMTrainer（各 seed）
    """
    t_lo = t_range[0] if t_range else 0
    t_hi = t_range[1] if t_range else float(trainers[0].t_end)
    t_grid = np.linspace(t_lo, t_hi, n_grid)

    colors_true = ["#1976D2", "#D32F2F", "#388E3C", "#F57C00", "#7B1FA2"]
    colors_hat = ["#64B5F6", "#EF5350", "#66BB6A", "#FFB74D", "#BA68C8"]

    for idx, (p, ax) in enumerate(zip(signal_idx, ax_list)):
        beta_true = beta_funcs[idx % len(beta_funcs)](t_grid)

        # 收集各 seed 的 beta_hat
        beta_hats = []
        for trainer in trainers:
            B_grid = bspline_design_matrix(t_grid, trainer.knots, trainer.k)
            coef_mat = np.stack(trainer.coef_blocks, axis=0)
            beta_hat = B_grid @ coef_mat[p]
            beta_hats.append(beta_hat)

        beta_hats = np.array(beta_hats)  # (n_seeds, n_grid)
        beta_mean = beta_hats.mean(axis=0)
        beta_std = beta_hats.std(axis=0)

        c_true = colors_true[idx % len(colors_true)]
        c_hat = colors_hat[idx % len(colors_hat)]

        ax.plot(t_grid, beta_true, "-", color=c_true, linewidth=2.5, label=f"β_{p}(t) true")
        ax.plot(t_grid, beta_mean, "--", color=c_hat, linewidth=2, label=f"β_{p}(t) mean est")
        ax.fill_between(t_grid, beta_mean - beta_std, beta_mean + beta_std,
                        alpha=0.2, color=c_hat, label="±1 std")

        ax.set_title(f"β_{p}(t)", fontsize=12, fontweight="bold")
        ax.set_xlabel("t", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # RMSE（均值 beta 的 RMSE）
        rmse_mean = float(np.sqrt(np.mean((beta_true - beta_mean) ** 2)))
        # 各 seed RMSE 的均值
        rmse_per_seed = [float(np.sqrt(np.mean((beta_true - bh) ** 2))) for bh in beta_hats]
        rmse_avg = np.mean(rmse_per_seed)
        rmse_std_val = np.std(rmse_per_seed)
        ax.text(0.02, 0.95,
                f"RMSE(mean)={rmse_mean:.4f}\nRMSE(avg)={rmse_avg:.4f}±{rmse_std_val:.4f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))


def plot_beta_error_heatmap_multiseed(trainers, beta_funcs, signal_idx, ax, 
                                       t_range=None, n_grid=500):
    """多 seed 平均误差热力图"""
    t_lo = t_range[0] if t_range else 0
    t_hi = t_range[1] if t_range else float(trainers[0].t_end)
    t_grid = np.linspace(t_lo, t_hi, n_grid)

    all_errors = []  # list of (n_signals, n_grid)
    for trainer in trainers:
        B_grid = bspline_design_matrix(t_grid, trainer.knots, trainer.k)
        coef_mat = np.stack(trainer.coef_blocks, axis=0)
        errors = []
        for idx, p in enumerate(signal_idx):
            beta_true = beta_funcs[idx % len(beta_funcs)](t_grid)
            beta_hat = B_grid @ coef_mat[p]
            errors.append(beta_true - beta_hat)
        all_errors.append(np.array(errors))

    mean_error = np.mean(all_errors, axis=0)  # (n_signals, n_grid)

    vmax = np.percentile(np.abs(mean_error), 95)
    im = ax.imshow(mean_error, aspect="auto", cmap="RdBu_r",
                   extent=[t_lo, t_hi, len(signal_idx) - 0.5, -0.5],
                   vmin=-vmax, vmax=vmax)
    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel("Signal Variable", fontsize=12)
    ax.set_yticks(range(len(signal_idx)))
    ax.set_yticklabels([f"β_{p}" for p in signal_idx], fontsize=10)
    ax.set_title(f"Mean β(t) Error (n={len(trainers)} seeds)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Error")


# ─────────────────── 主流程 ───────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-seed average visualization.")
    parser.add_argument("--config", default="", help="Config JSON (same as vcm.py).")
    parser.add_argument("--log-dir", required=True, help="Directory containing seed*.log files.")
    parser.add_argument("--checkpoint-base", default="",
                        help="Base checkpoint dir (without _seedN suffix). Auto-detected from config if not set.")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--t-range", default="",
                        help="Comma-separated t range, e.g. '90,100'.")
    parser.add_argument("--output-dir", default="",
                        help="Output dir for figures. Default: auto (next to checkpoint).")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    # --- 解析配置 ---
    if args.config:
        cfg = _load_config(args.config)
    else:
        cfg = {}

    signal_idx_cfg = _get_cfg(cfg, "model", "signal_idx", "1,2,3,4,5")
    signal_idx = _parse_int_list(signal_idx_cfg)

    beta_cfg = _get_cfg(cfg, "model", "beta", None)
    if beta_cfg is None:
        beta_scales_cfg = _get_cfg(cfg, "model", "beta_scales", "1,1,1,1,1")
        beta_funcs = true_beta_funcs_default(_parse_float_list(beta_scales_cfg))
    else:
        beta_funcs = build_beta_functions_from_config(beta_cfg)

    # --- 解析 t-range ---
    t_range = None
    stages_range = None
    if args.t_range:
        parts = args.t_range.split(",")
        t_range = (float(parts[0]), float(parts[1]))
        stages_range = (int(parts[0]), int(parts[1]))
        print(f"t-range: [{t_range[0]}, {t_range[1]}]")

    # --- 加载日志数据 ---
    print(f"Loading logs from {args.log_dir} ({args.n_seeds} seeds)...")
    all_records, all_rmse_maps = load_all_seeds(args.log_dir, args.n_seeds, args.seed_start)
    n_seeds_loaded = len(all_records)
    print(f"Loaded {n_seeds_loaded} seeds")

    if n_seeds_loaded == 0:
        print("[ERROR] No seed data loaded!", file=sys.stderr)
        sys.exit(1)

    # --- 确定 checkpoint 目录（用于加载 trainer 做 beta 可视化） ---
    checkpoint_root = _get_cfg(cfg, "experiment", "checkpoint_root", "checkpoints")
    if args.checkpoint_base:
        ckpt_base = args.checkpoint_base
    else:
        cfg_ckpt = _get_cfg(cfg, "experiment", "checkpoint_dir", "")
        if cfg_ckpt:
            ckpt_base = _resolve_checkpoint_base(cfg_ckpt, root=checkpoint_root)
        else:
            ckpt_base = ""

    # 尝试加载各 seed 的 trainer（如果有独立目录的话）
    trainers = []
    for i in range(args.seed_start, args.seed_start + args.n_seeds):
        # 先尝试 _seedN 目录
        seed_dir = f"{ckpt_base}_seed{i}"
        if not os.path.isdir(seed_dir):
            # 回退到共享目录（所有 seed 写到同一目录的情况）
            seed_dir = ckpt_base
        t_latest = _find_latest_checkpoint(seed_dir) if os.path.isdir(seed_dir) else None
        if t_latest is not None:
            try:
                trainer = IncrementalVCMTrainer.load_checkpoint(
                    os.path.join(seed_dir, f"ckpt_t{t_latest}"))
                trainers.append(trainer)
            except Exception as e:
                print(f"  [WARN] Failed to load trainer from {seed_dir}: {e}")
    print(f"Loaded {len(trainers)} trainers for beta visualization")

    # --- 输出目录 ---
    if args.output_dir:
        out_dir = args.output_dir
    elif ckpt_base:
        out_dir = ckpt_base
    else:
        out_dir = args.log_dir
    os.makedirs(out_dir, exist_ok=True)

    range_suffix = f"_t{int(t_range[0])}-{int(t_range[1])}" if t_range else ""
    saved_files = []

    # ========== 图1: 训练指标总览 (2×2) ==========
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))
    fig1.suptitle(
        f"Multi-Seed Training Summary ({n_seeds_loaded} seeds){' [t∈' + str(stages_range) + ']' if stages_range else ''}",
        fontsize=15, fontweight="bold"
    )

    # RMSE 均值±std
    s, m, sd = aggregate_metric(all_records, "train_rmse_cn", stages_range)
    if len(s) == 0:
        # 尝试 train_rmse
        s, m, sd = aggregate_metric(all_records, "train_rmse", stages_range)
    if len(s) > 0:
        plot_metric_band(axes1[0, 0], s, m, sd, "RMSE", "#2196F3",
                         ylabel="RMSE", title="RMSE by Stage (mean ± std)")
    else:
        axes1[0, 0].text(0.5, 0.5, "No RMSE data", ha="center", va="center", transform=axes1[0, 0].transAxes)

    # Lambda
    s, m, sd = aggregate_metric(all_records, "lambda_best", stages_range)
    if len(s) > 0:
        axes1[0, 1].semilogy(s, m, "-", color="#FF9800", linewidth=2, label="λ_best (mean)")
        axes1[0, 1].fill_between(s, np.maximum(m - sd, 1e-6), m + sd, alpha=0.2, color="#FF9800")
        axes1[0, 1].set_xlabel("Stage", fontsize=12)
        axes1[0, 1].set_ylabel("λ (log scale)", fontsize=12)
        axes1[0, 1].set_title("Selected λ by Stage (mean ± std)", fontsize=13, fontweight="bold")
        axes1[0, 1].grid(True, alpha=0.3)
        axes1[0, 1].legend(fontsize=9)
    else:
        axes1[0, 1].text(0.5, 0.5, "No lambda data", ha="center", va="center", transform=axes1[0, 1].transAxes)

    # 每个 seed 的 RMSE 独立曲线 + 均值
    plot_all_seeds_lines(axes1[1, 0], all_records, "train_rmse_cn", stages_range,
                         title="Per-Seed RMSE (individual + mean)", ylabel="RMSE")

    # Elapsed time
    s, m, sd = aggregate_metric(all_records, "elapsed_sec", stages_range)
    if len(s) > 0:
        plot_metric_band(axes1[1, 1], s, m, sd, "Time", "#9C27B0",
                         ylabel="Seconds", title="Time per Stage (mean ± std)")
    else:
        axes1[1, 1].text(0.5, 0.5, "No timing data", ha="center", va="center", transform=axes1[1, 1].transAxes)

    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = os.path.join(out_dir, f"viz_multiseed_1_summary{range_suffix}.png")
    fig1.savefig(p1, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    saved_files.append(p1)
    plt.close(fig1)

    # ========== 图2: Beta 函数拟合对比（多 seed 均值） ==========
    if trainers:
        n_signals = len(signal_idx)
        n_cols_beta = min(3, n_signals)
        n_rows_beta = (n_signals + n_cols_beta - 1) // n_cols_beta
        fig2, axes2 = plt.subplots(n_rows_beta, n_cols_beta,
                                    figsize=(6 * n_cols_beta, 4.5 * n_rows_beta),
                                    squeeze=False)
        range_label = f" (t∈[{t_range[0]:.0f},{t_range[1]:.0f}])" if t_range else ""
        seed_note = f" — {len(trainers)} seed(s)" if len(trainers) < n_seeds_loaded else f" — {len(trainers)} seeds"
        fig2.suptitle(f"Beta Functions: True vs Estimated (mean ± std){range_label}{seed_note}",
                      fontsize=14, fontweight="bold")

        ax_beta_list = []
        for i in range(n_signals):
            ax_beta_list.append(axes2[i // n_cols_beta, i % n_cols_beta])
        for i in range(n_signals, n_rows_beta * n_cols_beta):
            axes2[i // n_cols_beta, i % n_cols_beta].set_visible(False)

        if len(trainers) > 1:
            plot_beta_multiseed_avg(trainers, beta_funcs, signal_idx, ax_beta_list, t_range=t_range)
        else:
            # 只有一个 trainer，退化为单 seed 模式
            from visualize import plot_beta_functions
            t_final = float(trainers[0].t_end)
            plot_beta_functions(trainers[0], beta_funcs, signal_idx, t_final, ax_beta_list, t_range=t_range)

        fig2.tight_layout(rect=[0, 0, 1, 0.93])
        p2 = os.path.join(out_dir, f"viz_multiseed_2_beta{range_suffix}.png")
        fig2.savefig(p2, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        saved_files.append(p2)
        plt.close(fig2)

        # ========== 图3: Error Heatmap ==========
        fig3, axes3 = plt.subplots(1, 2, figsize=(16, 5))
        range_label = f" (t∈[{t_range[0]:.0f},{t_range[1]:.0f}])" if t_range else ""
        fig3.suptitle(f"Mean Estimation Error & Sparsity{range_label}", fontsize=15, fontweight="bold")

        plot_beta_error_heatmap_multiseed(trainers, beta_funcs, signal_idx, axes3[0], t_range=t_range)

        # Sparsity (from last trainer as representative)
        from visualize import plot_sparsity_pattern
        plot_sparsity_pattern(trainers[-1], signal_idx, axes3[1])

        fig3.tight_layout(rect=[0, 0, 1, 0.92])
        p3 = os.path.join(out_dir, f"viz_multiseed_3_error{range_suffix}.png")
        fig3.savefig(p3, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        saved_files.append(p3)
        plt.close(fig3)
    else:
        print("[INFO] No trainers loaded — skipping beta function plots.")
        print("       (All seeds wrote to same checkpoint dir; only last seed's data survives.)")

    # --- 输出 ---
    print()
    for fp in saved_files:
        print(f"[OK] Saved: {fp}")


if __name__ == "__main__":
    main()
