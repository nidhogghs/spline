"""
可视化 main_1.py 增量算法运行结果。

用法：
    python visualize_main1_result.py --config configs/main1_t20_p100_n100_simple5_server.json
    python visualize_main1_result.py --checkpoint-dir checkpoints/main1_t20_p100_n100_simple5_server
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 复用 vcm.py 中的核心函数
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


def load_history(history_json):
    """加载 history.json"""
    with open(history_json, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_rmse_curve(history, ax, title="RMSE by Stage (CN)"):
    """绘制 RMSE 随 stage 变化的曲线"""
    stages = []
    rmses = []
    for h in history:
        if isinstance(h, dict) and "stage" in h:
            rmse = h.get("train_rmse_cn", h.get("train_rmse", None))
            if rmse is not None:
                stages.append(int(h["stage"]))
                rmses.append(float(rmse))

    if not stages:
        ax.text(0.5, 0.5, "No RMSE data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.plot(stages, rmses, "o-", color="#2196F3", linewidth=2, markersize=5, label="RMSE (CN)")
    ax.set_xlabel("Stage", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # 标注首尾值
    ax.annotate(f"{rmses[0]:.4f}", (stages[0], rmses[0]), textcoords="offset points",
                xytext=(10, 10), fontsize=9, color="#F44336")
    ax.annotate(f"{rmses[-1]:.4f}", (stages[-1], rmses[-1]), textcoords="offset points",
                xytext=(10, -15), fontsize=9, color="#4CAF50")


def plot_lambda_curve(history, ax, title="Selected λ by Stage"):
    """绘制 lambda 随 stage 变化的曲线"""
    stages = []
    lambdas = []
    for h in history:
        if isinstance(h, dict) and "stage" in h and "lambda_best" in h:
            stages.append(int(h["stage"]))
            lambdas.append(float(h["lambda_best"]))

    if not stages:
        ax.text(0.5, 0.5, "No lambda data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.semilogy(stages, lambdas, "s-", color="#FF9800", linewidth=2, markersize=5, label="λ_best")
    ax.set_xlabel("Stage", fontsize=12)
    ax.set_ylabel("λ (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)


def plot_timing_curve(history, ax, title="Time per Stage (sec)"):
    """绘制每个 stage 耗时曲线"""
    stages = []
    times = []
    for h in history:
        if isinstance(h, dict) and "stage" in h and "elapsed_sec" in h:
            stages.append(int(h["stage"]))
            times.append(float(h["elapsed_sec"]))

    if not stages:
        ax.text(0.5, 0.5, "No timing data", ha="center", va="center", transform=ax.transAxes)
        return

    ax.bar(stages, times, color="#9C27B0", alpha=0.7, label="Elapsed (s)")
    ax.set_xlabel("Stage", fontsize=12)
    ax.set_ylabel("Seconds", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=10)

    # 总时间
    total = sum(times)
    ax.text(0.98, 0.95, f"Total: {total:.1f}s", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))


def plot_cn_data_count(history, ax, title="CN Data Points per Stage"):
    """绘制每个 stage 的 CN 数据量"""
    stages = []
    n_cn = []
    n_o = []
    n_c = []
    n_n = []
    for h in history:
        if isinstance(h, dict) and "stage" in h:
            stage = int(h["stage"])
            if "n_cn_data" in h:
                stages.append(stage)
                n_cn.append(int(h["n_cn_data"]))
                n_o.append(int(h.get("num_O", 0)))
                n_c.append(int(h.get("num_C", 0)))
                n_n.append(int(h.get("num_N", 0)))

    if not stages:
        ax.text(0.5, 0.5, "No CN data info", ha="center", va="center", transform=ax.transAxes)
        return

    ax.plot(stages, n_cn, "D-", color="#E91E63", linewidth=2, markersize=5, label="n_cn_data")
    ax.set_xlabel("Stage", fontsize=12)
    ax.set_ylabel("Data Points", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # 第二坐标轴显示 O/C/N basis 数量
    ax2 = ax.twinx()
    ax2.plot(stages, n_o, "^--", color="#607D8B", alpha=0.6, markersize=4, label="num_O")
    ax2.plot(stages, n_c, "v--", color="#00BCD4", alpha=0.6, markersize=4, label="num_C")
    ax2.plot(stages, n_n, "o--", color="#8BC34A", alpha=0.6, markersize=4, label="num_N")
    ax2.set_ylabel("Basis Count", fontsize=11, color="#607D8B")
    ax2.legend(loc="center right", fontsize=9)


def plot_beta_functions(trainer, beta_funcs, signal_idx, t_final, ax_list, n_grid=500):
    """
    绘制每个信号变量的 beta(t) 拟合对比：真实 vs 估计。

    参数：
        trainer: 训练好的 IncrementalVCMTrainer
        beta_funcs: 真实 beta 函数列表
        signal_idx: 信号变量索引
        t_final: 时间终点
        ax_list: Axes 列表，每个信号变量一个
        n_grid: 网格点数
    """
    t_grid = np.linspace(0, float(t_final), n_grid)
    B_grid = bspline_design_matrix(t_grid, trainer.knots, trainer.k)
    coef_mat = np.stack(trainer.coef_blocks, axis=0)  # (P, m)

    colors_true = ["#1976D2", "#D32F2F", "#388E3C", "#F57C00", "#7B1FA2"]
    colors_hat = ["#64B5F6", "#EF5350", "#66BB6A", "#FFB74D", "#BA68C8"]

    for idx, (p, ax) in enumerate(zip(signal_idx, ax_list)):
        # 真实 beta
        beta_true = beta_funcs[idx % len(beta_funcs)](t_grid)
        # 估计 beta
        beta_hat = B_grid @ coef_mat[p]

        c_true = colors_true[idx % len(colors_true)]
        c_hat = colors_hat[idx % len(colors_hat)]

        ax.plot(t_grid, beta_true, "-", color=c_true, linewidth=2, label=f"β_{p}(t) true")
        ax.plot(t_grid, beta_hat, "--", color=c_hat, linewidth=1.5, label=f"β_{p}(t) est")
        ax.set_title(f"β_{p}(t)", fontsize=12, fontweight="bold")
        ax.set_xlabel("t", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 计算全局 RMSE
        rmse_beta = float(np.sqrt(np.mean((beta_true - beta_hat) ** 2)))
        ax.text(0.02, 0.95, f"RMSE={rmse_beta:.4f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.8))


def plot_beta_error_heatmap(trainer, beta_funcs, signal_idx, t_final, ax, n_grid=500):
    """绘制各信号变量 beta(t) 误差的热力图 (t × signal_idx)"""
    t_grid = np.linspace(0, float(t_final), n_grid)
    B_grid = bspline_design_matrix(t_grid, trainer.knots, trainer.k)
    coef_mat = np.stack(trainer.coef_blocks, axis=0)

    errors = []
    for idx, p in enumerate(signal_idx):
        beta_true = beta_funcs[idx % len(beta_funcs)](t_grid)
        beta_hat = B_grid @ coef_mat[p]
        errors.append(beta_true - beta_hat)

    error_mat = np.array(errors)  # (num_signals, n_grid)

    im = ax.imshow(error_mat, aspect="auto", cmap="RdBu_r",
                   extent=[0, float(t_final), len(signal_idx) - 0.5, -0.5],
                   vmin=-np.percentile(np.abs(error_mat), 95),
                   vmax=np.percentile(np.abs(error_mat), 95))
    ax.set_xlabel("t", fontsize=12)
    ax.set_ylabel("Signal Variable", fontsize=12)
    ax.set_yticks(range(len(signal_idx)))
    ax.set_yticklabels([f"β_{p}" for p in signal_idx], fontsize=10)
    ax.set_title("β(t) Error Heatmap (True - Estimated)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Error")


def plot_sparsity_pattern(trainer, signal_idx, ax, title="Coefficient Sparsity"):
    """绘制系数矩阵的稀疏模式"""
    coef_mat = np.stack(trainer.coef_blocks, axis=0)  # (P, m)
    P, m = coef_mat.shape

    # 按变量的 L2 范数排序
    norms = np.linalg.norm(coef_mat, axis=1)
    active = norms > 1e-8

    ax.barh(range(P), norms, color=["#4CAF50" if a else "#E0E0E0" for a in active], height=0.8)
    ax.set_xlabel("||coef||₂", fontsize=12)
    ax.set_ylabel("Variable Index", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # 标注 signal 变量
    for p in signal_idx:
        if p < P:
            ax.axhline(y=p, color="#F44336", linestyle="--", alpha=0.3, linewidth=0.8)

    n_active = int(active.sum())
    n_signal = len(signal_idx)
    signal_detected = sum(1 for p in signal_idx if p < P and active[p])
    ax.text(0.98, 0.02,
            f"Active: {n_active}/{P}\nSignal detected: {signal_detected}/{n_signal}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))


def main():
    parser = argparse.ArgumentParser(description="Visualize incremental VCM results.")
    parser.add_argument("--config", default="", help="Path to JSON config file (same as vcm.py).")
    parser.add_argument("--checkpoint-dir", default="", help="Override checkpoint directory.")
    parser.add_argument("--history-json", default="", help="Override history JSON path.")
    parser.add_argument("--output", default="", help="Output PNG path (default: auto-generated).")
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI.")
    args = parser.parse_args()

    # --- 解析配置 ---
    if args.config:
        cfg = _load_config(args.config)
    else:
        cfg = {}

    checkpoint_root = _get_cfg(cfg, "experiment", "checkpoint_root", "checkpoints")
    checkpoint_dir_cfg = _get_cfg(cfg, "experiment", "checkpoint_dir", args.checkpoint_dir)

    if checkpoint_dir_cfg:
        checkpoint_dir = _resolve_checkpoint_base(checkpoint_dir_cfg, root=checkpoint_root)
    elif args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
    else:
        raise ValueError("Must provide --config or --checkpoint-dir.")

    history_json = args.history_json or _get_cfg(cfg, "experiment", "history_json", "")
    if not history_json:
        history_json = os.path.join(checkpoint_dir, "history.json")

    # --- 加载数据 ---
    # 加载 trainer
    t_latest = _find_latest_checkpoint(checkpoint_dir)
    if t_latest is None:
        print(f"[ERROR] No checkpoints found in {checkpoint_dir}", file=sys.stderr)
        sys.exit(1)
    trainer = IncrementalVCMTrainer.load_checkpoint(
        os.path.join(checkpoint_dir, f"ckpt_t{t_latest}")
    )
    print(f"Loaded checkpoint: t_end={trainer.t_end}, P={trainer.P}, "
          f"num_basis={len(trainer.knots) - (trainer.k + 1)}")

    # 加载 history
    if os.path.exists(history_json):
        history = load_history(history_json)
        print(f"Loaded history: {len(history)} entries from {history_json}")
    else:
        print(f"[WARN] History file not found: {history_json}")
        history = []

    # --- 解析 beta 函数和 signal_idx ---
    signal_idx_cfg = _get_cfg(cfg, "model", "signal_idx", "1,2,3,4,5")
    signal_idx = _parse_int_list(signal_idx_cfg)

    beta_cfg = _get_cfg(cfg, "model", "beta", None)
    if beta_cfg is None:
        beta_scales_cfg = _get_cfg(cfg, "model", "beta_scales", "1,1,1,1,1")
        beta_funcs = true_beta_funcs_default(_parse_float_list(beta_scales_cfg))
    else:
        beta_funcs = build_beta_functions_from_config(beta_cfg)

    t_final = float(trainer.t_end)
    n_signals = len(signal_idx)

    out_dir = checkpoint_dir if not args.output else os.path.dirname(args.output) or "."
    os.makedirs(out_dir, exist_ok=True)
    saved_files = []

    # ========== 图1: 训练指标总览 (2x2) ==========
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))
    fig1.suptitle(
        f"Incremental VCM Training Summary\n"
        f"t_final={int(t_final)}, P={trainer.P}, n_basis={len(trainer.knots) - (trainer.k + 1)}",
        fontsize=15, fontweight="bold"
    )

    if history:
        plot_rmse_curve(history, axes1[0, 0])
        plot_lambda_curve(history, axes1[0, 1])
        plot_timing_curve(history, axes1[1, 0])
        plot_cn_data_count(history, axes1[1, 1])
    else:
        for ax in axes1.flat:
            ax.text(0.5, 0.5, "No history data", ha="center", va="center", transform=ax.transAxes)

    fig1.tight_layout(rect=[0, 0, 1, 0.93])
    p1 = os.path.join(out_dir, "viz_1_training_summary.png")
    fig1.savefig(p1, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    saved_files.append(p1)
    plt.close(fig1)

    # ========== 图2: Beta 函数拟合对比 ==========
    n_cols_beta = min(3, n_signals)
    n_rows_beta = (n_signals + n_cols_beta - 1) // n_cols_beta
    fig2, axes2 = plt.subplots(n_rows_beta, n_cols_beta,
                                figsize=(6 * n_cols_beta, 4.5 * n_rows_beta),
                                squeeze=False)
    fig2.suptitle("Beta Functions: True vs Estimated", fontsize=15, fontweight="bold")

    ax_beta_list = []
    for i in range(n_signals):
        row = i // n_cols_beta
        col = i % n_cols_beta
        ax_beta_list.append(axes2[row, col])

    # 隐藏多余的子图
    for i in range(n_signals, n_rows_beta * n_cols_beta):
        row = i // n_cols_beta
        col = i % n_cols_beta
        axes2[row, col].set_visible(False)

    plot_beta_functions(trainer, beta_funcs, signal_idx, t_final, ax_beta_list)

    fig2.tight_layout(rect=[0, 0, 1, 0.93])
    p2 = os.path.join(out_dir, "viz_2_beta_functions.png")
    fig2.savefig(p2, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    saved_files.append(p2)
    plt.close(fig2)

    # ========== 图3: Error Heatmap + Sparsity ==========
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 5))
    fig3.suptitle("Estimation Error & Sparsity Analysis", fontsize=15, fontweight="bold")

    plot_beta_error_heatmap(trainer, beta_funcs, signal_idx, t_final, axes3[0])
    plot_sparsity_pattern(trainer, signal_idx, axes3[1])

    fig3.tight_layout(rect=[0, 0, 1, 0.92])
    p3 = os.path.join(out_dir, "viz_3_error_sparsity.png")
    fig3.savefig(p3, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    saved_files.append(p3)
    plt.close(fig3)

    # --- 输出 ---
    for fp in saved_files:
        print(f"[OK] Saved: {fp}")


if __name__ == "__main__":
    main()
