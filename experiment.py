"""
统一实验运行器：支持增量/Batch 对比，自动按时间戳管理输出到 experiments/ 目录。

用法示例：
  # 最优配置（MinCV, ks=0.1, n=600, P=15, t=8）
  python experiment.py --knot-step 0.1 --n-per-seg 600 --P 15 --t-final 8

  # 自定义标签
  python experiment.py --knot-step 0.1 --n-per-seg 600 --P 15 --t-final 8 --tag best_config

  # 只跑增量（不跑 Batch 对比）
  python experiment.py --knot-step 0.1 --n-per-seg 600 --P 15 --t-final 8 --no-batch

  # 复现密节点实验的全部 6 种配置
  python experiment.py --preset dense_knots
"""
import os
import sys
import argparse
import json
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vcm import (
    VCMSimulator,
    IncrementalVCMTrainer,
    make_global_knots,
    partition_OCN,
    collect_cn_data,
    bspline_design_matrix,
    build_vcm_design,
    true_beta_funcs_default,
    fista_group_lasso,
    group_weights,
    gram_R,
    lambda_max_R,
    make_lambda_path,
    cv_select_lambda_plain,
    split_blocks,
)


# ============================================================
# 输出目录管理
# ============================================================
def make_output_dir(tag=""):
    """创建带时间戳的实验输出目录: experiments/YYYYMMDD_HHMMSS_tag/"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dirname = f"{ts}_{tag}" if tag else ts
    outdir = os.path.join("experiments", dirname)
    os.makedirs(outdir, exist_ok=True)
    return outdir


# ============================================================
# 增量运行
# ============================================================
def run_incremental(k, P, signal_idx, beta_funcs, sigma, t_final,
                    knot_step, n_per_seg, use_1se, seed_data=0, label=""):
    print(f"\n{'=' * 70}")
    print(f"增量算法: {label}")
    print(f"  knot_step={knot_step}, n_per_seg={n_per_seg}, use_1se={use_1se}")
    print(f"{'=' * 70}")

    sim = VCMSimulator(P=P, signal_idx=signal_idx, beta_funcs=beta_funcs,
                       sigma=sigma, seed_base=seed_data)
    trainer = IncrementalVCMTrainer(
        k=k, knot_step=knot_step, P=P,
        seed_cv=2025, use_1se=use_1se,
        r_relax=0, adaptive=False, debug=False,
    )

    # Stage 1
    t0, X0, y0, _ = sim.sample_segment(0.0, 1.0, n_per_seg, segment_id=0)
    info = trainer.fit_stage1(t0, X0, y0)
    print(f"  Stage 1: RMSE={info['train_rmse']:.6f}, lambda={info['lambda_best']:.6f}")

    prev_t, prev_X, prev_y = t0, X0, y0
    for stage in range(2, t_final + 1):
        old_end = float(stage - 1)
        next_end = float(stage)
        knots_new = make_global_knots(0.0, next_end, k, knot_step)
        part = partition_OCN(trainer.knots, knots_new, k, old_end=old_end)
        idx_cn = part["idx_cn_new"]
        t_cn, X_cn, y_cn, cur_t, cur_X, cur_y = collect_cn_data(
            sim, knots_new, k, idx_cn, old_end, next_end, n_per_seg,
            prev_seg_t=prev_t, prev_seg_X=prev_X, prev_seg_y=prev_y,
        )
        info = trainer.extend_one_stage(next_end, t_cn, X_cn, y_cn)
        print(f"  Stage {stage}: RMSE_cn={info['train_rmse_cn']:.6f}, lambda={info['lambda_best']:.6f}")
        prev_t, prev_X, prev_y = cur_t, cur_X, cur_y

    return trainer


def run_batch(k, P, signal_idx, beta_funcs, sigma, t_final,
              knot_step, n_per_seg, use_1se, seed_data=0, label=""):
    print(f"\n{'=' * 70}")
    print(f"Batch 全局拟合: {label}")
    print(f"  knot_step={knot_step}, n_per_seg={n_per_seg}, use_1se={use_1se}")
    print(f"{'=' * 70}")

    sim = VCMSimulator(P=P, signal_idx=signal_idx, beta_funcs=beta_funcs,
                       sigma=sigma, seed_base=seed_data)

    t_all_parts, X_all_parts, y_all_parts = [], [], []
    for seg_id in range(t_final):
        a, b = float(seg_id), float(seg_id) + 1.0
        ts, Xs, ys, _ = sim.sample_segment(a, b, n_per_seg, segment_id=seg_id)
        t_all_parts.append(ts)
        X_all_parts.append(Xs)
        y_all_parts.append(ys)
    t_all = np.concatenate(t_all_parts)
    X_all = np.vstack(X_all_parts)
    y_all = np.concatenate(y_all_parts)

    knots_batch = make_global_knots(0.0, float(t_final), k, knot_step)
    B_batch = bspline_design_matrix(t_all, knots_batch, k)
    m_batch = B_batch.shape[1]
    R_batch = gram_R(knots_batch, k, 0.0, float(t_final))
    Phi_batch = build_vcm_design(B_batch, X_all)
    w_batch = group_weights(B_batch, X_all)
    lam_max_b = lambda_max_R(Phi_batch, y_all, m_batch, R_batch)
    lam_path_b = make_lambda_path(lam_max_b)
    lam_best_b = cv_select_lambda_plain(B_batch, X_all, y_all, R_batch, lam_path_b,
                                         K=5, seed=2025, use_1se=use_1se)
    c_batch = fista_group_lasso(Phi_batch.T @ Phi_batch, Phi_batch.T @ y_all,
                                lam_best_b, m_batch, P, R_batch, w_batch)
    coef_batch = np.asarray(c_batch).reshape(P, m_batch)
    print(f"  Batch: lambda={lam_best_b:.6f}, m={m_batch}")

    return knots_batch, coef_batch, lam_best_b


# ============================================================
# 可视化
# ============================================================
def generate_visualizations(configs_list, results, beta_funcs, signal_idx,
                            k, t_final, P, sigma, outdir):
    """生成所有可视化图到 outdir 目录"""
    t_eval = np.linspace(0, float(t_final), 2000)
    n_signals = len(signal_idx)

    beta_labels = [
        r"$-0.5+0.6\cos(2\pi t)+0.15\ln(1+0.3t)$",
        r"$-0.5+0.6\cos(2\pi t)$",
        r"$0.7\sin(4\pi t)$",
        r"$0.7\sin(4\pi t)$",
        r"$0.4\cos(3\pi t)$",
    ]

    # 预计算 MISE
    fitted = {}
    mise_data = {cfg["name"]: [] for cfg in configs_list}
    for cfg in configs_list:
        name = cfg["name"]
        r = results[name]
        B_eval = bspline_design_matrix(t_eval, r["knots"], k)
        fitted[name] = B_eval
        for idx, p in enumerate(signal_idx):
            beta_true = beta_funcs[idx](t_eval)
            beta_hat = B_eval @ r["coef"][p]
            mise = float(np.mean((beta_true - beta_hat) ** 2))
            mise_data[name].append(mise)

    # --- 图1: Beta 函数拟合 ---
    fig, axes = plt.subplots(n_signals, 1, figsize=(20, 4 * n_signals), sharex=True)
    if n_signals == 1:
        axes = [axes]
    fig.suptitle(
        f"Beta Function Fitting Comparison\n"
        f"P={P}, t_final={t_final}, sigma={sigma}",
        fontsize=14, fontweight="bold"
    )
    for idx, p in enumerate(signal_idx):
        ax = axes[idx]
        beta_true = beta_funcs[idx](t_eval)
        ax.plot(t_eval, beta_true, "k-", linewidth=2.5, label="True", alpha=0.85)
        for cfg in configs_list:
            name = cfg["name"]
            r = results[name]
            beta_hat = fitted[name] @ r["coef"][p]
            mise_val = mise_data[name][idx]
            ax.plot(t_eval, beta_hat, cfg["ls"], color=cfg["color"],
                    linewidth=cfg["lw"], label=f"{name} (MISE={mise_val:.6f})",
                    alpha=cfg["alpha"])
        for bd in range(1, t_final):
            ax.axvline(x=float(bd), color="gray", linestyle=":", alpha=0.3)
        ax.set_ylabel(f"$\\beta_{p}(t)$", fontsize=12)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.2)
        if idx < len(beta_labels):
            ax.set_title(f"p={p}: {beta_labels[idx]}", fontsize=11)
    axes[-1].set_xlabel("t", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out1 = os.path.join(outdir, "beta_fit.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {out1}")

    # --- 图2: MISE 柱状图 ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(16, 7))
    x_pos = np.arange(n_signals)
    n_methods = len(configs_list)
    w = 0.8 / n_methods
    for i, cfg in enumerate(configs_list):
        name = cfg["name"]
        vals = mise_data[name]
        ax2.bar(x_pos + (i - n_methods / 2 + 0.5) * w, vals, w,
                label=name, color=cfg["color"], alpha=0.85,
                edgecolor="black", linewidth=0.5)
    ax2.set_xlabel("Signal variable p", fontsize=12)
    ax2.set_ylabel("MISE", fontsize=12)
    ax2.set_title("MISE Comparison (lower is better)", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"p={p}" for p in signal_idx], fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    out2 = os.path.join(outdir, "mise_bar.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"[OK] {out2}")

    # --- 图3: Shrinkage ---
    t_dense = np.linspace(0, float(t_final), 10000)
    fig3, axes3 = plt.subplots(2, 1, figsize=(18, 10))

    ax = axes3[0]
    for cfg in configs_list:
        name = cfg["name"]
        r = results[name]
        B_ev = bspline_design_matrix(t_eval, r["knots"], k)
        for idx_p, p in enumerate(signal_idx):
            beta_true = beta_funcs[idx_p](t_eval)
            err = (B_ev @ r["coef"][p]) - beta_true
            ax.plot(t_eval, err, color=cfg["color"], linewidth=cfg["lw"],
                    alpha=cfg["alpha"] * 0.6, linestyle=cfg["ls"])
    for cfg in configs_list:
        ax.plot([], [], cfg["ls"], color=cfg["color"], linewidth=cfg["lw"],
                label=cfg["name"], alpha=0.8)
    ax.axhline(y=0, color="k", linewidth=0.5)
    for bd in range(1, t_final):
        ax.axvline(x=float(bd), color="gray", linestyle=":", alpha=0.3)
    ax.set_ylabel("$\\hat{\\beta}(t) - \\beta(t)$", fontsize=12)
    ax.set_title("Signed Error (all signals overlaid)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.2)

    ax2b = axes3[1]
    shrinkage_summary = {cfg["name"]: {} for cfg in configs_list}
    markers = ["o", "s", "^", "D", "v", "x", "P", "*"]
    for idx_p, p in enumerate(signal_idx):
        beta_true_dense = beta_funcs[idx_p](t_dense)
        d1 = np.diff(beta_true_dense)
        sign_change = np.where(d1[:-1] * d1[1:] < 0)[0] + 1
        peak_t = t_dense[sign_change]
        peak_vals = beta_true_dense[sign_change]
        for ci, cfg in enumerate(configs_list):
            name = cfg["name"]
            r = results[name]
            B_pk = bspline_design_matrix(peak_t, r["knots"], k)
            pk_hat = B_pk @ r["coef"][p]
            eps_v = 1e-10
            sh = np.abs(pk_hat) / (np.abs(peak_vals) + eps_v)
            shrinkage_summary[name][p] = float(np.mean(sh))
            if p in [3, 4]:
                ax2b.scatter(peak_t, sh, s=20 + ci * 5, c=cfg["color"],
                            marker=markers[ci % len(markers)], alpha=0.5 + ci * 0.06, zorder=3 + ci)
    for ci, cfg in enumerate(configs_list):
        ax2b.scatter([], [], s=30, c=cfg["color"], marker=markers[ci % len(markers)],
                    label=cfg["name"], alpha=0.8)
    ax2b.axhline(y=1.0, color="k", linewidth=1, linestyle="--", alpha=0.5, label="Perfect (1.0)")
    for bd in range(1, t_final):
        ax2b.axvline(x=float(bd), color="gray", linestyle=":", alpha=0.3)
    ax2b.set_xlabel("t (at extrema)", fontsize=12)
    ax2b.set_ylabel("|hat| / |true|", fontsize=12)
    ax2b.set_title("Peak Shrinkage (sin(4πt), closer to 1.0 is better)", fontsize=13, fontweight="bold")
    ax2b.legend(fontsize=7, ncol=3)
    ax2b.grid(True, alpha=0.2)
    ax2b.set_ylim(0.85, 1.08)
    fig3.tight_layout()
    out3 = os.path.join(outdir, "shrinkage.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    print(f"[OK] {out3}")

    # --- 图4: Zoom-in ---
    fig4, axes4 = plt.subplots(2, 2, figsize=(18, 10))
    zoom_ranges = [(1.5, 2.5), (4.0, 5.0)]
    zoom_ps = [p for p in [3, 4] if p in signal_idx]
    for row, p_val in enumerate(zoom_ps[:2]):
        idx_p = signal_idx.index(p_val)
        for col, (za, zb) in enumerate(zoom_ranges):
            ax = axes4[row, col]
            mask = (t_eval >= za) & (t_eval <= zb)
            t_z = t_eval[mask]
            bt = beta_funcs[idx_p](t_z)
            ax.plot(t_z, bt, "k-", linewidth=2.5, label="True", alpha=0.85)
            for cfg in configs_list:
                name = cfg["name"]
                r = results[name]
                B_z = bspline_design_matrix(t_z, r["knots"], k)
                beta_hat_z = B_z @ r["coef"][p_val]
                ax.plot(t_z, beta_hat_z, cfg["ls"], color=cfg["color"],
                        linewidth=cfg["lw"], label=name, alpha=cfg["alpha"])
            if idx_p < len(beta_labels):
                ax.set_title(f"p={p_val}: {beta_labels[idx_p]}, t∈[{za},{zb}]", fontsize=10)
            ax.legend(fontsize=6)
            ax.grid(True, alpha=0.2)
    fig4.suptitle("Zoom-in: Peak Detail", fontsize=13, fontweight="bold")
    fig4.tight_layout(rect=[0, 0, 1, 0.93])
    out4 = os.path.join(outdir, "zoom_in.png")
    fig4.savefig(out4, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig4)
    print(f"[OK] {out4}")

    return mise_data, shrinkage_summary


def print_summary(configs_list, mise_data, shrinkage_summary, signal_idx, P, outdir):
    """打印并保存汇总表"""
    lines = []

    lines.append(f"{'=' * 90}")
    lines.append("MISE Summary")
    lines.append(f"{'=' * 90}")
    header = f"{'p':>4}"
    for cfg in configs_list:
        header += f" {cfg['name'][:22]:>24}"
    lines.append(header)

    for idx, p in enumerate(signal_idx):
        row = f"{p:>4}"
        for cfg in configs_list:
            v = mise_data[cfg["name"]][idx]
            row += f" {v:>24.6f}"
        lines.append(row)

    row_avg = f"{'avg':>4}"
    for cfg in configs_list:
        v = np.mean(mise_data[cfg["name"]])
        row_avg += f" {v:>24.6f}"
    lines.append(row_avg)

    # Shrinkage
    lines.append(f"\n{'=' * 90}")
    lines.append("Peak Shrinkage (mean |hat|/|true| at extrema)")
    lines.append(f"{'=' * 90}")
    header = f"{'p':>4}"
    for cfg in configs_list:
        header += f" {cfg['name'][:22]:>24}"
    lines.append(header)

    for p in signal_idx:
        row = f"{p:>4}"
        for cfg in configs_list:
            v = shrinkage_summary[cfg["name"]].get(p, float("nan"))
            row += f" {v:>24.4f}"
        lines.append(row)

    row_avg = f"{'avg':>4}"
    for cfg in configs_list:
        v = np.mean([shrinkage_summary[cfg["name"]].get(p, 0) for p in signal_idx])
        row_avg += f" {v:>24.4f}"
    lines.append(row_avg)
    lines.append(f"{'=' * 90}")

    text = "\n".join(lines)
    print(text)

    # 保存到文件
    with open(os.path.join(outdir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[OK] {os.path.join(outdir, 'summary.txt')}")


# ============================================================
# Preset 配置
# ============================================================
PRESETS = {
    "dense_knots": {
        "description": "密节点+大n对比实验（6种配置）",
        "configs": [
            {"name": "Incr MinCV (ks=0.1, n=300)",  "knot_step": 0.1,  "n_per_seg": 300, "use_1se": False, "mode": "incr",  "color": "#2196F3", "ls": "-",  "lw": 1.2, "alpha": 0.6},
            {"name": "Incr MinCV (ks=0.05, n=300)", "knot_step": 0.05, "n_per_seg": 300, "use_1se": False, "mode": "incr",  "color": "#FF9800", "ls": "-",  "lw": 1.5, "alpha": 0.8},
            {"name": "Incr MinCV (ks=0.1, n=600)",  "knot_step": 0.1,  "n_per_seg": 600, "use_1se": False, "mode": "incr",  "color": "#4CAF50", "ls": "-",  "lw": 1.5, "alpha": 0.8},
            {"name": "Incr MinCV (ks=0.05, n=600)", "knot_step": 0.05, "n_per_seg": 600, "use_1se": False, "mode": "incr",  "color": "#E91E63", "ls": "-",  "lw": 2.0, "alpha": 0.9},
            {"name": "Batch MinCV (ks=0.05, n=600)","knot_step": 0.05, "n_per_seg": 600, "use_1se": False, "mode": "batch", "color": "#9C27B0", "ls": "--", "lw": 1.5, "alpha": 0.8},
            {"name": "Incr 1SE (ks=0.1, n=300)",    "knot_step": 0.1,  "n_per_seg": 300, "use_1se": True,  "mode": "incr",  "color": "#607D8B", "ls": ":",  "lw": 1.0, "alpha": 0.5},
        ],
    },
    "mincv_vs_1se": {
        "description": "MinCV vs 1SE对比实验",
        "configs": [
            {"name": "Incr MinCV", "knot_step": 0.1, "n_per_seg": 300, "use_1se": False, "mode": "incr",  "color": "#4CAF50", "ls": "-",  "lw": 2.0, "alpha": 0.9},
            {"name": "Incr 1SE",   "knot_step": 0.1, "n_per_seg": 300, "use_1se": True,  "mode": "incr",  "color": "#F44336", "ls": "-",  "lw": 2.0, "alpha": 0.9},
            {"name": "Batch MinCV","knot_step": 0.1, "n_per_seg": 300, "use_1se": False, "mode": "batch", "color": "#9C27B0", "ls": "--", "lw": 1.5, "alpha": 0.8},
            {"name": "Batch 1SE",  "knot_step": 0.1, "n_per_seg": 300, "use_1se": True,  "mode": "batch", "color": "#FF9800", "ls": "--", "lw": 1.5, "alpha": 0.8},
        ],
    },
}


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="统一实验运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                        help="使用预设配置")
    parser.add_argument("--knot-step", type=float, default=0.1)
    parser.add_argument("--n-per-seg", type=int, default=600)
    parser.add_argument("--P", type=int, default=15)
    parser.add_argument("--t-final", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="", help="实验标签（用于输出目录名）")
    parser.add_argument("--no-batch", action="store_true", help="不跑 Batch 对比")
    parser.add_argument("--use-1se", action="store_true", help="使用 1SE（默认 MinCV）")

    args = parser.parse_args()

    k = 3
    signal_idx = [1, 2, 3, 4, 5]
    beta_funcs = true_beta_funcs_default()

    if args.preset:
        preset = PRESETS[args.preset]
        tag = args.tag or args.preset
        configs_list = preset["configs"]
        print(f"使用预设: {args.preset} - {preset['description']}")
    else:
        # 单配置模式：Incremental + (可选)Batch
        tag = args.tag or f"ks{args.knot_step}_n{args.n_per_seg}_P{args.P}_t{args.t_final}"
        configs_list = [
            {"name": f"Incr {'1SE' if args.use_1se else 'MinCV'} (ks={args.knot_step}, n={args.n_per_seg})",
             "knot_step": args.knot_step, "n_per_seg": args.n_per_seg,
             "use_1se": args.use_1se, "mode": "incr",
             "color": "#4CAF50", "ls": "-", "lw": 2.0, "alpha": 0.9},
        ]
        if not args.no_batch:
            configs_list.append({
                "name": f"Batch {'1SE' if args.use_1se else 'MinCV'} (ks={args.knot_step}, n={args.n_per_seg})",
                "knot_step": args.knot_step, "n_per_seg": args.n_per_seg,
                "use_1se": args.use_1se, "mode": "batch",
                "color": "#9C27B0", "ls": "--", "lw": 1.5, "alpha": 0.8,
            })

    outdir = make_output_dir(tag)
    print(f"\n输出目录: {outdir}")

    # 保存实验配置
    exp_meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "args": vars(args),
        "k": k, "P": args.P, "t_final": args.t_final,
        "signal_idx": signal_idx, "sigma": args.sigma,
        "configs": [{k_: v for k_, v in c.items() if k_ not in ("color", "ls", "lw", "alpha")}
                    for c in configs_list],
    }
    with open(os.path.join(outdir, "experiment_config.json"), "w") as f:
        json.dump(exp_meta, f, indent=2, ensure_ascii=False)

    # 运行
    results = {}
    for cfg in configs_list:
        name = cfg["name"]
        if cfg["mode"] == "incr":
            trainer = run_incremental(
                k=k, P=args.P, signal_idx=signal_idx, beta_funcs=beta_funcs,
                sigma=args.sigma, t_final=args.t_final,
                knot_step=cfg["knot_step"], n_per_seg=cfg["n_per_seg"],
                use_1se=cfg["use_1se"], seed_data=args.seed, label=name,
            )
            results[name] = {
                "knots": trainer.knots,
                "coef": np.stack(trainer.coef_blocks, axis=0),
                "type": "incr",
            }
        else:
            knots_b, coef_b, lam_b = run_batch(
                k=k, P=args.P, signal_idx=signal_idx, beta_funcs=beta_funcs,
                sigma=args.sigma, t_final=args.t_final,
                knot_step=cfg["knot_step"], n_per_seg=cfg["n_per_seg"],
                use_1se=cfg["use_1se"], seed_data=args.seed, label=name,
            )
            results[name] = {
                "knots": knots_b,
                "coef": coef_b,
                "type": "batch",
            }

    # 可视化
    mise_data, shrinkage_summary = generate_visualizations(
        configs_list, results, beta_funcs, signal_idx,
        k, args.t_final, args.P, args.sigma, outdir
    )

    # 汇总表
    print_summary(configs_list, mise_data, shrinkage_summary, signal_idx, args.P, outdir)

    print(f"\n{'=' * 70}")
    print(f"实验完成！所有输出保存在: {outdir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
