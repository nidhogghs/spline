"""
多 seed 并行调度器：为 vcm.py 提供并行运行支持。

用法示例：
  # 通过配置文件运行 10 个 seed
  python parallel.py --n-seeds 10 --config configs/main1_t20_p100_n300_simple5_server.json

  # 通过命令行参数运行
  python parallel.py --n-seeds 5 --t-final 8 --P 15 --knot-step 0.1 --n-per-segment 600

  # 指定并行度和日志
  python parallel.py --n-seeds 10 --max-workers 4 --log-dir logs/run1 --config configs/xxx.json
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def _str2bool(s):
    if isinstance(s, bool):
        return s
    s = str(s).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def _append_arg(cmd, name, value):
    """将非 None 参数追加到命令行。"""
    if value is None:
        return
    cmd.append(name)
    cmd.append(str(value))


def _build_checkpoint_dir(tag, checkpoint_dir, seed):
    if checkpoint_dir:
        base = _resolve_checkpoint_base(checkpoint_dir)
    else:
        default_name = f"checkpoints_vcm_{tag}" if tag else "checkpoints_vcm"
        base = os.path.join(DEFAULT_CHECKPOINT_ROOT, default_name)
    return f"{base}_seed{seed}"


def _run_one(seed, args, extra_args):
    """运行单个 seed 的 vcm.py 进程。"""
    vcm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcm.py")
    ckpt_dir = _build_checkpoint_dir(args.tag, args.checkpoint_dir, seed)

    cmd = [sys.executable, vcm_path]

    # 配置文件
    if args.config:
        _append_arg(cmd, "--config", args.config)

    # checkpoint 与实验
    _append_arg(cmd, "--checkpoint-dir", ckpt_dir)
    if args.history_json:
        history_path = args.history_json.replace(".json", f"_seed{seed}.json")
        _append_arg(cmd, "--history-json", history_path)

    # 数据参数
    _append_arg(cmd, "--t-final", args.t_final)
    _append_arg(cmd, "--n-per-segment", args.n_per_segment)
    _append_arg(cmd, "--P", args.P)
    _append_arg(cmd, "--signal-idx", args.signal_idx)
    _append_arg(cmd, "--beta-scales", args.beta_scales)
    _append_arg(cmd, "--sigma", args.sigma)

    # 模型参数
    _append_arg(cmd, "--k", args.k)
    _append_arg(cmd, "--knot-step", args.knot_step)

    # 训练参数
    _append_arg(cmd, "--seed-data", seed)
    _append_arg(cmd, "--seed-cv", args.seed_cv)
    _append_arg(cmd, "--use-1se", str(bool(args.use_1se)).lower())
    _append_arg(cmd, "--r-relax", args.r_relax)
    _append_arg(cmd, "--adaptive", str(bool(args.adaptive)).lower())

    # 调试与心跳
    _append_arg(cmd, "--debug", str(bool(args.debug)).lower())
    _append_arg(cmd, "--heartbeat-sec", args.heartbeat_sec)

    # 额外参数（透传）
    if extra_args:
        cmd.extend(extra_args)

    # 日志
    log_file = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = os.path.join(args.log_dir, f"seed{seed}.log")

    if log_file:
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    else:
        proc = subprocess.run(cmd, check=False)

    return seed, proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Parallel multi-seed runner for vcm.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- 调度参数 ----
    parser.add_argument("--n-seeds", type=int, required=True, help="Number of seeds to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed (default: 0).")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count() or 1,
                        help="Max parallel workers (default: CPU count).")
    parser.add_argument("--tag", default="default", help="Experiment tag (used if --checkpoint-dir not set).")
    parser.add_argument("--checkpoint-dir", default="", help="Override base checkpoint directory.")
    parser.add_argument("--log-dir", default="", help="Directory to save per-seed logs.")
    parser.add_argument("--config", default="", help="Path to JSON config file (passed to vcm.py).")
    parser.add_argument("--history-json", default="", help="Base path for per-seed history JSON output.")

    # ---- 数据参数（与 vcm.py 对齐） ----
    parser.add_argument("--t-final", type=float, default=3.0)
    parser.add_argument("--n-per-segment", type=int, default=600)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--signal-idx", default="1,2,3,4,5")
    parser.add_argument("--beta-scales", default="1,1,1,1,1")
    parser.add_argument("--sigma", type=float, default=0.1)

    # ---- 模型参数（与 vcm.py 对齐） ----
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--knot-step", type=float, default=0.1, help="Knot spacing.")

    # ---- 训练参数（与 vcm.py 对齐） ----
    parser.add_argument("--seed-cv", type=int, default=2025)
    parser.add_argument("--use-1se", type=_str2bool, default=False)
    parser.add_argument("--r-relax", type=int, default=0, help="Number of bases near boundary to release.")
    parser.add_argument("--adaptive", type=_str2bool, default=False, help="Enable adaptive group-lasso refit.")

    # ---- 调试与心跳（与 vcm.py 对齐） ----
    parser.add_argument("--debug", type=_str2bool, default=False)
    parser.add_argument("--heartbeat-sec", type=float, default=90.0, help="Heartbeat interval; <=0 disables.")

    args, extra_args = parser.parse_known_args()

    seeds = [args.seed_start + i for i in range(args.n_seeds)]
    max_workers = max(1, int(args.max_workers))

    print(f"[parallel] seeds={args.n_seeds} range=[{seeds[0]}, {seeds[-1]}] "
          f"max_workers={max_workers}", flush=True)
    if args.config:
        print(f"[parallel] config={args.config}", flush=True)

    failures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for seed in seeds:
            print(f"[submit] seed={seed}", flush=True)
            futures[ex.submit(_run_one, seed, args, extra_args)] = seed
        for fut in as_completed(futures):
            seed, code = fut.result()
            if code != 0:
                failures.append((seed, code))
                print(f"[done] seed={seed} code={code} (FAILED)", flush=True)
            else:
                print(f"[done] seed={seed} ok", flush=True)

    n_ok = len(seeds) - len(failures)
    print(f"[parallel] finished: {n_ok}/{len(seeds)} succeeded", flush=True)

    if failures:
        msg = ", ".join([f"seed={s}(code={c})" for s, c in sorted(failures)])
        raise SystemExit(f"[parallel] {len(failures)} runs failed: {msg}")


if __name__ == "__main__":
    main()
