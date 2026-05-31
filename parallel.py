"""
多 seed 并行调度器：为 vcm.py 提供并行运行支持。

用法示例：
  # 通过配置文件运行 5 个 seed（推荐）
  python parallel.py --config configs/longcycle_shift_beta2.json --n-seeds 5

  # 指定并行度、日志目录和 seed 范围
  python parallel.py --config configs/longcycle_shift_beta2.json --n-seeds 5 --max-workers 5 --log-dir logs/longcycle

  # 不用配置文件，通过命令行参数运行
  python parallel.py --n-seeds 5 --t-final 8 --P 15 --knot-step 0.1 --n-per-segment 600 --tag myexp

  # 从指定 seed 开始
  python parallel.py --config configs/xxx.json --n-seeds 5 --seed-start 10
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


DEFAULT_CHECKPOINT_ROOT = "checkpoints"


def _compute_threads_per_worker(max_workers):
    """根据 CPU 核心数和并行 worker 数，计算每个子进程应使用的线程数。
    
    避免 N 个子进程各自调用 BLAS/OpenMP 开满线程导致严重的资源争抢。
    """
    n_cpu = os.cpu_count() or 1
    threads = max(1, n_cpu // max_workers)
    return str(threads)


def _thread_limit_env(max_workers):
    """返回限制 BLAS/OpenMP/MKL 线程数的环境变量字典。"""
    t = _compute_threads_per_worker(max_workers)
    return {
        "OMP_NUM_THREADS": t,
        "MKL_NUM_THREADS": t,
        "OPENBLAS_NUM_THREADS": t,
        "NUMEXPR_MAX_THREADS": t,
        "VECLIB_MAXIMUM_THREADS": t,
    }


def _resolve_checkpoint_base(checkpoint_dir, root=DEFAULT_CHECKPOINT_ROOT):
    ckpt_dir = str(checkpoint_dir).strip()
    if not ckpt_dir:
        return ckpt_dir
    norm_dir = os.path.normpath(ckpt_dir)
    if os.path.isabs(ckpt_dir):
        return ckpt_dir
    if norm_dir == root or norm_dir.startswith(root + os.sep):
        return ckpt_dir
    return os.path.join(root, ckpt_dir)


def _str2bool(s):
    if isinstance(s, bool):
        return s
    s = str(s).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_cfg(cfg, section, key, default):
    sec = cfg.get(section, {})
    return sec.get(key, default) if isinstance(sec, dict) else default


def _build_checkpoint_dir_for_seed(base_dir, seed):
    """为每个 seed 生成独立的 checkpoint 目录。"""
    return f"{base_dir}_seed{seed}"


def _build_history_json_for_seed(base_history_json, seed):
    """为每个 seed 生成独立的 history.json 路径。"""
    if not base_history_json:
        return ""
    root, ext = os.path.splitext(base_history_json)
    return f"{root}_seed{seed}{ext}"


def _run_one_with_config(seed, config_path, base_ckpt_dir, base_history_json,
                         log_dir, heartbeat_sec, env_extra=None):
    """使用配置文件模式运行单个 seed。
    
    只传给 vcm.py 以下参数：
    - --config: 配置文件路径
    - --checkpoint-dir: 加了 _seedN 后缀的目录
    - --history-json: 加了 _seedN 后缀的路径
    - --seed-data: 当前 seed
    - --heartbeat-sec: 心跳间隔
    
    其余参数全部由 config 决定，避免默认值覆盖。
    """
    vcm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcm.py")
    ckpt_dir = _build_checkpoint_dir_for_seed(base_ckpt_dir, seed)
    history_json = _build_history_json_for_seed(base_history_json, seed)

    cmd = [sys.executable, vcm_path]
    cmd.extend(["--config", config_path])
    cmd.extend(["--checkpoint-dir", ckpt_dir])
    cmd.extend(["--seed-data", str(seed)])
    if history_json:
        cmd.extend(["--history-json", history_json])
    if heartbeat_sec is not None:
        cmd.extend(["--heartbeat-sec", str(heartbeat_sec)])

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    log_file = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"seed{seed}.log")

    if log_file:
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  check=False, env=env)
    else:
        proc = subprocess.run(cmd, check=False, env=env)

    return seed, proc.returncode


def _run_one_with_args(seed, args, extra_args, env_extra=None):
    """使用命令行参数模式运行单个 seed（无 config 文件时）。"""
    vcm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vcm.py")

    # 构建 checkpoint 目录
    tag = str(args.tag).strip() if args.tag else "default"
    if args.checkpoint_dir:
        base = _resolve_checkpoint_base(args.checkpoint_dir)
    else:
        base = os.path.join(DEFAULT_CHECKPOINT_ROOT, f"checkpoints_vcm_{tag}")
    ckpt_dir = _build_checkpoint_dir_for_seed(base, seed)

    cmd = [sys.executable, vcm_path]
    cmd.extend(["--checkpoint-dir", ckpt_dir])

    # history-json
    if args.history_json:
        hj = _build_history_json_for_seed(args.history_json, seed)
        cmd.extend(["--history-json", hj])

    # 数据参数
    cmd.extend(["--t-final", str(args.t_final)])
    cmd.extend(["--n-per-segment", str(args.n_per_segment)])
    cmd.extend(["--P", str(args.P)])
    cmd.extend(["--signal-idx", str(args.signal_idx)])
    cmd.extend(["--beta-scales", str(args.beta_scales)])
    cmd.extend(["--sigma", str(args.sigma)])

    # 模型参数
    cmd.extend(["--k", str(args.k)])
    cmd.extend(["--knot-step", str(args.knot_step)])

    # 训练参数
    cmd.extend(["--seed-data", str(seed)])
    cmd.extend(["--seed-cv", str(args.seed_cv)])
    cmd.extend(["--use-1se", str(bool(args.use_1se)).lower()])
    cmd.extend(["--r-relax", str(args.r_relax)])
    cmd.extend(["--adaptive", str(bool(args.adaptive)).lower()])

    # 调试与心跳
    cmd.extend(["--debug", str(bool(args.debug)).lower()])
    cmd.extend(["--heartbeat-sec", str(args.heartbeat_sec)])

    # 额外参数（透传）
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)

    log_file = None
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_file = os.path.join(args.log_dir, f"seed{seed}.log")

    if log_file:
        with open(log_file, "w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                  check=False, env=env)
    else:
        proc = subprocess.run(cmd, check=False, env=env)

    return seed, proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Parallel multi-seed runner for vcm.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- 调度参数 ----
    parser.add_argument("--n-seeds", type=int, required=True, help="Number of seeds to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed (default: 0).")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max parallel workers (default: min(n_seeds, CPU_count)).")
    parser.add_argument("--tag", default="default", help="Experiment tag (used if --checkpoint-dir not set).")
    parser.add_argument("--checkpoint-dir", default="", help="Override base checkpoint directory.")
    parser.add_argument("--log-dir", default="", help="Directory to save per-seed logs.")
    parser.add_argument("--config", default="", help="Path to JSON config file (passed to vcm.py).")
    parser.add_argument("--history-json", default="", help="Base path for per-seed history JSON output.")

    # ---- 数据参数（与 vcm.py 对齐，仅在无 config 时使用） ----
    parser.add_argument("--t-final", type=float, default=3.0)
    parser.add_argument("--n-per-segment", type=int, default=600)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--signal-idx", default="1,2,3,4,5")
    parser.add_argument("--beta-scales", default="1,1,1,1,1")
    parser.add_argument("--sigma", type=float, default=0.1)

    # ---- 模型参数 ----
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--knot-step", type=float, default=0.1, help="Knot spacing.")

    # ---- 训练参数 ----
    parser.add_argument("--seed-cv", type=int, default=2025)
    parser.add_argument("--use-1se", type=_str2bool, default=False)
    parser.add_argument("--r-relax", type=int, default=0, help="Number of bases near boundary to release.")
    parser.add_argument("--adaptive", type=_str2bool, default=False, help="Enable adaptive group-lasso refit.")

    # ---- 调试与心跳 ----
    parser.add_argument("--debug", type=_str2bool, default=False)
    parser.add_argument("--heartbeat-sec", type=float, default=120.0, help="Heartbeat interval; <=0 disables.")

    args, extra_args = parser.parse_known_args()

    seeds = [args.seed_start + i for i in range(args.n_seeds)]
    max_workers = args.max_workers
    if max_workers is None:
        max_workers = min(args.n_seeds, os.cpu_count() or 1)
    max_workers = max(1, int(max_workers))

    # 用于消除 gcda profiling 噪音 + 限制每个子进程的 BLAS/OpenMP 线程数
    env_extra = {"GCOV_PREFIX": "/tmp/gcov_dummy"}
    env_extra.update(_thread_limit_env(max_workers))

    use_config_mode = bool(args.config)

    # 确定 base checkpoint 目录和 history json
    if use_config_mode:
        cfg = _load_config(args.config)
        ckpt_root = _get_cfg(cfg, "experiment", "checkpoint_root", DEFAULT_CHECKPOINT_ROOT)

        if args.checkpoint_dir:
            base_ckpt_dir = _resolve_checkpoint_base(args.checkpoint_dir, root=ckpt_root)
        else:
            cfg_ckpt_dir = _get_cfg(cfg, "experiment", "checkpoint_dir", "")
            if cfg_ckpt_dir:
                base_ckpt_dir = _resolve_checkpoint_base(cfg_ckpt_dir, root=ckpt_root)
            else:
                tag = str(args.tag).strip() if args.tag else "default"
                base_ckpt_dir = os.path.join(ckpt_root, f"checkpoints_vcm_{tag}")

        if args.history_json:
            base_history_json = args.history_json
        else:
            cfg_hj = _get_cfg(cfg, "experiment", "history_json", "")
            if cfg_hj:
                base_history_json = cfg_hj
            else:
                base_history_json = os.path.join(base_ckpt_dir, "history.json")

        log_dir = args.log_dir
        if not log_dir:
            # 自动设置日志目录到 checkpoint 目录的 parallel_logs 子目录
            log_dir = os.path.join(os.path.dirname(base_ckpt_dir), "logs_" + os.path.basename(base_ckpt_dir))

        t_final_display = _get_cfg(cfg, "data", "t_final", "?")
        P_display = _get_cfg(cfg, "model", "P", "?")
    else:
        tag = str(args.tag).strip() if args.tag else "default"
        if args.checkpoint_dir:
            base_ckpt_dir = _resolve_checkpoint_base(args.checkpoint_dir)
        else:
            base_ckpt_dir = os.path.join(DEFAULT_CHECKPOINT_ROOT, f"checkpoints_vcm_{tag}")
        base_history_json = args.history_json
        log_dir = args.log_dir
        t_final_display = args.t_final
        P_display = args.P

    # 打印运行信息
    print(f"{'=' * 70}", flush=True)
    print(f"[parallel] Multi-Seed Parallel Runner", flush=True)
    print(f"{'=' * 70}", flush=True)
    if use_config_mode:
        print(f"  config     = {args.config}", flush=True)
    print(f"  n_seeds    = {args.n_seeds} (seeds {seeds[0]}..{seeds[-1]})", flush=True)
    print(f"  workers    = {max_workers}", flush=True)
    print(f"  t_final    = {t_final_display}", flush=True)
    print(f"  P          = {P_display}", flush=True)
    print(f"  base_ckpt  = {base_ckpt_dir}", flush=True)
    print(f"  log_dir    = {log_dir}", flush=True)
    print(f"  threads/worker = {_compute_threads_per_worker(max_workers)} (of {os.cpu_count()} CPUs)", flush=True)
    print(f"{'=' * 70}", flush=True)

    failures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for seed in seeds:
            if use_config_mode:
                fut = ex.submit(
                    _run_one_with_config,
                    seed=seed,
                    config_path=args.config,
                    base_ckpt_dir=base_ckpt_dir,
                    base_history_json=base_history_json,
                    log_dir=log_dir,
                    heartbeat_sec=args.heartbeat_sec,
                    env_extra=env_extra,
                )
            else:
                fut = ex.submit(
                    _run_one_with_args,
                    seed=seed,
                    args=args,
                    extra_args=extra_args,
                    env_extra=env_extra,
                )
            print(f"[submit] seed={seed} -> {_build_checkpoint_dir_for_seed(base_ckpt_dir, seed)}", flush=True)
            futures[fut] = seed

        for fut in as_completed(futures):
            seed, code = fut.result()
            if code != 0:
                failures.append((seed, code))
                print(f"[done] seed={seed} code={code} (FAILED)", flush=True)
            else:
                print(f"[done] seed={seed} ok", flush=True)

    n_ok = len(seeds) - len(failures)
    print(f"\n{'=' * 70}", flush=True)
    print(f"[parallel] finished: {n_ok}/{len(seeds)} succeeded", flush=True)
    if log_dir:
        print(f"[parallel] logs in: {log_dir}", flush=True)
    print(f"{'=' * 70}", flush=True)

    if failures:
        msg = ", ".join([f"seed={s}(code={c})" for s, c in sorted(failures)])
        raise SystemExit(f"[parallel] {len(failures)} runs failed: {msg}")


if __name__ == "__main__":
    main()
