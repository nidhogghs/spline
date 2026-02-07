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
    main_path = os.path.join(os.path.dirname(__file__), "main.py")
    ckpt_dir = _build_checkpoint_dir(args.tag, args.checkpoint_dir, seed)

    cmd = [sys.executable, main_path]
    _append_arg(cmd, "--checkpoint-dir", ckpt_dir)
    _append_arg(cmd, "--t-final", args.t_final)
    _append_arg(cmd, "--n-per-segment", args.n_per_segment)
    _append_arg(cmd, "--P", args.P)
    _append_arg(cmd, "--signal-idx", args.signal_idx)
    _append_arg(cmd, "--beta-scales", args.beta_scales)
    _append_arg(cmd, "--sigma", args.sigma)
    _append_arg(cmd, "--k", args.k)
    _append_arg(cmd, "--n-inner-per-unit", args.n_inner_per_unit)
    _append_arg(cmd, "--seed-data", seed)
    _append_arg(cmd, "--seed-cv", args.seed_cv)
    _append_arg(cmd, "--use-1se", str(bool(args.use_1se)).lower())
    _append_arg(cmd, "--r-relax", args.r_relax)
    _append_arg(cmd, "--use-adaptive-cn", str(bool(args.use_adaptive_cn)).lower())
    _append_arg(cmd, "--save-checkpoints", str(bool(args.save_checkpoints)).lower())
    _append_arg(cmd, "--save-checkpoint-data", str(bool(args.save_checkpoint_data)).lower())

    if extra_args:
        cmd.extend(extra_args)

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
    parser = argparse.ArgumentParser(description="Parallel multi-seed runner for main.py")
    parser.add_argument("--n-seeds", type=int, required=True, help="Number of seeds to run.")
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed.")
    parser.add_argument("--max-workers", type=int, default=os.cpu_count() or 1,
                        help="Max parallel workers.")
    parser.add_argument("--tag", default="default", help="Experiment tag (used if checkpoint-dir not set).")
    parser.add_argument("--checkpoint-dir", default="", help="Override base checkpoint directory.")
    parser.add_argument("--log-dir", default="", help="Optional dir to save per-seed logs.")

    parser.add_argument("--t-final", type=float, default=3.0)
    parser.add_argument("--n-per-segment", type=int, default=400)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--signal-idx", default="1,2,3,4,5")
    parser.add_argument("--beta-scales", default="1,1,1,1,1")
    parser.add_argument("--sigma", type=float, default=0.1)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--n-inner-per-unit", type=int, default=10)
    parser.add_argument("--seed-cv", type=int, default=2025)
    parser.add_argument("--use-1se", type=_str2bool, default=True)
    parser.add_argument("--r-relax", type=int, default=2)
    parser.add_argument("--use-adaptive-cn", type=_str2bool, default=True)
    parser.add_argument("--save-checkpoints", type=_str2bool, default=True)
    parser.add_argument("--save-checkpoint-data", type=_str2bool, default=True)

    args, extra_args = parser.parse_known_args()

    seeds = [args.seed_start + i for i in range(args.n_seeds)]
    max_workers = max(1, int(args.max_workers))

    print(f"[parallel] seeds={args.n_seeds} max_workers={max_workers} start={args.seed_start}")
    failures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for seed in seeds:
            print(f"[submit] seed={seed}")
            futures[ex.submit(_run_one, seed, args, extra_args)] = seed
        for fut in as_completed(futures):
            seed, code = fut.result()
            if code != 0:
                failures.append((seed, code))
                print(f"[done] seed={seed} code={code} (failed)")
            else:
                print(f"[done] seed={seed} code=0")

    if failures:
        msg = ", ".join([f"seed={s} code={c}" for s, c in failures])
        raise SystemExit(f"Some runs failed: {msg}")


if __name__ == "__main__":
    main()
