import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime


def _format_seconds(seconds):
    s = max(0, int(round(float(seconds))))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_workers(total_cpu, args, scenario_workers_cap):
    total_cpu = max(1, int(total_cpu))

    if args.cpu_mode == "manual":
        if args.max_workers <= 0:
            raise ValueError("cpu_mode=manual 时必须设置 --max-workers > 0")
        workers = int(args.max_workers)
    else:
        reserve = max(0, int(args.reserve_cpus))
        usable = max(1, total_cpu - reserve)
        util = float(args.cpu_utilization)
        if util <= 0 or util > 1:
            raise ValueError("--cpu-utilization 必须在 (0,1] 范围")
        workers = max(1, int(math.floor(usable * util)))

    if scenario_workers_cap is not None and int(scenario_workers_cap) > 0:
        workers = min(workers, int(scenario_workers_cap))

    if args.max_workers_cap > 0:
        workers = min(workers, int(args.max_workers_cap))

    workers = max(1, min(workers, total_cpu))
    return workers


def _resolve_blas_threads(total_cpu, workers, args, scenario_blas_threads):
    if scenario_blas_threads is not None and int(scenario_blas_threads) > 0:
        return int(scenario_blas_threads)

    if args.blas_threads > 0:
        return int(args.blas_threads)

    if args.cpu_mode == "manual":
        budget = total_cpu
    else:
        budget = max(1, total_cpu - max(0, int(args.reserve_cpus)))

    # 自动模式下给每个 worker 分配线程预算，避免过度超卖
    return max(1, int(budget // max(1, workers)))


def _ensure_section(cfg, key):
    if key not in cfg or not isinstance(cfg[key], dict):
        cfg[key] = {}
    return cfg[key]


def _prepare_benchmark_config(suite_cfg, scenario_cfg, output_root, workers):
    global_cfg = suite_cfg.get("global", {})
    merged = _deep_merge(global_cfg, scenario_cfg)

    for sec in ("experiment", "model", "data", "train", "old_algo", "main1_algo", "main2_algo"):
        _ensure_section(merged, sec)

    name = str(merged.get("name", scenario_cfg.get("name", "unnamed_scenario"))).strip()
    if not name:
        raise ValueError("每个 scenario 必须有非空 name")

    run_root = os.path.join(output_root, name)
    exp = merged["experiment"]
    exp["run_root"] = run_root
    exp["out_json"] = os.path.join(run_root, "benchmark_summary_three.json")
    exp["max_workers"] = int(workers)
    exp.setdefault("clean_seed_dir", False)
    exp.setdefault("seed_start", 0)
    exp.setdefault("n_seeds", 10)

    return name, merged


def _build_env(blas_threads):
    env = os.environ.copy()
    bt = str(int(blas_threads))
    env["OMP_NUM_THREADS"] = bt
    env["OPENBLAS_NUM_THREADS"] = bt
    env["MKL_NUM_THREADS"] = bt
    env["NUMEXPR_NUM_THREADS"] = bt
    env["VECLIB_MAXIMUM_THREADS"] = bt
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_one_scenario(python_bin, project_root, name, cfg, env, dry_run=False):
    run_root = cfg["experiment"]["run_root"]
    os.makedirs(run_root, exist_ok=True)

    cfg_path = os.path.join(run_root, "generated_benchmark_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    cmd = [python_bin, os.path.join(project_root, "benchmark_compare_three.py"), "--config", cfg_path]
    log_path = os.path.join(run_root, "runner.log")

    started = time.time()
    if dry_run:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[dry-run] {' '.join(cmd)}\n")
        return {
            "scenario": name,
            "ok": True,
            "returncode": 0,
            "duration_sec": 0.0,
            "config_path": cfg_path,
            "log_path": log_path,
            "cmd": cmd,
        }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n===== start {datetime.now().isoformat()} =====\n")
        f.write(f"cmd: {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=project_root, env=env, stdout=f, stderr=subprocess.STDOUT, check=False)

    elapsed = time.time() - started
    return {
        "scenario": name,
        "ok": (proc.returncode == 0),
        "returncode": int(proc.returncode),
        "duration_sec": float(elapsed),
        "config_path": cfg_path,
        "log_path": log_path,
        "cmd": cmd,
    }


def main():
    parser = argparse.ArgumentParser(description="Run paper simulation suite with CPU/resource control.")
    parser.add_argument("--suite-config", required=True, help="Path to simulation suite JSON.")
    parser.add_argument("--output-root", default="checkpoints/paper_simulation_suite", help="Output root directory.")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable.")

    parser.add_argument("--cpu-mode", choices=["auto", "manual"], default="auto",
                        help="auto=按预留核与利用率计算；manual=直接使用 --max-workers")
    parser.add_argument("--max-workers", type=int, default=0, help="manual 模式下并行 worker 数")
    parser.add_argument("--reserve-cpus", type=int, default=24, help="auto 模式下预留给同学/系统的核数")
    parser.add_argument("--cpu-utilization", type=float, default=0.7, help="auto 模式下可用核利用率")
    parser.add_argument("--max-workers-cap", type=int, default=0, help="对最终 worker 数再做全局上限限制")

    parser.add_argument("--blas-threads", type=int, default=1,
                        help="每个 worker 的 BLAS 线程数；<=0 表示自动估计")

    parser.add_argument("--scenarios", default="", help="仅运行指定场景名，逗号分隔")
    parser.add_argument("--continue-on-error", type=_str2bool, default=True)
    parser.add_argument("--dry-run", type=_str2bool, default=False)

    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    suite_cfg = _load_json(args.suite_config)

    scenarios = suite_cfg.get("scenarios", [])
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("suite-config 中 scenarios 必须是非空数组")

    pick_names = None
    if args.scenarios.strip():
        pick_names = {x.strip() for x in args.scenarios.split(",") if x.strip()}

    os.makedirs(args.output_root, exist_ok=True)

    total_cpu = os.cpu_count() or 1
    print(f"[suite] total_cpu={total_cpu} cpu_mode={args.cpu_mode}")

    selected_scenarios = []
    for sc in scenarios:
        sc_name = str(sc.get("name", "")).strip()
        if not sc_name:
            raise ValueError("每个 scenario 必须包含 name")
        if pick_names is not None and sc_name not in pick_names:
            continue
        selected_scenarios.append(sc)

    run_results = []
    failures = []
    suite_started = time.time()

    for idx, sc in enumerate(selected_scenarios, start=1):
        sc_resource = sc.get("resource", {}) if isinstance(sc.get("resource", {}), dict) else {}
        workers = _resolve_workers(total_cpu, args, sc_resource.get("max_workers_cap", None))
        blas_threads = _resolve_blas_threads(total_cpu, workers, args, sc_resource.get("blas_threads", None))

        name, bench_cfg = _prepare_benchmark_config(suite_cfg, sc, args.output_root, workers)
        env = _build_env(blas_threads)

        elapsed_suite = time.time() - suite_started
        done_before = idx - 1
        rate = (done_before / elapsed_suite) if elapsed_suite > 1e-9 else 0.0
        remain = max(0, len(selected_scenarios) - done_before)
        eta = (remain / rate) if rate > 1e-9 else float("inf")
        eta_text = _format_seconds(eta) if math.isfinite(eta) else "--:--"

        print(
            f"[scenario] ({idx}/{len(selected_scenarios)}) name={name} seeds={bench_cfg['experiment'].get('n_seeds')} "
            f"workers={workers} blas_threads={blas_threads} run_root={bench_cfg['experiment']['run_root']} eta={eta_text}"
        )

        result = _run_one_scenario(
            python_bin=args.python_bin,
            project_root=project_root,
            name=name,
            cfg=bench_cfg,
            env=env,
            dry_run=bool(args.dry_run),
        )
        result["workers"] = int(workers)
        result["blas_threads"] = int(blas_threads)
        run_results.append(result)

        if not result["ok"]:
            failures.append(name)
            print(f"[scenario] failed name={name} returncode={result['returncode']}")
            if not bool(args.continue_on_error):
                break
        else:
            print(f"[scenario] done name={name} duration_sec={result['duration_sec']:.1f}")

        done_count = len(run_results)
        pct = (100.0 * done_count / len(selected_scenarios)) if selected_scenarios else 100.0
        print(
            f"[suite-progress] done={done_count}/{len(selected_scenarios)} ({pct:.1f}%) "
            f"elapsed={_format_seconds(time.time() - suite_started)}"
        )

    suite_summary = {
        "suite_config": os.path.abspath(args.suite_config),
        "output_root": os.path.abspath(args.output_root),
        "cpu_mode": args.cpu_mode,
        "total_cpu": int(total_cpu),
        "reserve_cpus": int(args.reserve_cpus),
        "cpu_utilization": float(args.cpu_utilization),
        "max_workers_manual": int(args.max_workers),
        "blas_threads_arg": int(args.blas_threads),
        "dry_run": bool(args.dry_run),
        "num_scenarios_run": len(run_results),
        "num_failures": len(failures),
        "failed_scenarios": failures,
        "results": run_results,
    }

    summary_path = os.path.join(args.output_root, "suite_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, ensure_ascii=False, indent=2)

    print(f"[suite] summary_saved={summary_path}")
    if failures:
        raise SystemExit(f"Simulation suite finished with failures: {failures}")


if __name__ == "__main__":
    main()
