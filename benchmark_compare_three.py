import argparse
import inspect
import json
import os
import shutil
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import main as alg_old
import main_1 as alg_m1
import main_2 as alg_m2


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_stage_rmse(history):
    out = {}
    for h in history:
        if isinstance(h, dict) and ("stage" in h) and ("train_rmse" in h):
            out[int(h["stage"])] = float(h["train_rmse"])
    return out


def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**supported)


def _run_one_seed(seed, cfg):
    exp = cfg["experiment"]
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["train"]
    old_cfg = cfg.get("old_algo", {})
    m1_cfg = cfg.get("main1_algo", {})
    m2_cfg = cfg.get("main2_algo", {})

    run_root = exp["run_root"]
    seed_name = f"seed{seed}"
    old_dir = os.path.join(run_root, "main", seed_name)
    m1_dir = os.path.join(run_root, "main1", seed_name)
    m2_dir = os.path.join(run_root, "main2", seed_name)

    if bool(exp.get("clean_seed_dir", False)):
        for d in (old_dir, m1_dir, m2_dir):
            if os.path.isdir(d):
                shutil.rmtree(d)

    os.makedirs(old_dir, exist_ok=True)
    os.makedirs(m1_dir, exist_ok=True)
    os.makedirs(m2_dir, exist_ok=True)

    beta_scales = model.get("beta_scales", [1, 1, 1, 1, 1])
    beta_funcs = alg_old.true_beta_funcs_default(scales=beta_scales)

    result = {"seed": int(seed), "ok": False}
    try:
        def _progress(algo, info):
            stg = info.get("stage", info.get("t_end", "?"))
            rmse = info.get("train_rmse", None)
            if rmse is None:
                print(f"[progress] seed={seed} algo={algo} stage={stg}", flush=True)
            else:
                print(f"[progress] seed={seed} algo={algo} stage={stg} rmse={float(rmse):.6g}", flush=True)

        _, h_old = _call_with_supported_kwargs(
            alg_old.run_or_resume_incremental,
            checkpoint_dir=old_dir,
            t_final=float(data["t_final"]),
            n_per_segment=int(data["n_per_segment"]),
            P=int(model["P"]),
            signal_idx=list(model["signal_idx"]),
            beta_funcs=beta_funcs,
            sigma=float(data["sigma"]),
            k=int(model.get("k", 3)),
            n_inner_per_unit=int(old_cfg.get("n_inner_per_unit", 10)),
            seed_data=int(seed),
            seed_cv=int(train.get("seed_cv", 2025)),
            use_1se=bool(old_cfg.get("use_1se", True)),
            r_relax=int(old_cfg.get("r_relax", 2)),
            use_adaptive_cn=bool(old_cfg.get("use_adaptive_cn", True)),
            save_checkpoints=True,
            save_checkpoint_data=bool(train.get("save_checkpoint_data", False)),
            progress_hook=lambda info: _progress("main", info),
        )
        print(f"[progress] seed={seed} algo=main done", flush=True)

        _, h_m1 = _call_with_supported_kwargs(
            alg_m1.run_or_resume_incremental,
            checkpoint_dir=m1_dir,
            t_final=float(data["t_final"]),
            n_per_segment=int(data["n_per_segment"]),
            P=int(model["P"]),
            signal_idx=list(model["signal_idx"]),
            beta_funcs=beta_funcs,
            sigma=float(data["sigma"]),
            k=int(model.get("k", 3)),
            knot_step=float(m1_cfg.get("knot_step", 0.1)),
            seed_data=int(seed),
            seed_cv=int(train.get("seed_cv", 2025)),
            use_1se=bool(m1_cfg.get("use_1se", True)),
            save_checkpoint_data=bool(train.get("save_checkpoint_data", False)),
            debug=False,
            progress_hook=lambda info: _progress("main1", info),
        )
        print(f"[progress] seed={seed} algo=main1 done", flush=True)

        _, h_m2 = _call_with_supported_kwargs(
            alg_m2.run_or_resume_incremental,
            checkpoint_dir=m2_dir,
            t_final=float(data["t_final"]),
            n_per_segment=int(data["n_per_segment"]),
            P=int(model["P"]),
            signal_idx=list(model["signal_idx"]),
            beta_funcs=beta_funcs,
            sigma=float(data["sigma"]),
            k=int(model.get("k", 3)),
            knot_step=float(m2_cfg.get("knot_step", 0.1)),
            seed_data=int(seed),
            seed_cv=int(train.get("seed_cv", 2025)),
            use_1se=bool(m2_cfg.get("use_1se", True)),
            save_checkpoint_data=bool(train.get("save_checkpoint_data", False)),
            debug=False,
            local_window_units=float(m2_cfg.get("local_window_units", 2.0)),
            local_support_margin=float(m2_cfg.get("local_support_margin", 0.0)),
            progress_hook=lambda info: _progress("main2", info),
        )
        print(f"[progress] seed={seed} algo=main2 done", flush=True)

        result["main_rmse"] = _extract_stage_rmse(h_old)
        result["main1_rmse"] = _extract_stage_rmse(h_m1)
        result["main2_rmse"] = _extract_stage_rmse(h_m2)
        result["ok"] = True
        return result
    except Exception as e:
        result["error"] = repr(e)
        result["traceback"] = traceback.format_exc(limit=6)
        return result


def _aggregate(seed_results, t_final):
    stages = list(range(1, int(t_final) + 1))
    keys = ["main", "main1", "main2"]
    out = {"stages": stages}
    for k in keys:
        out[f"{k}_mean"] = {}
        out[f"{k}_std"] = {}

    for s in stages:
        vals = {k: [] for k in keys}
        for r in seed_results:
            if not r.get("ok", False):
                continue
            for k in keys:
                rm = r.get(f"{k}_rmse", {})
                if s in rm:
                    vals[k].append(float(rm[s]))
        for k in keys:
            out[f"{k}_mean"][s] = float(np.mean(vals[k])) if vals[k] else None
            out[f"{k}_std"][s] = float(np.std(vals[k], ddof=1)) if len(vals[k]) > 1 else 0.0

    return out


def _trend_stats(curve):
    vals = [v for _, v in sorted(curve.items()) if v is not None]
    if len(vals) <= 1:
        return {"delta": None, "increase_steps": 0, "n_steps": 0, "non_decreasing": None}
    diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    return {
        "delta": float(vals[-1] - vals[0]),
        "increase_steps": int(sum(d > 0 for d in diffs)),
        "n_steps": int(len(diffs)),
        "non_decreasing": bool(all(d >= -1e-12 for d in diffs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark main.py vs main_1.py vs main_2.py")
    parser.add_argument("--config", required=True)
    parser.add_argument("--t-final", type=int, default=0, help="Override t_final from config when > 0")
    parser.add_argument("--clean-seed-dir", type=int, default=-1,
                        help="Override clean_seed_dir: 1=true, 0=false, -1=use config")
    args = parser.parse_args()

    cfg = _load_json(args.config)
    if args.t_final and args.t_final > 0:
        cfg["data"]["t_final"] = int(args.t_final)
    if args.clean_seed_dir in (0, 1):
        cfg["experiment"]["clean_seed_dir"] = bool(args.clean_seed_dir)
    exp = cfg["experiment"]
    run_root = exp["run_root"]
    os.makedirs(run_root, exist_ok=True)

    n_seeds = int(exp["n_seeds"])
    seed_start = int(exp.get("seed_start", 0))
    max_workers = int(exp.get("max_workers", 1))
    seeds = [seed_start + i for i in range(n_seeds)]

    print(f"[benchmark3] run_root={run_root}")
    print(f"[benchmark3] seeds={n_seeds} seed_start={seed_start} max_workers={max_workers}")

    results = []
    failures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {}
        for seed in seeds:
            print(f"[submit] seed={seed}")
            futs[ex.submit(_run_one_seed, seed, cfg)] = seed
        for fut in as_completed(futs):
            seed = futs[fut]
            r = fut.result()
            results.append(r)
            if r.get("ok", False):
                print(
                    f"[done] seed={seed} ok "
                    f"main={len(r.get('main_rmse', {}))} "
                    f"main1={len(r.get('main1_rmse', {}))} "
                    f"main2={len(r.get('main2_rmse', {}))}"
                )
            else:
                failures.append(seed)
                print(f"[done] seed={seed} failed: {r.get('error')}")

    agg = _aggregate(results, cfg["data"]["t_final"])
    stats = {
        "main": _trend_stats(agg["main_mean"]),
        "main1": _trend_stats(agg["main1_mean"]),
        "main2": _trend_stats(agg["main2_mean"]),
    }

    summary = {
        "config": cfg,
        "results": results,
        "aggregate": agg,
        "trend_stats": stats,
        "num_failures": len(failures),
        "failed_seeds": failures,
    }

    out_json = exp.get("out_json", os.path.join(run_root, "benchmark_summary_three.json"))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[saved] {out_json}")
    print(f"[trend] main={stats['main']}")
    print(f"[trend] main1={stats['main1']}")
    print(f"[trend] main2={stats['main2']}")

    if failures:
        raise SystemExit(f"Benchmark finished with failures: {failures}")


if __name__ == "__main__":
    main()
