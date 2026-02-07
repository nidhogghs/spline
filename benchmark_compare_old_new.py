import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import main as old_main
import main_1 as new_main


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_stage_rmse(history):
    out = {}
    for h in history:
        if isinstance(h, dict) and ("stage" in h) and ("train_rmse" in h):
            out[int(h["stage"])] = float(h["train_rmse"])
    return out


def _run_one_seed(seed, cfg):
    exp = cfg["experiment"]
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["train"]
    old_cfg = cfg.get("old_algo", {})
    new_cfg = cfg.get("new_algo", {})

    run_root = exp["run_root"]
    seed_name = f"seed{seed}"
    old_dir = os.path.join(run_root, "old", seed_name)
    new_dir = os.path.join(run_root, "new", seed_name)
    os.makedirs(old_dir, exist_ok=True)
    os.makedirs(new_dir, exist_ok=True)

    beta_scales = model.get("beta_scales", [1, 1, 1, 1, 1])
    beta_funcs = old_main.true_beta_funcs_default(scales=beta_scales)

    result = {"seed": int(seed), "ok": False}
    try:
        _, h_old = old_main.run_or_resume_incremental(
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
        )
        old_rmse = _extract_stage_rmse(h_old)

        _, h_new = new_main.run_or_resume_incremental(
            checkpoint_dir=new_dir,
            t_final=float(data["t_final"]),
            n_per_segment=int(data["n_per_segment"]),
            P=int(model["P"]),
            signal_idx=list(model["signal_idx"]),
            beta_funcs=beta_funcs,
            sigma=float(data["sigma"]),
            k=int(model.get("k", 3)),
            knot_step=float(new_cfg.get("knot_step", 0.1)),
            seed_data=int(seed),
            seed_cv=int(train.get("seed_cv", 2025)),
            use_1se=bool(new_cfg.get("use_1se", True)),
            save_checkpoint_data=bool(train.get("save_checkpoint_data", False)),
            debug=False,
        )
        new_rmse = _extract_stage_rmse(h_new)

        result["old_rmse"] = old_rmse
        result["new_rmse"] = new_rmse
        result["ok"] = True
        return result
    except Exception as e:
        result["error"] = repr(e)
        result["traceback"] = traceback.format_exc(limit=4)
        return result


def _aggregate(seed_results, t_final):
    stages = list(range(1, int(t_final) + 1))
    old_mean = {}
    new_mean = {}
    old_std = {}
    new_std = {}

    for s in stages:
        old_vals = []
        new_vals = []
        for r in seed_results:
            if not r.get("ok", False):
                continue
            if s in r.get("old_rmse", {}):
                old_vals.append(float(r["old_rmse"][s]))
            if s in r.get("new_rmse", {}):
                new_vals.append(float(r["new_rmse"][s]))

        old_mean[s] = float(np.mean(old_vals)) if old_vals else None
        new_mean[s] = float(np.mean(new_vals)) if new_vals else None
        old_std[s] = float(np.std(old_vals, ddof=1)) if len(old_vals) > 1 else 0.0
        new_std[s] = float(np.std(new_vals, ddof=1)) if len(new_vals) > 1 else 0.0

    return {
        "stages": stages,
        "old_mean": old_mean,
        "new_mean": new_mean,
        "old_std": old_std,
        "new_std": new_std,
    }


def _trend_stats(curve):
    vals = [v for _, v in sorted(curve.items()) if v is not None]
    if len(vals) <= 1:
        return {"delta": None, "increase_steps": 0, "n_steps": 0}
    diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    return {
        "delta": float(vals[-1] - vals[0]),
        "increase_steps": int(sum(d > 0 for d in diffs)),
        "n_steps": int(len(diffs)),
        "non_decreasing": bool(all(d >= -1e-12 for d in diffs)),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark old main.py vs new main_1.py")
    parser.add_argument("--config", required=True, help="Benchmark config json path")
    args = parser.parse_args()

    cfg = _load_json(args.config)
    exp = cfg["experiment"]
    run_root = exp["run_root"]
    os.makedirs(run_root, exist_ok=True)

    n_seeds = int(exp["n_seeds"])
    seed_start = int(exp.get("seed_start", 0))
    max_workers = int(exp.get("max_workers", 1))
    seeds = [seed_start + i for i in range(n_seeds)]

    print(f"[benchmark] run_root={run_root}")
    print(f"[benchmark] seeds={n_seeds} seed_start={seed_start} max_workers={max_workers}")

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
                n_old = len(r.get("old_rmse", {}))
                n_new = len(r.get("new_rmse", {}))
                print(f"[done] seed={seed} ok old_stages={n_old} new_stages={n_new}")
            else:
                failures.append(seed)
                print(f"[done] seed={seed} failed: {r.get('error')}")

    agg = _aggregate(results, cfg["data"]["t_final"])
    stats = {
        "old": _trend_stats(agg["old_mean"]),
        "new": _trend_stats(agg["new_mean"]),
    }

    summary = {
        "config": cfg,
        "results": results,
        "aggregate": agg,
        "trend_stats": stats,
        "num_failures": len(failures),
        "failed_seeds": failures,
    }

    out_json = exp.get("out_json", os.path.join(run_root, "benchmark_summary.json"))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[saved] {out_json}")
    print(f"[trend] old={stats['old']}")
    print(f"[trend] new={stats['new']}")

    if failures:
        raise SystemExit(f"Benchmark finished with failures: {failures}")


if __name__ == "__main__":
    main()
