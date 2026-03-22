"""
真正的增量 VCM (Varying-Coefficient Model) 算法。

核心思想：
- Stage 1: 在 [0,1] 上做完整的 group-lasso 拟合
- Stage s>1: 扩展到 [0,s]
  - O (Old) 区间的 basis 系数冻结不动
  - 只在 C+N (Cross + New) 区间的数据上做优化
  - C = 跨越旧边界的 basis，N = 完全在新区间的 basis
  - 不使用 Old 区间的数据做误差计算或参数优化
"""
import argparse
import json
import os
import threading
import time

import numpy as np
from numpy.linalg import norm
from scipy.interpolate import BSpline


# =========================================================
# 0) 工具函数
# =========================================================

def _str2bool(s):
    if isinstance(s, bool):
        return s
    s = str(s).strip().lower()
    if s in ("1", "true", "t", "yes", "y"):
        return True
    if s in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Expected a boolean value.")


def _parse_list_int(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def _parse_list_float(s):
    return [float(x) for x in str(s).split(",") if x.strip()]


def _format_seconds(seconds):
    s = max(0, int(round(float(seconds))))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _start_heartbeat_watcher(algo_name, stage_getter, total_stages, stop_event, interval_sec=90.0, start_time=None):
    interval = float(interval_sec)
    if interval <= 0:
        return None

    run_start = float(start_time) if start_time is not None else time.time()

    def _watch():
        while not stop_event.wait(interval):
            now = time.time()
            try:
                cur_stage = int(round(float(stage_getter())))
            except Exception:
                cur_stage = 0
            cur_stage = max(0, cur_stage)
            elapsed = max(0.0, now - run_start)
            avg_stage = (elapsed / float(cur_stage)) if cur_stage > 0 else None
            avg_stage_text = f"{avg_stage:.1f}s" if avg_stage is not None else "NA"

            if total_stages and total_stages > 0:
                pct = 100.0 * float(cur_stage) / float(total_stages)
                print(
                    f"[heartbeat] algo={algo_name} latest_stage={cur_stage}/{int(total_stages)} ({pct:.1f}%) "
                    f"elapsed={_format_seconds(elapsed)} avg_stage={avg_stage_text}",
                    flush=True,
                )
            else:
                print(
                    f"[heartbeat] algo={algo_name} latest_stage={cur_stage} "
                    f"elapsed={_format_seconds(elapsed)} avg_stage={avg_stage_text}",
                    flush=True,
                )

    t = threading.Thread(target=_watch, daemon=True)
    t.start()
    return t


def true_beta_funcs_default(scales=None):
    if scales is None:
        scales = [1.0, 1.0, 1.0, 1.0, 1.0]
    return [
        lambda t: scales[0] * (-0.5 + 0.6 * np.cos(2 * np.pi * t)) + 0.15 * np.log1p(0.3 * np.asarray(t, dtype=float)),
        lambda t: scales[1] * (-0.5 + 0.6 * np.cos(2 * np.pi * t)),
        lambda t: scales[2] * (0.7 * np.sin(4 * np.pi * t)),
        lambda t: scales[3] * (0.7 * np.sin(4 * np.pi * t)),
        lambda t: scales[4] * (0.4 * np.cos(3 * np.pi * t)),
    ]


def _parse_int_list(v):
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return _parse_list_int(v)


def _parse_float_list(v):
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return _parse_list_float(v)


def _build_single_beta_function(spec):
    def _log_tail_term(t, tail_cfg):
        if not isinstance(tail_cfg, dict):
            return 0.0
        start = float(tail_cfg.get("start", 0.0))
        amp = float(tail_cfg.get("amplitude", 0.0))
        slope = float(tail_cfg.get("slope", 1.0))
        if slope <= 0:
            slope = 1.0
        sign = float(tail_cfg.get("sign", -1.0))
        tt = np.asarray(t, dtype=float)
        dt = np.maximum(tt - start, 0.0)
        return sign * amp * np.log1p(slope * dt)

    kind = str(spec.get("type", "sin")).strip().lower()

    if kind in ("sin", "cos"):
        amp = float(spec.get("amplitude", spec.get("amp", 1.0)))
        freq_pi = float(spec.get("frequency_pi", spec.get("freq_pi", 2.0)))
        phase = float(spec.get("phase", 0.0))
        bias = float(spec.get("bias", 0.0))
        tail_cfg = spec.get("log_tail", None)
        if kind == "sin":
            return (
                lambda t, a=amp, f=freq_pi, ph=phase, b=bias, tc=tail_cfg:
                b + a * np.sin(f * np.pi * t + ph) + _log_tail_term(t, tc)
            )
        return (
            lambda t, a=amp, f=freq_pi, ph=phase, b=bias, tc=tail_cfg:
            b + a * np.cos(f * np.pi * t + ph) + _log_tail_term(t, tc)
        )

    if kind == "trig_mix":
        terms = list(spec.get("terms", []))
        if not terms:
            raise ValueError("beta function spec type='trig_mix' requires non-empty 'terms'.")
        bias = float(spec.get("bias", 0.0))
        tail_cfg = spec.get("log_tail", None)

        parsed = []
        for term in terms:
            tk = str(term.get("kind", "sin")).strip().lower()
            if tk not in ("sin", "cos"):
                raise ValueError(f"Unsupported trig term kind: {tk}")
            a = float(term.get("amplitude", term.get("amp", 1.0)))
            f = float(term.get("frequency_pi", term.get("freq_pi", 2.0)))
            ph = float(term.get("phase", 0.0))
            parsed.append((tk, a, f, ph))

        def _fn(t, b=bias, ps=parsed, tc=tail_cfg):
            out = np.zeros_like(np.asarray(t, dtype=float), dtype=float) + b
            for tk, a, f, ph in ps:
                if tk == "sin":
                    out = out + a * np.sin(f * np.pi * t + ph)
                else:
                    out = out + a * np.cos(f * np.pi * t + ph)
            out = out + _log_tail_term(t, tc)
            return out

        return _fn

    raise ValueError(f"Unsupported beta function spec type: {kind}")


def build_beta_functions_from_config(beta_cfg):
    if beta_cfg is None:
        return true_beta_funcs_default()

    if isinstance(beta_cfg, dict) and "specs" in beta_cfg:
        specs = list(beta_cfg.get("specs", []))
        if not specs:
            raise ValueError("beta config has empty 'specs'.")
        return [_build_single_beta_function(s) for s in specs]

    if isinstance(beta_cfg, dict) and "scales" in beta_cfg:
        scales = _parse_float_list(beta_cfg.get("scales", []))
        return true_beta_funcs_default(scales=scales)

    if isinstance(beta_cfg, list):
        if not beta_cfg:
            raise ValueError("beta config list is empty.")
        return [_build_single_beta_function(s) for s in beta_cfg]

    raise ValueError("Unsupported beta config format. Use {'specs': [...]} or {'scales': [...]} or a list.")


# =========================================================
# 1) 数据模拟器
# =========================================================

class VCMSimulator:
    def __init__(self, P, signal_idx, beta_funcs, sigma, seed_base=0, standardize_X=True):
        self.P = int(P)
        self.signal_idx = list(signal_idx)
        self.beta_funcs = list(beta_funcs)
        self.sigma = float(sigma)
        self.standardize_X = bool(standardize_X)
        self.seed_base = int(seed_base)

    def sample_segment(self, a, b, n, segment_id):
        rng_t = np.random.default_rng(self.seed_base + 10_000 + int(segment_id))
        rng_x = np.random.default_rng(self.seed_base + 20_000 + int(segment_id))
        rng_e = np.random.default_rng(self.seed_base + 30_000 + int(segment_id))

        t = np.sort(rng_t.uniform(float(a), float(b), int(n)))
        X = rng_x.standard_normal((int(n), self.P))
        if self.standardize_X:
            X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

        beta_true = np.zeros((int(n), self.P), dtype=float)
        for j, p in enumerate(self.signal_idx):
            beta_true[:, p] = self.beta_funcs[j % len(self.beta_funcs)](t)

        y = np.sum(X * beta_true, axis=1) + self.sigma * rng_e.standard_normal(int(n))
        return t, X, y, beta_true


# =========================================================
# 2) B-spline / VCM 基础函数
# =========================================================

def make_global_knots(a, b, k, knot_step):
    a = float(a)
    b = float(b)
    k = int(k)
    h = float(knot_step)
    if b <= a:
        raise ValueError("Require b > a.")
    if h <= 0:
        raise ValueError("knot_step must be positive.")

    ratio = (b - a) / h
    n_steps = int(round(ratio))
    if abs(ratio - n_steps) > 1e-10:
        raise ValueError("(b-a) must be a multiple of knot_step.")

    interior = np.array([a + i * h for i in range(1, n_steps)], dtype=float)
    knots = np.r_[np.repeat(a, k + 1), interior, np.repeat(b, k + 1)]
    return np.asarray(knots, dtype=float)


def bspline_design_matrix(x, knots, k):
    x = np.asarray(x, dtype=float)
    t = np.asarray(knots, dtype=float)
    m = len(t) - (int(k) + 1)
    coeff = np.eye(m, dtype=float)
    B = np.asarray(BSpline(t, coeff, int(k), extrapolate=False, axis=0)(x), dtype=float)
    B[np.isnan(B)] = 0.0
    return B


def build_vcm_design(B, X):
    B = np.asarray(B, dtype=float)
    X = np.asarray(X, dtype=float)
    n, m = B.shape
    P = X.shape[1]
    return (X[:, :, None] * B[:, None, :]).reshape(n, P * m)


def split_blocks(vec, m, P):
    return [vec[p * m:(p + 1) * m] for p in range(P)]


def _coef_to_block_matrix(vec, m, P):
    return np.asarray(vec, dtype=float).reshape(int(P), int(m))


def _predict_from_coef_matrix(B, X, coef_mat):
    beta_hat = B @ np.asarray(coef_mat, dtype=float).T
    return np.sum(np.asarray(X, dtype=float) * beta_hat, axis=1)


def post_lasso_debias(Phi, y, c_lasso, m, P, active_thresh=1e-8):
    """
    Post-Lasso Debiasing: 对 Group-Lasso 选出的活跃组做无惩罚 OLS 重拟合。

    参数:
        Phi: (n, P*m) 的 VCM 设计矩阵
        y: (n,) 响应变量
        c_lasso: (P*m,) Group-Lasso 得到的系数
        m: B-spline basis 数量
        P: 协变量数量
        active_thresh: 判断组是否活跃的阈值

    返回:
        c_debias: (P*m,) debiased 后的系数
        active_groups: 活跃组的索引列表
    """
    blocks = split_blocks(c_lasso, m, P)
    active_groups = []
    for p in range(P):
        if float(norm(blocks[p])) > active_thresh:
            active_groups.append(p)

    if len(active_groups) == 0:
        return c_lasso.copy(), active_groups

    # 构建活跃组的列索引
    active_cols = []
    for p in active_groups:
        active_cols.extend(range(p * m, (p + 1) * m))
    active_cols = np.array(active_cols, dtype=int)

    # 提取活跃组的设计矩阵子集
    Phi_active = Phi[:, active_cols]

    # OLS 重拟合 (使用 lstsq 确保数值稳定)
    c_active, _, _, _ = np.linalg.lstsq(Phi_active, y, rcond=None)

    # 组装完整的 debiased 系数
    c_debias = np.zeros_like(c_lasso)
    c_debias[active_cols] = c_active

    return c_debias, active_groups


def post_lasso_debias_cn(Phi_cn, y_cn_res, c_cn_lasso, m_cn, P, active_thresh=1e-8):
    """
    Post-Lasso Debiasing (增量版): 对 CN 区间的活跃组做 OLS 重拟合。

    与 post_lasso_debias 逻辑完全一致，只是作用于 CN 子空间。

    参数:
        Phi_cn: (n_cn, P*m_cn) CN 区间的 VCM 设计矩阵
        y_cn_res: (n_cn,) 减去 frozen(O) 后的残差
        c_cn_lasso: (P*m_cn,) CN 空间的 Group-Lasso 系数
        m_cn: CN basis 数量
        P: 协变量数量
        active_thresh: 判断组是否活跃的阈值

    返回:
        c_cn_debias: (P*m_cn,) debiased 后的系数
        active_groups: 活跃组的索引列表
    """
    return post_lasso_debias(Phi_cn, y_cn_res, c_cn_lasso, m_cn, P, active_thresh)








def gram_R(knots, k, a, b, grid=3000):
    gx = np.linspace(float(a), float(b), int(grid))
    B = bspline_design_matrix(gx, knots, k)
    w = np.ones(gx.shape[0], dtype=float)
    w[0] *= 0.5
    w[-1] *= 0.5
    w *= (float(b) - float(a)) / (gx.shape[0] - 1)
    return B.T @ (B * w[:, None])


def group_weights(B, X):
    row_energy = np.sum(B * B, axis=1)
    return np.sqrt((X * X * row_energy[:, None]).sum(axis=0) / B.shape[0] + 1e-12)


def _safe_cholesky(R, jitter=1e-10):
    R = np.asarray(R, dtype=float)
    eye = np.eye(R.shape[0], dtype=float)
    for mul in (0.0, 1.0, 10.0, 100.0, 1000.0):
        try:
            return np.linalg.cholesky(R + (jitter * mul) * eye)
        except np.linalg.LinAlgError:
            continue
    return np.linalg.cholesky(R + 1e-6 * eye)


def lambda_max_R(Phi, y, m, R):
    """计算 lambda_max（与 main.py 原始版本一致，接受 VCM 设计矩阵 Phi）。"""
    L = _safe_cholesky(R)
    P = Phi.shape[1] // m
    lam = 0.0
    for p in range(P):
        g = Phi[:, p * m:(p + 1) * m].T @ y
        u = np.linalg.solve(L.T, g)
        lam = max(lam, float(norm(u)))
    return lam


def group_soft_thresh(blocks, tau, R, lam, w):
    out = []
    for v, wp in zip(blocks, w):
        nr = float(np.sqrt(max(0.0, v.T @ R @ v)))
        thr = tau * lam * wp
        out.append(np.zeros_like(v) if nr <= thr else (1 - thr / nr) * v)
    return out


def fista_group_lasso(XtX, Xty, lam, m, P, R, w, max_iter=6000, tol=1e-7):
    """显式 Gram 矩阵版 FISTA group-lasso（与 main.py 原始版本一致）。"""
    d = m * P
    Ls = float(np.linalg.norm(XtX, 2))
    Ls = Ls if (np.isfinite(Ls) and Ls > 0) else 1.0
    step = 1.0 / Ls

    c = np.zeros(d)
    z = c.copy()
    tN = 1.0

    for _ in range(max_iter):
        grad = XtX @ z - Xty
        yv = z - step * grad
        yb = split_blocks(yv, m, P)
        c_new = np.concatenate(group_soft_thresh(yb, step, R, lam, w))

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * tN * tN))
        z = c_new + (tN - 1) / t_new * (c_new - c)

        if norm(c_new - c) <= tol * max(1.0, norm(c)):
            return c_new
        c, tN = c_new, t_new

    return c


# =========================================================
# 3) CV 和 lambda 路径
# =========================================================

def kfold_indices(n, K=5, seed=2025):
    rng = np.random.default_rng(int(seed))
    idx = np.arange(int(n))
    rng.shuffle(idx)
    return np.array_split(idx, int(K))


def make_lambda_path(lam_max, n_lam=30, min_ratio=5e-4):
    lam_max = float(lam_max)
    if (not np.isfinite(lam_max)) or lam_max <= 1e-12:
        lam_max = 1.0
    return np.geomspace(lam_max, lam_max * float(min_ratio), int(n_lam))


def cv_select_lambda_plain(B, X, y, R, lam_path, K=5, seed=2025, use_1se=True):
    """Stage 1 的 plain CV（无 frozen 部分），使用显式 Gram 矩阵。"""
    n = len(y)
    P = X.shape[1]
    m = B.shape[1]
    folds = kfold_indices(n, K=K, seed=seed)
    mse = np.zeros((len(lam_path), len(folds)), dtype=float)
    all_idx = np.arange(n)

    for kf, val in enumerate(folds):
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[val] = False
        tr = all_idx[tr_mask]

        Btr, Bv = B[tr], B[val]
        Xtr, Xv = X[tr], X[val]
        ytr, yv = y[tr], y[val]

        Phi_tr = build_vcm_design(Btr, Xtr)
        Phi_v = build_vcm_design(Bv, Xv)
        XtX = Phi_tr.T @ Phi_tr
        Xty = Phi_tr.T @ ytr
        w_tr = group_weights(Btr, Xtr)

        for li, lam in enumerate(lam_path):
            c = fista_group_lasso(XtX, Xty, float(lam), m, P, R, w_tr)
            mse[li, kf] = float(np.mean((yv - Phi_v @ c) ** 2))

    mm = mse.mean(axis=1)
    best = int(np.argmin(mm))
    if use_1se and len(folds) > 1:
        se = mse.std(axis=1, ddof=1) / np.sqrt(len(folds))
        thr = mm[best] + se[best]
        best = int(np.where(mm <= thr)[0][0])
    return float(lam_path[best])


def cv_select_lambda_frozen_cn(B_cn, X_cn, y_cn_res, R_cn, lam_path, K=5, seed=2025, use_1se=True):
    """
    真正增量的 Frozen-CV：只在 CN 区间数据上做交叉验证。
    使用显式 Gram 矩阵版 FISTA。

    y_cn_res 已经是减去 frozen(O+C) 贡献后的残差。
    只需在优化区间的 basis 上做普通的 group-lasso CV。
    """
    n = len(y_cn_res)
    P = X_cn.shape[1]
    m_cn = B_cn.shape[1]
    folds = kfold_indices(n, K=K, seed=seed)
    mse = np.zeros((len(lam_path), len(folds)), dtype=float)
    all_idx = np.arange(n)

    for kf, val in enumerate(folds):
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[val] = False
        tr = all_idx[tr_mask]

        Btr, Bv = B_cn[tr], B_cn[val]
        Xtr, Xv = X_cn[tr], X_cn[val]
        ytr, yv = y_cn_res[tr], y_cn_res[val]

        Phi_tr = build_vcm_design(Btr, Xtr)
        Phi_v = build_vcm_design(Bv, Xv)
        XtX = Phi_tr.T @ Phi_tr
        Xty = Phi_tr.T @ ytr
        w_tr = group_weights(Btr, Xtr)

        for li, lam in enumerate(lam_path):
            c = fista_group_lasso(XtX, Xty, float(lam), m_cn, P, R_cn, w_tr)
            y_pred = Phi_v @ c
            mse[li, kf] = float(np.mean((yv - y_pred) ** 2))

    mm = mse.mean(axis=1)
    best = int(np.argmin(mm))
    if use_1se and len(folds) > 1:
        se = mse.std(axis=1, ddof=1) / np.sqrt(len(folds))
        thr = mm[best] + se[best]
        best = int(np.where(mm <= thr)[0][0])

    return float(lam_path[best])


def apply_relax(idx_new_matched, knots_new, k, r_relax):
    """
    将最靠近边界的 r_relax 个 basis 从 O 中释放出来（加入 CN）。
    与 main.py 原始版本一致。
    """
    idx_new_matched = np.asarray(idx_new_matched, dtype=int)
    if r_relax <= 0 or len(idx_new_matched) <= r_relax:
        return idx_new_matched, np.array([], dtype=int)

    right_end = np.array([knots_new[j + k] for j in idx_new_matched], dtype=float)
    order = np.argsort(right_end)

    move = np.zeros_like(idx_new_matched, dtype=bool)
    move[order[-r_relax:]] = True

    idx_o = idx_new_matched[~move]
    idx_relaxed = idx_new_matched[move]
    return idx_o, idx_relaxed


def adaptive_cn_refit(B_cn, X, y_res, R_cn, lam_path,
                      seed_cv=2025, use_1se=True, eps=1e-6, delta=1.0):
    """
    One-step adaptive group-lasso in CN space（与 main.py 原始版本一致）:
    - warm fit (use weak lambda) -> r0 -> w_ad
    - CV over lam_path with w_ad
    - final refit
    """
    n = len(y_res)
    P = X.shape[1]
    m_cn = B_cn.shape[1]

    Phi = build_vcm_design(B_cn, X)

    # Warm fit: use the smallest lambda (weak penalty) as initialization
    lam_warm = float(lam_path[-1])
    w_warm = group_weights(B_cn, X)
    c0 = fista_group_lasso(Phi.T @ Phi, Phi.T @ y_res, lam_warm, m_cn, P, R_cn, w_warm)

    blocks0 = split_blocks(c0, m_cn, P)
    r0 = np.array([float(np.sqrt(max(0.0, b.T @ R_cn @ b))) for b in blocks0])
    w_ad = 1.0 / (r0 + eps) ** delta

    folds = kfold_indices(n, K=5, seed=seed_cv)
    mse = np.zeros((len(lam_path), 5))

    for li, lam in enumerate(lam_path):
        for kf, val in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), val)
            Btr, Bv = B_cn[tr], B_cn[val]
            Xtr, Xv = X[tr], X[val]
            ytr, yv = y_res[tr], y_res[val]

            Phi_tr = build_vcm_design(Btr, Xtr)
            Phi_v = build_vcm_design(Bv, Xv)

            c_tmp = fista_group_lasso(Phi_tr.T @ Phi_tr, Phi_tr.T @ ytr,
                                      float(lam), m_cn, P, R_cn, w_ad)
            mse[li, kf] = np.mean((yv - Phi_v @ c_tmp) ** 2)

    mm = mse.mean(1)
    best = int(np.argmin(mm))
    if use_1se:
        se = mse.std(1, ddof=1) / np.sqrt(5)
        thr = mm[best] + se[best]
        best = int(np.where(mm <= thr)[0][0])

    lam_best = float(lam_path[best])
    c_final = fista_group_lasso(Phi.T @ Phi, Phi.T @ y_res, lam_best, m_cn, P, R_cn, w_ad)
    return lam_best, c_final


# =========================================================
# 4) OCN 分区
# =========================================================

def local_span_matrix(knots, k):
    t = np.asarray(knots, dtype=float)
    m = len(t) - (int(k) + 1)
    return np.stack([t[j:j + int(k) + 2] for j in range(m)], axis=0)


def basis_supports(knots, k):
    t = np.asarray(knots, dtype=float)
    m = len(t) - (int(k) + 1)
    left = np.array([t[j] for j in range(m)], dtype=float)
    right = np.array([t[j + int(k) + 1] for j in range(m)], dtype=float)
    return left, right


def partition_OCN(knots_old, knots_new, k, old_end, eps=1e-10):
    spans_old = local_span_matrix(knots_old, k)
    spans_new = local_span_matrix(knots_new, k)

    old_to_new = {}
    new_to_old = {}

    for i in range(spans_old.shape[0]):
        hit = np.where(np.all(np.abs(spans_new - spans_old[i]) <= eps, axis=1))[0]
        if hit.size == 1:
            j = int(hit[0])
            old_to_new[int(i)] = j
            new_to_old[j] = int(i)

    idx_o_new = np.array(sorted(new_to_old.keys()), dtype=int)
    idx_o_old = np.array([new_to_old[j] for j in idx_o_new], dtype=int)

    idx_all_new = np.arange(spans_new.shape[0], dtype=int)
    idx_unmatched_new = np.setdiff1d(idx_all_new, idx_o_new)

    left_new, right_new = basis_supports(knots_new, k)
    idx_n_new = idx_unmatched_new[left_new[idx_unmatched_new] >= float(old_end) - eps]
    idx_c_new = np.setdiff1d(idx_unmatched_new, idx_n_new)

    idx_all_old = np.arange(spans_old.shape[0], dtype=int)
    idx_c_old = np.setdiff1d(idx_all_old, idx_o_old)

    idx_cn_new = np.sort(np.r_[idx_c_new, idx_n_new])

    return {
        "idx_o_new": idx_o_new,
        "idx_o_old": idx_o_old,
        "idx_c_new": idx_c_new,
        "idx_n_new": idx_n_new,
        "idx_cn_new": idx_cn_new,
        "idx_c_old": idx_c_old,
        "spans_old": spans_old,
        "spans_new": spans_new,
    }


def _span_to_tuple(arr, nd=12):
    return tuple(np.round(np.asarray(arr, dtype=float), nd).tolist())


def debug_partition_report(part, old_end, max_show=3):
    spans_old = part["spans_old"]
    spans_new = part["spans_new"]
    idx_o_new = part["idx_o_new"]
    idx_o_old = part["idx_o_old"]
    idx_c_new = part["idx_c_new"]
    idx_n_new = part["idx_n_new"]

    print("[debug] old_end=", float(old_end))
    print(
        "[debug] counts: O=", len(idx_o_new),
        " C=", len(idx_c_new),
        " N=", len(idx_n_new),
        " CN=", len(part["idx_cn_new"]),
    )

    def show_examples(label, idx, spans):
        use = idx[:max_show]
        ex = [_span_to_tuple(spans[j]) for j in use]
        print(f"[debug] {label} examples ({len(use)}): {ex}")

    show_examples("O(new)", idx_o_new, spans_new)
    show_examples("C(new)", idx_c_new, spans_new)
    show_examples("N(new)", idx_n_new, spans_new)

    ok_o = True
    for old_i, new_j in zip(idx_o_old.tolist(), idx_o_new.tolist()):
        if not np.allclose(spans_old[old_i], spans_new[new_j], atol=1e-10, rtol=0):
            ok_o = False
            break
    print(f"[debug] verify O spans identical: {ok_o}")

    old_span_set = {_span_to_tuple(x) for x in spans_old}
    c_has_old = any(_span_to_tuple(spans_new[j]) in old_span_set for j in idx_c_new.tolist())
    n_has_old = any(_span_to_tuple(spans_new[j]) in old_span_set for j in idx_n_new.tolist())

    print(f"[debug] verify C spans differ from old: {not c_has_old}")
    print(f"[debug] verify N spans absent in old: {not n_has_old}")


# =========================================================
# 5) 真正的增量 Trainer
# =========================================================

def _compute_cn_data_range(knots_new, k, idx_cn):
    """
    计算 CN basis 函数的 support 覆盖的数据范围 [cn_left, cn_right]。
    任何 t 值落入此区间的数据点才与 CN basis 有关。
    """
    left_new, right_new = basis_supports(knots_new, k)
    cn_left = float(np.min(left_new[idx_cn]))
    cn_right = float(np.max(right_new[idx_cn]))
    return cn_left, cn_right


class IncrementalVCMTrainer:
    """
    真正的增量 VCM Trainer。

    关键特性：
    - 不累积全部历史数据 (t_all / X_all / y_all)
    - extend_one_stage 只在 C+N 区间的数据上做优化和 CV
    - O 区间的系数冻结不动，误差计算只在 CN 区间上
    """

    def __init__(self, k, knot_step, P, seed_cv=2025, use_1se=False, r_relax=0, adaptive=False, debug=False):
        self.k = int(k)
        self.knot_step = float(knot_step)
        self.P = int(P)
        self.seed_cv = int(seed_cv)
        self.use_1se = bool(use_1se)
        self.r_relax = int(r_relax)
        self.adaptive = bool(adaptive)
        self.debug = bool(debug)

        self.t_end = None
        self.knots = None
        self.coef_blocks = None  # list of length P, each is (m,) array

    def save_checkpoint(self, prefix_path):
        """保存模型状态（不保存数据，因为真增量不需要累积数据）。"""
        if self.t_end is None:
            raise RuntimeError("Nothing to save.")

        meta = {
            "t_end": float(self.t_end),
            "k": int(self.k),
            "knot_step": float(self.knot_step),
            "P": int(self.P),
            "seed_cv": int(self.seed_cv),
            "use_1se": bool(self.use_1se),
            "r_relax": int(self.r_relax),
            "adaptive": bool(self.adaptive),
        }

        coef_mat = np.stack(self.coef_blocks, axis=0)

        os.makedirs(os.path.dirname(prefix_path) or ".", exist_ok=True)
        np.savez_compressed(
            prefix_path + ".npz",
            knots=self.knots,
            coef_mat=coef_mat,
        )

        with open(prefix_path + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_checkpoint(cls, prefix_path, debug=False):
        with open(prefix_path + ".json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        obj = cls(
            k=meta["k"],
            knot_step=meta["knot_step"],
            P=meta["P"],
            seed_cv=meta["seed_cv"],
            use_1se=meta["use_1se"],
            r_relax=meta.get("r_relax", 0),
            adaptive=meta.get("adaptive", False),
            debug=debug,
        )

        arr = np.load(prefix_path + ".npz", allow_pickle=False)
        obj.knots = arr["knots"]
        coef_mat = arr["coef_mat"]
        if int(coef_mat.shape[0]) != int(obj.P):
            raise ValueError(
                f"Checkpoint 系数维度不一致: meta.P={obj.P}, coef_mat_rows={coef_mat.shape[0]}。"
                f"请清理该 checkpoint 目录后重跑。"
            )
        obj.coef_blocks = [coef_mat[p].copy() for p in range(coef_mat.shape[0])]
        obj.t_end = float(meta["t_end"])

        return obj

    def fit_stage1(self, t, X, y):
        """
        Stage 1: 在 [0,1] 上做完整的 group-lasso 拟合。
        这是初始阶段，使用全部数据。
        使用显式 Gram 矩阵版 FISTA（与 main.py 原始版本一致）。
        """
        self.t_end = 1.0
        t = np.asarray(t, dtype=float)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self.knots = make_global_knots(0.0, 1.0, self.k, self.knot_step)
        B = bspline_design_matrix(t, self.knots, self.k)
        m = B.shape[1]

        Phi = build_vcm_design(B, X)
        R = gram_R(self.knots, self.k, 0.0, 1.0)
        w_full = group_weights(B, X)

        lam_max = lambda_max_R(Phi, y, m, R)
        lam_path = make_lambda_path(lam_max)

        lam_best = cv_select_lambda_plain(
            B, X, y, R, lam_path,
            K=5, seed=self.seed_cv, use_1se=self.use_1se,
        )

        c = fista_group_lasso(Phi.T @ Phi, Phi.T @ y, lam_best, m, self.P, R, w_full)
        self.coef_blocks = split_blocks(c, m, self.P)

        yhat = Phi @ c
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

        return {
            "stage": 1,
            "t_end": 1.0,
            "m": int(m),
            "lambda_best": float(lam_best),
            "train_rmse": rmse,
        }

    def extend_one_stage(self, next_end, t_cn, X_cn, y_cn):
        """
        真正的增量扩展：从 [0, t_end] 扩展到 [0, next_end]。

        关键特性（保持增量）：
        - t_cn / X_cn / y_cn 只包含 C+N 区间的数据（不是全部历史数据）
        - O basis 的系数冻结不动
        - 只在 CN 数据上做优化、CV、误差计算

        算法质量改进（与 main.py 原始版一致）：
        - 使用显式 Gram 矩阵 (Φ^TΦ) + 精确谱范数的 FISTA
        - 支持 r_relax：将最靠近边界的 basis 从 O 中释放加入 CN
        - 支持 adaptive_cn_refit：自适应 group-lasso 二次拟合

        参数：
            next_end: 新的时间终点（= t_end + 1）
            t_cn: CN 区间的时间数据
            X_cn: CN 区间的协变量数据
            y_cn: CN 区间的响应数据
        """
        if self.t_end is None:
            raise RuntimeError("Call fit_stage1 first.")

        old_end = float(self.t_end)
        next_end = float(next_end)
        if next_end <= old_end + 1e-12:
            raise ValueError("next_end must be > current t_end.")

        t_cn = np.asarray(t_cn, dtype=float)
        X_cn = np.asarray(X_cn, dtype=float)
        y_cn = np.asarray(y_cn, dtype=float)

        knots_old = self.knots.copy()
        coef_old = [c.copy() for c in self.coef_blocks]

        # 构造新的全局 knot 序列
        knots_new = make_global_knots(0.0, next_end, self.k, self.knot_step)

        # OCN 分区
        part = partition_OCN(knots_old, knots_new, self.k, old_end=old_end)
        idx_o_new_matched = part["idx_o_new"]
        idx_o_old_matched = part["idx_o_old"]
        m_full = len(knots_new) - (self.k + 1)

        # r_relax：将最靠近边界的 basis 从 O 中释放
        idx_o_new, idx_relaxed = apply_relax(idx_o_new_matched, knots_new, self.k, self.r_relax)

        # 建立 new -> old 映射，过滤出对应的 old idx
        new2old = {int(j): int(i) for i, j in zip(idx_o_old_matched.tolist(), idx_o_new_matched.tolist())}
        idx_o_old = np.array([new2old[int(j)] for j in idx_o_new], dtype=int)

        # 重新计算 CN = 全部 - O（含被 relax 释放的）
        idx_all_new = np.arange(m_full, dtype=int)
        idx_cn = np.setdiff1d(idx_all_new, idx_o_new)

        if self.debug:
            debug_partition_report(part, old_end=old_end, max_show=3)
            print(f"[debug] r_relax={self.r_relax}, relaxed={len(idx_relaxed)}, "
                  f"O_final={len(idx_o_new)}, CN_final={len(idx_cn)}")

        # 冻结的 O 系数矩阵: (P, |O|)
        c_o_mat = np.stack([coef_old[p][idx_o_old] for p in range(self.P)], axis=0)

        # 只在 CN 数据上构造设计矩阵
        B_cn_data = bspline_design_matrix(t_cn, knots_new, self.k)

        # O basis 在 CN 数据上的冻结贡献
        B_cn_data_o = B_cn_data[:, idx_o_new]
        frozen_y = np.sum(X_cn * (B_cn_data_o @ c_o_mat.T), axis=1)

        # 残差 = y_cn - frozen(O)
        y_cn_res = y_cn - frozen_y

        # CN basis 在 CN 数据上的设计矩阵
        B_cn_data_cn = B_cn_data[:, idx_cn]
        m_cn = B_cn_data_cn.shape[1]

        # Gram 矩阵：在全局 [0, next_end] 上积分
        R_full = gram_R(knots_new, self.k, 0.0, next_end)
        R_cn = R_full[np.ix_(idx_cn, idx_cn)]

        # VCM 设计矩阵
        Phi_cn = build_vcm_design(B_cn_data_cn, X_cn)

        # group weights
        w_cn = group_weights(B_cn_data_cn, X_cn)

        # lambda 路径
        lam_max = lambda_max_R(Phi_cn, y_cn_res, m_cn, R_cn)
        lam_path = make_lambda_path(lam_max)

        # CV 选择 lambda（只在 CN 数据上做 CV）
        lam_best = cv_select_lambda_frozen_cn(
            B_cn=B_cn_data_cn,
            X_cn=X_cn,
            y_cn_res=y_cn_res,
            R_cn=R_cn,
            lam_path=lam_path,
            K=5,
            seed=self.seed_cv,
            use_1se=self.use_1se,
        )

        # 初始 CN 拟合
        c_cn_vec = fista_group_lasso(
            Phi_cn.T @ Phi_cn, Phi_cn.T @ y_cn_res,
            lam_best, m_cn, self.P, R_cn, w_cn,
        )

        # Adaptive refit（可选）
        lam_ad = None
        if self.adaptive:
            lam_ad, c_cn_use = adaptive_cn_refit(
                B_cn_data_cn, X_cn, y_cn_res, R_cn, lam_path,
                seed_cv=self.seed_cv, use_1se=self.use_1se,
            )
        else:
            c_cn_use = c_cn_vec

        # 组装完整系数向量
        blocks_cn = split_blocks(c_cn_use, m_cn, self.P)
        coef_new = []
        for p in range(self.P):
            cp = np.zeros(m_full, dtype=float)
            cp[idx_o_new] = c_o_mat[p]
            cp[idx_cn] = blocks_cn[p]
            coef_new.append(cp)

        # 更新状态
        self.t_end = next_end
        self.knots = knots_new
        self.coef_blocks = coef_new

        # RMSE 只在 CN 数据上计算
        yhat_cn = frozen_y + Phi_cn @ c_cn_use
        rmse_cn = float(np.sqrt(np.mean((y_cn - yhat_cn) ** 2)))

        return {
            "stage": int(round(next_end)),
            "t_end": float(next_end),
            "m_full": int(m_full),
            "num_O": int(len(idx_o_new)),
            "num_C": int(len(part["idx_c_new"])),
            "num_N": int(len(part["idx_n_new"])),
            "num_CN": int(len(idx_cn)),
            "num_relaxed": int(len(idx_relaxed)),
            "n_cn_data": int(len(t_cn)),
            "cn_data_range": [float(t_cn.min()), float(t_cn.max())],
            "lambda_best": float(lam_best),
            "lambda_ad": float(lam_ad) if lam_ad is not None else None,
            "train_rmse_cn": rmse_cn,
        }

    def predict(self, t, X):
        t = np.asarray(t, dtype=float)
        X = np.asarray(X, dtype=float)
        B = bspline_design_matrix(t, self.knots, self.k)
        beta_hat = np.column_stack([B @ self.coef_blocks[p] for p in range(self.P)])
        return np.sum(X * beta_hat, axis=1)


# =========================================================
# 6) 数据采集辅助：为 CN 区间采集数据
# =========================================================

def collect_cn_data(sim, knots_new, k, idx_cn, old_end, next_end, n_per_segment,
                    prev_seg_t=None, prev_seg_X=None, prev_seg_y=None):
    """
    为 CN 区间收集数据。

    C basis 跨越 old_end 边界，其 support 可能延伸到 [old_end - k*knot_step, old_end]。
    N basis 完全在新区间 [old_end, next_end]。

    数据来源：
    1. 上一段数据中落在 CN support 范围内的部分（用于 C basis 的旧侧覆盖）
    2. 当前段的全部新采样数据（用于 C+N basis）

    参数：
        prev_seg_t/X/y: 上一个 segment 的数据（如果可用），用于给 C basis 在旧侧提供数据
    """
    cn_left, cn_right = _compute_cn_data_range(knots_new, k, idx_cn)

    t_parts = []
    X_parts = []
    y_parts = []

    # 上一段数据中落在 CN support 范围内的部分
    if prev_seg_t is not None and prev_seg_X is not None and prev_seg_y is not None:
        prev_t = np.asarray(prev_seg_t, dtype=float)
        prev_X = np.asarray(prev_seg_X, dtype=float)
        prev_y = np.asarray(prev_seg_y, dtype=float)
        # 筛选落在 CN basis support 范围 [cn_left, old_end) 的数据
        mask_prev = (prev_t >= cn_left - 1e-10) & (prev_t < old_end + 1e-10)
        if mask_prev.sum() > 0:
            t_parts.append(prev_t[mask_prev])
            X_parts.append(prev_X[mask_prev])
            y_parts.append(prev_y[mask_prev])

    # 当前段的新采样数据 [old_end, next_end]
    seg_id = int(round(old_end))
    t_seg, X_seg, y_seg, _ = sim.sample_segment(old_end, next_end, int(n_per_segment), segment_id=seg_id)
    t_parts.append(t_seg)
    X_parts.append(X_seg)
    y_parts.append(y_seg)

    t_cn = np.concatenate(t_parts)
    X_cn_data = np.vstack(X_parts)
    y_cn = np.concatenate(y_parts)

    return t_cn, X_cn_data, y_cn, t_seg, X_seg, y_seg


# =========================================================
# 7) Driver 函数
# =========================================================

def run_or_resume_incremental(
    checkpoint_dir,
    t_final,
    n_per_segment,
    P,
    signal_idx,
    beta_funcs,
    sigma,
    k=3,
    knot_step=0.1,
    seed_data=0,
    seed_cv=2025,
    use_1se=False,
    r_relax=0,
    adaptive=False,
    debug=False,
    heartbeat_sec=90.0,
    progress_hook=None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    def ckpt_path(t_end):
        return os.path.join(checkpoint_dir, f"ckpt_t{int(round(float(t_end)))}")

    t_final = float(t_final)
    if abs(t_final - round(t_final)) > 1e-12 or t_final < 1:
        raise ValueError("t_final must be integer >= 1.")

    stage_grid = np.arange(1, int(round(t_final)) + 1, dtype=int)
    start_end = None
    for s in stage_grid[::-1]:
        if os.path.exists(ckpt_path(s) + ".json") and os.path.exists(ckpt_path(s) + ".npz"):
            start_end = float(s)
            break

    sim = VCMSimulator(P=P, signal_idx=signal_idx, beta_funcs=beta_funcs, sigma=sigma, seed_base=seed_data)

    history = []
    run_started = time.time()
    total_stages = int(round(t_final))
    hb_stop = threading.Event()
    hb_thread = None

    # 保留上一段数据，用于给下一阶段的 C basis 提供旧侧数据
    prev_seg_t = None
    prev_seg_X = None
    prev_seg_y = None

    if start_end is None:
        trainer = IncrementalVCMTrainer(k=k, knot_step=knot_step, P=P, seed_cv=seed_cv, use_1se=use_1se, r_relax=r_relax, adaptive=adaptive, debug=debug)
        hb_thread = _start_heartbeat_watcher(
            "main1",
            stage_getter=lambda: trainer.t_end or 0.0,
            total_stages=total_stages,
            stop_event=hb_stop,
            interval_sec=heartbeat_sec,
            start_time=run_started,
        )
        print(f"[stage-start] stage=1/{int(round(t_final))} P={P} n={int(n_per_segment)}", flush=True)
        stage_started = time.time()
        t0, X0, y0, _ = sim.sample_segment(0.0, 1.0, int(n_per_segment), segment_id=0)
        info = trainer.fit_stage1(t0, X0, y0)
        trainer.save_checkpoint(ckpt_path(1))
        info["elapsed_sec"] = float(time.time() - stage_started)
        history.append(info)
        if callable(progress_hook):
            progress_hook(dict(info))
        # 保留 stage1 的数据作为 prev_seg
        prev_seg_t = t0
        prev_seg_X = X0
        prev_seg_y = y0

    else:
        trainer = IncrementalVCMTrainer.load_checkpoint(ckpt_path(start_end), debug=debug)
        hb_thread = _start_heartbeat_watcher(
            "main1",
            stage_getter=lambda: trainer.t_end or 0.0,
            total_stages=total_stages,
            stop_event=hb_stop,
            interval_sec=heartbeat_sec,
            start_time=run_started,
        )
        if int(trainer.P) != int(P):
            raise ValueError(
                f"Checkpoint P={trainer.P} 与当前参数 P={P} 不一致。"
                f"请使用新的 --checkpoint-dir，或先删除目录 {checkpoint_dir} 后重跑。"
            )
        if int(trainer.k) != int(k):
            raise ValueError(
                f"Checkpoint k={trainer.k} 与当前参数 k={k} 不一致。"
            )
        if abs(float(trainer.knot_step) - float(knot_step)) > 1e-12:
            raise ValueError(
                f"Checkpoint knot_step={trainer.knot_step} 与当前参数 knot_step={knot_step} 不一致。"
            )
        history.append({"loaded_from": ckpt_path(start_end), "t_end": trainer.t_end})

        # 恢复上一段数据（用于给下一阶段的 C basis 提供数据）
        prev_end = float(trainer.t_end)
        prev_seg_id = int(round(prev_end)) - 1
        prev_a = float(prev_seg_id)
        prev_b = prev_a + 1.0
        prev_seg_t, prev_seg_X, prev_seg_y, _ = sim.sample_segment(
            prev_a, prev_b, int(n_per_segment), segment_id=prev_seg_id
        )

    while trainer.t_end + 1e-12 < t_final:
        cur = float(trainer.t_end)
        nxt = cur + 1.0
        seg_id = int(round(cur))
        print(f"[stage-start] stage={int(round(nxt))}/{int(round(t_final))} P={P} n={int(n_per_segment)}", flush=True)

        stage_started = time.time()

        # 构造下一阶段的 knots 和分区，确定 CN 数据范围
        knots_new = make_global_knots(0.0, nxt, trainer.k, trainer.knot_step)
        part = partition_OCN(trainer.knots, knots_new, trainer.k, old_end=cur)
        idx_cn = part["idx_cn_new"]

        # 收集 CN 区间的数据
        t_cn, X_cn, y_cn, cur_seg_t, cur_seg_X, cur_seg_y = collect_cn_data(
            sim, knots_new, trainer.k, idx_cn, cur, nxt, n_per_segment,
            prev_seg_t=prev_seg_t, prev_seg_X=prev_seg_X, prev_seg_y=prev_seg_y,
        )

        if debug:
            cn_left, cn_right = _compute_cn_data_range(knots_new, trainer.k, idx_cn)
            print(f"[debug] CN data: n={len(t_cn)} range=[{t_cn.min():.4f}, {t_cn.max():.4f}]", flush=True)
            print(f"[debug] CN basis support: [{cn_left:.4f}, {cn_right:.4f}]", flush=True)

        # 执行增量扩展（只用 CN 数据）
        info = trainer.extend_one_stage(nxt, t_cn, X_cn, y_cn)

        trainer.save_checkpoint(ckpt_path(nxt))
        info["elapsed_sec"] = float(time.time() - stage_started)
        history.append(info)
        if callable(progress_hook):
            progress_hook(dict(info))

        # 当前段数据变为下一阶段的 prev_seg
        prev_seg_t = cur_seg_t
        prev_seg_X = cur_seg_X
        prev_seg_y = cur_seg_y

    hb_stop.set()
    if hb_thread is not None:
        hb_thread.join(timeout=1.0)

    return trainer, history


# =========================================================
# 8) 辅助函数
# =========================================================

def _resolve_checkpoint_base(checkpoint_dir, root="checkpoints"):
    ck = str(checkpoint_dir).strip()
    if not ck:
        return ck
    norm_ck = os.path.normpath(ck)
    if os.path.isabs(ck):
        return ck
    if norm_ck == root or norm_ck.startswith(root + os.sep):
        return ck
    return os.path.join(root, ck)


def _load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_cfg(cfg, section, key, default):
    sec = cfg.get(section, {})
    return sec.get(key, default) if isinstance(sec, dict) else default


def _summarize_rmse_history(history):
    stage_rmse = []
    for h in history:
        if isinstance(h, dict) and ("stage" in h):
            rmse = h.get("train_rmse_cn", h.get("train_rmse", None))
            if rmse is not None:
                stage_rmse.append((int(h["stage"]), float(rmse)))
    stage_rmse.sort(key=lambda x: x[0])
    if len(stage_rmse) <= 1:
        return stage_rmse, []
    diffs = [stage_rmse[i][1] - stage_rmse[i - 1][1] for i in range(1, len(stage_rmse))]
    return stage_rmse, diffs


def _summarize_timing_history(history):
    rows = []
    for h in history:
        if isinstance(h, dict) and ("stage" in h) and ("elapsed_sec" in h):
            try:
                rows.append((int(h["stage"]), float(h["elapsed_sec"])))
            except Exception:
                pass
    rows.sort(key=lambda x: x[0])
    if not rows:
        return None
    vals = [x[1] for x in rows]
    return {
        "num_stages": int(len(vals)),
        "total_elapsed_sec": float(sum(vals)),
        "mean_stage_sec": float(np.mean(vals)),
        "max_stage_sec": float(np.max(vals)),
        "min_stage_sec": float(np.min(vals)),
        "stage_elapsed_sec": {int(s): float(v) for s, v in rows},
    }


def _find_latest_checkpoint(checkpoint_dir):
    if not os.path.isdir(checkpoint_dir):
        return None
    best = None
    for name in os.listdir(checkpoint_dir):
        if not (name.startswith("ckpt_t") and name.endswith(".json")):
            continue
        token = name[len("ckpt_t"):-len(".json")]
        if not token.isdigit():
            continue
        t_val = int(token)
        npz = os.path.join(checkpoint_dir, f"ckpt_t{t_val}.npz")
        if os.path.exists(npz):
            if best is None or t_val > best:
                best = t_val
    return best


# =========================================================
# 9) main 入口
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="True Incremental VCM: only uses C+N data for optimization.")
    parser.add_argument("--config", default="", help="Path to JSON config file.")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--checkpoint-dir", default="")

    parser.add_argument("--t-final", type=float, default=3.0)
    parser.add_argument("--n-per-segment", type=int, default=600)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--signal-idx", default="1,2,3,4,5")
    parser.add_argument("--beta-scales", default="1,1,1,1,1")
    parser.add_argument("--sigma", type=float, default=0.1)

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--knot-step", type=float, default=0.1)

    parser.add_argument("--seed-data", type=int, default=0)
    parser.add_argument("--seed-cv", type=int, default=2025)
    parser.add_argument("--use-1se", type=_str2bool, default=False)
    parser.add_argument("--r-relax", type=int, default=0, help="Number of bases near boundary to release from freezing.")
    parser.add_argument("--adaptive", type=_str2bool, default=False, help="Enable adaptive group-lasso refit.")

    parser.add_argument("--history-json", default="")
    parser.add_argument("--debug", type=_str2bool, default=False)
    parser.add_argument("--heartbeat-sec", type=float, default=90.0, help="Heartbeat interval in seconds; <=0 disables.")

    parser.add_argument("--load-only", type=_str2bool, default=False,
                        help="Only load latest checkpoint and print basic info.")

    args = parser.parse_args()

    if args.config:
        cfg = _load_config(args.config)
    else:
        cfg = {}

    signal_idx_cfg = _get_cfg(cfg, "model", "signal_idx", args.signal_idx)
    signal_idx = _parse_int_list(signal_idx_cfg)

    beta_cfg = _get_cfg(cfg, "model", "beta", None)
    if beta_cfg is None:
        beta_scales_cfg = _get_cfg(cfg, "model", "beta_scales", args.beta_scales)
        beta_funcs = true_beta_funcs_default(_parse_float_list(beta_scales_cfg))
    else:
        beta_funcs = build_beta_functions_from_config(beta_cfg)

    checkpoint_root = _get_cfg(cfg, "experiment", "checkpoint_root", "checkpoints")
    checkpoint_dir_cfg = _get_cfg(cfg, "experiment", "checkpoint_dir", args.checkpoint_dir)
    tag_cfg = _get_cfg(cfg, "experiment", "tag", args.tag)

    if checkpoint_dir_cfg:
        checkpoint_dir = _resolve_checkpoint_base(checkpoint_dir_cfg, root=checkpoint_root)
    else:
        tag = str(tag_cfg).strip()
        name = f"checkpoints_vcm_{tag}" if tag else "checkpoints_vcm"
        checkpoint_dir = os.path.join(checkpoint_root, name)

    load_only = bool(_get_cfg(cfg, "run", "load_only", args.load_only))

    if load_only:
        t_latest = _find_latest_checkpoint(checkpoint_dir)
        if t_latest is None:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        trainer = IncrementalVCMTrainer.load_checkpoint(os.path.join(checkpoint_dir, f"ckpt_t{t_latest}"), debug=args.debug)
        print(f"checkpoint_dir={checkpoint_dir}")
        print(f"loaded_t_end={trainer.t_end}")
        print(f"num_basis={len(trainer.knots) - (trainer.k + 1)}")
        return

    def _progress(info):
        stg = info.get("stage", info.get("t_end", "?"))
        rmse = info.get("train_rmse_cn", info.get("train_rmse", None))
        mse = None
        if rmse is not None:
            try:
                mse = float(rmse) ** 2
            except Exception:
                mse = None

        if rmse is not None and mse is not None:
            print(f"[progress] stage={stg} rmse_cn={float(rmse):.6g} mse_cn={float(mse):.6g}", flush=True)
        elif rmse is not None:
            print(f"[progress] stage={stg} rmse_cn={float(rmse):.6g}", flush=True)
        else:
            print(f"[progress] stage={stg}", flush=True)

    trainer, history = run_or_resume_incremental(
        checkpoint_dir=checkpoint_dir,
        t_final=float(_get_cfg(cfg, "data", "t_final", args.t_final)),
        n_per_segment=int(_get_cfg(cfg, "data", "n_per_segment", args.n_per_segment)),
        P=int(_get_cfg(cfg, "model", "P", args.P)),
        signal_idx=signal_idx,
        beta_funcs=beta_funcs,
        sigma=float(_get_cfg(cfg, "data", "sigma", args.sigma)),
        k=int(_get_cfg(cfg, "model", "k", args.k)),
        knot_step=float(_get_cfg(cfg, "model", "knot_step", args.knot_step)),
        seed_data=int(_get_cfg(cfg, "data", "seed_data", args.seed_data)),
        seed_cv=int(_get_cfg(cfg, "train", "seed_cv", args.seed_cv)),
        use_1se=bool(_get_cfg(cfg, "train", "use_1se", args.use_1se)),
        r_relax=int(_get_cfg(cfg, "train", "r_relax", args.r_relax)),
        adaptive=bool(_get_cfg(cfg, "train", "adaptive", args.adaptive)),
        debug=bool(_get_cfg(cfg, "train", "debug", args.debug)),
        heartbeat_sec=float(_get_cfg(cfg, "train", "heartbeat_sec", args.heartbeat_sec)),
        progress_hook=_progress,
    )

    print(f"checkpoint_dir={checkpoint_dir}")
    for h in history:
        print(h)

    stage_rmse, diffs = _summarize_rmse_history(history)
    if stage_rmse:
        print("rmse_by_stage=", {s: r for s, r in stage_rmse})
    if diffs:
        print("rmse_stage_diffs=", diffs)

    timing_stats = _summarize_timing_history(history)
    if timing_stats is not None:
        print("timing_summary=", {
            "num_stages": timing_stats["num_stages"],
            "total_elapsed_sec": timing_stats["total_elapsed_sec"],
            "mean_stage_sec": timing_stats["mean_stage_sec"],
            "max_stage_sec": timing_stats["max_stage_sec"],
            "min_stage_sec": timing_stats["min_stage_sec"],
        })

    history_json = _get_cfg(cfg, "experiment", "history_json", args.history_json)
    if history_json:
        with open(history_json, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
