import argparse
import json
import os

import numpy as np
from numpy.linalg import norm
from scipy.interpolate import BSpline


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


def true_beta_funcs_default(scales=None):
    if scales is None:
        scales = [1.0, 1.0, 1.0, 1.0, 1.0]
    return [
        lambda t: scales[0] * (-0.5 + 0.6 * np.cos(2 * np.pi * t)),
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
    kind = str(spec.get("type", "sin")).strip().lower()

    if kind in ("sin", "cos"):
        amp = float(spec.get("amplitude", spec.get("amp", 1.0)))
        freq_pi = float(spec.get("frequency_pi", spec.get("freq_pi", 2.0)))
        phase = float(spec.get("phase", 0.0))
        bias = float(spec.get("bias", 0.0))
        if kind == "sin":
            return lambda t, a=amp, f=freq_pi, ph=phase, b=bias: b + a * np.sin(f * np.pi * t + ph)
        return lambda t, a=amp, f=freq_pi, ph=phase, b=bias: b + a * np.cos(f * np.pi * t + ph)

    if kind == "trig_mix":
        terms = list(spec.get("terms", []))
        if not terms:
            raise ValueError("beta function spec type='trig_mix' requires non-empty 'terms'.")
        bias = float(spec.get("bias", 0.0))

        parsed = []
        for term in terms:
            tk = str(term.get("kind", "sin")).strip().lower()
            if tk not in ("sin", "cos"):
                raise ValueError(f"Unsupported trig term kind: {tk}")
            a = float(term.get("amplitude", term.get("amp", 1.0)))
            f = float(term.get("frequency_pi", term.get("freq_pi", 2.0)))
            ph = float(term.get("phase", 0.0))
            parsed.append((tk, a, f, ph))

        def _fn(t, b=bias, ps=parsed):
            out = np.zeros_like(np.asarray(t, dtype=float), dtype=float) + b
            for tk, a, f, ph in ps:
                if tk == "sin":
                    out = out + a * np.sin(f * np.pi * t + ph)
                else:
                    out = out + a * np.cos(f * np.pi * t + ph)
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
    B = np.zeros((x.shape[0], m), dtype=float)
    for j in range(m):
        c = np.zeros(m, dtype=float)
        c[j] = 1.0
        B[:, j] = BSpline(t, c, int(k), extrapolate=False)(x)
    B[np.isnan(B)] = 0.0
    return B


def build_vcm_design(B, X):
    B = np.asarray(B, dtype=float)
    X = np.asarray(X, dtype=float)
    n, m = B.shape
    P = X.shape[1]
    Phi = np.zeros((n, m * P), dtype=float)
    for p in range(P):
        Phi[:, p * m:(p + 1) * m] = X[:, [p]] * B
    return Phi


def split_blocks(vec, m, P):
    return [vec[p * m:(p + 1) * m] for p in range(P)]


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


def lambda_max_R(Phi, y, m, R, w):
    L = _safe_cholesky(R)
    P = Phi.shape[1] // int(m)
    lam = 0.0
    for p in range(P):
        g = Phi[:, p * m:(p + 1) * m].T @ y
        u = np.linalg.solve(L.T, g)
        lam = max(lam, float(norm(u) / max(float(w[p]), 1e-12)))
    return lam


def group_soft_thresh(blocks, tau, R, lam, w):
    out = []
    for b, wp in zip(blocks, w):
        nrm = float(np.sqrt(max(0.0, b.T @ R @ b)))
        thr = float(tau * lam * wp)
        if nrm <= thr:
            out.append(np.zeros_like(b))
        else:
            out.append((1.0 - thr / nrm) * b)
    return out


def fista_group_lasso(XtX, Xty, lam, m, P, R, w, max_iter=6000, tol=1e-7):
    m = int(m)
    P = int(P)
    d = m * P

    Ls = float(np.linalg.norm(XtX, 2))
    if (not np.isfinite(Ls)) or Ls <= 0:
        Ls = 1.0
    step = 1.0 / Ls

    c = np.zeros(d, dtype=float)
    z = c.copy()
    tN = 1.0

    for _ in range(int(max_iter)):
        grad = XtX @ z - Xty
        yv = z - step * grad
        y_blocks = split_blocks(yv, m, P)
        c_new = np.concatenate(group_soft_thresh(y_blocks, step, R, lam, w))

        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tN * tN))
        z = c_new + ((tN - 1.0) / t_new) * (c_new - c)

        if norm(c_new - c) <= tol * max(1.0, norm(c)):
            return c_new

        c = c_new
        tN = t_new

    return c


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
    n = len(y)
    P = X.shape[1]
    m = B.shape[1]
    folds = kfold_indices(n, K=K, seed=seed)
    mse = np.zeros((len(lam_path), len(folds)), dtype=float)

    for li, lam in enumerate(lam_path):
        for kf, val in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), val)
            Btr, Bv = B[tr], B[val]
            Xtr, Xv = X[tr], X[val]
            ytr, yv = y[tr], y[val]

            w_tr = group_weights(Btr, Xtr)
            Phi_tr = build_vcm_design(Btr, Xtr)
            Phi_v = build_vcm_design(Bv, Xv)

            c = fista_group_lasso(
                Phi_tr.T @ Phi_tr,
                Phi_tr.T @ ytr,
                float(lam),
                m,
                P,
                R,
                w_tr,
            )
            mse[li, kf] = float(np.mean((yv - Phi_v @ c) ** 2))

    mm = mse.mean(axis=1)
    best = int(np.argmin(mm))
    if use_1se and len(folds) > 1:
        se = mse.std(axis=1, ddof=1) / np.sqrt(len(folds))
        thr = mm[best] + se[best]
        best = int(np.where(mm <= thr)[0][0])
    return float(lam_path[best])


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

    # Checks requested by theory
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


def cv_select_lambda_frozen_local(B_local_full, X_local, y_local, R_cn, lam_path, idx_o, c_o_mat, K=5, seed=2025, use_1se=True):
    n_all, m_full = B_local_full.shape
    P = X_local.shape[1]
    folds = kfold_indices(n_all, K=K, seed=seed)

    idx_all = np.arange(m_full, dtype=int)
    idx_cn = np.setdiff1d(idx_all, np.asarray(idx_o, dtype=int))

    mse = np.zeros((len(lam_path), len(folds)), dtype=float)

    for li, lam in enumerate(lam_path):
        for kf, val in enumerate(folds):
            tr = np.setdiff1d(np.arange(n_all), val)

            Btr, Bv = B_local_full[tr], B_local_full[val]
            Xtr, Xv = X_local[tr], X_local[val]
            ytr, yv = y_local[tr], y_local[val]

            Btr_o, Bv_o = Btr[:, idx_o], Bv[:, idx_o]
            Btr_cn, Bv_cn = Btr[:, idx_cn], Bv[:, idx_cn]

            frozen_tr = np.sum(Xtr * (Btr_o @ c_o_mat.T), axis=1)
            frozen_v = np.sum(Xv * (Bv_o @ c_o_mat.T), axis=1)

            ytr_res = ytr - frozen_tr
            w_tr = group_weights(Btr_cn, Xtr)

            Phi_tr = build_vcm_design(Btr_cn, Xtr)
            Phi_v = build_vcm_design(Bv_cn, Xv)

            c_cn = fista_group_lasso(
                Phi_tr.T @ Phi_tr,
                Phi_tr.T @ ytr_res,
                float(lam),
                Btr_cn.shape[1],
                P,
                R_cn,
                w_tr,
            )
            y_pred = frozen_v + Phi_v @ c_cn
            mse[li, kf] = float(np.mean((yv - y_pred) ** 2))

    mm = mse.mean(axis=1)
    best = int(np.argmin(mm))
    if use_1se and len(folds) > 1:
        se = mse.std(axis=1, ddof=1) / np.sqrt(len(folds))
        thr = mm[best] + se[best]
        best = int(np.where(mm <= thr)[0][0])

    return float(lam_path[best])


class IncrementalVCMTrainer:
    def __init__(self, k, knot_step, P, seed_cv=2025, use_1se=True, debug=False,
                 local_window_units=2.0, local_support_margin=0.0):
        self.k = int(k)
        self.knot_step = float(knot_step)
        self.P = int(P)
        self.seed_cv = int(seed_cv)
        self.use_1se = bool(use_1se)
        self.debug = bool(debug)
        self.local_window_units = float(local_window_units)
        self.local_support_margin = float(local_support_margin)

        self.t_end = None
        self.knots = None
        self.coef_blocks = None

        self.t_all = None
        self.X_all = None
        self.y_all = None

    def save_checkpoint(self, prefix_path, save_data=True):
        if self.t_end is None:
            raise RuntimeError("Nothing to save.")

        meta = {
            "t_end": float(self.t_end),
            "k": int(self.k),
            "knot_step": float(self.knot_step),
            "P": int(self.P),
            "seed_cv": int(self.seed_cv),
            "use_1se": bool(self.use_1se),
            "local_window_units": float(self.local_window_units),
            "local_support_margin": float(self.local_support_margin),
        }

        coef_mat = np.stack(self.coef_blocks, axis=0)

        os.makedirs(os.path.dirname(prefix_path) or ".", exist_ok=True)
        if bool(save_data):
            np.savez_compressed(
                prefix_path + ".npz",
                t_all=self.t_all,
                X_all=self.X_all,
                y_all=self.y_all,
                knots=self.knots,
                coef_mat=coef_mat,
            )
        else:
            np.savez_compressed(
                prefix_path + ".npz",
                knots=self.knots,
                coef_mat=coef_mat,
            )

        meta["save_data"] = bool(save_data)
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
            debug=debug,
            local_window_units=meta.get("local_window_units", 2.0),
            local_support_margin=meta.get("local_support_margin", 0.0),
        )

        arr = np.load(prefix_path + ".npz", allow_pickle=False)
        obj.knots = arr["knots"]
        coef_mat = arr["coef_mat"]
        obj.coef_blocks = [coef_mat[p].copy() for p in range(coef_mat.shape[0])]
        obj.t_end = float(meta["t_end"])

        if "t_all" in arr and "X_all" in arr and "y_all" in arr:
            obj.t_all = arr["t_all"]
            obj.X_all = arr["X_all"]
            obj.y_all = arr["y_all"]
        else:
            obj.t_all = None
            obj.X_all = None
            obj.y_all = None

        return obj

    def fit_stage1(self, t, X, y):
        self.t_end = 1.0
        self.t_all = np.asarray(t, dtype=float).copy()
        self.X_all = np.asarray(X, dtype=float).copy()
        self.y_all = np.asarray(y, dtype=float).copy()

        self.knots = make_global_knots(0.0, 1.0, self.k, self.knot_step)
        B = bspline_design_matrix(self.t_all, self.knots, self.k)
        m = B.shape[1]

        Phi = build_vcm_design(B, self.X_all)
        R = gram_R(self.knots, self.k, 0.0, 1.0)
        w_full = group_weights(B, self.X_all)

        lam_max = lambda_max_R(Phi, self.y_all, m, R, w_full)
        lam_path = make_lambda_path(lam_max)

        lam_best = cv_select_lambda_plain(
            B,
            self.X_all,
            self.y_all,
            R,
            lam_path,
            K=5,
            seed=self.seed_cv,
            use_1se=self.use_1se,
        )

        c = fista_group_lasso(Phi.T @ Phi, Phi.T @ self.y_all, lam_best, m, self.P, R, w_full)
        self.coef_blocks = split_blocks(c, m, self.P)

        yhat = self.predict(self.t_all, self.X_all)
        rmse = float(np.sqrt(np.mean((self.y_all - yhat) ** 2)))

        return {
            "stage": 1,
            "t_end": 1.0,
            "m": int(m),
            "lambda_best": float(lam_best),
            "train_rmse": rmse,
        }

    def extend_one_stage(self, next_end, t_seg, X_seg, y_seg):
        if self.t_end is None:
            raise RuntimeError("Call fit_stage1 first.")

        old_end = float(self.t_end)
        next_end = float(next_end)
        if next_end <= old_end + 1e-12:
            raise ValueError("next_end must be > current t_end.")

        self.t_all = np.r_[self.t_all, np.asarray(t_seg, dtype=float)]
        self.X_all = np.vstack([self.X_all, np.asarray(X_seg, dtype=float)])
        self.y_all = np.r_[self.y_all, np.asarray(y_seg, dtype=float)]

        knots_old = self.knots.copy()
        coef_old = [c.copy() for c in self.coef_blocks]

        knots_new = make_global_knots(0.0, next_end, self.k, self.knot_step)
        B_full = bspline_design_matrix(self.t_all, knots_new, self.k)
        m_full = B_full.shape[1]

        part = partition_OCN(knots_old, knots_new, self.k, old_end=old_end)
        idx_o_new = part["idx_o_new"]
        idx_o_old = part["idx_o_old"]
        idx_cn = part["idx_cn_new"]

        if self.debug:
            debug_partition_report(part, old_end=old_end, max_show=3)

        c_o_mat = np.stack([coef_old[p][idx_o_old] for p in range(self.P)], axis=0)

        left_new, right_new = basis_supports(knots_new, self.k)
        cn_left = float(np.min(left_new[idx_cn]))
        cn_right = float(np.max(right_new[idx_cn]))
        h = float(self.knot_step)
        fit_a = max(0.0, old_end - float(self.local_window_units))
        fit_b = next_end

        mask_win = (self.t_all >= fit_a) & (self.t_all <= fit_b)
        mask_sup = (self.t_all >= (cn_left - self.local_support_margin * h)) & (
            self.t_all <= (cn_right + self.local_support_margin * h)
        )
        mask = mask_win & mask_sup

        min_rows = max(40, 2 * int(len(idx_cn)))
        if int(mask.sum()) < min_rows:
            mask = mask_win
        if int(mask.sum()) < min_rows:
            mask = np.ones_like(self.t_all, dtype=bool)

        t_loc = self.t_all[mask]
        X_loc = self.X_all[mask]
        y_loc = self.y_all[mask]
        B_loc_full = bspline_design_matrix(t_loc, knots_new, self.k)

        B_loc_o = B_loc_full[:, idx_o_new]
        frozen_loc = np.sum(X_loc * (B_loc_o @ c_o_mat.T), axis=1)
        y_loc_res = y_loc - frozen_loc

        B_loc_cn = B_loc_full[:, idx_cn]
        Phi_cn = build_vcm_design(B_loc_cn, X_loc)
        w_cn = group_weights(B_loc_cn, X_loc)

        r_grid = int(max(800, 800 * (fit_b - fit_a)))
        R_full_loc = gram_R(knots_new, self.k, fit_a, fit_b, grid=r_grid)
        R_cn = R_full_loc[np.ix_(idx_cn, idx_cn)]

        lam_max = lambda_max_R(Phi_cn, y_loc_res, B_loc_cn.shape[1], R_cn, w_cn)
        lam_path = make_lambda_path(lam_max)

        lam_best = cv_select_lambda_frozen_local(
            B_local_full=B_loc_full,
            X_local=X_loc,
            y_local=y_loc,
            R_cn=R_cn,
            lam_path=lam_path,
            idx_o=idx_o_new,
            c_o_mat=c_o_mat,
            K=5,
            seed=self.seed_cv,
            use_1se=self.use_1se,
        )

        c_cn = fista_group_lasso(
            Phi_cn.T @ Phi_cn,
            Phi_cn.T @ y_loc_res,
            lam_best,
            B_loc_cn.shape[1],
            self.P,
            R_cn,
            w_cn,
        )

        blocks_cn = split_blocks(c_cn, B_loc_cn.shape[1], self.P)
        coef_new = []
        for p in range(self.P):
            cp = np.zeros(m_full, dtype=float)
            cp[idx_o_new] = c_o_mat[p]
            cp[idx_cn] = blocks_cn[p]
            coef_new.append(cp)

        self.t_end = next_end
        self.knots = knots_new
        self.coef_blocks = coef_new

        yhat = self.predict(self.t_all, self.X_all)
        rmse = float(np.sqrt(np.mean((self.y_all - yhat) ** 2)))

        return {
            "stage": int(round(next_end)),
            "t_end": float(next_end),
            "m_full": int(m_full),
            "num_O": int(len(idx_o_new)),
            "num_C": int(len(part["idx_c_new"])),
            "num_N": int(len(part["idx_n_new"])),
            "num_CN": int(len(idx_cn)),
            "num_local_rows": int(mask.sum()),
            "fit_interval": [float(fit_a), float(fit_b)],
            "lambda_best": float(lam_best),
            "train_rmse": rmse,
        }

    def predict(self, t, X):
        t = np.asarray(t, dtype=float)
        X = np.asarray(X, dtype=float)
        B = bspline_design_matrix(t, self.knots, self.k)
        beta_hat = np.column_stack([B @ self.coef_blocks[p] for p in range(self.P)])
        return np.sum(X * beta_hat, axis=1)


def rebuild_cached_data_until(sim, t_end, n_per_segment):
    t_list = []
    X_list = []
    y_list = []
    stage_end = int(round(float(t_end)))
    for seg in range(stage_end):
        a = float(seg)
        b = float(seg + 1)
        t_seg, X_seg, y_seg, _ = sim.sample_segment(a, b, int(n_per_segment), segment_id=seg)
        t_list.append(t_seg)
        X_list.append(X_seg)
        y_list.append(y_seg)

    return np.concatenate(t_list), np.vstack(X_list), np.concatenate(y_list)


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
    use_1se=True,
    save_checkpoint_data=True,
    debug=False,
    local_window_units=2.0,
    local_support_margin=0.0,
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
    if start_end is None:
        trainer = IncrementalVCMTrainer(
            k=k, knot_step=knot_step, P=P, seed_cv=seed_cv, use_1se=use_1se, debug=debug,
            local_window_units=local_window_units, local_support_margin=local_support_margin
        )
        t0, X0, y0, _ = sim.sample_segment(0.0, 1.0, int(n_per_segment), segment_id=0)
        info = trainer.fit_stage1(t0, X0, y0)
        trainer.save_checkpoint(ckpt_path(1), save_data=save_checkpoint_data)
        history.append(info)
    else:
        trainer = IncrementalVCMTrainer.load_checkpoint(ckpt_path(start_end), debug=debug)
        history.append({"loaded_from": ckpt_path(start_end), "t_end": trainer.t_end})

    if trainer.t_all is None or trainer.X_all is None or trainer.y_all is None:
        t_all, X_all, y_all = rebuild_cached_data_until(sim, trainer.t_end, n_per_segment)
        trainer.t_all = t_all
        trainer.X_all = X_all
        trainer.y_all = y_all

    while trainer.t_end + 1e-12 < t_final:
        cur = float(trainer.t_end)
        nxt = cur + 1.0
        seg_id = int(round(cur))

        t_seg, X_seg, y_seg, _ = sim.sample_segment(cur, nxt, int(n_per_segment), segment_id=seg_id)
        info = trainer.extend_one_stage(nxt, t_seg, X_seg, y_seg)
        trainer.save_checkpoint(ckpt_path(nxt), save_data=save_checkpoint_data)
        history.append(info)

    return trainer, history


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
        if isinstance(h, dict) and ("stage" in h) and ("train_rmse" in h):
            stage_rmse.append((int(h["stage"]), float(h["train_rmse"])))
    stage_rmse.sort(key=lambda x: x[0])
    if len(stage_rmse) <= 1:
        return stage_rmse, []
    diffs = [stage_rmse[i][1] - stage_rmse[i - 1][1] for i in range(1, len(stage_rmse))]
    return stage_rmse, diffs


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


def main():
    parser = argparse.ArgumentParser(description="Incremental VCM with strict O/C/N update.")
    parser.add_argument("--config", default="", help="Path to JSON config file.")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--checkpoint-dir", default="")

    parser.add_argument("--t-final", type=float, default=3.0)
    parser.add_argument("--n-per-segment", type=int, default=400)
    parser.add_argument("--P", type=int, default=100)
    parser.add_argument("--signal-idx", default="1,2,3,4,5")
    parser.add_argument("--beta-scales", default="1,1,1,1,1")
    parser.add_argument("--sigma", type=float, default=0.1)

    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--knot-step", type=float, default=0.1)

    parser.add_argument("--seed-data", type=int, default=0)
    parser.add_argument("--seed-cv", type=int, default=2025)
    parser.add_argument("--use-1se", type=_str2bool, default=True)
    parser.add_argument("--local-window-units", type=float, default=2.0)
    parser.add_argument("--local-support-margin", type=float, default=0.0)

    parser.add_argument("--save-checkpoint-data", type=_str2bool, default=True)
    parser.add_argument("--history-json", default="")
    parser.add_argument("--debug", type=_str2bool, default=False)

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
        save_checkpoint_data=bool(_get_cfg(cfg, "train", "save_checkpoint_data", args.save_checkpoint_data)),
        debug=bool(_get_cfg(cfg, "train", "debug", args.debug)),
        local_window_units=float(_get_cfg(cfg, "train", "local_window_units", args.local_window_units)),
        local_support_margin=float(_get_cfg(cfg, "train", "local_support_margin", args.local_support_margin)),
    )

    print(f"checkpoint_dir={checkpoint_dir}")
    for h in history:
        print(h)

    stage_rmse, diffs = _summarize_rmse_history(history)
    if stage_rmse:
        print("rmse_by_stage=", {s: r for s, r in stage_rmse})
    if diffs:
        print("rmse_stage_diffs=", diffs)

    history_json = _get_cfg(cfg, "experiment", "history_json", args.history_json)
    if history_json:
        with open(history_json, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
