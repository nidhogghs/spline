# ============================ 阶段 I：在 [0,1] 上训练 ============================
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.interpolate import BSpline

# ---------- 更平缓的 C^2 掩码（接口不变） ----------
def ghost_mask(u, a=0.3, b=0.6):
    """
    在 [a,b] 上严格为 0，区间外平滑升至 1；C^2，统一尺度。
    调整 ALPHA 可让过渡更“缓”（大）或更“陡”（小）。
    """
    u = np.asarray(u, dtype=float)
    if a > b: a, b = b, a
    ALPHA = 0.20
    Uspan = float(np.ptp(u)) if np.ptp(u) > 0 else 1.0
    delta = ALPHA * Uspan
    def s(x):
        x = np.clip(x, 0.0, 1.0)
        return x*x*x*(x*(x*6 - 15) + 10)
    t = np.maximum(0.0, np.maximum(a - u, u - b))
    return s(t / (delta + 1e-12))

# ---------- 核心工具 ----------

def make_open_uniform_knots(a, b, k, n_inner):
    """
    生成区间 [a,b] 上的开区间（open / clamped）均匀结点向量。

    作用：
    - 用于构造 k 次 B 样条（degree=k）的 knots。
    - 两端各重复 (k+1) 次，使样条在边界处“夹持/贴边”（clamped），便于稳定拟合。

    数学直观：
    - 内部结点在 (a,b) 上均匀放置 n_inner 个；
    - knots 结构为：
        [a,...,a,  inner_1,...,inner_{n_inner},  b,...,b]
      其中 a 与 b 各重复 (k+1) 次。

    参数：
    - a, b: 区间端点
    - k: B 样条次数（degree）
    - n_inner: 内部结点数

    返回：
    - knots: 结点向量（numpy 数组）
    """
    inner = np.linspace(a, b, n_inner+2)[1:-1] if n_inner > 0 else np.array([])
    return np.r_[np.repeat(a, k+1), inner, np.repeat(b, k+1)]


def bspline_design_matrix(x, knots, k):
    """
    构造 B 样条基函数设计矩阵 B。

    数学定义（大致）：
    - 记第 j 个 B 样条基函数为 B_j(t)，j=1,...,m
    - 设计矩阵 B ∈ R^{n×m}：
        B[i,j] = B_j(x_i)

    实现要点：
    - SciPy 的 BSpline(knots, c, k) 中，c 为系数向量；
      这里通过 one-hot 的 c（只有第 j 位为 1）来“抽取”单个基函数 B_j。
    - extrapolate=False：超出结点定义区间时返回 NaN；这里统一把 NaN 置 0。

    参数：
    - x: 评价点（长度 n）
    - knots: 结点向量
    - k: B 样条次数（degree）

    返回：
    - B: 设计矩阵 (n×m)
    """
    m = len(knots) - (k+1)  # 基函数个数
    B = np.zeros((len(x), m))
    for j in range(m):
        c = np.zeros(m)
        c[j] = 1.0
        B[:, j] = BSpline(knots, c, k, extrapolate=False)(x)
    B[np.isnan(B)] = 0.0
    return B


def build_vcm_design(B, X):
    """
    构造变系数模型（VCM）的整体设计矩阵 Φ（块结构）。

    模型与样条展开（大致）：
    - 观测为 (t_i, x_i, y_i)，其中 x_i ∈ R^P
    - 变系数模型：
        y_i ≈ Σ_{p=1}^P x_{ip} β_p(t_i)
    - 用 B 样条基展开每个系数函数：
        β_p(t) ≈ Σ_{j=1}^m c_{pj} B_j(t)
      记 c_p = (c_{p1},...,c_{pm})^T ∈ R^m
      把所有系数拼接：c = (c_1^T,...,c_P^T)^T ∈ R^{mP}

    设计矩阵结构：
    - 令 B(t_i) = (B_1(t_i),...,B_m(t_i))^T
    - 则第 i 行：
        Φ_i = ( x_{i1} B(t_i)^T, ..., x_{iP} B(t_i)^T ) ∈ R^{mP}
    - 等价实现（对每组 p）：
        Φ[:, p*m:(p+1)*m] = X[:,[p]] * B

    参数：
    - B: B 样条基设计矩阵 (n×m)
    - X: 协变量矩阵 (n×P)

    返回：
    - out: VCM 设计矩阵 Φ (n×(mP))
    """
    n, m = B.shape
    P = X.shape[1]
    out = np.zeros((n, m * P))
    for p in range(P):
        out[:, p*m:(p+1)*m] = X[:, [p]] * B
    return out


def split_blocks(vec, m, P):
    """
    将拼接的系数向量 vec ∈ R^{mP} 按组拆成 P 个块，每块长度为 m。

    数学含义：
    - vec = (c_1^T,...,c_P^T)^T
    - 返回 [c_1, ..., c_P]

    参数：
    - vec: 长度 mP 的向量
    - m: 每组长度
    - P: 组数

    返回：
    - blocks: list，每个元素为 shape (m,) 的数组
    """
    return [vec[p*m:(p+1)*m] for p in range(P)]


def gram_R(knots, k, a, b, grid=2500):
    """
    用数值积分近似计算 Gram 矩阵 R = ∫_a^b B(t) B(t)^T dt。

    数学定义（大致）：
    - R_{ij} = ∫_a^b B_i(t) B_j(t) dt
    - 这里用等距网格 + 梯形法：
        R ≈ B(g)^T diag(w) B(g)
      其中 g 为网格点，w 为梯形法权重。

    注意：
    - 这里的 R 是基函数的 L2 Gram（不是常见的二阶导惩罚 ∫ B''B''）。
    - 后续组范数定义为：
        ||v||_R = sqrt(v^T R v)

    参数：
    - knots, k, a, b: 同上
    - grid: 数值积分网格点数

    返回：
    - R: (m×m) 矩阵
    """
    gx = np.linspace(a, b, grid)
    B = bspline_design_matrix(gx, knots, k)
    w = np.ones(grid)
    w[0] *= 0.5
    w[-1] *= 0.5
    w *= (b - a) / (grid - 1)
    return B.T @ (B * w[:, None])


def group_weights(B, X):
    """
    计算每个组的尺度权重 w_p，用于 group-lasso 惩罚项的尺度校正。

    直观目的：
    - 如果不同协变量 x_{·p} 的尺度差异很大，group-lasso 的选择会偏向尺度更大的变量；
      因此引入 w_p 做标准化/再加权。

    本实现（等价表达）：
    - 对每行 i，row_e[i] = ||B_i||_2^2 = Σ_j B[i,j]^2
    - 对每个变量 p：
        w_p = sqrt( mean_i [ x_{ip}^2 * row_e[i] ] + 1e-12 )

    参数：
    - B: (n×m)
    - X: (n×P)

    返回：
    - w: (P,) 向量
    """
    row_e = np.sum(B * B, axis=1)
    return np.sqrt((X**2 * row_e[:, None]).sum(0) / B.shape[0] + 1e-12)


def lambda_max_R(XtB, y, m, R):
    """
    计算 λ_max：当 λ ≥ λ_max 时，group-lasso 的解会变成全零（或非常接近全零）。

    背景（KKT 阈值的“对应量”，大致）：
    - 目标函数：
        min_c  1/2 ||y - Φc||_2^2 + λ Σ_p w_p ||c_p||_R
      其中 ||c_p||_R = sqrt(c_p^T R c_p)
    - 若忽略 w_p（本函数未乘 w_p），全零解成立需要：
        max_p || g_p ||_{R^{-1}} ≤ λ
      其中 g_p = Φ_p^T y，且 ||g||_{R^{-1}} = sqrt(g^T R^{-1} g)

    实现方式：
    - Cholesky: R = L L^T
    - 对每组 p：
        g = Φ_p^T y
        u = solve(L^T, g) = L^{-T} g
        ||u||_2 = sqrt(g^T R^{-1} g)
      取最大值作为 λ_max。

    参数：
    - XtB: 这里实际是 Φ（设计矩阵），shape (n×(mP))
    - y: (n,)
    - m: 每组维度
    - R: (m×m)

    返回：
    - lam_max: 标量
    """
    L = np.linalg.cholesky(R)
    P = XtB.shape[1] // m
    lam = 0.0
    for p in range(P):
        g = XtB[:, p*m:(p+1)*m].T @ y
        u = np.linalg.solve(L.T, g)
        lam = max(lam, float(norm(u)))
    return lam


def group_soft_thresh(blocks, tau, R, lam, w):
    """
    对每一组做“组软阈值”（group soft-thresholding），即 group-lasso 的 proximal 算子。

    惩罚项（大致）：
    - Pen(c) = λ Σ_p w_p ||c_p||_R
      其中 ||v||_R = sqrt(v^T R v)

    Prox（大致）：
    - 给定梯度步后的 v_p，计算：
        prox_{tau*λ*w_p*||·||_R}(v_p)
    - 形式与标准 group-lasso 类似：
        若 ||v_p||_R ≤ tau*λ*w_p  => 0
        否则 (1 - (tau*λ*w_p)/||v_p||_R) * v_p

    参数：
    - blocks: list，每个元素 v_p ∈ R^m
    - tau: 步长（通常为 1/L）
    - R: (m×m) 定义组范数的矩阵
    - lam: λ
    - w: (P,) 每组权重

    返回：
    - out: list，阈值后的 blocks
    """
    out = []
    for v, wp in zip(blocks, w):
        nr = float(np.sqrt(max(0.0, v.T @ R @ v)))  # ||v||_R
        thr = tau * lam * wp
        out.append(np.zeros_like(v) if nr <= thr else (1 - thr / nr) * v)
    return out


def fista_group_lasso(XtX, Xty, lam, m, P, R, w, max_iter=3000, tol=1e-6):
    """
    用 FISTA 求解带权 group-lasso（组范数由 R 定义）的凸优化问题。

    优化目标（与实现一致的大致形式）：
    - 令 Φ 为设计矩阵，c 为系数向量（长度 mP）
    - 损失：
        1/2 ||y - Φc||_2^2
      展开后（忽略常数项）等价于：
        Q(c) = 1/2 c^T (Φ^TΦ) c - (Φ^T y)^T c
      这里 XtX = Φ^TΦ, Xty = Φ^T y
    - 惩罚：
        Pen(c) = λ Σ_{p=1}^P w_p ||c_p||_R
      其中 ||c_p||_R = sqrt(c_p^T R c_p)

    算法要点（FISTA）：
    - 梯度步：z - step * ∇Q(z)，其中 ∇Q(z)=XtX z - Xty
    - 近端步：对每组做 group_soft_thresh
    - Nesterov 加速：
        t_{k+1} = (1 + sqrt(1+4 t_k^2))/2
        z_{k+1} = c_{k+1} + (t_k-1)/t_{k+1} (c_{k+1}-c_k)
    - step 取 1/L，L≈||XtX||_2（谱范数）

    参数：
    - XtX: (mP×mP)
    - Xty: (mP,)
    - lam: λ
    - m, P: 组结构
    - R: (m×m)
    - w: (P,)
    - max_iter: 最大迭代
    - tol: 收敛阈值（相对变化）

    返回：
    - c: (mP,) 最终系数
    """
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

        # 收敛判据：相对变化足够小
        if norm(c_new - c) <= tol * max(1.0, norm(c)):
            return c_new
        c, tN = c_new, t_new

    return c


def kfold_indices(n, K=5, seed=0):
    """
    生成 K 折交叉验证的索引划分。

    参数：
    - n: 样本数
    - K: 折数
    - seed: 随机种子

    返回：
    - folds: list，每个元素是一折验证集的索引数组
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, K)


def cv_select_lambda(B, X, y, R, w, lams, K=5, seed=2025, use_1se=True):
    """
    用 K 折交叉验证选择 λ（可选 1-SE 规则）。

    评价方式：
    - 对每个 λ：
        对每折 k：
            用训练集拟合 c^(λ,-k)
            在验证集算 MSE_k(λ) = mean( (y_val - Φ_val c)^2 )
        聚合 mm(λ) = 平均 MSE
    - 选择：
        - 若 use_1se=False：取最小 mm(λ) 的 λ
        - 若 use_1se=True：用 1-SE 规则（更保守、更稀疏）
            令 λ_min 为最小误差点
            阈值 thr = mm(λ_min) + se(λ_min)
            选择满足 mm(λ) ≤ thr 的“最小 λ 索引”（你的代码按网格方向返回更偏稀疏的一端）

    参数：
    - B: (n×m) 基函数矩阵
    - X: (n×P) 协变量
    - y: (n,)
    - R: (m×m)
    - w: (P,)
    - lams: 候选 λ 列表/数组
    - K, seed, use_1se: 同上

    返回：
    - 选定的 λ（float）
    """
    n, m = B.shape
    P = X.shape[1]
    folds = kfold_indices(n, K, seed)
    mse = np.zeros((len(lams), K))

    for li, lam in enumerate(lams):
        for kf, val in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), val)
            Btr, Bv = B[tr], B[val]
            Xtr, Xv = X[tr], X[val]
            ytr, yv = y[tr], y[val]

            XtBtr = build_vcm_design(Btr, Xtr)
            XtBv = build_vcm_design(Bv, Xv)

            XtX = XtBtr.T @ XtBtr
            Xty = XtBtr.T @ ytr

            c = fista_group_lasso(XtX, Xty, lam, m, P, R, w)
            mse[li, kf] = np.mean((yv - XtBv @ c)**2)

    mm = mse.mean(1)
    best = int(np.argmin(mm))

    if use_1se:
        se = mse.std(1, ddof=1) / np.sqrt(K)
        thr = mm[best] + se[best]
        best = int(np.where(mm <= thr)[0][0])

    return float(lams[best])


# —— 选择/评估/Adaptive —— #

def group_selection_by_Rnorm(coef_blocks, R, tol=1e-6):
    """
    通过每组系数的 R-范数判断“被选中”的组（变量选择结果）。

    数学定义：
    - r_p = ||c_p||_R = sqrt(c_p^T R c_p)
    - 若 r_p > tol，则认为第 p 组非零（被选中）

    参数：
    - coef_blocks: [c_1,...,c_P]，每个 c_p ∈ R^m
    - R: (m×m)
    - tol: 判零阈值

    返回：
    - sel: 被选中组的索引列表
    - r: 每组的 R-范数数组
    """
    r = np.array([float(np.sqrt(max(0.0, cb.T @ R @ cb))) for cb in coef_blocks])
    sel = np.where(r > tol)[0].tolist()
    return sel, r


def eval_selection(selected, signal_idx, P):
    """
    对变量选择结果做指标评估（TP/FP/FN、Precision/Recall/F1）。

    参数：
    - selected: 选中的组索引列表
    - signal_idx: 真正的信号组索引列表
    - P: 组总数（这里不直接使用，只保留接口一致性）

    返回：
    - dict: 包含 TP, FP, FN, precision, recall, f1
    """
    S = set(signal_idx)
    Sel = set(selected)
    TP = len(S & Sel)
    FP = len(Sel - S)
    FN = len(S - Sel)
    precision = TP / max(1, TP + FP)
    recall = TP / max(1, TP + FN)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return dict(TP=TP, FP=FP, FN=FN, precision=precision, recall=recall, f1=f1)


def print_selection_report(tag, selected, metrics, max_show=30):
    """
    打印变量选择结果摘要（用于仿真/对比实验）。

    参数：
    - tag: 方法标签
    - selected: 选中组
    - metrics: eval_selection 的返回
    - max_show: 最多展示多少个索引
    """
    print(f"\n[{tag}] 选中 {len(selected)} 组（前{max_show}）：{sorted(selected)[:max_show]}")
    print(f"[{tag}] TP={metrics['TP']}, FP={metrics['FP']}, FN={metrics['FN']} | "
          f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")


def adaptive_once(B, X, y, R, coef_blocks_init, use_1se=True, delta=1.0, eps=1e-6):
    """
    做一次“一步自适应 group-lasso”（adaptive group-lasso）的拟合与 λ 选择。

    自适应思想（大致）：
    - 先有初始估计 c^(0)，按组得到 c_p^(0)
    - 用组大小构造自适应权重：
        r_p^(0) = ||c_p^(0)||_R
        w_p^(ad) = 1 / (r_p^(0) + eps)^delta
      含义：初始越小的组惩罚越强，初始越大的组惩罚越弱（更利于变量选择、降低偏差）。

    自适应目标函数（大致）：
        min_c  1/2 ||y - Φc||_2^2 + λ Σ_p w_p^(ad) ||c_p||_R
    其中 ||c_p||_R = sqrt(c_p^T R c_p)

    代码流程：
    1) 由 coef_blocks_init 计算 r0 与 w_adapt
    2) 构造 Φ = build_vcm_design(B,X)
    3) 计算 λ_max（使解全零的阈值上界）
    4) 在 [λ_max, λ_max*5e-4] 上做几何网格
    5) 用 5 折 CV 选 λ（可选 1-SE）
    6) 用全数据在该 λ 下再拟合一次得到最终 c_ad

    参数：
    - B: (n×m)
    - X: (n×P)
    - y: (n,)
    - R: (m×m)
    - coef_blocks_init: 初始块系数列表 [c_1^(0),...,c_P^(0)]
    - use_1se: 是否使用 1-SE 规则
    - delta: 自适应指数 δ（常用 1）
    - eps: 防止除零的小常数

    返回：
    - lam_ad: 选定 λ
    - c_ad: 最终系数向量 (mP,)
    - blocks_ad: 拆分后的块系数列表
    """
    n, m = B.shape
    P = X.shape[1]

    # 初始每组大小：r0_p = ||c_p^(0)||_R
    r0 = np.array([float(np.sqrt(max(0.0, cb.T @ R @ cb))) for cb in coef_blocks_init])

    # 自适应权重：w_p = 1/(r0_p + eps)^delta
    w_adapt = 1.0 / (r0 + eps)**delta

    # 构造整体设计矩阵 Φ
    XtB = build_vcm_design(B, X)

    # 计算 λ_max（KKT 阈值对应量）：使解趋于全零的上界
    lam_max = lambda_max_R(XtB, y, m, R)

    # λ 候选网格：从 lam_max 到 lam_max*5e-4（几何递减）
    lambdas = np.geomspace(lam_max, lam_max * 5e-4, 30)

    # 5 折 CV 评估每个 λ 的验证误差
    folds = kfold_indices(n, 5, 2025)
    mse = np.zeros((len(lambdas), 5))

    for li, lam in enumerate(lambdas):
        for kf, val in enumerate(folds):
            tr = np.setdiff1d(np.arange(n), val)
            Btr, Bv = B[tr], B[val]
            Xtr, Xv = X[tr], X[val]
            ytr, yv = y[tr], y[val]

            XtBtr = build_vcm_design(Btr, Xtr)
            XtBv = build_vcm_design(Bv, Xv)

            XtX = XtBtr.T @ XtBtr
            Xty = XtBtr.T @ ytr

            # 注意：这里传入的是自适应权重 w_adapt
            c = fista_group_lasso(XtX, Xty, lam, m, P, R, w_adapt)
            mse[li, kf] = np.mean((yv - XtBv @ c)**2)

    # 选 λ：最小 CV 或 1-SE 规则
    mm = mse.mean(1)
    best = int(np.argmin(mm))
    if use_1se:
        se = mse.std(1, ddof=1) / np.sqrt(5)
        thr = mm[best] + se[best]
        best = int(np.where(mm <= thr)[0][0])
    lam_ad = lambdas[best]

    # 用全数据在选定 λ 下重新拟合，得到最终自适应估计
    c_ad = fista_group_lasso(
        XtB.T @ XtB,
        XtB.T @ y,
        lam_ad, m, P, R, w_adapt,
        max_iter=6000, tol=1e-7
    )
    blocks_ad = split_blocks(c_ad, m, P)
    return lam_ad, c_ad, blocks_ad



# ---------- 数据与超参 ----------
def true_funcs(scales=None):
    if scales is None: scales=[1.0,1.0,1.0,1.0,1.0]  # 控制幅度
    return [
        lambda u: scales[0]*(-0.5+0.6*np.cos(2*np.pi*u)),
        lambda u: scales[1]*(-0.5+0.6*np.cos(2*np.pi*u)),
        lambda u: scales[2]*(u-0.5),
        lambda u: scales[3]*(0.7*np.sin(4*np.pi*u)),
        lambda u: scales[4]*(0.4*np.cos(3*np.pi*u)),
    ]

def gen_interval(funcs,a,b,n,P,signal_idx,sigma,seed,normalize_within=True):
    rng=np.random.default_rng(seed)
    t=np.sort(rng.uniform(a,b,n))
    X=rng.standard_normal((n,P)); X=(X-X.mean(0))/(X.std(0)+1e-12)
    u=(t-a)/(b-a) if normalize_within else t
    beta=np.zeros((n,P))
    for j,idx in enumerate(signal_idx):
        beta[:,idx]=funcs[j%len(funcs)](u)
    y=(X*beta).sum(1)+sigma*rng.standard_normal(n)
    return t,X,y,beta

# 超参
k=3; n_inner=10
P=100; n=400; noise_sigma=0.1
seed_data=0; seed_cv=2025; use_1se=True
signal_idx=[1,2,3,4,5]

# —— 阶段 I —— #
funcs=true_funcs()
t1,X1,y1,beta1=gen_interval(funcs,0,1,n,P,signal_idx,noise_sigma,seed_data,normalize_within=True)
N = n_inner + 1
knots1 = make_open_uniform_knots(0.0,1.0,k,n_inner)     # m1 = N + k
B1     = bspline_design_matrix(t1, knots1, k); m1=B1.shape[1]
XBt1   = build_vcm_design(B1, X1)
R1     = gram_R(knots1, k, 0.0, 1.0)
w1     = group_weights(B1, X1)
lammax = lambda_max_R(XBt1, y1, m1, R1)
lams   = np.geomspace(lammax, lammax*5e-4, 30)

lam1 = cv_select_lambda(B1, X1, y1, R1, w1, lams, K=5, seed=seed_cv, use_1se=use_1se)
c1   = fista_group_lasso(XBt1.T@XBt1, XBt1.T@y1, lam1, m1, P, R1, w1)
coef1_blocks = split_blocks(c1, m1, P)

# 可视化（前 r 个有效组）
r = min(len(signal_idx), 5)
betahat1 = np.column_stack([B1 @ coef1_blocks[p] for p in range(P)])
order = np.argsort(t1)
fig,axs=plt.subplots(r,1,figsize=(7,2.0*r+3),sharex=True)
for ax,p in zip(axs, signal_idx[:r]):
    ax.plot(t1[order], beta1[order,p], 'b', lw=2, label='True β(t) [0,1]')
    ax.plot(t1[order], betahat1[order,p], 'r--', lw=2, label='Stage I fit')
    ax.grid(alpha=0.3); ax.legend()
axs[-1].set_xlabel("t"); plt.tight_layout(); plt.show()
# ===================== 单独画第一个有效组 =====================
p0 = signal_idx[0]           # 第一个有效组
betahat1_all = np.column_stack([B1 @ coef1_blocks[p] for p in range(P)])
order = np.argsort(t1)

plt.figure(figsize=(7, 3))
plt.plot(t1[order], beta1[order, p0], 'b', lw=2, label='True β(t) [0,1]')
plt.plot(t1[order], betahat1_all[order, p0], 'r--', lw=2, label='Stage I fit')
plt.grid(alpha=0.3)
plt.xlabel("t")
plt.title(f"First active group: p = {p0}")
plt.legend()
plt.tight_layout()
plt.show()

# 选择 & 性能
sel1, _ = group_selection_by_Rnorm(coef1_blocks, R1, tol=1e-6)
m1x = eval_selection(sel1, signal_idx, P)
print(f"[Stage I] m1={m1} (应为 N+k={N+k}), λ1*={lam1:.3e}")
print(f"[Stage I] 选中的变量组：{sorted(sel1)[:30]}")
print(f"[Stage I] 指标：TP={m1x['TP']}, FP={m1x['FP']}, FN={m1x['FN']}, "
      f"Precision={m1x['precision']:.3f}, Recall={m1x['recall']:.3f}, F1={m1x['f1']:.3f}")

# 若误选，做一次 Adaptive
if m1x["FP"]>0 or m1x["FN"]>0:
    lam1_ad, c1_ad, blocks1_ad = adaptive_once(B1, X1, y1, R1, coef1_blocks, use_1se=use_1se)
    sel1_ad, _ = group_selection_by_Rnorm(blocks1_ad, R1, tol=1e-6)
    m1x_ad = eval_selection(sel1_ad, signal_idx, P)
    print(f"[Stage I (Adaptive)] λ1_ad={lam1_ad:.3e}")
    print(f"[Stage I (Adaptive)] 选中的变量组：{sorted(sel1_ad)[:30]}")
    print(f"[Stage I (Adaptive)] 指标：TP={m1x_ad['TP']}, FP={m1x_ad['FP']}, FN={m1x_ad['FN']}, "
          f"Precision={m1x_ad['precision']:.3f}, Recall={m1x_ad['recall']:.3f}, F1={m1x_ad['f1']:.3f}")
