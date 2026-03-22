# Incremental VCM — 增量变系数模型

基于 B-spline Group Lasso 的增量 VCM（Varying-Coefficient Model，变系数模型）实现。

核心思想：随着数据区间逐步扩展，冻结旧区间（O）的 basis 系数，仅在 Cross + New（C+N）区间上做 group-lasso 优化，实现真正的增量学习——内存恒定、不依赖历史数据。

---

## 项目结构

```
├── vcm.py            # 核心算法（数据模拟、B-spline 构造、FISTA 优化、增量训练）
├── experiment.py     # 实验运行器（增量 / Batch 对比，自动输出管理）
├── parallel.py       # 并行调度器（多 seed / 多场景并行）
├── visualize.py      # 结果可视化（β 曲线、RMSE 趋势、分组选择）
├── configs/          # JSON 配置文件
├── checkpoints/      # 训练 checkpoint（模型参数）
├── experiments/      # 实验输出（图表、汇总）
├── logs/             # 运行日志
├── archive/          # 历史版本存档（旧版 main.py / notebook）
└── requirements.txt  # Python 依赖
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 单次训练

直接运行核心算法（使用 JSON 配置）：

```bash
python vcm.py --config configs/main1_t20_p100_n300_simple5_server.json
```

或通过命令行参数指定：

```bash
python vcm.py --t-final 10 --P 15 --n-per-seg 300 --knot-step 0.2
```

### 3. 实验对比

使用 `experiment.py` 运行增量 vs Batch 对比实验：

```bash
# 推荐配置
python experiment.py --knot-step 0.1 --n-per-seg 600 --P 15 --t-final 8

# 只跑增量（不跑 Batch 对比）
python experiment.py --knot-step 0.1 --n-per-seg 600 --P 15 --t-final 8 --no-batch

# 自定义输出标签
python experiment.py --knot-step 0.1 --n-per-seg 600 --P 15 --t-final 8 --tag my_exp
```

### 4. 并行运行

多 seed 或多配置并行：

```bash
python parallel.py --config configs/main1_t20_p100_n300_simple5_server.json
```

### 5. 可视化

```bash
python visualize.py --config configs/main1_t20_p100_n300_simple5_server.json
python visualize.py --checkpoint-dir checkpoints/main1_t20_p100_n300_simple5_server
```

---

## 核心算法概要

### 增量策略（O / C / N 三分法）

每个 stage 将 B-spline basis 分为三类：

| 分区 | 含义 | 处理方式 |
|------|------|----------|
| **O** (Old) | support 完全落在旧区间内 | 系数冻结，不参与优化 |
| **C** (Cross) | support 跨越新旧边界 | 参与优化，使用旧侧 + 新侧数据 |
| **N** (New) | support 完全在新区间 | 参与优化，仅用新数据 |

### 训练流程

1. **Stage 1**：在 `[0, 1]` 上做完整的 group-lasso 拟合
2. **Stage s > 1**：扩展到 `[0, s]`
   - 冻结 O 区间系数
   - 通过 `collect_cn_data()` 收集 C+N 区间的数据
   - Frozen-CV 选择正则化参数 λ
   - FISTA + Group Lasso 优化 C+N 的系数
   - 可选 Adaptive Refit（自适应重拟合）

### 关键参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `knot_step` | 节点间距 | 0.1 ~ 0.5 |
| `k` | B-spline 阶数 | 3 |
| `n_per_seg` | 每段样本量 | 100 ~ 600 |
| `P` | 协变量维度 | 5 ~ 100 |
| `t_final` | 最终区间长度 | 8 ~ 40 |
| `r_relax` | 边界释放的 basis 数 | 0 ~ 2 |
| `lambda_grid` | 正则化参数搜索网格 | 自动生成 |

---

## 配置文件

配置文件位于 `configs/` 目录，JSON 格式，支持以下字段：

```jsonc
{
  "experiment": { "n_seeds": 5 },
  "model": { "P": 15, "k": 3, "knot_step": 0.2 },
  "data": { "t_final": 20, "n_per_segment": 300, "sigma": 0.1 },
  "incremental": { "r_relax": 1, "adaptive": true }
}
```

---

## 依赖

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
