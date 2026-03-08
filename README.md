# VCM Simulation 训练与监控指南

本文档说明四件事：
1. 如何启动训练
2. 投入服务器前如何在本地快速验证
3. 如何指定 `setting`（配置）
4. 输出信息代表什么、如何实时监控并及时停机调整

---

## 1. 推荐入口（服务器）

在 Linux/Aliyun 服务器，优先使用：

```bash
bash run_simulation_aliyun.sh
```

该脚本会调用 `simulation_suite_runner.py`，按场景批量运行 `benchmark_compare_three.py`。

---

## 2. 如何启动训练

### 2.1 最常用：自动资源分配（共享机器推荐）

```bash
CPU_MODE=auto \
RESERVE_CPUS=24 \
CPU_UTILIZATION=0.7 \
BLAS_THREADS=1 \
bash run_simulation_aliyun.sh
```

含义：
- `CPU_MODE=auto`：自动估算并行进程数（worker）
- `RESERVE_CPUS=24`：预留给同学/系统
- `CPU_UTILIZATION=0.7`：可用核只用 70%
- `BLAS_THREADS=1`：每个 worker 内部线性代数线程数，建议先设 1 防止线程超卖

### 2.2 手动指定并行度

```bash
CPU_MODE=manual \
MAX_WORKERS=48 \
BLAS_THREADS=1 \
bash run_simulation_aliyun.sh
```

### 2.3 只跑部分场景

```bash
SCENARIOS="S2_p600_t40_n180_sigma015_seed20,S3_p400_t60_n140_sigma020_seed16" \
bash run_simulation_aliyun.sh
```

### 2.4 空跑检查（不真正训练）

```bash
DRY_RUN=true bash run_simulation_aliyun.sh
```

### 2.5 三档计算量快速切换

```bash
# A. 本地快测（推荐先跑）
SUITE_CONFIG=configs/simulation_suite_local_fastcheck_v1.json \
CPU_MODE=manual MAX_WORKERS=2 BLAS_THREADS=1 \
bash run_simulation_aliyun.sh

# B. 降维正式（当前默认，推荐）
SUITE_CONFIG=configs/simulation_suite_paper_v3_lowdim.json \
CPU_MODE=auto RESERVE_CPUS=24 CPU_UTILIZATION=0.7 BLAS_THREADS=1 \
bash run_simulation_aliyun.sh

# C. 平衡正式（比降维版更重）
SUITE_CONFIG=configs/simulation_suite_paper_v2_balanced.json \
CPU_MODE=auto RESERVE_CPUS=24 CPU_UTILIZATION=0.7 BLAS_THREADS=1 \
bash run_simulation_aliyun.sh

# D. 原始满配正式（耗时最长）
SUITE_CONFIG=configs/simulation_suite_paper_v1.json \
CPU_MODE=auto RESERVE_CPUS=24 CPU_UTILIZATION=0.7 BLAS_THREADS=1 \
bash run_simulation_aliyun.sh
```

---

## 3. 投入服务器前：本地快速验证（推荐必做）

下面把“测试脚本 + 流程 + 命令行”一次性汇总。

### 3.1 测试脚本清单

- `benchmark_compare_three.py`：单场景三算法对比主脚本（最核心功能验证）
- `simulation_suite_runner.py`：多场景调度器（资源控制与场景编排验证）
- `configs/benchmark_three_tiny_smoke.json`：极小规模 smoke 配置（冒烟测试）

### 3.2 本地验证流程（4 步）

1. **环境检查**：确认依赖可导入、关键脚本可编译
2. **功能冒烟**：运行 tiny smoke，确认训练链路、checkpoint、summary 正常
3. **调度器空跑**：用 `--dry-run true` 验证场景展开、资源参数和日志输出
4. **（可选）小规模真跑**：只跑 1~2 个场景，低并发确认稳定性后再上服务器

### 3.3 所需命令行（Windows PowerShell）

```powershell
cd g:/Work/spline

# 0) 激活虚拟环境（若你已激活可跳过）
.\.venv\Scripts\Activate.ps1

# 1) 语法与依赖基础检查
python -m py_compile g:/Work/spline/main.py g:/Work/spline/main_1.py g:/Work/spline/main_2.py g:/Work/spline/benchmark_compare_three.py g:/Work/spline/simulation_suite_runner.py

# 2) 冒烟测试（2个阶段，最快）
python g:/Work/spline/benchmark_compare_three.py --config g:/Work/spline/configs/benchmark_three_tiny_smoke.json --t-final 2 --clean-seed-dir 1

# 3) 调度器 dry-run（不训练，只检查编排和参数）
python g:/Work/spline/simulation_suite_runner.py --suite-config g:/Work/spline/configs/simulation_suite_paper_v3_lowdim.json --output-root g:/Work/spline/checkpoints/paper_simulation_suite_dryrun_local --dry-run true --cpu-mode auto --reserve-cpus 2 --cpu-utilization 0.5 --blas-threads 1

# 4) 可选：小规模真跑（仅1个轻量场景，低并发）
python g:/Work/spline/simulation_suite_runner.py --suite-config g:/Work/spline/configs/simulation_suite_local_fastcheck_v1.json --output-root g:/Work/spline/checkpoints/paper_simulation_suite_local_quick --scenarios L1_p80_t4_n30_sigma020_seed2 --cpu-mode manual --max-workers 2 --blas-threads 1
```

### 3.4 每一步通过标准

- 第 1 步通过：无 `ImportError` / 无语法报错
- 第 2 步通过：出现 `[progress]`、`[overall]`、`[done]`，并生成 `benchmark_summary*.json`
- 第 3 步通过：出现 `[scenario]`、`[suite-progress]`，并生成 `generated_benchmark_config.json`
- 第 4 步通过：有阶段 `rmse/mse` 输出，且场景目录下存在 `runner.log` 与 summary

---

## 4. 如何指定 setting

### 4.1 配置文件位置

默认配置文件：

- `configs/simulation_suite_paper_v3_lowdim.json`（降维正式配置，当前推荐）
- `configs/simulation_suite_paper_v2_balanced.json`（平衡正式配置）
- `configs/simulation_suite_paper_v1.json`（原始重载正式配置，计算量最大）
- `configs/simulation_suite_local_fastcheck_v1.json`（本地快速验证配置）

可通过环境变量替换：

```bash
SUITE_CONFIG=configs/your_suite.json bash run_simulation_aliyun.sh
```

推荐选择：

- 本地调试：`simulation_suite_local_fastcheck_v1.json`
- 服务器正式（默认推荐）：`simulation_suite_paper_v3_lowdim.json`
- 服务器加压版：`simulation_suite_paper_v2_balanced.json`
- 最终满配实验：`simulation_suite_paper_v1.json`

### 4.2 配置结构（`global + scenarios`）

- `global`：所有场景共用默认配置
- `scenarios`：每个场景只写差异项（覆盖 `global`）

关键字段：

- `experiment.n_seeds`：多种子平均的 seed 数
- `model.P`：维度（协变量个数）
- `data.t_final`：训练到的总区间长度
- `data.n_per_segment`：每个区间样本数
- `data.sigma`：噪声强度
- `resource.max_workers_cap`：该场景的并行上限
- `resource.blas_threads`：该场景每 worker 的 BLAS 线程

### 4.3 资源参数优先级

1. 场景内 `resource.*`（若设置）
2. 命令行参数（`--max-workers-cap` / `--blas-threads` 等）
3. 脚本默认值

---

## 5. 实时输出怎么看（进度/健康度）


训练期间你会看到以下几类日志：

### 5.1 套件级进度（场景维度）

- `[scenario] (i/N) ... eta=...`：当前场景序号、总场景数、预计剩余时间
- `[suite-progress] done=x/N ...`：场景级总体完成比例

### 5.2 单 seed 级进度

- `[submit] seed=...`：seed 已提交
- `[done] seed=... ok/failed`：seed 完成或失败
- `[overall] seeds_done=x/N ... eta=... avg_seed=...`：该场景种子总体进度与预计剩余时间

### 5.3 算法阶段级进度（最关键）

- `[progress] seed=... algo=... stage=s/T (...) rmse=... mse=...`：
  - 当前算法（`main/main1/main2`）
  - 当前阶段 `s` / 总阶段 `T`
  - 当前阶段训练 `RMSE (均方根误差)` / `MSE (均方误差)`
- `[heartbeat] ... latest_stage=...`：长时间无新增阶段时的心跳提示，避免“看起来卡死”

### 5.4 checkpoint 进度

- `[progress] ... (ckpt)`：检测到新 `ckpt_t*.json`，说明阶段已落盘

---

## 6. 输出文件与含义


假设输出根目录是 `checkpoints/paper_simulation_suite`：

- `suite_summary.json`
  - 整体运行摘要（跑了哪些场景、失败数、每场景耗时等）
- `<scenario>/generated_benchmark_config.json`
  - 实际生效配置（含合并后的 `global + scenario`）
- `<scenario>/runner.log`
  - 该场景完整控制台日志
- `<scenario>/benchmark_summary_three.json`
  - 三算法结果汇总
  - `aggregate.main_mean/main1_mean/main2_mean`：各阶段 seed 平均 RMSE
  - `aggregate.*_std`：各阶段 seed 标准差
  - `trend_stats`：趋势统计（是否随阶段上升等）

同时会在 `run_root/main|main1|main2/seed*/` 下保留 checkpoint（`ckpt_t*.json/.npz`）。

---

## 7. 监控与及时调整建议


1. 首次跑新设置时，先 `DRY_RUN=true`。
2. 正式跑先用较保守资源：
   - `CPU_MODE=auto`
   - `RESERVE_CPUS` 设大一些（如 24/32）
   - `BLAS_THREADS=1`
3. 观察日志中的：
   - `stage` 是否持续推进
   - `rmse/mse` 是否出现异常跳变
   - 是否出现大量 `failed`
4. 若需及时停机：直接 `Ctrl+C`。
   - checkpoint 已按阶段保存，后续可继续跑（`clean_seed_dir=false`）
5. 若想重头跑某场景：将该场景 `clean_seed_dir=true` 或手动清理该场景目录。

---

## 8. 直接调用 Python 入口（可选）

不走 shell 脚本也可直接调用：

```bash
python simulation_suite_runner.py \
  --suite-config configs/simulation_suite_paper_v1.json \
  --output-root checkpoints/paper_simulation_suite \
  --cpu-mode auto \
  --reserve-cpus 24 \
  --cpu-utilization 0.7 \
  --blas-threads 1
```

---

## 9. 常见问题

- Q: 日志一直不动，是不是卡死？
  - A: 先看是否有 `[heartbeat]` 输出；有心跳通常表示进程仍在运行。
- Q: 如何减少对同学的影响？
  - A: 增大 `RESERVE_CPUS`，降低 `CPU_UTILIZATION`，并保持 `BLAS_THREADS=1`。
- Q: 如何快速定位失败原因？
  - A: 先看对应场景目录下 `runner.log`，再看 `suite_summary.json` 的 `failed_scenarios`。
