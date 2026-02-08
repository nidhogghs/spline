# VCM Incremental Benchmarks

本仓库当前核心用途：比较三种算法在统一 setting 下的阶段 RMSE 与系数拟合表现。

- `main.py`：原算法
- `main_1.py`：修正版（严格 O/C/N）
- `main_2.py`：局部窗口 CN 更新版

## 1) 一键三算法对比（推荐入口）

脚本：`run_benchmark_complex.sh`  
默认配置：`configs/benchmark_three_simple.json`

```bash
bash run_benchmark_complex.sh
```

运行后会自动执行：
1. `benchmark_compare_three.py`（三算法 benchmark）
2. `benchmark_visualize_three.py`（自动画图）

输出目录（由配置 `experiment.run_root` 决定），默认：
- `checkpoints/benchmark_three_simple/benchmark_summary.json`
- `checkpoints/benchmark_three_simple/rmse_compare_three.png`
- `checkpoints/benchmark_three_simple/coef_fit_main_*.png`
- `checkpoints/benchmark_three_simple/coef_fit_main1_*.png`
- `checkpoints/benchmark_three_simple/coef_fit_main2_*.png`

## 2) 运行环境（venv / conda）

两种方式都支持，选一种即可。

### venv
```bash
source /path/to/venv/bin/activate
bash run_benchmark_complex.sh
```

### conda
```bash
conda activate <env_name>
bash run_benchmark_complex.sh
```

如需显式指定解释器：
```bash
PYTHON_BIN=/path/to/python bash run_benchmark_complex.sh
```

## 3) 常用命令行覆盖参数（无需改 JSON）

`run_benchmark_complex.sh` 支持：

- `CONFIG_PATH`：配置文件路径
- `T_FINAL`：覆盖配置里的 `data.t_final`
- `CLEAN_SEED_DIR`：
  - `1` 全新跑（先删旧 seed 目录）
  - `0` 续跑（保留旧 checkpoint）
  - `-1` 按配置文件
- `COEF_INTERVAL`：系数图时间区间，例如 `90,100`
- `CKPT_STAGE`：系数图读取的 checkpoint 阶段（`0`=自动选择各算法可用的最新共同阶段）
- `PYTHON_BIN`：Python 路径

示例：
```bash
CONFIG_PATH=configs/benchmark_three_simple.json \
T_FINAL=5 \
CLEAN_SEED_DIR=1 \
COEF_INTERVAL=1,5 \
CKPT_STAGE=5 \
bash run_benchmark_complex.sh
```

## 4) 实验推进建议（本地到服务器）

1. **先冒烟**（确认链路）
```bash
CONFIG_PATH=configs/benchmark_three_tiny_smoke.json \
T_FINAL=2 \
CLEAN_SEED_DIR=1 \
COEF_INTERVAL=1,2 \
bash run_benchmark_complex.sh
```

2. **再简单对比**（单 seed，短区间）
```bash
CONFIG_PATH=configs/benchmark_three_simple.json \
T_FINAL=5 \
CLEAN_SEED_DIR=1 \
COEF_INTERVAL=1,5 \
bash run_benchmark_complex.sh
```

3. **续跑到更长区间**（不清目录）
```bash
CONFIG_PATH=configs/benchmark_three_simple.json \
T_FINAL=30 \
CLEAN_SEED_DIR=0 \
COEF_INTERVAL=20,30 \
bash run_benchmark_complex.sh
```

4. **上服务器大实验**（多种子/复杂 setting）
```bash
CONFIG_PATH=configs/benchmark_complex_t100_s30.json \
T_FINAL=100 \
CLEAN_SEED_DIR=1 \
COEF_INTERVAL=90,100 \
bash run_benchmark_complex.sh
```

## 5) 配置文件说明（最小集合）

benchmark 配置分 6 段：

- `experiment`：输出路径、seed 数、并行数、是否清目录
- `model`：`P`、`k`、`signal_idx`、`beta_scales`
- `data`：`t_final`、`n_per_segment`、`sigma`
- `train`：`seed_cv`、`save_checkpoint_data`
- `old_algo`：原算法特有参数
- `main1_algo` / `main2_algo`：新算法参数

可参考：
- `configs/benchmark_three_tiny_smoke.json`
- `configs/benchmark_three_simple.json`
- `configs/benchmark_complex_t100_s30.json`

## 6) 进度与排错

看实时日志：
```bash
tail -f logs/benchmark_*.log
```

看 checkpoint 是否在增长：
```bash
find checkpoints/benchmark_three_simple -name "ckpt_t*.json" | wc -l
```

若改过维度/参数后报 shape mismatch，优先清旧 seed 目录，或设 `CLEAN_SEED_DIR=1`。
