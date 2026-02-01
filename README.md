# 递增式 VCM 训练（Spline）使用说明

本程序实现分阶段（每次扩展 1 个时间区间）的 VCM 训练，并支持检查点保存与断点续跑。

## 快速开始

单次训练：
```bash
python main.py --tag default --t-final 3 --n-per-segment 400 --P 100
```

断点续跑（自动寻找最新检查点）：
```bash
python main.py --tag default --t-final 5
```

只保存模型、不保存全量数据（节省存储）：
```bash
python main.py --save-checkpoint-data false
```

多种子（串行）批量：
```bash
python main.py --n-seeds 5 --seed-data-start 0 --tag batch
```

## 服务器并行多种子（新增脚本）

当前 `main.py` 的多种子逻辑是串行执行。若服务器资源充足，可用并行脚本同时跑多个种子。

并行脚本：`run_parallel.py`

示例（并行 4 进程，跑 8 个种子）：
```bash
python run_parallel.py --n-seeds 8 --seed-start 0 --max-workers 4 --t-final 5 --tag parallel
```

脚本会为每个种子创建独立的检查点目录，避免冲突。

## 多种子平均 beta(t) 画图

脚本：`plot_avg_beta.py`

示例（画 signal_idx 对应的平均曲线 vs 真实曲线）：
```bash
python plot_avg_beta.py --n-seeds 10 --seed-start 0 --t-final 5 --tag server_parallel --out avg_beta.png
```

如需画所有协变量（可能很大）：
```bash
python plot_avg_beta.py --n-seeds 10 --seed-start 0 --t-final 5 --tag server_parallel --plot-all --P 100 --out avg_beta_all.png
```

## 检查点说明

默认每个阶段都会保存：
```
checkpoints_vcm_<tag>/ckpt_t1.{json,npz}, ckpt_t2.{json,npz}, ...
```

只保存模型（不保存 t_all/X_all/y_all）：
- 使用 `--save-checkpoint-data false`
- 断点续跑时会用 `seed_data` 与 `n_per_segment` 重新生成历史数据
- 如果续跑时修改了 `seed_data` 或 `n_per_segment`，结果会不一致

## main.py 全部命令行参数

### 实验 / 输出
- `--tag` (str, 默认: `default`)
  - 用于默认检查点目录名
- `--checkpoint-dir` (str, 默认: 空)
  - 手动指定检查点目录；指定后 `--tag` 被忽略
- `--history-json` (str, 默认: 空)
  - 保存每阶段历史信息的 JSON
- `--results-json` (str, 默认: 空)
  - 多种子批量结果的 JSON
- `--verbose` (bool, 默认: `false`)
  - 批量模式下打印每个种子的历史信息

### 训练范围 / 数据
- `--t-final` (float, 默认: `3.0`)
  - 训练终点；建议取整数
- `--n-per-segment` (int, 默认: `400`)
  - 每个区间样本数
- `--P` (int, 默认: `100`)
  - 协变量数量
- `--signal-idx` (str, 默认: `1,2,3,4,5`)
  - 有效变量索引（逗号分隔）
- `--beta-scales` (str, 默认: `1,1,1,1,1`)
  - Beta 函数缩放（逗号分隔）
- `--sigma` (float, 默认: `0.1`)
  - 噪声标准差

### Spline / 模型
- `--k` (int, 默认: `3`)
  - 样条阶数
- `--n-inner-per-unit` (int, 默认: `10`)
  - 每个单位区间的内部结点数

### 随机种子
- `--seed-data` (int, 默认: `0`)
  - 数据生成随机种子
- `--seed-data-start` (int, 默认: `0`)
  - 多种子批量的起始种子
- `--n-seeds` (int, 默认: `1`)
  - 多种子数量；>1 时进入批量模式（串行）
- `--seed-cv` (int, 默认: `2025`)
  - 交叉验证分割的随机种子

### 正则 / 交叉验证
- `--use-1se` (bool, 默认: `true`)
  - 是否使用 1-SE 规则选 lambda
- `--r-relax` (int, 默认: `2`)
  - 每次扩展时边界放松的基函数数量
- `--use-adaptive-cn` (bool, 默认: `true`)
  - 是否使用自适应 CN refit

### 检查点
- `--save-checkpoints` (bool, 默认: `true`)
  - 是否保存检查点
- `--save-checkpoint-data` (bool, 默认: `true`)
  - 是否在检查点内保存全量数据（t_all/X_all/y_all）

## run_parallel.py 参数说明

### 并行控制
- `--max-workers` (int, 默认: CPU 核心数)
  - 并行进程数量
- `--n-seeds` (int, 必填)
  - 要跑的种子数量
- `--seed-start` (int, 默认: `0`)
  - 起始种子
- `--log-dir` (str, 默认: 空)
  - 可选日志目录；每个种子写一个日志文件

### 训练参数（会传给 main.py）
- `--t-final` (float)
- `--n-per-segment` (int)
- `--P` (int)
- `--signal-idx` (str)
- `--beta-scales` (str)
- `--sigma` (float)
- `--k` (int)
- `--n-inner-per-unit` (int)
- `--seed-cv` (int)
- `--use-1se` (bool)
- `--r-relax` (int)
- `--use-adaptive-cn` (bool)
- `--save-checkpoints` (bool)
- `--save-checkpoint-data` (bool)
- `--tag` (str)
- `--checkpoint-dir` (str)

## plot_avg_beta.py 参数说明

- `--n-seeds` (int, 必填)
  - 要平均的种子数量
- `--seed-start` (int, 默认: `0`)
  - 起始种子
- `--tag` (str, 默认: `default`)
  - 实验标签（生成 checkpoint 目录名）
- `--checkpoint-dir` (str, 默认: 空)
  - 手动指定 checkpoint 基目录
- `--t-final` (float, 默认: `3.0`)
  - 终点（加载对应 ckpt_tX）
- `--signal-idx` (str, 默认: `1,2,3,4,5`)
  - 用于对比真实曲线的协变量索引
- `--beta-scales` (str, 默认: `1,1,1,1,1`)
  - 真实曲线缩放系数
- `--grid` (int, 默认: `400`)
  - 绘图网格密度
- `--out` (str, 默认: `avg_beta.png`)
  - 输出图像路径
- `--plot-all` (flag)
  - 画出所有协变量的平均曲线
- `--P` (int, 默认: `100`)
  - 协变量数量（`--plot-all` 时需要）

## 说明

- 布尔参数支持：`true/false`, `1/0`, `yes/no`, `y/n`
- 阶段按长度 1 的区间递增：`[0,1]`, `[0,2]`, ...
