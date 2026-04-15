# STRF

本仓库包含 **STRF** 的实现代码（时间序列预测），对应论文已被 **CCFC 2026** 录用。

> 说明：仓库内 `ablation/` 目录包含若干第三方基线（如 Autoformer / FEDformer / PatchTST / LTSF-Linear / TimeMixer 等），用于对比与消融。STRF 主体代码位于仓库根目录（`run.py`, `models/`, `layers/`, `data_provider/`, `exp/`, `utils/`）。

## 功能特性

- 长序列时间序列预测实验流程（train/val/test）
- STRF（代码中模型名为 `xPatch`）支持：patching + 分解（reg/ema/dema）+ 可选 RevIN
- 支持多种公开数据集（ETT、Weather、Traffic、Electricity、Exchange、Solar）与自定义 CSV
- 提供批量复现实验脚本与参数搜索脚本

## 目录结构

- `run.py`：主入口（参数解析、训练与测试流程）
- `exp/`：实验封装
  - `exp/exp_basic.py`：设备选择与基类
  - `exp/exp_main.py`：STRF 训练/验证/测试逻辑
- `models/`
  - `models/xPatch.py`：STRF 主模型实现（模型名：`xPatch`）
- `layers/`：模型组件（分解、patch 网络、RevIN 等）
- `data_provider/`：数据集与 DataLoader
- `utils/`：metrics、时间特征、EarlyStopping、学习率调整等
- `scripts/`：复现实验用的 bash 脚本
- `ablation/`：第三方基线实现（用于对比/消融）

## 安装

### 方式 A：pip（推荐）

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `requirements.txt` 固定了主要 Python 依赖；PyTorch 建议按你的 CUDA/CPU 环境自行安装匹配版本。

### 方式 B：conda

```bash
conda env create -f environment.yml
conda activate xPatch
```

## 数据准备

默认约定数据在 `./dataset/` 下。

脚本中常用路径示例：

- ETT：`./dataset/ETT-small/ETTh1.csv`, `ETTh2.csv`, `ETTm1.csv`, `ETTm2.csv`
- Weather：`./dataset/weather/weather.csv`
- Traffic：`./dataset/traffic/traffic.csv`
- Electricity：`./dataset/electricity/electricity.csv`
- Exchange：`./dataset/exchange_rate/exchange_rate.csv`
- Solar：`./dataset/solar.txt`

自定义数据集（CSV）用法：`--data custom --root_path <dir> --data_path <csv>`。

## 快速开始

### 1）以 ETTh1 为例训练 + 测试

```bash
python run.py \
  --is_training 1 \
  --model_id ETTh1_96_ema \
  --model xPatch \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --batch_size 2048 \
  --learning_rate 5e-4 \
  --lradj sigmoid \
  --ma_type ema \
  --alpha 0.3 \
  --beta 0.3
```

### 2）仅测试（从 checkpoints 加载）

将 `--is_training` 设为 `0`，代码会从 `./checkpoints/<setting>/checkpoint.pth` 加载权重。

```bash
python run.py \
  --is_training 0 \
  --model_id ETTh1_96_ema \
  --model xPatch \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv
```

## 复现实验

`scripts/` 下提供了批量实验脚本：

- `scripts/xPatch_unified.sh`：跨数据集、不同 `pred_len` 的统一设置复现
- `scripts/xPatch_search.sh`：另一组设置/搜索配置
- `scripts/xPatch_fair.sh`：用于公平对比的设置
- `scripts/time.sh`：用于计时的短跑实验

示例：

```bash
bash scripts/xPatch_unified.sh
```

日志默认输出：`./logs/<ma_type>/`。

## 主要参数说明

来自 `run.py`：

- 训练控制：`--is_training`, `--itr`, `--train_epochs`, `--patience`, `--train_only`
- 数据相关：`--data`, `--root_path`, `--data_path`, `--features`, `--target`, `--freq`, `--embed`
- 预测任务：`--seq_len`, `--label_len`, `--pred_len`, `--enc_in`
- Patching：`--patch_len`, `--stride`, `--padding_patch`
- 分解/滑动平均：`--ma_type`（`reg`/`ema`/`dema`）, `--alpha`, `--beta`
- 优化：`--batch_size`, `--learning_rate`, `--lradj`, `--use_amp`
- 归一化：`--revin`（1/0）
- 硬件：`--use_gpu`, `--gpu`, `--use_multi_gpu`, `--devices`

## 输出文件

训练/测试过程中会生成：

- `./checkpoints/<setting>/`：模型权重
- `./test_results/<setting>/`：预测结果/可视化（由 `exp/exp_main.py` 产出）
- `./results/`：若工具函数启用，会写入汇总指标文件

## 基线（`ablation/`）

`ablation/` 目录包含多个第三方基线实现，用于对比/消融（非 STRF 主入口）。

包含（以目录名为准）：

- Autoformer（`ablation/Autoformer-main`，MIT License，Copyright (c) 2021 THUML @ Tsinghua University）
- FEDformer（`ablation/FEDformer-master`，MIT License，Copyright (c) 2021 DAMO Academy @ Alibaba）
- LTSF-Linear / DLinear（`ablation/LTSF-Linear-main`，Apache-2.0 License，Copyright 2022 DLinear Authors）
- PatchTST（`ablation/PatchTST-main`，Apache-2.0 License）
- TimeMixer（`ablation/TimeMixer-main`，Apache-2.0 License）

> 由于这些子目录未包含上游 README/引用信息，本 README 仅在“引用”部分给出**方法级别**的 BibTeX 占位符。若你希望补齐到“带链接/带作者”的标准引用，请提供对应上游仓库链接或论文信息。

## 许可证

本项目使用 **Apache License 2.0**，见 [LICENSE](./LICENSE)。

## Citation

```bibtex
@inproceedings{autoformer,
  title = {Autoformer},
  year  = {2021}
}

@inproceedings{fedformer,
  title = {FEDformer},
  year  = {2022}
}

@inproceedings{patchtst,
  title = {PatchTST},
  year  = {2023}
}

@inproceedings{dlinear,
  title = {DLinear},
  year  = {2022}
}

@inproceedings{timemixer,
  title = {TimeMixer},
  year  = {2024}
}
```
