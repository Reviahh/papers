# CMSAN 使用指南

本文档提供 CMSAN 项目的详细使用说明，涵盖三种实验模式的具体操作步骤。

---

## 目录

1. [环境准备](#环境准备)
2. [三种实验模式](#三种实验模式)
   - [维度一: 作者原文实验](#维度一-作者原文实验)
   - [维度二: 我自己的复现](#维度二-我自己的复现)
   - [维度三: 框架应用](#维度三-框架应用)
3. [数据准备](#数据准备)
4. [常见问题](#常见问题)

---

## 环境准备

### 1. 安装依赖

```bash
cd src/
pip install -r requirements.txt
```

### 2. 验证安装

```bash
# 快速测试（使用假数据）
python main.py --epochs 2
```

如果成功运行，说明环境配置正确。

---

## 三种实验模式

本项目按照实验目的分为三个维度，每个维度有独立的脚本和配置。

### 维度一: 作者原文实验

**目的**: 使用作者提供的数据和固定参数，复现论文结果，验证代码正确性。

#### 步骤

1. **准备数据**

   将作者提供的 `.npz` 文件放入 `data/author_original/` 目录：

   ```bash
   cp /path/to/eeg_data.npz data/author_original/
   ```

2. **运行实验**

   ```bash
   # 方法一: 直接运行专用脚本
   python scripts/reproduce_paper.py --data data/author_original/eeg_data.npz
   
   # 方法二: 通过主入口
   python main.py --mode paper --data data/author_original/eeg_data.npz
   ```

3. **查看结果**

   实验完成后：
   - 终端显示训练和验证准确率
   - 模型保存在 `checkpoints/paper_model.pkl`

#### 参数说明

**固定参数** (在 `configs/paper_config.yaml` 中定义，不可修改):
- `C=22`: 通道数
- `T=438`: 时间点
- `D=20`: 特征维度
- `S=3`: 流形段数
- `K=4`: 类别数
- `epochs=100`: 训练轮数
- `batch_size=16`: 批大小
- `lr=5e-4`: 学习率
- `seed=42`: 随机种子

---

### 维度二: 我自己的复现

**目的**: 使用自己下载的数据，进行完整的 10-fold 交叉验证实验。

#### 步骤

1. **准备数据**

   有两种方式：

   **方式 A: 使用下载脚本**
   ```bash
   python scripts/data_utils/download_data.py --subject 1 --output data/my_custom/eeg_data.npz
   ```

   **方式 B: 手动放置数据**
   ```bash
   # 将数据集放入 data/my_custom/
   data/my_custom/
   ├── BCICIV_2a_mat/
   ├── MAMEM/
   └── BCIcha/
   ```

2. **运行实验**

   ```bash
   # 方法一: 直接运行专用脚本
   python scripts/my_reproduction.py --data data/my_custom --dataset bcic
   
   # 方法二: 通过主入口
   python main.py --mode reproduce --data data/my_custom --dataset bcic
   ```

3. **支持的数据集**

   - `bcic`: BCI Competition IV 2a (运动想象)
   - `mamem`: MAMEM (SSVEP)
   - `bcicha`: BCI Challenge (ERN)
   - `all`: 运行所有数据集

4. **自定义参数**

   编辑 `configs/custom_config.yaml` 调整超参数：
   ```yaml
   bcic:
     D: 20          # 可以尝试 15, 25, 30
     S: 3           # 可以尝试 2, 4, 5
     epochs: 100    # 可以调整
     batch_size: 16 # 可以尝试 8, 32
     lr: 5.0e-4     # 可以尝试 1e-3, 1e-4
   ```

5. **查看结果**

   实验完成后：
   - 终端显示每个被试的 10-fold CV 结果
   - 总体平均准确率和标准差
   - 与论文结果对比

---

### 维度三: 框架应用

**目的**: 展示框架的通用性和扩展性，使用 CPU 优化进行快速实验。

#### 步骤

1. **运行快速实验**

   ```bash
   # 方法一: 直接运行专用脚本
   python scripts/run_application.py --data data/my_custom --dataset all
   
   # 方法二: 通过主入口
   python main.py --mode fast --data data/my_custom --dataset all
   ```

2. **优化策略**

   该模式使用以下优化：
   - 5-fold CV（而非 10-fold）
   - 50 epochs（而非 100）
   - 批大小 32（而非 16）
   - 学习率 1e-3（稍大）

   **目标**: 1小时内完成所有数据集实验

3. **修改优化参数**

   编辑 `configs/custom_config.yaml` 中的 `fast` 部分：
   ```yaml
   fast:
     epochs: 50
     batch_size: 32
     lr: 1.0e-3
     n_folds: 5
   ```

---

## 数据准备

### 数据目录结构

```
data/
├── author_original/      # 作者提供的数据
│   └── eeg_data.npz
├── my_custom/            # 自己下载的数据
│   ├── BCICIV_2a_mat/
│   ├── MAMEM/
│   └── BCIcha/
└── raw/                  # 原始未处理数据
```

### 下载 BCI Competition IV 2a 数据

使用 MOABB 自动下载：

```bash
python scripts/data_utils/download_data.py --subject 1 --output data/my_custom/s1.npz
```

**参数说明**:
- `--subject`: 被试编号 (1-9)
- `--output`: 输出文件路径

**首次运行**: 会自动下载数据（约 1.5GB），之后会使用缓存。

### 探索数据格式

```bash
python scripts/data_utils/explore_data.py
```

---

## 常见问题

### Q1: 如何更换数据集？

**A**: 只需修改 `--dataset` 参数：

```bash
# BCIC 数据集
python scripts/my_reproduction.py --data data/my_custom --dataset bcic

# MAMEM 数据集
python scripts/my_reproduction.py --data data/my_custom --dataset mamem

# BCI Challenge 数据集
python scripts/my_reproduction.py --data data/my_custom --dataset bcicha
```

### Q2: 如何调整超参数？

**A**: 编辑 `configs/custom_config.yaml`：

```yaml
bcic:
  D: 25          # 增加特征维度
  S: 4           # 增加流形段数
  epochs: 150    # 增加训练轮数
  lr: 1.0e-3     # 调整学习率
```

### Q3: 训练太慢怎么办？

**A**: 使用快速模式：

```bash
python scripts/run_application.py --data data/my_custom --dataset bcic
```

或手动调整参数：
```bash
python scripts/my_reproduction.py --data data/my_custom --dataset bcic --epochs 50 --folds 5
```

### Q4: 如何保存和加载模型？

**A**: 使用 CMSAN API：

```python
from cmsan import save_model, load_model

# 保存
save_model(model, 'checkpoints/my_model.pkl')

# 加载
model = load_model('checkpoints/my_model.pkl')
```

### Q5: 维度一、二、三有什么区别？

**A**:
- **维度一**: 作者原文实验，参数固定，用于验证代码正确性
- **维度二**: 我的复现，完整实验，可调参数，用于科研对比
- **维度三**: 框架应用，快速迭代，展示通用性和扩展性

### Q6: 为什么要重构目录结构？

**A**: 重构后的结构：
1. **清晰区分**: 代码、数据、配置、权重分离
2. **易于管理**: 三种实验模式独立，互不干扰
3. **可扩展**: 新增数据集或实验只需添加脚本，不影响核心库
4. **可复现**: 固定参数确保实验可重复

---

## 高级用法

### 自定义实验脚本

参考 `scripts/reproduce_paper.py` 创建自己的实验脚本：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cmsan import CMSAN, fit, evaluate
import jax

# 创建模型
model = CMSAN(jax.random.key(0), C=22, T=438, D=20, S=3, K=4)

# 训练
trained_model = fit(model, train_data, val_data, epochs=100)

# 评估
acc = evaluate(trained_model, X_test, y_test)
print(f"准确率: {acc:.2%}")
```

### 使用配置文件

```python
import yaml

with open('configs/custom_config.yaml') as f:
    config = yaml.safe_load(f)

bcic_cfg = config['bcic']
model = CMSAN(
    key, 
    C=bcic_cfg['C'], 
    T=bcic_cfg['T'], 
    D=bcic_cfg['D'],
    S=bcic_cfg['S'],
    K=bcic_cfg['K']
)
```

---

## 联系与支持

如有问题或建议，请：
1. 查看项目 README.md
2. 阅读代码注释和文档字符串
3. 提交 GitHub Issue

---

**祝实验顺利！** 🚀
