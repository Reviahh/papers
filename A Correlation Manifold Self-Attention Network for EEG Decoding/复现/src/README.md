# CMSAN: Correlation Manifold Self-Attention Network

> 基于相关流形自注意力机制的 EEG 解码网络 (JAX + Equinox + Optax)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-green.svg)](https://github.com/google/jax)

---

## 📁 项目结构

```
src/
├── main.py                 # 🎯 唯一入口，全局调度
├── requirements.txt
├── README.md
│
├── cmsan/                  # 🧠 核心模块
│   ├── __init__.py         #    统一导出 API
│   ├── model.py            #    CMSAN 模型定义
│   ├── engine.py           #    🔥 训练引擎 (SCAN/REDUCE)
│   ├── data.py             #    📦 数据加载器
│   └── layers/             #    底层可插拔模块
│       ├── fem.py          #    特征提取
│       ├── mmm.py          #    流形映射
│       ├── hom.py          #    李群同态
│       ├── att.py          #    流形注意力
│       ├── prj.py          #    切空间投影
│       ├── cls.py          #    分类器
│       ├── loss.py         #    损失函数
│       ├── ops.py          #    基础算子
│       └── manifold.py     #    流形运算
│
├── configs/                # ⚙️ 配置管理
│   ├── __init__.py
│   ├── presets.py          #    训练配置 (FAST/PAPER/DEBUG)
│   └── experiments.py      #    实验配置 (消融/超参搜索)
│
├── data/                   # 📊 数据集
│   ├── BCICIV_2a_mat/
│   ├── BCIcha/
│   └── MAMEM/
│
└── checkpoints/            # 💾 模型存档
```

---

## 🚀 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 运行模式

| 模式 | 命令 | 用途 |
|------|------|------|
| **FAST** | `python main.py --mode fast` | 本地开发，单被试 |
| **PAPER** | `python main.py --mode paper` | 基准测试，全被试 |
| **EXPERIMENT** | `python main.py --mode experiment` | 消融/超参搜索 |
| **DEBUG** | `python main.py --mode debug` | 快速验证 |

### 示例

```bash
# 快速开发 (i5-12500H)
python main.py --mode fast --dataset bcic --subject 1

# 论文基准 (全被试)
python main.py --mode paper --dataset bcic

# 所有数据集
python main.py --mode paper --dataset all

# 消融实验
python main.py --mode experiment --exp ablation_all

# 自定义超参
python main.py --mode fast --override "lr=0.002,epochs=150"
```

---

## ⚙️ 配置系统

### 层次结构

```
配置 = 训练配置 + 数据配置 + 模型配置
```

### 修改配置

**方式 1: 命令行覆盖**
```bash
python main.py --override "lr=0.002,epochs=150,d_model=64"
```

**方式 2: 修改 presets.py**
```python
# configs/presets.py

FAST = TrainConfig(
    epochs=100,
    batch_size=64,
    lr=1.5e-3,
    d_model=32,
    slices=4,
    ...
)
```

**方式 3: 编程接口**
```python
from configs import get_full_config

config = get_full_config(
    mode='fast',
    dataset='bcic',
    model='default',
    # 覆盖任意参数
    lr=0.002,
    epochs=150,
)
```

### 配置参数说明

| 参数 | FAST | PAPER | 说明 |
|------|------|-------|------|
| `epochs` | 100 | 200 | 训练轮数 |
| `batch_size` | 64 | 128 | 批大小 |
| `lr` | 1.5e-3 | 1e-3 | 学习率 |
| `d_model` | 32 | 32 | 特征维度 |
| `slices` | 4 | 4 | 时间切片数 |
| `engine` | reduce | scan | 训练引擎 |

---

## 🔬 实验系统

### 消融实验

```bash
# 运行所有消融
python main.py --mode experiment --exp ablation_all

# 单个消融
python main.py --mode experiment --exp ablation_euclidean_att
```

可用消融:
- `ablation_euclidean_att`: 欧氏注意力
- `ablation_no_hom`: 无同态映射
- `ablation_cov`: 协方差代替相关
- `ablation_linear_fem`: 线性 FEM
- `ablation_flatten`: 直接展平

### 超参搜索

```bash
python main.py --mode experiment --exp hyperparam_search
```

修改搜索空间: `configs/experiments.py`
```python
HYPERPARAM_GRID = {
    'lr': [1e-4, 5e-4, 1e-3, 2e-3],
    'batch_size': [32, 64, 128],
    'd_model': [16, 32, 64],
    'slices': [2, 4, 8],
}
```

---

## 🛠️ 扩展指南

### 添加新模块

```python
# cmsan/layers/fem.py

def init_my_fem(key, C, D, **kw):
    """初始化自定义 FEM"""
    return {...}

def my_fem(x, θ):
    """自定义前向传播"""
    return ...

# 注册
FEM['my_fem'] = (init_my_fem, my_fem)
```

使用:
```bash
python main.py --override "model.fem=my_fem"
```

### 添加新数据集

```python
# configs/presets.py

DATASETS['my_dataset'] = DatasetConfig(
    name='my_dataset',
    channels=32,
    timepoints=500,
    classes=3,
    subjects=list(range(1, 11)),
    folder='MyDataset',
)
```

```python
# cmsan/data.py

def _load_my_dataset(search_paths, subject):
    """自定义加载逻辑"""
    ...

# 在 load_unified 中添加分支
```

### 添加新实验

```python
# configs/experiments.py

ABLATIONS['my_ablation'] = {
    'name': 'My Custom Ablation',
    'model': {
        'fem': 'conv',
        'att': 'my_attention',  # 自定义模块
        ...
    },
}
```

---

## 🖥️ 平台优化

### Windows (i5-12500H)

自动进行:
- P-Core 锁定 (0-7)
- 进程优先级 HIGH
- `OMP_NUM_THREADS=8`

### TPU/GPU

```bash
# 设置环境
export TPU_NAME=your-tpu
export XLA_PYTHON_CLIENT_PREALLOCATE=true

# 使用 SCAN 引擎
python main.py --mode paper
```

---

## 📐 数学框架

### 完整 Pipeline

$$
f_\theta: \mathbb{R}^{C \times T} \xrightarrow{\text{FEM}} \mathbb{R}^{D \times T} \xrightarrow{\text{MMM}} (\text{Corr}^{++}_D)^S \xrightarrow{\text{HOM}} \text{QKV} \xrightarrow{\text{ATT}} (\text{Corr}^{++}_D)^S \xrightarrow{\text{PRJ}} \mathbb{R}^d \xrightarrow{\text{CLS}} \Delta^{K-1}
$$

### OLM 几何

| 操作 | 公式 |
|------|------|
| 对数映射 | $\text{Logo}(C) = \text{Off}(\log C)$ |
| 指数映射 | $\text{Expo}(S) = \exp(S + D^\circ)$ |
| 测地距离 | $d(P, Q) = \|\text{Logo}(P) - \text{Logo}(Q)\|_F$ |
| Fréchet 均值 | $\bar{P} = \text{Expo}(\sum_i w_i \cdot \text{Logo}(P_i))$ |

---

## 📚 API 参考

### 核心函数

```python
from cmsan import (
    CMSAN,           # 模型类
    train_session,   # 训练入口
    evaluate,        # 评估
    load_unified,    # 数据加载
    save_checkpoint, # 保存
    load_checkpoint, # 加载
)

from configs import (
    get_full_config,      # 获取完整配置
    get_train_config,     # 训练配置
    get_dataset_config,   # 数据集配置
    get_model_config,     # 模型配置
)
```

### 训练流程

```python
import jax
from cmsan import train_session, load_unified
from configs import get_full_config

# 1. 配置
config = get_full_config(mode='fast', dataset='bcic')

# 2. 数据
X, y = load_unified('bcic', subject_id=1)

# 3. 训练
key = jax.random.PRNGKey(42)
result = train_session(X_train, y_train, config, key, X_test, y_test)

# 4. 使用
model = result.model
print(f"Test Acc: {result.test_acc:.2%}")
```

---

## 📄 License

MIT License
