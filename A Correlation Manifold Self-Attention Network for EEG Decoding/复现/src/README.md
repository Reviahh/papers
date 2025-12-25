# CMSAN: Correlation Manifold Self-Attention Network

> 基于相关流形自注意力机制的 EEG 解码网络 (Equinox + Optax)

## 📐 数学框架

### 核心映射

$$
f_\theta: \mathbb{R}^{C \times T} \to \Delta^{K-1}
$$

其中 $C$ 是通道数，$T$ 是时间点数，$K$ 是类别数。

### 完整 Composition

$$
f_\theta = \underbrace{\text{Cls}}_{\text{分类}} \circ \underbrace{\text{Prj}}_{\text{投影}} \circ \underbrace{\text{Att}}_{\text{注意力}} \circ \underbrace{\text{Hom}}_{\text{同态}} \circ \underbrace{\text{MMM}}_{\text{流形映射}} \circ \underbrace{\text{FEM}}_{\text{特征提取}}
$$

---

## 📁 项目结构 (重构后)

本项目已重构为清晰的三维度实验框架：

```
src/
├── cmsan/                     # 🎯 核心算法库 (保持纯净，不放数据)
│   ├── __init__.py            #    导出 CMSAN, train, fit
│   ├── model.py               #    模型定义 (eqx.Module)
│   ├── train_engine.py        #    通用训练逻辑 (Equinox + Optax)
│   ├── README.md              #    API 文档
│   └── layers/                # 🔧 可插拔模块组合
│       ├── fem.py, mmm.py, hom.py, att.py, prj.py, cls.py
│       ├── manifold.py        #    OLM 流形几何
│       ├── ops.py, loss.py    #    基础算子和损失函数
│       └── ...
│
├── data/                      # 📦 数据存放区
│   ├── author_original/       #    作者提供的数据
│   ├── my_custom/             #    自己下载的数据
│   └── raw/                   #    原始未处理数据
|
├── logs/                      #    日志区
|
├── scripts/                   # 📝 脚本区 (三维度实验)
│   ├── reproduce_paper.py     #    【维度一】作者原文实验
│   ├── my_reproduction.py     #    【维度二】我的复现
│   ├── run_application.py     #    【维度三】框架应用
│   └── data_utils/            #    数据处理工具
│       ├── download_data.py   #    数据下载
│       ├── load_data.py       #    数据加载
│       └── explore_data.py    #    数据探索
│
├── checkpoints/               # 💾 模型权重存放
│   └── (*.pkl 文件)
│
├── configs/                   # ⚙️  配置文件区
│   ├── paper_config.yaml      #    论文固定参数
│   └── custom_config.yaml     #    自定义参数
│
├── main.py                    # 🚀 统一入口
└── requirements.txt           # 📦 依赖
```

---

## 🎯 三维度实验框架

### 维度一: 作者原文实验 (Official Benchmark)

**目的**: 使用作者提供的数据和参数，复现论文中的实验结果，作为"定海神针"。

```bash
# 1. 将作者提供的数据放入 data/author_original/
# 2. 运行作者原文实验
python scripts/reproduce_paper.py --data data/author_original/eeg_data.npz

# 或通过主入口
python main.py --mode paper --data data/author_original/eeg_data.npz
```

**特点**:
- 参数固定，不可修改
- 保证可重复性 (固定种子)
- 验证代码实现正确性

---

### 维度二: 我自己的复现 (My Validation)

**目的**: 使用自己下载的数据，进行完整的 10-fold CV 实验。

```bash
# 1. 将下载的数据放入 data/my_custom/
# 2. 运行 10-fold CV 实验
python scripts/my_reproduction.py --data data/my_custom --dataset bcic

# 或通过主入口
python main.py --mode reproduce --data data/my_custom --dataset bcic
```

**特点**:
- 支持多数据集 (bcic, mamem, bcicha)
- 完整 10-fold 交叉验证
- 可调整超参数

---

### 维度三: 框架应用 (Extension)

**目的**: 展示框架的通用性和扩展性，CPU 优化快速实验。

```bash
# 运行快速实验 (5-fold, 50 epochs)
python scripts/run_application.py --data data/my_custom --dataset all

# 或通过主入口
python main.py --mode fast --data data/my_custom --dataset all
```

**特点**:
- CPU 优化 (多线程，大批次)
- 快速迭代 (1小时完成所有数据集)
- 证明框架低耦合，通用性强

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 测试模式 (使用假数据)

```bash
# 快速测试代码是否正常工作
python main.py
```

### 最简示例 (Equinox 风格)

```python
import jax
from cmsan import CMSAN, fit

# 创建模型 (Equinox Module，参数内嵌)
model = CMSAN(jax.random.key(0), C=22, T=438, D=20, S=3, K=4)

# 训练 (完全函数式，无 for 循环)
model = fit(model, (X_train, y_train), epochs=100, verbose=True)

# 推理 (直接调用)
logits = model(x)
pred = model.predict(x)
```

---

## 🔧 数据准备

### 方法一: 使用作者提供的数据

```bash
# 将 .npz 文件放入 data/author_original/
cp /path/to/eeg_data.npz data/author_original/
```

### 方法二: 自己下载数据

```bash
# 使用 MOABB 下载 BCI Competition IV 2a
python scripts/data_utils/download_data.py --subject 1 --output data/my_custom/eeg_data.npz
```

---

## 📊 配置文件说明

### paper_config.yaml

论文固定参数，用于维度一实验，**不可修改**。

### custom_config.yaml

自定义参数，用于维度二和维度三实验，**可自由调整**。

---

## 🧮 流形几何基础

### OLM 流形 (Oblique Log-Euclidean Manifold)

**切空间映射（对数映射）**：
$$
\text{Log}_I(P) = \log(P) - \text{off}(\log(P))
$$

**指数映射**：
$$
\text{Exp}_I(\xi) = \exp(\xi + \text{off}(\xi))
$$

**测地距离**：
$$
d(P, Q) = \|\text{Log}_I(P) - \text{Log}_I(Q)\|_F
$$

**加权 Fréchet 均值**：
$$
\bar{P} = \text{Exp}_I\left(\sum_i w_i \cdot \text{Log}_I(P_i)\right)
$$

---

## 🔬 模块详解

| 模块 | 数学表示 | 功能 |
|------|----------|------|
| **FEM** | $x \mapsto h = Wx$ | 线性特征提取 |
| **MMM** | $h \mapsto \{C_i\}_{i=1}^S$ | 分段相关矩阵计算 |
| **HOM** | $C \mapsto (Q, K, V)$ | Cayley 线性同态 |
| **ATT** | $(Q, K, V) \mapsto R$ | 流形自注意力 |
| **PRJ** | $\{R_i\} \mapsto f$ | 切空间投影 + 展平 |
| **CLS** | $f \mapsto \hat{y}$ | 线性分类 + Softmax |

---

## 📚 参考文献

- 原论文: *A Correlation Manifold Self-Attention Network for EEG Decoding*
- JAX 文档: https://jax.readthedocs.io/
- Equinox 文档: https://docs.kidger.site/equinox/
- Optax 文档: https://optax.readthedocs.io/

---

## 📄 License

MIT License
