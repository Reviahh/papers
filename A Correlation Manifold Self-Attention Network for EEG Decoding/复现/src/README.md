# CMSAN: Correlation Manifold Self-Attention Network

> 基于相关流形自注意力机制的 EEG 解码网络 (JAX + Equinox + Optax)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4+-green.svg)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

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

## 📁 项目结构

```
src/
├── main.py                 # 🚀 唯一入口 (纯函数式)
├── requirements.txt
├── README.md
│
├── cmsan/                  # 🧠 核心模块
│   ├── __init__.py         #    导出 CMSAN, data
│   ├── model.py            #    CMSAN 模型定义
│   ├── engine.py           #    🔥 训练引擎 (lax.scan)
│   ├── data.py             #    📦 数据加载器
│   └── layers/             #    流形层实现
│
├── configs/                # ⚙️ 配置预设
│   └── presets.py          #    fast / paper 参数
│
├── data/                   # 📊 数据集
│   ├── BCICIV_2a_mat/      #    BCI Competition IV 2a
│   ├── BCIcha/             #    BCI Challenge
│   ├── MAMEM/              #    MAMEM SSVEP
│   └── data_utils/         #    数据处理工具
│
├── checkpoints/            # 💾 模型存档
└── logs/                   # 📝 训练日志
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行模式

| 模式 | 命令 | 用途 | 硬件优化 |
|------|------|------|----------|
| **FAST** | `python main.py --mode fast` | 单被试快速训练 | i5-12500H P-Core 锁定 |
| **PAPER** | `python main.py --mode paper` | 全量基准测试 | TPU/GPU 集群 |

### FAST 模式 (本地开发)

```bash
# 默认: BCIC 数据集, 被试 1
python main.py --mode fast --dataset bcic --sub 1

# 输出示例:
# 🔒 [System] Process locked to P-Cores: [0, 1, 2, 3, 4, 5, 6, 7]
# 🚀 [System] Priority set to HIGH. E-Cores are banned.
# 15:45:22 | 🔥 MODE: FAST | P-Cores Only | Threads: 8
# ...
# 🎓 Train Acc: 98.04%
# 🏆 Test Acc:  75.86%
```

### PAPER 模式 (基准测试)

```bash
# 单数据集全被试
python main.py --mode paper --dataset bcic

# 所有数据集
python main.py --mode paper --dataset all

# 输出: SCI 格式报表
# ══════════════════════════════════════════════════════════════════════
# 🏁 BENCHMARK REPORT | Time: 45.2 min
# ══════════════════════════════════════════════════════════════════════
# Dataset      | N    | Mean ± Std         | Best
# --------------------------------------------------
# bcic         | 9    | 72.34% ± 8.21%     | 85.71%
# ══════════════════════════════════════════════════════════════════════
```

---

## ⚙️ 配置参数

| 参数 | FAST | PAPER | 说明 |
|------|------|-------|------|
| `epochs` | 100 | 200 | 训练轮数 |
| `batch_size` | 64 | 128 | 批大小 |
| `lr` | 1e-3 | 5e-4 | 学习率 |
| `d_model` | 32 | 64 | 隐藏维度 |
| `slices` | 4 | 8 | 时间切片数 |
| `save_model` | ✅ | ❌ | 保存检查点 |
| `verbose` | ✅ | ❌ | 进度输出 |

---

## 🖥️ 硬件自适应

### Intel 12代+ (i5-12500H)

```
自动检测 → P-Core 锁定 (Core 0-7) → 进程优先级 HIGH → E-Core 禁用
```

- `OMP_NUM_THREADS=8`
- `XLA_FLAGS='--xla_cpu_multi_thread_eigen=true'`
- 实测吞吐: ~32 samples/s

### Cloud TPU

```
自动检测 TPU_NAME 环境变量 → 跳过 CPU 亲和性 → 使用 TPU 调度
```

- `XLA_PYTHON_CLIENT_PREALLOCATE='true'`
- 大 batch (128) 利用并行

---

## 🧮 代码风格

### 纯函数式设计

```python
# ❌ 传统风格
for epoch in range(100):
    for batch in dataloader:
        loss = train_step(batch)

# ✅ 函数式风格 (本项目)
final_state, history = lax.scan(epoch_step, init_state, jnp.arange(epochs))
```

### 零 if/else 分支

```python
# ❌ 传统风格
if mode == 'fast':
    run_fast()
elif mode == 'paper':
    run_paper()

# ✅ 派发表风格 (本项目)
MODE_HANDLERS = {'fast': run_fast, 'paper': run_paper}
MODE_HANDLERS[mode](args)
```

---

## 🧪 最简示例

```python
import jax
from cmsan import CMSAN, data
from cmsan.engine import fit_unified, evaluate_pure

# 1. 加载数据
X, y = data.load_unified('bcic', subject=1)

# 2. 创建模型
key = jax.random.PRNGKey(42)
model = CMSAN(key, C=22, T=1000, K=4, D=32, S=4)

# 3. 训练 (全图编译，无 Python 循环)
model, history = fit_unified(model, X, y, key, epochs=100, batch_size=64, lr=1e-3)

# 4. 评估
acc = evaluate_pure(model, X_test, y_test)
print(f"Accuracy: {acc:.2%}")
```

---

## 🔬 流形几何

### OLM 流形 (Oblique Log-Euclidean Manifold)

| 操作 | 公式 |
|------|------|
| **对数映射** | $\text{Log}_I(P) = \log(P) - \text{off}(\log(P))$ |
| **指数映射** | $\text{Exp}_I(\xi) = \exp(\xi + \text{off}(\xi))$ |
| **测地距离** | $d(P, Q) = \|\text{Log}_I(P) - \text{Log}_I(Q)\|_F$ |
| **Fréchet 均值** | $\bar{P} = \text{Exp}_I\left(\sum_i w_i \cdot \text{Log}_I(P_i)\right)$ |

### 模块功能

| 模块 | 映射 | 功能 |
|------|------|------|
| **FEM** | $x \mapsto h = Wx$ | 线性特征提取 |
| **MMM** | $h \mapsto \{C_i\}_{i=1}^S$ | 分段相关矩阵 |
| **HOM** | $C \mapsto (Q, K, V)$ | Cayley 同态 |
| **ATT** | $(Q, K, V) \mapsto R$ | 流形自注意力 |
| **PRJ** | $\{R_i\} \mapsto f$ | 切空间投影 |
| **CLS** | $f \mapsto \hat{y}$ | 线性分类 |

---

## 📊 数据集支持

| 数据集 | 被试数 | 类别 | 任务 |
|--------|--------|------|------|
| `bcic` | 9 | 4 | Motor Imagery |
| `bcicha` | 9 | 4 | Motor Imagery |
| `mamem` | 11 | 5 | SSVEP |

### 数据格式

```
data/
├── BCICIV_2a_mat/
│   ├── BCIC_S01_T.mat    # 训练集
│   ├── BCIC_S01_E.mat    # 测试集
│   └── ...
```

---

## 📚 参考文献

- **原论文**: *A Correlation Manifold Self-Attention Network for EEG Decoding*
- **JAX**: https://jax.readthedocs.io/
- **Equinox**: https://docs.kidger.site/equinox/
- **Optax**: https://optax.readthedocs.io/

---

## 📄 License

MIT License