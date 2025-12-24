# CMSAN API Reference (Equinox)

> Correlation Manifold Self-Attention Network for EEG Decoding
> 使用 Equinox + Optax 实现完全函数式深度学习

## 📐 数学概述

CMSAN 实现从 EEG 信号到类别概率的端到端映射：

$$
f_\theta: \mathbb{R}^{C \times T} \xrightarrow{\text{CMSAN}} \Delta^{K-1}
$$

---

## 🎯 核心类

### `CMSAN` - Equinox 模型

```python
import jax
from cmsan import CMSAN
```

#### 构造方法 (参数内嵌)

```python
# Equinox 风格：key 用于初始化，参数内嵌在模型中
model = CMSAN(
    jax.random.key(0),  # 初始化密钥
    C=22,               # EEG 通道数
    T=438,              # 时间点数
    D=20,               # 特征维度
    S=3,                # 分段数
    K=4,                # 类别数
    fem='conv',         # FEM 变体
    mmm='corr',         # MMM 变体
    hom='olm',          # HOM 变体
    att='manifold',     # ATT 变体
    prj='tangent',      # PRJ 变体
    cls='linear',       # CLS 变体
)
```

#### 预设配置

| 预设 | C | T | D | S | K | 特点 |
|------|---|---|---|---|---|------|
| `'light'` | 8 | 128 | 10 | 2 | 4 | 快速测试 |
| `'bcic'` | 22 | 438 | 20 | 3 | 4 | BCI 竞赛 |
| `'physionet'` | 64 | 640 | 32 | 4 | 4 | 高密度 |

```python
from cmsan import create_from_preset
model = create_from_preset(jax.random.key(0), 'bcic')
```

#### 方法 (Equinox 风格)

```python
# 直接调用 (无需 params，参数内嵌)
logits = model(x)  # x: (C, T) -> logits: (K,)
pred = model.predict(x)  # -> int

# 批量操作
from cmsan import batch_forward, batch_predict
logits = batch_forward(model, xs)  # xs: (N, C, T) -> (N, K)
preds = batch_predict(model, xs)   # -> (N,)

# JIT 编译
import equinox as eqx
jit_model = eqx.filter_jit(model)
logits = jit_model(x)
```

---

## 🚀 训练函数

### `train()` - 完全函数式训练

```python
from cmsan import train

# 完全函数式，无 for 循环 (使用 lax.fori_loop)
trained_model, losses = train(
    model,
    xs_train, ys_train,
    epochs=100,
    batch_size=16,
    lr=5e-4,
    key=jax.random.key(42),
)
```

### `fit()` - 带日志的训练

```python
from cmsan import fit

# 带日志输出 (内部用 Python for 循环)
trained_model = fit(
    model,
    (xs_train, ys_train),
    (xs_val, ys_val),  # 可选验证集
    epochs=100,
    batch_size=16,
    lr=5e-4,
    verbose=True,
    log_every=5,
)
```

---

## 📊 完整训练流程 (Equinox)

```python
import jax
from cmsan import CMSAN, fit, evaluate, save_model, load_model

# 1. 创建模型 (Equinox Module，参数内嵌)
model = CMSAN(jax.random.key(0), C=22, T=438, D=20, S=3, K=4)

# 2. 训练
trained_model = fit(
    model,
    (X_train, y_train),
    (X_val, y_val),
    epochs=100,
)

# 3. 评估
acc = evaluate(trained_model, X_test, y_test)

# 4. 推理 (直接调用)
logits = trained_model(x)
pred = trained_model.predict(x)

# 5. 保存/加载
save_model(trained_model, 'model.eqx')
loaded_model = load_model('model.eqx', model)  # 需要模板
```

---

## 🧮 数学细节

### 损失函数

**交叉熵损失**：
$$
\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{i=1}^N \log \hat{y}_{i, y_i}
$$

### 优化器 (Optax)

$$
\text{Optimizer} = \text{ClipNorm} \circ \text{AdamW} \circ \text{CosineDecay}
$$

```python
import optax

schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=lr,
    warmup_steps=warmup_steps,
    decay_steps=total_steps,
)
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(schedule, weight_decay=weight_decay),
)
```

---

## 🔄 训练状态 (不可变)

```python
from typing import NamedTuple
import equinox as eqx

class TrainState(NamedTuple):
    model: CMSAN            # Equinox Module (参数内嵌)
    opt_state: optax.OptState  # 优化器状态
    step: int               # 当前步数
    key: jax.Array          # PRNG 密钥
```

状态转移：
$$
(\text{model}_{t+1}, s_{t+1}) = \text{step}(\text{model}_t, s_t, x, y)
$$

---

## 📁 模块结构

```
cmsan/
├── __init__.py     # 导出: CMSAN, train, fit, evaluate, ...
├── model.py        # CMSAN (eqx.Module), FEMLayer, HOMLayer, CLSLayer
├── train.py        # train, fit, TrainState, make_optimizer, ...
├── README.md       # 本文档
└── layers/         # 底层可插拔模块
    ├── fem.py      # Feature Extraction Module
    ├── mmm.py      # Manifold Mapping Module
    ├── hom.py      # Homogeneous Mapping
    ├── att.py      # Attention Module
    ├── prj.py      # Projection Module
    ├── cls.py      # Classification Head
    ├── manifold.py # 流形算子
    ├── ops.py      # 基础算子
    └── loss.py     # 损失函数
```

---

## 🔗 与 Layers 的关系

```
┌─────────────────────────────────────────────────────┐
│              cmsan (Equinox API)                    │
│  ┌─────────┐  ┌─────────┐  ┌──────────────────┐   │
│  │ CMSAN   │  │  train  │  │ eqx.filter_jit   │   │
│  │(Module) │  │   fit   │  │ eqx.filter_grad  │   │
│  └────┬────┘  └────┬────┘  └────────┬─────────┘   │
│       │            │                │              │
│       ▼            ▼                ▼              │
├───────┴────────────┴────────────────┴──────────────┤
│              cmsan.layers (纯函数模块)              │
│  ┌─────┬─────┬─────┬─────┬─────┬─────┬──────────┐ │
│  │ FEM │ MMM │ HOM │ ATT │ PRJ │ CLS │ manifold │ │
│  └─────┴─────┴─────┴─────┴─────┴─────┴──────────┘ │
└─────────────────────────────────────────────────────┘
```

- **cmsan**: Equinox Module，参数内嵌，直接调用
- **cmsan.layers**: 底层纯函数模块，可自由组合

---

## 📚 参考

- [layers/README.md](./layers/README.md) - 底层模块数学文档
- [main.py](../main.py) - 完整使用示例
- [Equinox 文档](https://docs.kidger.site/equinox/)
