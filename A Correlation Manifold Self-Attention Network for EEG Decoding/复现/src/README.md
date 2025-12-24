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

## 📁 项目结构

```
src/
├── cmsan/               # 🎯 用户 API (Equinox)
│   ├── __init__.py      #    导出 CMSAN, train, fit
│   ├── model.py         #    模型 (eqx.Module)
│   ├── train.py         #    训练器 (Equinox + Optax)
│   ├── README.md        #    API 文档
│   └── layers/          # 🔧 可插拔模块组合
│       ├── __init__.py  #    模块注册表导出
│       ├── README.md    #    数学文档
│       ├── pipe.py      #    组装器
│       ├── fem.py       #    特征提取模块 (FEM)
│       ├── mmm.py       #    流形映射模块 (MMM)
│       ├── hom.py       #    同态映射模块 (HOM)
│       ├── att.py       #    注意力模块 (ATT)
│       ├── prj.py       #    投影模块 (PRJ)
│       ├── cls.py       #    分类模块 (CLS)
│       ├── manifold.py  #    OLM 流形几何
│       ├── ops.py       #    基础算子
│       └── loss.py      #    损失函数
│
├── main.py              # 🚀 入口示例
└── requirements.txt     # 📦 依赖
```

---

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 最简示例 (Equinox 风格)

```python
import jax
from cmsan import CMSAN, train, fit

# 创建模型 (Equinox Module，参数内嵌)
model = CMSAN(jax.random.key(0), C=22, T=438, D=20, S=3, K=4)

# 训练 (完全函数式，无 for 循环)
model, losses = train(model, X_train, y_train, epochs=100)

# 或带日志的训练
model = fit(model, (X_train, y_train), epochs=100, verbose=True)

# 推理 (直接调用)
logits = model(x)
pred = model.predict(x)
```

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

## 📊 训练范式

### Equinox 函数式训练

使用 `eqx.filter_grad` + `lax.scan` 实现完全函数式训练：

$$
\text{model}_{t+1} = \text{update}(\text{model}_t, \nabla_\theta \mathcal{L})
$$

```python
import equinox as eqx
import optax

@eqx.filter_jit
def step(state, batch):
    xs, ys = batch
    # eqx.filter_grad 只对数组求梯度
    loss, grads = eqx.filter_value_and_grad(compute_loss)(state.model, xs, ys)
    
    # optax 更新
    updates, new_opt_state = optimizer.update(grads, state.opt_state)
    new_model = eqx.apply_updates(state.model, updates)
    
    return TrainState(new_model, new_opt_state, state.step + 1), loss

# lax.scan 替代 for 循环
final_state, losses = jax.lax.scan(step, init_state, batches)
```

### Equinox 优势

- **不可变模型**: `eqx.Module` 是 PyTree，参数作为属性自动追踪
- **自动静态/动态分离**: `eqx.filter_jit` 自动处理
- **CPU 兼容**: 纯 Python，无 C++ 依赖
- **无控制流**: 完全函数式，无 for/while/if-else

---

## 📚 参考文献

- 原论文: *A Correlation Manifold Self-Attention Network for EEG Decoding*
- JAX 文档: https://jax.readthedocs.io/
- Equinox 文档: https://docs.kidger.site/equinox/
- Optax 文档: https://optax.readthedocs.io/

---

## 📄 License

MIT License
