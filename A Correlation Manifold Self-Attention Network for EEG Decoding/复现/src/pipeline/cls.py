r"""
分类模块 (Classifier)
═══════════════════════════════════════════════════════════════════════════════

## 数学描述

### 接口

$$
\text{Cls}: \mathbb{R}^d \times \Theta_{\text{Cls}} \to \mathbb{R}^K
$$

将特征向量映射到类别 logits.

### 线性分类器

$$
z = W_c \cdot f + b_c
$$

其中 $W_c \in \mathbb{R}^{K \times d}$, $b_c \in \mathbb{R}^K$.

### 预测概率

$$
\hat{y} = \text{softmax}(z) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}
$$

### 损失函数

交叉熵损失:

$$
\mathcal{L} = -\log(\hat{y}_y) = -z_y + \log\sum_{k=1}^K e^{z_k}
$$

### MLP 分类器 (可选)

$$
z = W_2 \cdot \sigma(W_1 \cdot f + b_1) + b_2
$$

其中 $\sigma$ 是 ReLU 激活函数.

### 可选实现
- `linear`: 线性分类器 (默认)
- `mlp`: 两层 MLP
"""

import jax
import jax.numpy as jnp


# ═══════════════════════════════════════
#           线性分类器 (默认)
# ═══════════════════════════════════════

def init_linear(key, d, K):
    """初始化线性分类器"""
    return {
        'W': jax.random.normal(key, (K, d)) * jnp.sqrt(2/d),
        'b': jnp.zeros(K),
    }


def linear(f, θ):
    """logits = W·f + b"""
    return θ['W'] @ f + θ['b']


# ═══════════════════════════════════════
#           MLP 分类器
# ═══════════════════════════════════════

def init_mlp(key, d, K, hidden=64):
    """初始化 MLP 分类器"""
    k1, k2 = jax.random.split(key)
    return {
        'W1': jax.random.normal(k1, (hidden, d)) * jnp.sqrt(2/d),
        'b1': jnp.zeros(hidden),
        'W2': jax.random.normal(k2, (K, hidden)) * jnp.sqrt(2/hidden),
        'b2': jnp.zeros(K),
    }


def mlp(f, θ):
    """两层 MLP"""
    h = jax.nn.relu(θ['W1'] @ f + θ['b1'])
    return θ['W2'] @ h + θ['b2']


# ═══════════════════════════════════════
#           接口
# ═══════════════════════════════════════

CLS = {
    'linear': (init_linear, linear),
    'mlp':    (init_mlp, mlp),
}