r"""
特征提取模块 (FEM)
═══════════════════════════════════════════════════════════════════════════════

## 数学描述

### 接口

$$
\text{FEM}: \mathbb{R}^{C \times T} \times \Theta_{\text{FEM}} \to \mathbb{R}^{D \times T}
$$

其中:
- $x \in \mathbb{R}^{C \times T}$: EEG 输入信号 (C 通道, T 时间点)
- $h \in \mathbb{R}^{D \times T}$: 特征表示 (D 特征维度)

### 卷积实现 (默认)

**空间卷积** (跨通道线性组合):
$$
h^{(1)} = \sigma(W_s \cdot x + b_s)
$$

其中 $W_s \in \mathbb{R}^{D \times C}$, $\sigma$ 是 ELU 激活函数.

**时间卷积** (逐通道 1D 卷积):
$$
h^{(2)}_d[t] = \sigma\left(\sum_{k=0}^{K-1} W_t[d, k] \cdot h^{(1)}_d[t + k - K/2] + b_t[d]\right)
$$

其中 $W_t \in \mathbb{R}^{D \times K}$ 是卷积核.

### 完整公式

$$
h = \text{FEM}(x; \theta) = \sigma(W_t * \sigma(W_s \cdot x + b_s) + b_t)
$$

### 可选实现
- `conv`: 空间+时间卷积 (默认)
- `linear`: 纯线性变换 $h = W \cdot x + b$
"""

import jax
import jax.numpy as jnp
from jax import vmap


# ═══════════════════════════════════════
#           卷积 FEM (默认)
# ═══════════════════════════════════════

def init_conv(key, C, D, kernel=25):
    """初始化卷积 FEM 参数"""
    k1, k2 = jax.random.split(key)
    return {
        'Ws': jax.random.normal(k1, (D, C)) * jnp.sqrt(2/C),
        'bs': jnp.zeros(D),
        'Wt': jax.random.normal(k2, (D, kernel)) * jnp.sqrt(2/(D*kernel)),
        'bt': jnp.zeros(D),
    }


def conv(x, θ):
    """
    h = σ(W_t * σ(W_s · x + b_s) + b_t)
    
    x: (C, T)
    θ: {Ws, bs, Wt, bt}
    h: (D, T)
    """
    # 空间: (C, T) → (D, T)
    h = θ['Ws'] @ x + θ['bs'][:, None]
    h = jax.nn.elu(h)
    
    # 时间: 逐通道 1D 卷积
    D, T = h.shape
    K = θ['Wt'].shape[1]
    pad = K // 2
    h_pad = jnp.pad(h, ((0, 0), (pad, pad)), mode='edge')
    
    def conv1d_at(t):
        window = h_pad[:, t:t+K]
        return jnp.sum(θ['Wt'] * window, axis=1)
    
    h = vmap(conv1d_at)(jnp.arange(T)).T + θ['bt'][:, None]
    h = jax.nn.elu(h)
    
    return h


# ═══════════════════════════════════════
#           线性 FEM (简单)
# ═══════════════════════════════════════

def init_linear(key, C, D, **_):
    """初始化线性 FEM 参数"""
    return {
        'W': jax.random.normal(key, (D, C)) * jnp.sqrt(2/C),
        'b': jnp.zeros(D),
    }


def linear(x, θ):
    """
    h = W · x + b
    """
    return θ['W'] @ x + θ['b'][:, None]


# ═══════════════════════════════════════
#           接口
# ═══════════════════════════════════════

FEM = {
    'conv':   (init_conv, conv),
    'linear': (init_linear, linear),
}