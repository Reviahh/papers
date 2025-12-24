r"""
损失函数
═══════════════════════════════════════════════════════════════════════════════

## 数学描述

### 接口

$$
\ell: \mathbb{R}^K \times \{0, 1, \ldots, K-1\} \to \mathbb{R}^+
$$

### 交叉熵损失 (默认)

$$
\ell_{\text{CE}}(z, y) = -\log\left(\frac{e^{z_y}}{\sum_{k=1}^K e^{z_k}}\right) = -z_y + \log\sum_{k=1}^K e^{z_k}
$$

数值稳定实现: $\ell = -\text{log\_softmax}(z)_y$

### Focal Loss

用于处理类别不平衡:

$$
\ell_{\text{focal}}(z, y) = -(1 - p_y)^\gamma \log(p_y)
$$

其中 $p = \text{softmax}(z)$, $\gamma$ 是聚焦参数.

### Label Smoothing

正则化技术，避免过度自信:

$$
\ell_{\text{smooth}}(z, y) = -\sum_{k=1}^K q_k \log(p_k)
$$

其中软标签:
$$
q_k = \begin{cases}
1 - \alpha + \alpha/K & k = y \\
\alpha/K & k \neq y
\end{cases}
$$

### 可选实现
- `ce`: 交叉熵 (默认)
- `focal`: Focal Loss
- `smooth_ce`: Label Smoothing CE
"""

import jax
import jax.numpy as jnp


# ═══════════════════════════════════════
#           交叉熵 (默认)
# ═══════════════════════════════════════

def cross_entropy(logits, label):
    """L = -log(softmax(logits)[label])"""
    log_probs = jax.nn.log_softmax(logits)
    return -log_probs[label]


# ═══════════════════════════════════════
#           Focal Loss
# ═══════════════════════════════════════

def focal(logits, label, gamma=2.0):
    """L = -(1-p)^γ log(p)"""
    probs = jax.nn.softmax(logits)
    p = probs[label]
    return -((1 - p) ** gamma) * jnp.log(p + 1e-8)


# ═══════════════════════════════════════
#           Label Smoothing
# ═══════════════════════════════════════

def smooth_ce(logits, label, α=0.1):
    """Label Smoothing CE"""
    K = logits.shape[0]
    log_probs = jax.nn.log_softmax(logits)
    smooth = jnp.ones(K) * (α / K)
    smooth = smooth.at[label].set(1 - α + α/K)
    return -jnp.sum(smooth * log_probs)


# ═══════════════════════════════════════
#           接口
# ═══════════════════════════════════════

LOSS = {
    'ce':        cross_entropy,
    'focal':     focal,
    'smooth_ce': smooth_ce,
}