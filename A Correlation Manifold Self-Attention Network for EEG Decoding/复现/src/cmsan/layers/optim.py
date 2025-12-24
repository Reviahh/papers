r"""
优化器
═══════════════════════════════════════════════════════════════════════════════

## 数学描述

### 函数式接口

$$
\begin{aligned}
\text{init}&: \Theta \to \mathcal{S} \\
\text{step}&: \Theta \times \mathcal{S} \times \nabla\Theta \to \Theta \times \mathcal{S}
\end{aligned}
$$

其中 $\mathcal{S}$ 是优化器状态空间.

### SGD with Momentum

状态: $s = (v,)$

更新规则:
$$
\begin{aligned}
v_t &= \mu \cdot v_{t-1} + g_t \\
\theta_t &= \theta_{t-1} - \eta \cdot v_t
\end{aligned}
$$

### Adam 优化器

状态: $s = (m, v, t)$

一阶矩估计 (动量):
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

二阶矩估计 (自适应学习率):
$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

偏差修正:
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

参数更新:
$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

### 余弦学习率衰减

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

### 可选实现
- `sgd`: SGD with Momentum
- `adam`: Adam (默认)
"""

import jax
import jax.numpy as jnp


# ═══════════════════════════════════════
#           SGD + Momentum
# ═══════════════════════════════════════

def init_sgd(θ):
    return {'v': jax.tree.map(jnp.zeros_like, θ)}


def sgd(θ, state, g, lr=1e-3, μ=0.9):
    """v ← μv + g, θ ← θ - lr·v"""
    v = jax.tree.map(lambda vi, gi: μ * vi + gi, state['v'], g)
    θ = jax.tree.map(lambda p, vi: p - lr * vi, θ, v)
    return θ, {'v': v}


# ═══════════════════════════════════════
#           Adam
# ═══════════════════════════════════════

def init_adam(θ):
    return {
        'm': jax.tree.map(jnp.zeros_like, θ),
        'v': jax.tree.map(jnp.zeros_like, θ),
        't': 0,
    }


def adam(θ, state, g, lr=1e-3, β1=0.9, β2=0.999, eps=1e-8):
    """Adam"""
    t = state['t'] + 1
    m = jax.tree.map(lambda mi, gi: β1*mi + (1-β1)*gi, state['m'], g)
    v = jax.tree.map(lambda vi, gi: β2*vi + (1-β2)*gi**2, state['v'], g)
    
    m_hat = jax.tree.map(lambda mi: mi / (1 - β1**t), m)
    v_hat = jax.tree.map(lambda vi: vi / (1 - β2**t), v)
    
    θ = jax.tree.map(lambda p, mi, vi: p - lr * mi / (jnp.sqrt(vi) + eps), θ, m_hat, v_hat)
    
    return θ, {'m': m, 'v': v, 't': t}


# ═══════════════════════════════════════
#           学习率调度
# ═══════════════════════════════════════

def cosine_lr(step, total, lr_max, lr_min=1e-6):
    """余弦退火"""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + jnp.cos(jnp.pi * step / total))


# ═══════════════════════════════════════
#           接口
# ═══════════════════════════════════════

OPTIM = {
    'sgd':  (init_sgd, sgd),
    'adam': (init_adam, adam),
}