r"""
优化器 (纯函数式)
═══════════════════════════════════════════════════════════════════════════════

## 函数式接口

$$
\begin{aligned}
\text{init}&: \Theta \to \mathcal{S} \\
\text{step}&: \Theta \times \mathcal{S} \times \nabla\Theta \to \Theta \times \mathcal{S}
\end{aligned}
$$

## Adam 优化器

状态: $s = (m, v, t)$

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= m_t / (1 - \beta_1^t) \\
\hat{v}_t &= v_t / (1 - \beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
\end{aligned}
$$

## 余弦学习率衰减

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

## Warmup + 余弦退火

$$
\eta_t = \begin{cases}
\eta_{\max} \cdot t / t_{\text{warmup}} & t < t_{\text{warmup}} \\
\text{cosine\_decay}(t - t_{\text{warmup}}) & t \geq t_{\text{warmup}}
\end{cases}
$$
"""

import jax
import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════════
#                              SGD + Momentum
# ═══════════════════════════════════════════════════════════════════════════

def init_momentum(θ):
    """初始化动量"""
    return jax.tree.map(jnp.zeros_like, θ)


def sgd_momentum(θ, v, g, lr=1e-3, mu=0.9):
    """
    SGD with Momentum
    
    v ← μv + g
    θ ← θ - lr * v
    """
    v = jax.tree.map(lambda vi, gi: mu * vi + gi, v, g)
    θ = jax.tree.map(lambda p, vi: p - lr * vi, θ, v)
    return θ, v


# ═══════════════════════════════════════════════════════════════════════════
#                              Adam
# ═══════════════════════════════════════════════════════════════════════════

def init_adam(θ):
    """初始化 Adam 状态"""
    return {
        'm': jax.tree.map(jnp.zeros_like, θ),  # 一阶矩
        'v': jax.tree.map(jnp.zeros_like, θ),  # 二阶矩
        't': 0                                   # 步数
    }


def adam(θ, state, g, lr=1e-3, β1=0.9, β2=0.999, eps=1e-8):
    """
    Adam 优化器
    
    m ← β₁m + (1-β₁)g
    v ← β₂v + (1-β₂)g²
    m̂ = m / (1-β₁ᵗ)
    v̂ = v / (1-β₂ᵗ)
    θ ← θ - lr * m̂ / (√v̂ + ε)
    """
    t = state['t'] + 1
    
    m = jax.tree.map(lambda mi, gi: β1 * mi + (1 - β1) * gi, state['m'], g)
    v = jax.tree.map(lambda vi, gi: β2 * vi + (1 - β2) * gi**2, state['v'], g)
    
    # 偏差修正
    m_hat = jax.tree.map(lambda mi: mi / (1 - β1**t), m)
    v_hat = jax.tree.map(lambda vi: vi / (1 - β2**t), v)
    
    # 更新参数
    θ = jax.tree.map(
        lambda p, mi, vi: p - lr * mi / (jnp.sqrt(vi) + eps),
        θ, m_hat, v_hat
    )
    
    return θ, {'m': m, 'v': v, 't': t}


# ═══════════════════════════════════════════════════════════════════════════
#                              学习率调度
# ═══════════════════════════════════════════════════════════════════════════

def cosine_decay(step, total_steps, lr_init, lr_min=1e-6):
    """余弦退火"""
    progress = step / total_steps
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + jnp.cos(jnp.pi * progress))


def warmup_cosine(step, warmup_steps, total_steps, lr_init):
    """Warmup + 余弦退火"""
    warmup_lr = lr_init * step / warmup_steps
    decay_lr = cosine_decay(step - warmup_steps, total_steps - warmup_steps, lr_init)
    return jnp.where(step < warmup_steps, warmup_lr, decay_lr)