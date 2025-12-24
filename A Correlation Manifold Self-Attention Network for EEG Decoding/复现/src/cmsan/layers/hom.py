r"""
李群同态模块 (Hom: Lie Group Homomorphism)
═══════════════════════════════════════════════════════════════════════════════

## 数学描述

### 接口

$$
\text{Hom}: \text{Corr}^{++}_D \times \Theta_{\text{Hom}} \to \text{Corr}^{++}_D
$$

在 OLM 几何下实现可学习的李群变换.

### 核心公式

$$
\text{hom}(C; A) = \text{Expo}(\text{Off}(M^\top \cdot \text{Logo}(C) \cdot M))
$$

其中:
- $A \in \mathbb{R}^{D \times D}$: 可学习参数
- $M = \text{Cayley}(A) \in O(D)$: 正交变换矩阵

### Cayley 参数化

将任意矩阵参数化为正交矩阵:

$$
M = \text{Cayley}(A) = (I - S)(I + S)^{-1}, \quad S = A - A^\top
$$

这保证了 $M \in O(D)$，即 $M^\top M = I$.

### Q, K, V 生成

类似 Transformer 的多头注意力，使用三组独立参数:

$$
\begin{aligned}
Q_i &= \text{hom}(C_i; A_Q) \\
K_i &= \text{hom}(C_i; A_K) \\
V_i &= \text{hom}(C_i; A_V)
\end{aligned}
$$

### 几何意义

同态保持了流形结构:
- 正交变换 $M$ 作用于切空间 $\text{Hol}(D)$
- 通过 Logo/Expo 映射回流形
- 保持了相关矩阵的正定性和对角约束

### 可选实现
- `olm`: OLM 同态 (默认)
- `identity`: 恒等映射 (baseline)
"""

import jax
import jax.numpy as jnp
from jax import vmap

from .ops import off
from .manifold import logo, expo, cayley


# ═══════════════════════════════════════
#           OLM 同态 (默认)
# ═══════════════════════════════════════

def init_hom(key, D):
    """初始化同态参数"""
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        'Aq': jax.random.normal(k1, (D, D)) * 0.01,
        'Ak': jax.random.normal(k2, (D, D)) * 0.01,
        'Av': jax.random.normal(k3, (D, D)) * 0.01,
    }


def hom(C, A):
    """
    hom(C) = Expo(Off(Mᵀ Logo(C) M))
    
    C: (D, D) 相关矩阵
    A: (D, D) 可学习参数
    """
    M = cayley(A)
    L = logo(C)
    return expo(off(M.T @ L @ M))


def transform(Cs, θ):
    """
    生成 Q, K, V
    
    Cs: (S, D, D)
    θ:  {Aq, Ak, Av}
    
    返回: Qs, Ks, Vs 各 (S, D, D)
    """
    Qs = vmap(hom, in_axes=(0, None))(Cs, θ['Aq'])
    Ks = vmap(hom, in_axes=(0, None))(Cs, θ['Ak'])
    Vs = vmap(hom, in_axes=(0, None))(Cs, θ['Av'])
    return Qs, Ks, Vs


# ═══════════════════════════════════════
#           恒等同态 (baseline)
# ═══════════════════════════════════════

def init_identity(key, D):
    """无参数"""
    return {}


def transform_identity(Cs, θ):
    """Q = K = V = C"""
    return Cs, Cs, Cs


# ═══════════════════════════════════════
#           接口
# ═══════════════════════════════════════

HOM = {
    'olm':      (init_hom, transform),
    'identity': (init_identity, transform_identity),
}