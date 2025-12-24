r"""
流形映射模块 (MMM: Manifold Mapping Module)
═══════════════════════════════════════════════════════════════════════════════

## 数学描述

### 接口

$$
\text{MMM}: \mathbb{R}^{D \times T} \times \mathbb{N} \to (\text{Corr}^{++}_D)^S
$$

将欧氏特征映射到相关矩阵流形.

### 时间分段

将特征沿时间轴均匀分割:

$$
h \mapsto [h_1, h_2, \ldots, h_S], \quad h_i \in \mathbb{R}^{D \times (T/S)}
$$

### 相关矩阵计算

对每个分段 $h_i$ 计算相关矩阵:

$$
C_i = D_i^{-1/2} P_i D_i^{-1/2}
$$

其中:
- 中心化: $\bar{h}_i = h_i - \mathbb{E}[h_i]$
- 协方差: $P_i = \frac{1}{T_s - 1} \bar{h}_i \bar{h}_i^\top$
- 对角归一化: $D_i = \text{diag}(\sqrt{\text{diag}(P_i)})$

### 正则化

为保证数值稳定性:

$$
C_i \leftarrow \frac{C_i + C_i^\top}{2} + \epsilon I
$$

### 可选实现
- `corr`: 皮尔逊相关矩阵 (默认)
- `cov`: 协方差矩阵 (不归一化对角)
"""

import jax
import jax.numpy as jnp
from jax import vmap

from .ops import sym


# ═══════════════════════════════════════
#           分段
# ═══════════════════════════════════════

def split(h, S):
    """
    h → [h₁, h₂, ..., h_S]
    
    h: (D, T)
    输出: (S, D, T//S)
    """
    D, T = h.shape
    L = T // S
    return jnp.stack([h[:, i*L:(i+1)*L] for i in range(S)])


# ═══════════════════════════════════════
#           相关矩阵 (默认)
# ═══════════════════════════════════════

def corr(h, eps=1e-6):
    """
    C = D⁻½ P D⁻½
    
    h: (D, T) 单段特征
    C: (D, D) 相关矩阵
    """
    h = h - h.mean(axis=1, keepdims=True)
    P = h @ h.T / (h.shape[1] - 1 + eps)
    d = jnp.sqrt(jnp.diag(P) + eps)
    D_inv = jnp.diag(1 / d)
    C = D_inv @ P @ D_inv
    return sym(C) + eps * jnp.eye(C.shape[0])


def to_corr(h, S):
    """h → [C₁, ..., C_S]"""
    segments = split(h, S)
    return vmap(corr)(segments)


# ═══════════════════════════════════════
#           协方差矩阵
# ═══════════════════════════════════════

def cov(h, eps=1e-6):
    """
    P = (1/T) Σ (hᵢ - μ)(hᵢ - μ)ᵀ
    """
    h = h - h.mean(axis=1, keepdims=True)
    P = h @ h.T / (h.shape[1] - 1 + eps)
    return sym(P) + eps * jnp.eye(P.shape[0])


def to_cov(h, S):
    """h → [P₁, ..., P_S]"""
    segments = split(h, S)
    return vmap(cov)(segments)


# ═══════════════════════════════════════
#           接口
# ═══════════════════════════════════════

MMM = {
    'corr': to_corr,
    'cov':  to_cov,
}