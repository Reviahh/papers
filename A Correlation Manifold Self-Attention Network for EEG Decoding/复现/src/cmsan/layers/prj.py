r"""
切空间投影模块 (Tangent Space Projection)
═══════════════════════════════════════════════════════════════════════════════

## 数学描述

### 接口

$$
\text{Prj}: (\text{Corr}^{++}_D)^S \to \mathbb{R}^{S \cdot D(D-1)/2}
$$

将流形表示投影到欧氏空间，用于下游分类.

### 切空间投影

对每个相关矩阵 $R_i$:

1. **对数映射到切空间**:
   $$L_i = \text{Logo}(R_i) = \text{Off}(\log R_i) \in \text{Hol}(D)$$

2. **提取下三角元素** (利用对称性):
   $$v_i = \text{tril}(L_i) \in \mathbb{R}^{D(D-1)/2}$$

3. **拼接所有段**:
   $$f = [v_1; v_2; \ldots; v_S] \in \mathbb{R}^{S \cdot D(D-1)/2}$$

### 维度分析

- 切空间 $\text{Hol}(D)$ 是 $D \times D$ 零对角对称矩阵空间
- 自由度: $D(D-1)/2$ (严格下三角元素)
- 总输出维度: $S \times D(D-1)/2$

例如: $D=20, S=3 \Rightarrow f \in \mathbb{R}^{570}$

### 几何意义

- 切空间是流形的局部线性近似
- 对数映射保持了流形结构信息
- 投影后可使用标准欧氏分类器

### 可选实现
- `tangent`: 切空间投影 (默认)
- `flatten`: 直接展平 (不做对数映射)
"""

import jax
import jax.numpy as jnp
from jax import vmap

from ops import tril, tril_dim
from manifold import logo


# ═══════════════════════════════════════
#           切空间投影 (默认)
# ═══════════════════════════════════════

def tangent(Rs):
    """
    Rs → f
    
    Rs: (S, D, D)
    f:  (S × D(D-1)/2,)
    """
    Ls = vmap(logo)(Rs)
    vs = vmap(tril)(Ls)
    return vs.flatten()


# ═══════════════════════════════════════
#           直接展平 (baseline)
# ═══════════════════════════════════════

def flatten(Rs):
    """直接展平，不做对数映射"""
    vs = vmap(tril)(Rs)
    return vs.flatten()


# ═══════════════════════════════════════
#           接口
# ═══════════════════════════════════════

PRJ = {
    'tangent': tangent,
    'flatten': flatten,
}


def output_dim(S, D, method='tangent'):
    """计算输出维度"""
    return S * tril_dim(D)