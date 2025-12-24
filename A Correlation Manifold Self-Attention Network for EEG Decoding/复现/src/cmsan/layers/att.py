r"""
流形注意力模块 (Manifold Attention)
═══════════════════════════════════════════════════════════════════════════════

## 数学描述

### 接口

$$
\text{Att}: (\text{Corr}^{++}_D)^S \times (\text{Corr}^{++}_D)^S \times (\text{Corr}^{++}_D)^S \to (\text{Corr}^{++}_D)^S
$$

在相关矩阵流形上实现自注意力机制.

### 测地距离

使用 OLM 测地距离计算相似度:

$$
d_{ij} = \|\text{Logo}(Q_i) - \text{Logo}(K_j)\|_F
$$

其中 $\|\cdot\|_F$ 是 Frobenius 范数.

### 注意力分数

使用对数阻尼变换将距离转换为相似度:

$$
s_{ij} = \frac{1}{1 + \log(1 + d_{ij})}
$$

这个变换满足:
- $d_{ij} = 0 \Rightarrow s_{ij} = 1$ (最相似)
- $d_{ij} \to \infty \Rightarrow s_{ij} \to 0$ (最不相似)
- 对数阻尼避免了极端值

### 注意力权重

$$
\alpha_{ij} = \frac{\exp(s_{ij})}{\sum_{k=1}^S \exp(s_{ik})} = \text{softmax}_j(s_{ij})
$$

### 加权 Fréchet 均值聚合

$$
R_i = \text{WFM}(\{\alpha_{ij}\}_{j=1}^S, \{V_j\}_{j=1}^S) = \text{Expo}\left(\sum_{j=1}^S \alpha_{ij} \cdot \text{Logo}(V_j)\right)
$$

### 几何意义

- 距离计算在切空间 (Logo) 进行
- 聚合通过切空间线性组合 + Expo 映射回流形
- 保持了相关矩阵的流形结构

### 可选实现
- `manifold`: 流形注意力 (默认)
- `euclidean`: 欧氏内积注意力 (baseline)
"""

import jax
import jax.numpy as jnp
from jax import vmap

from manifold import logo, dist, wfm


# ═══════════════════════════════════════
#           自注意力 (默认)
# ═══════════════════════════════════════

def scores(Qs, Ks):
    """
    sᵢⱼ = 1 / (1 + log(1 + dᵢⱼ))
    
    Qs, Ks: (S, D, D)
    返回: (S, S) 分数矩阵
    """
    def row(Qi):
        return vmap(lambda Kj: dist(Qi, Kj))(Ks)
    
    dists = vmap(row)(Qs)
    return 1.0 / (1.0 + jnp.log(1.0 + dists))


def self_attention(Qs, Ks, Vs):
    """
    Rᵢ = WFM(softmax(sᵢ), V)
    
    返回: (S, D, D)
    """
    S = scores(Qs, Ks)
    A = jax.nn.softmax(S, axis=1)
    
    return vmap(wfm, in_axes=(0, None))(A, Vs)


# ═══════════════════════════════════════
#           欧氏注意力 (baseline)
# ═══════════════════════════════════════

def euclidean_scores(Qs, Ks):
    """用 Frobenius 内积代替测地距离"""
    def row(Qi):
        return vmap(lambda Kj: jnp.sum(Qi * Kj))(Ks)
    return vmap(row)(Qs)


def euclidean_attention(Qs, Ks, Vs):
    """普通加权平均"""
    S = euclidean_scores(Qs, Ks)
    A = jax.nn.softmax(S, axis=1)
    return jnp.einsum('ij,jkl->ikl', A, Vs)


# ═══════════════════════════════════════
#           接口
# ═══════════════════════════════════════

ATT = {
    'manifold':  self_attention,
    'euclidean': euclidean_attention,
}