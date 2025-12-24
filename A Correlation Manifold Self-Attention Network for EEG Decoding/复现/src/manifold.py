r"""
相关矩阵流形几何
═══════════════════════════════════════════════════════════════════════════════

Corr⁺⁺ₙ 上的 Off-Log Metric (OLM)

## 流形定义

$$
\text{Corr}^{++}_n = \{C \in \text{Sym}^{++}_n : \text{diag}(C) = \mathbf{1}\}
$$

## OLM 映射

### 对数映射 (Logo)
$$
\text{Logo}: \text{Corr}^{++}_n \to \text{Hol}(n), \quad C \mapsto \text{Off}(\log C)
$$

### 指数映射 (Expo)
$$
\text{Expo}: \text{Hol}(n) \to \text{Corr}^{++}_n, \quad S \mapsto \exp(S + D^\circ)
$$

其中 $D^\circ$ 由固定点迭代求解:
$$
D^{(k+1)} = D^{(k)} - \log(\text{diag}(\exp(S + \text{diag}(D^{(k)}))))
$$

使用 `jax.lax.scan` 实现 (JIT 友好).

### 测地距离
$$
d(C_1, C_2) = \|\text{Logo}(C_1) - \text{Logo}(C_2)\|_F
$$

### 加权 Fréchet 均值
$$
\text{WFM}(\{w_i\}, \{C_i\}) = \text{Expo}\left(\sum_i w_i \cdot \text{Logo}(C_i)\right)
$$

## Cayley 变换

参数化正交矩阵:
$$
O = \text{Cayley}(A) = (I - S)(I + S)^{-1}, \quad S = A - A^\top
$$

保证 $O \in O(n)$, 即 $O^\top O = I$.
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm, logm
from functools import partial


# ═══════════════════════════════════════════════════════════════════════════
#                              基础操作
# ═══════════════════════════════════════════════════════════════════════════

def off(X):
    """Off: X ↦ X - diag(X)"""
    return X - jnp.diag(jnp.diag(X))


def sym(X):
    """对称化"""
    return (X + X.T) / 2


def corr(X, eps=1e-6):
    """
    数据 → 相关矩阵
    
    X: (D, T) 数据矩阵
    C = D⁻½ P D⁻½, P = 协方差
    """
    X = X - X.mean(axis=1, keepdims=True)
    P = X @ X.T / (X.shape[1] - 1 + eps)
    d = jnp.sqrt(jnp.diag(P) + eps)
    D_inv = jnp.diag(1 / d)
    C = D_inv @ P @ D_inv
    # 保证对称 + 正定
    return sym(C) + eps * jnp.eye(C.shape[0])


# ═══════════════════════════════════════════════════════════════════════════
#                              OLM 几何
# ═══════════════════════════════════════════════════════════════════════════

def logo(C, eps=1e-5):
    """
    Logo: Corr⁺⁺ → Hol(n)
    C ↦ Off(log C)
    """
    C_reg = C + eps * jnp.eye(C.shape[0])
    return off(logm(C_reg).real)


def expo(S, iters=12):
    """
    Expo: Hol(n) → Corr⁺⁺
    固定点迭代: D^{k+1} = D^k - log(diag(exp(S + D^k)))
    
    使用 scan 替代 for 循环 (JIT友好)
    """
    n = S.shape[0]
    
    def step(D, _):
        M = expm(S + jnp.diag(D))
        D_new = D - jnp.log(jnp.diag(M) + 1e-8)
        return D_new, None
    
    D_init = jnp.zeros(n)
    D_final, _ = jax.lax.scan(step, D_init, None, length=iters)
    
    return expm(S + jnp.diag(D_final))


def dist(C1, C2):
    """
    测地距离
    d(C₁, C₂) = ‖Logo(C₁) - Logo(C₂)‖_F
    """
    return jnp.linalg.norm(logo(C1) - logo(C2), 'fro')


def wfm(ws, Cs):
    """
    加权 Fréchet 均值
    WFM({wᵢ}, {Cᵢ}) = Expo(Σ wᵢ Logo(Cᵢ))
    
    ws: (S,) 权重
    Cs: (S, D, D) 相关矩阵堆叠
    """
    # 批量 Logo
    log_Cs = jax.vmap(logo)(Cs)  # (S, D, D)
    
    # 加权求和
    weighted_sum = jnp.einsum('s,sij->ij', ws, log_Cs)
    
    return expo(weighted_sum)


# ═══════════════════════════════════════════════════════════════════════════
#                              正交参数化
# ═══════════════════════════════════════════════════════════════════════════

def cayley(A):
    """
    Cayley 变换: 反对称 → 正交
    O = (I - S)(I + S)⁻¹, S = A - Aᵀ
    """
    S = A - A.T
    I = jnp.eye(A.shape[0])
    return jnp.linalg.solve(I + S, I - S)


# ═══════════════════════════════════════════════════════════════════════════
#                              向量化
# ═══════════════════════════════════════════════════════════════════════════

def vec_tril(X):
    """提取严格下三角并展平"""
    n = X.shape[0]
    idx = jnp.tril_indices(n, k=-1)
    return X[idx]


def tril_dim(n):
    """下三角维度"""
    return n * (n - 1) // 2