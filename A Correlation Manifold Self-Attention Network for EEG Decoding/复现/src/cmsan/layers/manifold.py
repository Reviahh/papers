r"""
相关矩阵流形 Corr⁺⁺ 上的运算
═══════════════════════════════════════════════════════════════════════════════

## 数学基础

### 相关矩阵流形 Corr⁺⁺ₙ

相关矩阵是对角元为 1 的对称正定矩阵:

$$
\text{Corr}^{++}_n = \{C \in \text{Sym}^{++}_n : \text{diag}(C) = \mathbf{1}\}
$$

### Off-Log Metric (OLM)

OLM 在 Corr⁺⁺ 上定义了一个黎曼度量，通过以下映射实现:

**对数映射 (Logo)**:
$$
\text{Logo}: \text{Corr}^{++}_n \to \text{Hol}(n), \quad C \mapsto \text{Off}(\log C)
$$

其中 $\text{Hol}(n)$ 是零对角对称矩阵空间 (hollow matrices).

**指数映射 (Expo)**:
$$
\text{Expo}: \text{Hol}(n) \to \text{Corr}^{++}_n, \quad S \mapsto \exp(S + D^\circ)
$$

其中 $D^\circ$ 是使结果对角元为 1 的对角矩阵，通过固定点迭代求解:

$$
D^{(k+1)} = D^{(k)} - \log(\text{diag}(\exp(S + \text{diag}(D^{(k)}))))
$$

**测地距离**:
$$
d(C_1, C_2) = \|\text{Logo}(C_1) - \text{Logo}(C_2)\|_F
$$

**加权 Fréchet 均值**:
$$
\text{WFM}(\{w_i\}, \{C_i\}) = \text{Expo}\left(\sum_i w_i \cdot \text{Logo}(C_i)\right)
$$

### Cayley 变换

将任意矩阵参数化为正交矩阵:

$$
O = \text{Cayley}(A) = (I - S)(I + S)^{-1}, \quad S = A - A^\top
$$
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from .ops import off, sym


# ═══════════════════════════════════════
#           矩阵对数 (JAX 无内置 logm)
# ═══════════════════════════════════════

def logm(A):
    r"""
    矩阵对数: $\log(A) = V \cdot \text{diag}(\log(\lambda)) \cdot V^{-1}$
    
    通过特征值分解实现，适用于正定矩阵。
    """
    eigvals, eigvecs = jnp.linalg.eigh(A)
    # 确保特征值为正
    eigvals = jnp.maximum(eigvals, 1e-8)
    log_eigvals = jnp.log(eigvals)
    return eigvecs @ jnp.diag(log_eigvals) @ eigvecs.T


# ═══════════════════════════════════════
#           OLM 几何
# ═══════════════════════════════════════

def logo(C, eps=1e-5):
    """
    Logo: Corr⁺⁺ → Hol(n)
    
    C ↦ Off(log(C))
    """
    return off(logm(C + eps * jnp.eye(C.shape[0])))


def expo(S, iters=12):
    """
    Expo: Hol(n) → Corr⁺⁺
    
    S ↦ exp(S + D°(S))
    
    D° 通过固定点迭代求解:
        D^{k+1} = D^k - log(diag(exp(S + diag(D^k))))
    """
    n = S.shape[0]
    
    def step(D, _):
        M = expm(S + jnp.diag(D))
        return D - jnp.log(jnp.diag(M) + 1e-8), None
    
    D, _ = jax.lax.scan(step, jnp.zeros(n), None, length=iters)
    return expm(S + jnp.diag(D))


def dist(C1, C2):
    """
    测地距离: d(C₁, C₂) = ‖Logo(C₁) - Logo(C₂)‖_F
    """
    return jnp.linalg.norm(logo(C1) - logo(C2), 'fro')


def wfm(ws, Cs):
    """
    加权 Fréchet 均值: WFM({wᵢ}, {Cᵢ}) = Expo(Σ wᵢ Logo(Cᵢ))
    
    ws: (S,) 权重
    Cs: (S, D, D) 相关矩阵
    """
    log_Cs = jax.vmap(logo)(Cs)
    weighted = jnp.einsum('s,sij->ij', ws, log_Cs)
    return expo(weighted)


# ═══════════════════════════════════════
#           正交参数化
# ═══════════════════════════════════════

def cayley(A):
    """
    Cayley 变换: A → O ∈ O(n)
    
    S = A - Aᵀ  (反对称)
    O = (I - S)(I + S)⁻¹
    """
    S = A - A.T
    I = jnp.eye(A.shape[0])
    return jnp.linalg.solve(I + S, I - S)