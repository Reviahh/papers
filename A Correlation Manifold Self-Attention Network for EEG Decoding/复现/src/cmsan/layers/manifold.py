"""
相关矩阵流形 Corr⁺⁺ 上的运算

OLM (Off-Log Metric) 几何:
    - logo: Corr⁺⁺ → Hol(n)  (对数映射)
    - expo: Hol(n) → Corr⁺⁺  (指数映射)
    - dist: 测地距离
    - wfm: 加权 Fréchet 均值
"""

import jax
import jax.numpy as jnp
from jax.scipy.linalg import expm

from .ops import off, sym


def logm(A):
    """矩阵对数 (通过特征分解)"""
    eigvals, eigvecs = jnp.linalg.eigh(A)
    eigvals = jnp.maximum(eigvals, 1e-8)
    return eigvecs @ jnp.diag(jnp.log(eigvals)) @ eigvecs.T


def logo(C, eps=1e-5):
    """Logo: Corr⁺⁺ → Hol(n), C ↦ Off(log(C))"""
    return off(logm(C + eps * jnp.eye(C.shape[0])))


def expo(S, iters=12):
    """
    Expo: Hol(n) → Corr⁺⁺, S ↦ exp(S + D°(S))
    
    D° 通过固定点迭代求解
    """
    n = S.shape[0]
    
    def step(D, _):
        M = expm(S + jnp.diag(D))
        return D - jnp.log(jnp.diag(M) + 1e-8), None
    
    D, _ = jax.lax.scan(step, jnp.zeros(n), None, length=iters)
    return expm(S + jnp.diag(D))


def dist(C1, C2):
    """测地距离: d(C₁, C₂) = ‖Logo(C₁) - Logo(C₂)‖_F"""
    return jnp.linalg.norm(logo(C1) - logo(C2), 'fro')


def wfm(ws, Cs):
    """
    加权 Fréchet 均值: WFM({wᵢ}, {Cᵢ}) = Expo(Σ wᵢ Logo(Cᵢ))
    """
    log_Cs = jax.vmap(logo)(Cs)
    weighted = jnp.einsum('s,sij->ij', ws, log_Cs)
    return expo(weighted)


def cayley(A):
    """Cayley 变换: A → O ∈ O(n)"""
    S = A - A.T
    I = jnp.eye(A.shape[0])
    return jnp.linalg.solve(I + S, I - S)
