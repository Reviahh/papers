"""基础算子：纯函数，无状态"""

import jax.numpy as jnp


def off(X):
    """去对角: X → X - diag(X)"""
    return X - jnp.diag(jnp.diag(X))


def sym(X):
    """对称化: X → (X + Xᵀ)/2"""
    return (X + X.T) / 2


def tril(X, k=-1):
    """下三角向量化: X → vec(tril(X))"""
    idx = jnp.tril_indices(X.shape[0], k=k)
    return X[idx]


def tril_dim(n):
    """下三角维度: n → n(n-1)/2"""
    return n * (n - 1) // 2


def normalize(X, axis=-1, eps=1e-8):
    """L2 规范化"""
    return X / (jnp.linalg.norm(X, axis=axis, keepdims=True) + eps)


def standardize(X, axis=-1, eps=1e-8):
    """标准化: (X - μ) / σ"""
    μ = X.mean(axis=axis, keepdims=True)
    σ = X.std(axis=axis, keepdims=True)
    return (X - μ) / (σ + eps)
