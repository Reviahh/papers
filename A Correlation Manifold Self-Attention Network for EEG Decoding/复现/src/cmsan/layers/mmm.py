"""流形映射模块 (MMM)"""

import jax
import jax.numpy as jnp
from jax import vmap

from .ops import sym


def split(h, S):
    """h → [h₁, h₂, ..., h_S]"""
    D, T = h.shape
    L = T // S
    return jnp.stack([h[:, i*L:(i+1)*L] for i in range(S)])


def corr(h, eps=1e-6):
    """相关矩阵: C = D⁻½ P D⁻½"""
    h = h - h.mean(axis=1, keepdims=True)
    P = h @ h.T / (h.shape[1] - 1 + eps)
    d = jnp.sqrt(jnp.diag(P) + eps)
    D_inv = jnp.diag(1 / d)
    C = D_inv @ P @ D_inv
    return sym(C) + eps * jnp.eye(C.shape[0])


def to_corr(h, S):
    """h → [C₁, ..., C_S]"""
    return vmap(corr)(split(h, S))


def cov(h, eps=1e-6):
    """协方差矩阵"""
    h = h - h.mean(axis=1, keepdims=True)
    P = h @ h.T / (h.shape[1] - 1 + eps)
    return sym(P) + eps * jnp.eye(P.shape[0])


def to_cov(h, S):
    """h → [P₁, ..., P_S]"""
    return vmap(cov)(split(h, S))


MMM = {
    'corr': to_corr,
    'cov': to_cov,
}
