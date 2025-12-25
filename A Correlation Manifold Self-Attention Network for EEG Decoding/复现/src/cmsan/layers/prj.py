"""切空间投影模块 (PRJ)"""

import jax.numpy as jnp
from jax import vmap

from .ops import tril, tril_dim
from .manifold import logo


def tangent(Rs):
    """Rs → f (切空间投影)"""
    Ls = vmap(logo)(Rs)
    vs = vmap(tril)(Ls)
    return vs.flatten()


def flatten(Rs):
    """直接展平"""
    vs = vmap(tril)(Rs)
    return vs.flatten()


PRJ = {
    'tangent': tangent,
    'flatten': flatten,
}


def output_dim(S, D, method='tangent'):
    """计算输出维度"""
    return S * tril_dim(D)
