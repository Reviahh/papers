"""李群同态模块 (HOM)"""

import jax
import jax.numpy as jnp
from jax import vmap

from .ops import off
from .manifold import logo, expo, cayley


def init_hom(key, D):
    """初始化同态参数"""
    k1, k2, k3 = jax.random.split(key, 3)
    return {
        'Aq': jax.random.normal(k1, (D, D)) * 0.01,
        'Ak': jax.random.normal(k2, (D, D)) * 0.01,
        'Av': jax.random.normal(k3, (D, D)) * 0.01,
    }


def hom(C, A):
    """hom(C) = Expo(Off(Mᵀ Logo(C) M))"""
    M = cayley(A)
    L = logo(C)
    return expo(off(M.T @ L @ M))


def transform(Cs, θ):
    """生成 Q, K, V"""
    Qs = vmap(hom, in_axes=(0, None))(Cs, θ['Aq'])
    Ks = vmap(hom, in_axes=(0, None))(Cs, θ['Ak'])
    Vs = vmap(hom, in_axes=(0, None))(Cs, θ['Av'])
    return Qs, Ks, Vs


def init_identity(key, D):
    """无参数"""
    return {}


def transform_identity(Cs, θ):
    """Q = K = V = C"""
    return Cs, Cs, Cs


HOM = {
    'olm': (init_hom, transform),
    'identity': (init_identity, transform_identity),
}
