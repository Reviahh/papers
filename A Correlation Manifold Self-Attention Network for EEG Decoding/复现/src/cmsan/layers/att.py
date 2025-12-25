"""流形注意力模块 (ATT)"""

import jax
import jax.numpy as jnp
from jax import vmap

from .manifold import logo, dist, wfm


def scores(Qs, Ks):
    """sᵢⱼ = 1 / (1 + log(1 + dᵢⱼ))"""
    def row(Qi):
        return vmap(lambda Kj: dist(Qi, Kj))(Ks)
    
    dists = vmap(row)(Qs)
    return 1.0 / (1.0 + jnp.log(1.0 + dists))


def self_attention(Qs, Ks, Vs):
    """Rᵢ = WFM(softmax(sᵢ), V)"""
    S = scores(Qs, Ks)
    A = jax.nn.softmax(S, axis=1)
    return vmap(wfm, in_axes=(0, None))(A, Vs)


def euclidean_scores(Qs, Ks):
    """Frobenius 内积"""
    def row(Qi):
        return vmap(lambda Kj: jnp.sum(Qi * Kj))(Ks)
    return vmap(row)(Qs)


def euclidean_attention(Qs, Ks, Vs):
    """普通加权平均"""
    S = euclidean_scores(Qs, Ks)
    A = jax.nn.softmax(S, axis=1)
    return jnp.einsum('ij,jkl->ikl', A, Vs)


ATT = {
    'manifold': self_attention,
    'euclidean': euclidean_attention,
}
