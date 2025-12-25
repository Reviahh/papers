"""分类模块 (CLS)"""

import jax
import jax.numpy as jnp


def init_linear(key, d, K):
    """初始化线性分类器"""
    return {
        'W': jax.random.normal(key, (K, d)) * jnp.sqrt(2/d),
        'b': jnp.zeros(K),
    }


def linear(f, θ):
    """logits = W·f + b"""
    return θ['W'] @ f + θ['b']


def init_mlp(key, d, K, hidden=64):
    """初始化 MLP 分类器"""
    k1, k2 = jax.random.split(key)
    return {
        'W1': jax.random.normal(k1, (hidden, d)) * jnp.sqrt(2/d),
        'b1': jnp.zeros(hidden),
        'W2': jax.random.normal(k2, (K, hidden)) * jnp.sqrt(2/hidden),
        'b2': jnp.zeros(K),
    }


def mlp(f, θ):
    """两层 MLP"""
    h = jax.nn.relu(θ['W1'] @ f + θ['b1'])
    return θ['W2'] @ h + θ['b2']


CLS = {
    'linear': (init_linear, linear),
    'mlp': (init_mlp, mlp),
}
