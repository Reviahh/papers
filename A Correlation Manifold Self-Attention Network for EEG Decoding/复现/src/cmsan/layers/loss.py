"""损失函数"""

import jax
import jax.numpy as jnp


def cross_entropy(logits, label):
    """交叉熵损失"""
    log_probs = jax.nn.log_softmax(logits)
    return -log_probs[label]


def focal(logits, label, gamma=2.0):
    """Focal Loss"""
    probs = jax.nn.softmax(logits)
    p = probs[label]
    return -((1 - p) ** gamma) * jnp.log(p + 1e-8)


def smooth_ce(logits, label, α=0.1):
    """Label Smoothing CE"""
    K = logits.shape[0]
    log_probs = jax.nn.log_softmax(logits)
    smooth = jnp.ones(K) * (α / K)
    smooth = smooth.at[label].set(1 - α + α/K)
    return -jnp.sum(smooth * log_probs)


LOSS = {
    'ce': cross_entropy,
    'focal': focal,
    'smooth_ce': smooth_ce,
}
