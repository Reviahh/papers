"""特征提取模块 (FEM)"""

import jax
import jax.numpy as jnp


def init_conv(key, C, D, kernel=25):
    """初始化卷积 FEM"""
    k1, k2 = jax.random.split(key)
    return {
        'Ws': jax.random.normal(k1, (D, C)) * jnp.sqrt(2/C),
        'bs': jnp.zeros(D),
        'Wt': jax.random.normal(k2, (D, kernel)) * jnp.sqrt(2/(D*kernel)),
        'bt': jnp.zeros(D),
    }


def conv(x, θ):
    """h = σ(W_t * σ(W_s · x + b_s) + b_t)"""
    # 空间: (C, T) → (D, T)
    h = θ['Ws'] @ x + θ['bs'][:, None]
    h = jax.nn.elu(h)
    
    # 时间: 逐通道 1D 卷积
    D, T = h.shape
    K = θ['Wt'].shape[1]
    
    h_expanded = h[None, :, :]
    Wt_expanded = θ['Wt'][:, None, :]
    
    h = jax.lax.conv_general_dilated(
        h_expanded, Wt_expanded,
        window_strides=(1,),
        padding=((K // 2, K // 2),),
        feature_group_count=D,
    )[0] + θ['bt'][:, None]
    
    return jax.nn.elu(h)


def init_linear(key, C, D, **_):
    """初始化线性 FEM"""
    return {
        'W': jax.random.normal(key, (D, C)) * jnp.sqrt(2/C),
        'b': jnp.zeros(D),
    }


def linear(x, θ):
    """h = W · x + b"""
    return θ['W'] @ x + θ['b'][:, None]


FEM = {
    'conv': (init_conv, conv),
    'linear': (init_linear, linear),
}
