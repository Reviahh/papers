r"""
CMSAN 模型封装 (Equinox)
═══════════════════════════════════════════════════════════════════════════════

Correlation Manifold Self-Attention Network

使用 Equinox 构建模块化神经网络，完全函数式，无 for/while/if-else。

## 数学描述

$$
f_\theta: \mathbb{R}^{C \times T} \to \Delta^{K-1}
$$

$$
f_\theta = \text{Cls} \circ \text{Prj} \circ \text{Att} \circ \text{Hom} \circ \text{MMM} \circ \text{FEM}
$$

## Equinox 特性

- `eqx.Module`: 不可变 PyTree，参数作为类属性自动追踪
- `eqx.filter_jit`: 仅 JIT 编译数组，自动处理静态字段
- `eqx.filter_grad`: 仅对数组求梯度
- `eqx.filter_vmap`: 批量映射，自动处理静态字段
"""

import jax
import jax.numpy as jnp
from jax import random
from typing import Dict, Any, Optional, Callable, Tuple

import equinox as eqx

# 从 layers 子包导入
from .layers import FEM, MMM, HOM, ATT, PRJ, CLS, output_dim


# ═══════════════════════════════════════════════════════════════════════════
#                       Equinox 子模块
# ═══════════════════════════════════════════════════════════════════════════

class FEMLayer(eqx.Module):
    r"""
    特征提取模块 (Feature Extraction Module)
    
    $$
    h = \text{FEM}(x) \in \mathbb{R}^{D \times T}
    $$
    """
    params: Dict[str, jax.Array]
    fn: Callable = eqx.field(static=True)
    
    def __init__(self, key: jax.Array, C: int, D: int, variant: str = 'conv', **kwargs):
        init_fn, self.fn = FEM[variant]
        self.params = init_fn(key, C, D, **kwargs)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fn(x, self.params)


class HOMLayer(eqx.Module):
    r"""
    同态映射模块 (Homomorphism Module)
    
    $$
    Q, K, V = \text{HOM}(C)
    $$
    """
    params: Dict[str, jax.Array]
    fn: Callable = eqx.field(static=True)
    
    def __init__(self, key: jax.Array, D: int, variant: str = 'olm'):
        init_fn, self.fn = HOM[variant]
        self.params = init_fn(key, D)
    
    def __call__(self, Cs: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        return self.fn(Cs, self.params)


class CLSLayer(eqx.Module):
    r"""
    分类模块 (Classification Module)
    
    $$
    \hat{y} = \text{softmax}(Wf + b)
    $$
    """
    params: Dict[str, jax.Array]
    fn: Callable = eqx.field(static=True)
    
    def __init__(self, key: jax.Array, in_dim: int, K: int, variant: str = 'linear'):
        init_fn, self.fn = CLS[variant]
        self.params = init_fn(key, in_dim, K)
    
    def __call__(self, f: jax.Array) -> jax.Array:
        return self.fn(f, self.params)


# ═══════════════════════════════════════════════════════════════════════════
#                       CMSAN 模型 (eqx.Module)
# ═══════════════════════════════════════════════════════════════════════════

class CMSAN(eqx.Module):
    r"""
    CMSAN 模型 (Equinox Module)
    
    Correlation Manifold Self-Attention Network
    相关矩阵流形上的自注意力网络。
    继承自 `eqx.Module`，参数作为类属性自动追踪，是不可变 PyTree。
    
    Example:
        >>> model = CMSAN(jax.random.key(0), C=22, T=438, D=20, S=3, K=4)
        >>> logits = model(x)  # 直接调用
        >>> logits = eqx.filter_jit(model)(x)  # JIT 编译
    """
    
    # ─────────────────────────────────────────────────────────────────────
    # 可训练子模块 (作为 PyTree 叶节点)
    # ─────────────────────────────────────────────────────────────────────
    fem: FEMLayer
    hom: HOMLayer
    cls: CLSLayer
    
    # ─────────────────────────────────────────────────────────────────────
    # 静态配置 (不参与梯度，eqx.field(static=True))
    # ─────────────────────────────────────────────────────────────────────
    C: int = eqx.field(static=True)
    T: int = eqx.field(static=True)
    D: int = eqx.field(static=True)
    S: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    
    # ─────────────────────────────────────────────────────────────────────
    # 无参数模块 (纯函数，静态)
    # ─────────────────────────────────────────────────────────────────────
    _mmm_fn: Callable = eqx.field(static=True)
    _att_fn: Callable = eqx.field(static=True)
    _prj_fn: Callable = eqx.field(static=True)
    
    def __init__(
        self,
        key: jax.Array,
        C: int,
        T: int,
        D: int,
        S: int,
        K: int,
        *,
        fem: str = 'conv',
        mmm: str = 'corr',
        hom: str = 'olm',
        att: str = 'manifold',
        prj: str = 'tangent',
        cls: str = 'linear',
        kernel: int = 25,
    ):
        """
        Args:
            key: JAX PRNG 密钥
            C: EEG 通道数
            T: 时间点数
            D: 特征维度
            S: 分段数
            K: 类别数
            fem/mmm/hom/att/prj/cls: 模块变体选择
            kernel: 卷积核大小
        """
        k1, k2, k3 = random.split(key, 3)
        
        # 静态配置
        self.C, self.T, self.D, self.S, self.K = C, T, D, S, K
        
        # 有参数模块 (Equinox 子模块)
        self.fem = FEMLayer(k1, C, D, variant=fem, kernel=kernel)
        self.hom = HOMLayer(k2, D, variant=hom)
        feat_dim = output_dim(S, D, prj)
        self.cls = CLSLayer(k3, feat_dim, K, variant=cls)
        
        # 无参数模块 (纯函数)
        self._mmm_fn = MMM[mmm]
        self._att_fn = ATT[att]
        self._prj_fn = PRJ[prj]
    
    def __call__(self, x: jax.Array) -> jax.Array:
        r"""
        前向传播 (单样本)
        
        $$
        f_\theta(x) = \text{Cls} \circ \text{Prj} \circ \text{Att} \circ \text{Hom} \circ \text{MMM} \circ \text{FEM}(x)
        $$
        
        Args:
            x: 输入 (C, T)
            
        Returns:
            logits: (K,)
        """
        # FEM: x ↦ h
        h = self.fem(x)
        
        # MMM: h ↦ {C_i}
        Cs = self._mmm_fn(h, self.S)
        
        # HOM: {C_i} ↦ (Q, K, V)
        Qs, Ks, Vs = self.hom(Cs)
        
        # ATT: (Q, K, V) ↦ R
        Rs = self._att_fn(Qs, Ks, Vs)
        
        # PRJ: R ↦ f
        f = self._prj_fn(Rs)
        
        # CLS: f ↦ logits
        logits = self.cls(f)
        
        return logits
    
    def predict(self, x: jax.Array) -> jax.Array:
        """预测类别 (单样本)"""
        return jnp.argmax(self(x))


# ═══════════════════════════════════════════════════════════════════════════
#                       批量操作 (使用 eqx.filter_vmap)
# ═══════════════════════════════════════════════════════════════════════════

def batch_forward(model: CMSAN, xs: jax.Array) -> jax.Array:
    """批量前向传播"""
    return eqx.filter_vmap(lambda m, x: m(x), in_axes=(None, 0))(model, xs)


def batch_predict(model: CMSAN, xs: jax.Array) -> jax.Array:
    """批量预测"""
    return eqx.filter_vmap(lambda m, x: m.predict(x), in_axes=(None, 0))(model, xs)


# ═══════════════════════════════════════════════════════════════════════════
#                       工厂函数
# ═══════════════════════════════════════════════════════════════════════════

def create_model(key: jax.Array, config: Optional[Dict] = None, **kwargs) -> CMSAN:
    """
    创建 CMSAN 模型的工厂函数
    
    Args:
        key: JAX PRNG 密钥
        config: 配置字典
        **kwargs: 覆盖配置
        
    Returns:
        CMSAN 模型实例
    """
    cfg = config or {}
    cfg.update(kwargs)
    return CMSAN(key, **cfg)


# ═══════════════════════════════════════════════════════════════════════════
#                       预设配置
# ═══════════════════════════════════════════════════════════════════════════

PRESETS = {
    'bcic': {
        'C': 22, 'T': 438, 'D': 20, 'S': 3, 'K': 4,
        'kernel': 25,
    },
    'light': {
        'C': 8, 'T': 128, 'D': 10, 'S': 2, 'K': 4,
        'kernel': 11,
    },
    'physionet': {
        'C': 64, 'T': 640, 'D': 32, 'S': 4, 'K': 4,
        'kernel': 31,
    },
}


def create_from_preset(key: jax.Array, name: str, **overrides) -> CMSAN:
    """从预设创建模型"""
    if name not in PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    cfg = {**PRESETS[name], **overrides}
    return CMSAN(key, **cfg)
