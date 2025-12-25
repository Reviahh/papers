"""
CMSAN Model Definition (Equinox)
全功能版: 包含所有 Layer 定义、工厂函数、批量工具，并修复了 cls 参数冲突。
"""
from typing import Dict, Any, Optional, Callable, Tuple, Union
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
from .layers import FEM, MMM, HOM, ATT, PRJ, CLS, output_dim

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 子模块封装 (显式定义类，以便 __init__.py 导出)
# ═══════════════════════════════════════════════════════════════════════════════

class FEMLayer(eqx.Module):
    """特征提取层包装器"""
    params: Any
    fn: Callable = eqx.field(static=True)
    
    def __init__(self, init_fn, fn, *args, **kwargs):
        self.params = init_fn(*args, **kwargs)
        self.fn = fn
        
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fn(x, self.params)

class HOMLayer(eqx.Module):
    """同态层包装器"""
    params: Any
    fn: Callable = eqx.field(static=True)
    
    def __init__(self, init_fn, fn, *args, **kwargs):
        self.params = init_fn(*args, **kwargs)
        self.fn = fn
        
    def __call__(self, x: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
        return self.fn(x, self.params)

class CLSLayer(eqx.Module):
    """分类层包装器"""
    params: Any
    fn: Callable = eqx.field(static=True)
    
    def __init__(self, init_fn, fn, *args, **kwargs):
        self.params = init_fn(*args, **kwargs)
        self.fn = fn
        
    def __call__(self, x: jax.Array) -> jax.Array:
        return self.fn(x, self.params)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. CMSAN 主模型
# ═══════════════════════════════════════════════════════════════════════════════

class CMSAN(eqx.Module):
    # 子模块
    fem: FEMLayer
    hom: HOMLayer
    cls_layer: CLSLayer  # 改名以避免与 python 关键字冲突
    
    # 静态配置
    C: int = eqx.field(static=True)
    T: int = eqx.field(static=True)
    D: int = eqx.field(static=True)
    S: int = eqx.field(static=True)
    K: int = eqx.field(static=True)
    
    # 静态函数引用
    _mmm_fn: Callable = eqx.field(static=True)
    _att_fn: Callable = eqx.field(static=True)
    _prj_fn: Callable = eqx.field(static=True)
    
    def __init__(
        self,
        key: jax.Array,
        C: int, T: int, D: int, S: int, K: int,
        # 避免使用 cls 作为参数名
        fem: str = 'conv',
        mmm: str = 'corr',
        hom: str = 'olm',
        att: str = 'manifold',
        prj: str = 'tangent',
        cls_type: str = 'linear',  # 默认名改成 cls_type
        kernel: int = 25,
        **kwargs # 吞掉可能传进来的 'cls' 参数
    ):
        # 兼容性处理：如果 config 里传的是 'cls'，在这里手动捞出来
        real_cls_type = kwargs.get('cls', cls_type)

        k1, k2, k3 = random.split(key, 3)
        self.C, self.T, self.D, self.S, self.K = C, T, D, S, K
        
        # 1. FEM
        fem_init, fem_fn = FEM[fem]
        self.fem = FEMLayer(fem_init, fem_fn, k1, C, D, kernel=kernel)
        
        # 2. HOM
        hom_init, hom_fn = HOM[hom]
        self.hom = HOMLayer(hom_init, hom_fn, k2, D)
        
        # 3. CLS
        feat_dim = output_dim(S, D, prj)
        cls_init, cls_fn = CLS[real_cls_type]
        self.cls_layer = CLSLayer(cls_init, cls_fn, k3, feat_dim, K)
        
        # 4. 无参模块
        self._mmm_fn = MMM[mmm]
        self._att_fn = ATT[att]
        self._prj_fn = PRJ[prj]

    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.fem(x)
        Cs = self._mmm_fn(h, self.S)
        Qs, Ks, Vs = self.hom(Cs)
        Rs = self._att_fn(Qs, Ks, Vs)
        f = self._prj_fn(Rs)
        return self.cls_layer(f)

    def predict(self, x: jax.Array) -> jax.Array:
        return jnp.argmax(self(x))

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 批量操作工具
# ═══════════════════════════════════════════════════════════════════════════════

def batch_forward(model: CMSAN, xs: jax.Array) -> jax.Array:
    """批量前向: (N, C, T) -> (N, K)"""
    return jax.vmap(model)(xs)

def batch_predict(model: CMSAN, xs: jax.Array) -> jax.Array:
    """批量预测: (N, C, T) -> (N,)"""
    return jax.vmap(model.predict)(xs)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 工厂函数与预设
# ═══════════════════════════════════════════════════════════════════════════════

# 内置预设，保证 create_from_preset 可用
PRESETS = {
    'bcic': {'C': 22, 'T': 438, 'D': 20, 'S': 3, 'K': 4, 'kernel': 25},
    'mamem': {'C': 8, 'T': 125, 'D': 10, 'S': 2, 'K': 5, 'kernel': 11},
    'physionet': {'C': 64, 'T': 640, 'D': 32, 'S': 4, 'K': 4, 'kernel': 31},
}

def create_model(key: jax.Array, config: Optional[Dict] = None, **kwargs) -> CMSAN:
    cfg = config or {}
    cfg.update(kwargs)
    return CMSAN(key, **cfg)

def create_from_preset(key: jax.Array, name: str, **overrides) -> CMSAN:
    """从预设创建模型 (保留给高级用户或旧代码使用)"""
    if name not in PRESETS:
        # 如果预设不存在，尝试用 default 参数创建一个空架子或者报错
        # 这里为了兼容性，如果传了足够的 overrides 也能跑
        if 'C' in overrides and 'T' in overrides and 'K' in overrides:
             base = {}
        else:
             raise ValueError(f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    else:
        base = PRESETS[name]
    
    full_cfg = base.copy()
    full_cfg.update(overrides)
    return CMSAN(key, **full_cfg)