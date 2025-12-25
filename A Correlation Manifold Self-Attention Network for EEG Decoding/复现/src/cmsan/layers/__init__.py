"""
CMSAN Layers - 可插拔模块框架
═══════════════════════════════════════════════════════════════════════════════

Pipeline: x → FEM → MMM → HOM → ATT → PRJ → CLS → ŷ

每个模块是一个字典，键是变体名称，值是 (init_fn, forward_fn) 元组:

    from cmsan.layers import FEM, MMM, HOM, ATT, PRJ, CLS
    
    # 获取卷积 FEM
    fem_init, fem_fn = FEM['conv']
    
    # 获取流形注意力
    att_fn = ATT['manifold']

添加新模块:
    # 在 fem.py 中
    def init_my_fem(key, C, D, **kw): ...
    def my_fem(x, θ): ...
    FEM['my_fem'] = (init_my_fem, my_fem)
"""

# 模块注册表
from .fem import FEM
from .mmm import MMM
from .hom import HOM
from .att import ATT
from .prj import PRJ, output_dim
from .cls import CLS
from .loss import LOSS

# 底层算子
from .ops import off, sym, tril, tril_dim, normalize, standardize
from .manifold import logo, expo, dist, wfm, cayley

__all__ = [
    # 模块注册表
    'FEM', 'MMM', 'HOM', 'ATT', 'PRJ', 'CLS', 'LOSS',
    'output_dim',
    # 流形算子
    'logo', 'expo', 'dist', 'wfm', 'cayley',
    # 基础算子
    'off', 'sym', 'tril', 'tril_dim', 'normalize', 'standardize',
]
