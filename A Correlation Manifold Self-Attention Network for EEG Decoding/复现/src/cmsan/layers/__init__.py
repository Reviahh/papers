r"""
CMSAN Layers - 可插拔模块框架
═══════════════════════════════════════════════════════════════════════════════

这个包提供了 CMSAN 的底层可插拔模块。每个模块都有多种实现，
可以通过注册表选择不同的变体进行消融实验。

## 模块结构

```
Composition: x → FEM → MMM → HOM → ATT → PRJ → CLS → ŷ
```

## 模块注册表

每个模块都是一个字典，键是变体名称，值是 (init_fn, forward_fn) 元组:

```python
from cmsan.layers import FEM, MMM, HOM, ATT, PRJ, CLS

# 获取卷积 FEM
fem_init, fem_fn = FEM['conv']

# 获取流形注意力
att_fn = ATT['manifold']
```

## 添加新模块

要添加新的模块实现，只需在对应文件中注册:

```python
# 在 fem.py 中
def init_my_fem(key, C, D, **kw):
    ...

def my_fem(x, θ):
    ...

FEM['my_fem'] = (init_my_fem, my_fem)
```
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
