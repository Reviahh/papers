r"""
CorAtt Pipeline 组装器
═══════════════════════════════════════════════════════════════════════════════

## 数学描述

### 完整 Pipeline

$$
f_\theta: \mathbb{R}^{C \times T} \to \Delta^{K-1}
$$

$$
f_\theta = \text{Cls} \circ \text{Prj} \circ \text{Att} \circ \text{Hom} \circ \text{MMM} \circ \text{FEM}
$$

### 模块组合

| 模块 | 功能 | 数学符号 |
|--------|------|------------|
| FEM | 特征提取 | $\mathbb{R}^{C \times T} \to \mathbb{R}^{D \times T}$ |
| MMM | 流形映射 | $\mathbb{R}^{D \times T} \to (\text{Corr}^{++}_D)^S$ |
| HOM | 李群同态 | $(\text{Corr}^{++}_D)^S \to (\text{Corr}^{++}_D)^{3S}$ |
| ATT | 流形注意力 | $(\text{Corr}^{++}_D)^{3S} \to (\text{Corr}^{++}_D)^S$ |
| PRJ | 切空间投影 | $(\text{Corr}^{++}_D)^S \to \mathbb{R}^d$ |
| CLS | 分类 | $\mathbb{R}^d \to \mathbb{R}^K$ |

### 可插拔设计

每个模块都有多种实现，通过配置字典选择:

```python
cfg = {
    'fem': 'conv',      # 或 'linear'
    'mmm': 'corr',      # 或 'cov'
    'hom': 'olm',       # 或 'identity'
    'att': 'manifold',  # 或 'euclidean'
    'prj': 'tangent',   # 或 'flatten'
    'cls': 'linear',    # 或 'mlp'
}
pipe = build(cfg)
θ = pipe.init(key, C, T, D, S, K)
logits = pipe.forward(θ, x)
```
"""

import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from functools import partial

from fem import FEM
from mmm import MMM
from hom import HOM
from att import ATT
from prj import PRJ, output_dim
from cls import CLS
from loss import LOSS
from optim import OPTIM


# ═══════════════════════════════════════
#           Pipeline 构建
# ═══════════════════════════════════════

class Pipeline:
    """可配置的 CorAtt Pipeline"""
    
    def __init__(self, cfg):
        """
        cfg: dict
            fem:   'conv' | 'linear'
            mmm:   'corr' | 'cov'
            hom:   'olm'  | 'identity'
            att:   'manifold' | 'euclidean'
            prj:   'tangent' | 'flatten'
            cls:   'linear' | 'mlp'
            loss:  'ce' | 'focal' | 'smooth_ce'
            optim: 'sgd' | 'adam'
        """
        self.cfg = {
            'fem': 'conv', 'mmm': 'corr', 'hom': 'olm',
            'att': 'manifold', 'prj': 'tangent', 'cls': 'linear',
            'loss': 'ce', 'optim': 'adam',
            **cfg
        }
        
        # 绑定模块
        self.fem_init, self.fem_fn = FEM[self.cfg['fem']]
        self.mmm_fn = MMM[self.cfg['mmm']]
        self.hom_init, self.hom_fn = HOM[self.cfg['hom']]
        self.att_fn = ATT[self.cfg['att']]
        self.prj_fn = PRJ[self.cfg['prj']]
        self.cls_init, self.cls_fn = CLS[self.cfg['cls']]
        self.loss_fn = LOSS[self.cfg['loss']]
        self.opt_init, self.opt_fn = OPTIM[self.cfg['optim']]
    
    def init(self, key, C, T, D, S, K, **kw):
        """初始化所有参数"""
        keys = random.split(key, 3)
        
        feat_dim = output_dim(S, D, self.cfg['prj'])
        
        return {
            'fem': self.fem_init(keys[0], C, D, **kw),
            'hom': self.hom_init(keys[1], D),
            'cls': self.cls_init(keys[2], feat_dim, K),
            '_S': S,  # 保存配置
        }
    
    def forward(self, θ, x):
        """
        前向传播
        
        x ──→ FEM ──→ MMM ──→ HOM ──→ ATT ──→ PRJ ──→ CLS ──→ logits
        """
        S = θ['_S']
        
        # Step 1: 特征提取
        h = self.fem_fn(x, θ['fem'])
        
        # Step 2: 流形映射
        Cs = self.mmm_fn(h, S)
        
        # Step 3: 李群同态
        Qs, Ks, Vs = self.hom_fn(Cs, θ['hom'])
        
        # Step 4: 流形注意力
        Rs = self.att_fn(Qs, Ks, Vs)
        
        # Step 5: 切空间投影
        f = self.prj_fn(Rs)
        
        # Step 6: 分类
        logits = self.cls_fn(f, θ['cls'])
        
        return logits
    
    def loss(self, θ, x, y):
        """单样本损失"""
        logits = self.forward(θ, x)
        return self.loss_fn(logits, y)
    
    def loss_batch(self, θ, xs, ys):
        """批量损失"""
        return jnp.mean(vmap(self.loss, in_axes=(None, 0, 0))(θ, xs, ys))
    
    def predict(self, θ, x):
        """预测"""
        return jnp.argmax(self.forward(θ, x))
    
    def accuracy(self, θ, xs, ys):
        """准确率"""
        preds = vmap(self.predict, in_axes=(None, 0))(θ, xs)
        return jnp.mean(preds == ys)


# ═══════════════════════════════════════
#           便捷构建函数
# ═══════════════════════════════════════

def build(cfg=None):
    """构建 Pipeline"""
    return Pipeline(cfg or {})


def default():
    """默认配置"""
    return build({
        'fem': 'conv',
        'mmm': 'corr',
        'hom': 'olm',
        'att': 'manifold',
        'prj': 'tangent',
        'cls': 'linear',
        'loss': 'ce',
        'optim': 'adam',
    })