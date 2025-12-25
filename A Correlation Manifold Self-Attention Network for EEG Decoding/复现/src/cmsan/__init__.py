"""
CMSAN - Correlation Manifold Self-Attention Network
═══════════════════════════════════════════════════════════════════════════════

基于 JAX/Equinox 的纯函数式 EEG 解码库。
针对性能进行了极致优化 (Scan Loop, No Python Overhead)。

核心理念:
    - 无状态类 (Stateless Classes): 模型是不可变的 PyTree。
    - 纯函数 (Pure Functions): 输入状态 -> 输出新状态。
    - 整体编译 (Whole-Graph Compilation): 整个训练过程编译为单一 XLA 内核。

用法:
    from cmsan import CMSAN, fit_unified, load_unified
    
    # 1. 加载数据
    X, y = load_unified('bcic', subject_id=1)

    # 2. 创建模型
    model = CMSAN(key, C=22, T=438, K=4)
    
    # 3. 极速训练 (自动根据 verbose 决定是否使用脉冲模式)
    model, history = fit_unified(model, X, y, key, epochs=50, batch_size=64, lr=1e-3)
"""

# 1. 模型定义 (Model Definition)
# 这些通常定义在 src/cmsan/model.py 中
from .model import (
    CMSAN,
    batch_forward,
    batch_predict,
    # 如果你的 model.py 里定义了这些层，可以保留暴露，方便魔改
    # FEMLayer, 
    # HOMLayer, 
    # CLSLayer,
)

# 2. 极速引擎 (The Engine)
# 对应 src/cmsan/engine.py
from .engine import (
    fit_unified,    # 唯一推荐的训练入口
    evaluate_pure,  # JIT 编译的快速评估
    compute_loss,   # 基础 Loss 函数
)

# 3. 数据适配器 (Data Adapter)
# 对应 src/cmsan/data.py
from .data import (
    load_unified,   # 统一加载接口
    DATASET_META,   # 数据集元数据
)

__all__ = [
    # Model
    "CMSAN",
    "batch_forward",
    "batch_predict",
    
    # Engine
    "fit_unified",
    "evaluate_pure",
    "compute_loss",
    
    # Data
    "load_unified",
    "DATASET_META",
]

__version__ = '2.0.0-fast'