"""
CMSAN - Correlation Manifold Self-Attention Network
═══════════════════════════════════════════════════════════════════════════════

基于 JAX/Equinox 的纯函数式 EEG 解码库。

核心理念:
    - 无状态类 (Stateless): 模型是不可变的 PyTree
    - 纯函数 (Pure Functions): 输入状态 → 输出新状态
    - 整体编译 (Whole-Graph): 训练循环编译为单一 XLA 内核

快速开始:
    ```python
    from cmsan import CMSAN, train_session, load_unified
    from configs import get_full_config
    import jax
    
    # 1. 加载数据
    X, y = load_unified('bcic', subject_id=1)
    
    # 2. 获取配置
    config = get_full_config(mode='fast', dataset='bcic')
    
    # 3. 训练
    key = jax.random.PRNGKey(42)
    result = train_session(X_train, y_train, config, key, X_test, y_test)
    
    # 4. 使用模型
    model = result.model
    pred = model.predict(x)
    ```

模块结构:
    cmsan/
    ├── model.py      # CMSAN 模型定义
    ├── engine.py     # 训练引擎
    ├── data.py       # 数据加载
    └── layers/       # 底层模块
"""

__version__ = '3.0.0'

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 模型 (Model)
# ═══════════════════════════════════════════════════════════════════════════════

from .model import (
    CMSAN,
    batch_forward,
    batch_predict,
    create_model,
    create_from_preset,
    # 子模块 (高级用户)
    FEMLayer,
    HOMLayer,
    CLSLayer,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 引擎 (Engine)
# ═══════════════════════════════════════════════════════════════════════════════

from .engine import (
    # 核心 API
    train_session,
    evaluate,
    
    # 检查点
    save_checkpoint,
    load_checkpoint,
    
    # 工具
    compute_loss,
    count_params,
    create_optimizer,
    
    # 类型
    TrainState,
    TrainResult,
    
    # 兼容旧 API
    fit_unified,
    evaluate_pure,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 数据 (Data)
# ═══════════════════════════════════════════════════════════════════════════════

from .data import (
    load_unified,
    DATASET_META,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 底层模块 (Layers) - 可选导入
# ═══════════════════════════════════════════════════════════════════════════════

# 显式导入底层模块 (用于自定义/消融)
from .layers import (
    # 模块注册表
    FEM, MMM, HOM, ATT, PRJ, CLS, LOSS,
    
    # 流形算子
    logo, expo, dist, wfm, cayley,
    
    # 基础算子
    off, sym, tril, tril_dim, normalize, standardize,
    
    # 输出维度计算
    output_dim,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 公开 API
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # 版本
    '__version__',
    
    # 模型
    'CMSAN',
    'batch_forward',
    'batch_predict',
    'create_model',
    'create_from_preset',
    
    # 引擎
    'train_session',
    'evaluate',
    'save_checkpoint',
    'load_checkpoint',
    'compute_loss',
    'count_params',
    
    # 数据
    'load_unified',
    'DATASET_META',
    
    # 类型
    'TrainState',
    'TrainResult',
    
    # 底层 (可选)
    'FEM', 'MMM', 'HOM', 'ATT', 'PRJ', 'CLS', 'LOSS',
    'logo', 'expo', 'dist', 'wfm', 'cayley',
]
