r"""
CMSAN - Correlation Manifold Self-Attention Network
═══════════════════════════════════════════════════════════════════════════════

相关矩阵流形自注意力网络，使用 Equinox + Optax 实现完全函数式深度学习。

用法:
    import jax
    from cmsan import CMSAN, train, fit
    
    # 创建模型 (Equinox Module，参数内嵌)
    model = CMSAN(jax.random.key(0), C=22, T=438, D=20, S=3, K=4)
    
    # 训练 (完全函数式，无 for 循环)
    model, losses = train(model, xs_train, ys_train, epochs=100)
    
    # 或带日志的训练
    model = fit(model, (xs_train, ys_train), epochs=100, verbose=True)
    
    # 推理
    logits = model(x)  # 直接调用
    pred = model.predict(x)
"""

from .model import (
    CMSAN,
    FEMLayer,
    HOMLayer,
    CLSLayer,
    batch_forward,
    batch_predict,
    create_model,
    create_from_preset,
    PRESETS,
)
from .train import (
    TrainState,
    train,
    fit,
    evaluate,
    make_optimizer,
    make_step_fn,
    make_epoch_fn,
    save_model,
    load_model,
)

__all__ = [
    # 模型
    'CMSAN',
    'FEMLayer',
    'HOMLayer',
    'CLSLayer',
    'batch_forward',
    'batch_predict',
    'create_model',
    'create_from_preset',
    'PRESETS',
    # 训练
    'TrainState',
    'train',
    'fit',
    'evaluate',
    'make_optimizer',
    'make_step_fn',
    'make_epoch_fn',
    'save_model',
    'load_model',
]
__version__ = '1.0.0'
