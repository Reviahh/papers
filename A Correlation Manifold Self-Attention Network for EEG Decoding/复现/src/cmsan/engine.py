"""
CMSAN Engine: Unified Training Core
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import time
import platform
import ctypes
from functools import partial, reduce
from typing import Dict, Any, Tuple, Optional, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import random, lax
import equinox as eqx
import optax

# 内部导入
from .model import CMSAN, batch_forward, batch_predict


# ═══════════════════════════════════════════════════════════════════════════════
# 0. 类型定义
# ═══════════════════════════════════════════════════════════════════════════════

class TrainState(NamedTuple):
    """不可变训练状态"""
    model: CMSAN
    opt_state: optax.OptState
    key: jax.Array
    step: int


class TrainResult(NamedTuple):
    """训练结果"""
    model: CMSAN
    train_acc: float
    test_acc: float
    loss_history: jax.Array
    duration: float
    params_count: int


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def get_core_info() -> str:
    """获取当前 CPU 核心 ID (Windows)"""
    if platform.system() == 'Windows':
        try:
            return f"#{ctypes.windll.kernel32.GetCurrentProcessorNumber()}"
        except:
            pass
    return "?"


def count_params(model: CMSAN) -> int:
    """统计模型参数量"""
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))


def create_optimizer(
    lr: float,
    total_steps: int,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    warmup_ratio: float = 0.1,
) -> optax.GradientTransformation:
    """
    创建优化器 (AdamW + Cosine Decay + Warmup)
    """
    warmup_steps = int(total_steps * warmup_ratio)
    
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.01,
    )
    
    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(schedule, weight_decay=weight_decay),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 核心计算函数 (纯函数)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_loss(model: CMSAN, xs: jax.Array, ys: jax.Array) -> jax.Array:
    """计算批量交叉熵损失"""
    logits = batch_forward(model, xs)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, ys))


@eqx.filter_jit
def evaluate(model: CMSAN, xs: jax.Array, ys: jax.Array) -> jax.Array:
    """计算准确率"""
    preds = batch_predict(model, xs)
    return jnp.mean(preds == ys)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SCAN 模式 (TPU/GPU 全图编译)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_scan_trainer(
    optimizer: optax.GradientTransformation,
    batch_size: int,
    n_epochs: int,
    log_interval: int = 10,
) -> Callable:
    
    @eqx.filter_jit
    def train_scan(
        model: CMSAN,
        opt_state: optax.OptState,
        key: jax.Array,
        X: jax.Array,
        y: jax.Array,
        start_ts: float,
    ) -> Tuple[CMSAN, optax.OptState, jax.Array]:
        """全图编译训练"""
        
        N = X.shape[0]
        n_batches = N // batch_size
        
        # 数据规整
        X_trimmed = X[:n_batches * batch_size]
        y_trimmed = y[:n_batches * batch_size]
        X_batched = X_trimmed.reshape(n_batches, batch_size, *X.shape[1:])
        y_batched = y_trimmed.reshape(n_batches, batch_size)
        
        def epoch_step(state, epoch_idx):
            m, o, k = state
            k, subk = random.split(k)
            perm = random.permutation(subk, n_batches)
            
            def batch_step(carry, batch_data):
                curr_m, curr_o = carry
                bx, by = batch_data
                
                loss, grads = eqx.filter_value_and_grad(compute_loss)(curr_m, bx, by)
                updates, new_o = optimizer.update(
                    grads, curr_o, eqx.filter(curr_m, eqx.is_array)
                )
                new_m = eqx.apply_updates(curr_m, updates)
                
                return (new_m, new_o), loss
            
            # 打乱批次顺序
            X_shuffled = jnp.take(X_batched, perm, axis=0)
            y_shuffled = jnp.take(y_batched, perm, axis=0)
            
            (m, o), losses = lax.scan(batch_step, (m, o), (X_shuffled, y_shuffled))
            avg_loss = jnp.mean(losses)
            
            # 条件日志回调
            def log_callback(args):
                ep, loss, ts = args
                elapsed = time.time() - float(ts)
                print(f"Ep {int(ep)+1:<4} | {elapsed:>6.1f}s | Loss: {loss:.4f}")
            
            lax.cond(
                (epoch_idx + 1) % log_interval == 0,
                lambda _: jax.debug.callback(log_callback, (epoch_idx, avg_loss, start_ts)),
                lambda _: None,
                operand=None,
            )
            
            return (m, o, k), avg_loss
        
        (final_m, final_o, _), loss_history = lax.scan(
            epoch_step,
            (model, opt_state, key),
            jnp.arange(n_epochs),
        )
        
        return final_m, final_o, loss_history
    
    return train_scan


# ═══════════════════════════════════════════════════════════════════════════════
# 4. REDUCE 模式 (Windows 混合模式)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_reduce_trainer(
    optimizer: optax.GradientTransformation,
    batch_size: int,
    n_epochs: int,
    log_interval: int = 10,
    verbose: bool = True,
) -> Callable:
    
    def train_reduce(
        model: CMSAN,
        opt_state: optax.OptState,
        key: jax.Array,
        X: jax.Array,
        y: jax.Array,
    ) -> Tuple[CMSAN, optax.OptState, jax.Array]:
        """混合模式训练"""
        
        N = X.shape[0]
        n_batches = N // batch_size
        X_batched = X[:n_batches * batch_size].reshape(n_batches, batch_size, *X.shape[1:])
        y_batched = y[:n_batches * batch_size].reshape(n_batches, batch_size)
        
        start_time = time.time()
        loss_history = []
        
        # JIT 编译的单 epoch 函数
        @eqx.filter_jit
        def run_epoch(carry, perm):
            m, o = carry
            
            def batch_step(s, idx):
                curr_m, curr_o = s
                bx = jnp.take(X_batched, idx, axis=0)
                by = jnp.take(y_batched, idx, axis=0)
                
                loss, grads = eqx.filter_value_and_grad(compute_loss)(curr_m, bx, by)
                updates, new_o = optimizer.update(
                    grads, curr_o, eqx.filter(curr_m, eqx.is_array)
                )
                return (eqx.apply_updates(curr_m, updates), new_o), loss
            
            return lax.scan(batch_step, (m, o), perm)
        
        # Reduce 调度
        def epoch_step(accum, epoch_idx):
            curr_m, curr_o, curr_k = accum
            new_k, subkey = random.split(curr_k)
            perm = random.permutation(subkey, n_batches)
            
            (new_m, new_o), batch_losses = run_epoch((curr_m, curr_o), perm)
            loss_val = float(jnp.mean(batch_losses))
            loss_history.append(loss_val)
            
            # 日志
            if verbose and (epoch_idx + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Ep {epoch_idx+1:<4}/{n_epochs} | {elapsed:>6.1f}s | "
                      f"Core {get_core_info():<5} | Loss: {loss_val:.4f}")
            
            return (new_m, new_o, new_k)
        
        final_m, final_o, _ = reduce(
            epoch_step,
            range(n_epochs),
            (model, opt_state, key),
        )
        
        return final_m, final_o, jnp.array(loss_history)
    
    return train_reduce


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 统一训练接口
# ═══════════════════════════════════════════════════════════════════════════════

def train_session(
    X_train: jax.Array,
    y_train: jax.Array,
    config: Dict[str, Any],
    key: jax.Array,
    X_test: Optional[jax.Array] = None,
    y_test: Optional[jax.Array] = None,
) -> TrainResult:
    """
    统一训练入口
    """
    start_time = time.time()
    
    # 解包配置
    # 注意: config 可能已经是扁平化的，或者包含子字典
    # 我们的 configs/__init__.py 现在返回的是嵌套结构:
    # { 'train': {...}, 'model': { 'D':20, 'C':22... } }
    
    # 提取训练参数 (优先从 train 字段取，如果没有就从根目录取)
    train_cfg = config.get('train', config)
    # 提取模型参数
    model_cfg = config.get('model', {})
    
    # 训练超参
    epochs = train_cfg.get('epochs', 100)
    batch_size = train_cfg.get('batch_size', 64)
    lr = train_cfg.get('lr', 1e-3)
    verbose = train_cfg.get('verbose', True)
    log_interval = train_cfg.get('log_interval', 10)
    engine_mode = train_cfg.get('engine', 'auto')
    weight_decay = train_cfg.get('weight_decay', 0.01)
    grad_clip = train_cfg.get('grad_clip', 1.0)
    
    # 数据信息
    N = X_train.shape[0]
    
    # 分割密钥
    k_model, k_train = random.split(key)
    
    # -------------------------------------------------------------------------
    # 1. 创建模型 (Fixed)
    # -------------------------------------------------------------------------
    # 直接使用 model_cfg 里的参数，它现在应该包含 C, T, K, D, S 等所有必要信息
    try:
        model = CMSAN(
            key=k_model,
            **model_cfg 
        )
    except TypeError as e:
        print("\n❌ Model Init Error: Maybe config is missing 'C', 'T', or 'K'?")
        print(f"Current model_cfg keys: {list(model_cfg.keys())}")
        raise e
    
    params_count = count_params(model)
    
    if verbose:
        print(f"🧠 Model: {params_count:,} params")
        print(f"📊 Data: N={N}, C={model.C}, T={model.T}")
    
    # 2. 创建优化器
    steps_per_epoch = N // batch_size
    total_steps = epochs * steps_per_epoch
    
    optimizer = create_optimizer(
        lr=lr,
        total_steps=total_steps,
        weight_decay=weight_decay,
        grad_clip=grad_clip,
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # 3. 选择引擎
    use_scan = (
        engine_mode == 'scan' or
        (engine_mode == 'auto' and platform.system() != 'Windows' and not verbose)
    )
    
    if verbose:
        engine_name = 'SCAN (Whole-Graph)' if use_scan else 'REDUCE (Hybrid)'
        print(f"🔧 Engine: {engine_name}")
        print(f"🚀 Training: {epochs} epochs, batch={batch_size}, lr={lr}")
        print("-" * 60)
    
    # 4. 训练
    if use_scan:
        trainer = _make_scan_trainer(optimizer, batch_size, epochs, log_interval)
        model, _, loss_history = trainer(
            model, opt_state, k_train, X_train, y_train, time.time()
        )
    else:
        trainer = _make_reduce_trainer(optimizer, batch_size, epochs, log_interval, verbose)
        model, _, loss_history = trainer(model, opt_state, k_train, X_train, y_train)
    
    # 确保计算完成
    jax.block_until_ready(eqx.filter(model, eqx.is_array))
    
    # 5. 评估
    train_acc = float(evaluate(model, X_train, y_train))
    
    if X_test is not None and y_test is not None:
        test_acc = float(evaluate(model, X_test, y_test))
    else:
        test_acc = 0.0
    
    duration = time.time() - start_time
    
    if verbose:
        print("-" * 60)
        print(f"✅ Done in {duration:.1f}s")
        print(f"🎓 Train Acc: {train_acc:.2%}")
        if X_test is not None:
            print(f"🏆 Test Acc:  {test_acc:.2%}")
    
    return TrainResult(
        model=model,
        train_acc=train_acc,
        test_acc=test_acc,
        loss_history=loss_history,
        duration=duration,
        params_count=params_count,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 检查点管理
# ═══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model: CMSAN, path: str) -> None:
    """保存模型检查点"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        eqx.tree_serialise_leaves(f, model)
    print(f"💾 Saved: {path}")


def load_checkpoint(path: str, model_template: CMSAN) -> CMSAN:
    """加载模型检查点 (需要模型模板)"""
    with open(path, 'rb') as f:
        return eqx.tree_deserialise_leaves(f, model_template)


# 兼容旧接口
fit_unified = train_session
evaluate_pure = evaluate