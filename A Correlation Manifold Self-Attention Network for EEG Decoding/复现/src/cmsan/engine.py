"""
CMSAN Engine: Functional Core
文件位置: src/cmsan/engine.py
职责: 全权负责模型生命周期（构建 -> 编译 -> 训练 -> 评估）。
"""
import jax
import jax.numpy as jnp
from jax import random, lax
import equinox as eqx
import optax
import time
import ctypes
import platform
from functools import partial, reduce

# 内部引用模型定义
from .model import CMSAN, batch_forward, batch_predict

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 纯函数式工具 (No For Loops)
# ═══════════════════════════════════════════════════════════════════════════════

def get_core_info():
    """Windows 核心 ID 获取 (Lambda版)"""
    return (lambda: f"#{ctypes.windll.kernel32.GetCurrentProcessorNumber()}" 
            if platform.system() == "Windows" else "?")()

def host_logger(args):
    """JAX 运行时回调打印"""
    epoch, loss, start_ts = args
    elapsed = time.time() - float(start_ts)
    print(f"Ep {int(epoch)+1:<4} | {elapsed:>6.1f}s | Core {get_core_info():<5} | Loss: {loss:.4f}")

def compute_loss(model, xs, ys):
    logits = batch_forward(model, xs)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, ys))

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 核心算子 (TPU/Whole-Graph Mode)
# ═══════════════════════════════════════════════════════════════════════════════

@eqx.filter_jit
def _train_scan(carrier, data, optimizer, batch_size, total_epochs, start_ts, log_int):
    """全图编译模式：使用 lax.scan 在 XLA 内部循环"""
    model, opt_state, key = carrier
    X, y = data
    n_batches = X.shape[0] // batch_size
    
    # 数据规整
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]
    X_b = X.reshape(n_batches, batch_size, *X.shape[1:])
    y_b = y.reshape(n_batches, batch_size)

    def epoch_step(state, epoch_idx):
        m, o, k = state
        k, subk = random.split(k)
        perm = random.permutation(subk, n_batches)
        
        def batch_step(s, batch_data):
            (curr_m, curr_o), (bx, by) = s, batch_data
            loss, grads = eqx.filter_value_and_grad(compute_loss)(curr_m, bx, by)
            updates, curr_o = optimizer.update(grads, curr_o, eqx.filter(curr_m, eqx.is_array))
            return (eqx.apply_updates(curr_m, updates), curr_o), loss

        # 乱序读取
        X_s, y_s = jnp.take(X_b, perm, axis=0), jnp.take(y_b, perm, axis=0)
        (m, o), losses = lax.scan(batch_step, (m, o), (X_s, y_s))
        
        # 仅在特定间隔回调打印 (副作用)
        avg_loss = jnp.mean(losses)
        lax.cond(
            (epoch_idx + 1) % log_int == 0,
            lambda _: jax.debug.callback(host_logger, (epoch_idx, avg_loss, start_ts)),
            lambda _: None,
            operand=None
        )
        return (m, o, k), avg_loss

    return lax.scan(epoch_step, (model, opt_state, key), jnp.arange(total_epochs))

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 混合算子 (Windows/Hybrid Mode)
# ═══════════════════════════════════════════════════════════════════════════════

def _train_reduce(model, X, y, key, epochs, batch_size, optimizer, opt_state, verbose):
    """混合模式：使用 functools.reduce 在 Python 端循环 (防卡死)"""
    n_batches = X.shape[0] // batch_size
    X_b = X[:n_batches*batch_size].reshape(n_batches, batch_size, *X.shape[1:])
    y_b = y[:n_batches*batch_size].reshape(n_batches, batch_size)
    start_global = time.time()

    # 单步 JIT 函数
    @eqx.filter_jit
    def step_jit(carrier, perm):
        def body(s, idx):
            (m, o), (bx, by) = s, (jnp.take(X_b, idx, axis=0), jnp.take(y_b, idx, axis=0))
            loss, grads = eqx.filter_value_and_grad(compute_loss)(m, bx, by)
            updates, o = optimizer.update(grads, o, eqx.filter(m, eqx.is_array))
            return (eqx.apply_updates(m, updates), o), loss
        return lax.scan(body, carrier, perm)

    # Reduce 调度器 (替代 for 循环)
    def reduce_step(accum, epoch_idx):
        curr_m, curr_o, curr_k, _ = accum
        new_k, subkey = random.split(curr_k)
        
        # 执行计算
        (new_m, new_o), batch_losses = step_jit((curr_m, curr_o), random.permutation(subkey, n_batches))
        loss_val = float(jnp.mean(batch_losses))
        
        # 打印日志 (利用短路逻辑)
        verbose and ((epoch_idx + 1) % 10 == 0) and print(
            f"Ep {epoch_idx+1:<4}/{epochs} | {time.time()-start_global:>6.1f}s | {get_core_info():<8} | {loss_val:.4f}"
        )
        return (new_m, new_o, new_k, loss_val)

    # 启动 Reduce
    final_m, final_o, _, _ = reduce(reduce_step, range(epochs), (model, opt_state, key, 0.0))
    return final_m

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 统一入口 (Public API)
# ═══════════════════════════════════════════════════════════════════════════════

def train_session(X_tr, y_tr, cfg, key_seed):
    """
    一站式服务：初始化 -> 训练 -> 返回模型
    """
    k_model, k_train = random.split(key_seed)
    
    # 1. 初始化模型
    n_class = len(jnp.unique(y_tr))
    model = CMSAN(
        key=k_model, C=X_tr.shape[1], T=X_tr.shape[2], K=n_class, 
        D=cfg['d_model'], S=cfg['slices']
    )
    
    # 2. 初始化优化器
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(optax.cosine_decay_schedule(cfg['lr'], cfg['epochs'] * (len(X_tr)//cfg['batch_size'])))
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # 3. 引擎路由 (Windows 强制 Hybrid)
    use_hybrid = (platform.system() == "Windows") or cfg['verbose']
    
    cfg['verbose'] and print(f"🔧 Engine: {'Hybrid (Reduce)' if use_hybrid else 'Whole-Graph (Scan)'}")
    
    if use_hybrid:
        return _train_reduce(model, X_tr, y_tr, k_train, cfg['epochs'], cfg['batch_size'], optimizer, opt_state, cfg['verbose'])
    else:
        log_int = 10 if cfg['verbose'] else 999999
        (final_m, _, _), _ = _train_scan(
            (model, opt_state, k_train), (X_tr, y_tr), optimizer, 
            cfg['batch_size'], cfg['epochs'], time.time(), log_int
        )
        return final_m

@eqx.filter_jit
def evaluate(model, xs, ys):
    return jnp.mean(batch_predict(model, xs) == ys)

def save_ckpt(model, path):
    with open(path, "wb") as f: eqx.tree_serialise_leaves(f, model)