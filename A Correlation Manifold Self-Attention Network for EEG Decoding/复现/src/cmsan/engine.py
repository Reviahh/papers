"""
CMSAN Engine: Pure Functional + IO Support
文件位置: src/cmsan/engine.py
"""
import jax
import jax.numpy as jnp
from jax import random, lax
import equinox as eqx
import optax
import time
import ctypes
import os
import platform
from typing import Tuple

from .model import CMSAN, batch_forward, batch_predict

# ─── 0. 宿主回调 (兼容 Windows/Linux) ───
def host_logger(args):
    epoch, loss, start_time = args
    elapsed = time.time() - float(start_time)
    
    core_info = "?"
    # 仅在 Windows 下尝试获取核心 ID
    if platform.system() == "Windows":
        try:
            core_id = ctypes.windll.kernel32.GetCurrentProcessorNumber()
            # i5-12500H: 0-7 P核, 8-15 E核
            c_type = "P" if core_id < 8 else "E"
            core_info = f"#{core_id}({c_type})"
        except: pass
    
    print(f"Ep {int(epoch)+1:<4} | {elapsed:>6.1f}s | Core {core_info:<5} | Loss: {loss:.4f}")

# ─── 1. 基础算子 ───
def compute_loss(model, xs, ys):
    logits = batch_forward(model, xs)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, ys))

def make_optimizer(lr, total_steps):
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps, alpha=0.01)
    return optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))

# ─── 2. 全图编译核心 (JIT) ───
@eqx.filter_jit
def train_all_pure(carrier, data, optimizer, batch_size, total_epochs, start_ts, log_interval):
    model, opt_state, key = carrier
    X, y = data
    N = X.shape[0]
    n_batches = N // batch_size
    n_keep = n_batches * batch_size
    X, y = X[:n_keep], y[:n_keep]
    X_batched = X.reshape(n_batches, batch_size, *X.shape[1:])
    y_batched = y.reshape(n_batches, batch_size)

    def batch_step(state, batch_data):
        m, o = state
        bx, by = batch_data
        loss, grads = eqx.filter_value_and_grad(compute_loss)(m, bx, by)
        updates, o = optimizer.update(grads, o, eqx.filter(m, eqx.is_array))
        m = eqx.apply_updates(m, updates)
        return (m, o), loss

    def epoch_step(state, epoch_idx):
        m, o, k = state
        k, subkey = random.split(k)
        perm = random.permutation(subkey, n_batches)
        X_s, y_s = jnp.take(X_batched, perm, axis=0), jnp.take(y_batched, perm, axis=0)
        (m, o), losses = lax.scan(batch_step, (m, o), (X_s, y_s))
        avg_loss = jnp.mean(losses)
        
        def do_log(_): jax.debug.callback(host_logger, (epoch_idx, avg_loss, start_ts))
        is_log_step = ((epoch_idx + 1) % log_interval == 0)
        lax.cond(is_log_step, do_log, lambda _: None, operand=None)

        return (m, o, k), avg_loss

    final_state, history = lax.scan(epoch_step, (model, opt_state, key), jnp.arange(total_epochs))
    return final_state, history

@eqx.filter_jit
def evaluate_pure(model, xs, ys):
    preds = batch_predict(model, xs)
    return jnp.mean(preds == ys)

# ─── 3. 统一训练接口 ───
def fit_unified(model, X, y, key, epochs, batch_size, lr, verbose=True):
    N = X.shape[0]
    optimizer = make_optimizer(lr, epochs * (N // batch_size))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    start_ts = time.time()
    # 如果不 verbose，设一个巨大的间隔，相当于不打印
    log_interval = 10 if verbose else 9999999
    
    (final_model, _, _), history = train_all_pure(
        (model, opt_state, key), (X, y), optimizer, batch_size, epochs, start_ts, log_interval
    )
    return final_model, history

# ─── 4. 🔥 模型 IO 工具 ───
def save_checkpoint(model, filename):
    """保存模型到 .eqx 文件"""
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)

def load_checkpoint(model_structure, filename):
    """加载模型"""
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model_structure)