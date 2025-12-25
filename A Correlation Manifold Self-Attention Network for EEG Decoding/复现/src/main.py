"""
CMSAN Functional Main: Pure FP Orchestration
═══════════════════════════════════════════════════════════════════════════════
设计: 
  ✅ 零 for/if/else - 纯派发表 + map/reduce
  ✅ 复用 engine.py 的训练核心
  ✅ FAST: P-Core 锁定 (engine.py 自动检测)
  ✅ PAPER: TPU 全量基准
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import os
import sys
import platform
import argparse
import time
import gc
import logging
from functools import partial, reduce
from typing import NamedTuple, Callable, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 0. 🛡️ Pre-JAX Bootstrap (P-Core Lock)
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_hardware():
    """硬件初始化 (必须在 import jax 前)"""
    import psutil
    
    # 硬件配置表
    profiles = {
        'i5-12500h': {'cores': list(range(8)), 'threads': 8, 'priority': 'high'},
        'tpu':       {'cores': None, 'threads': 0, 'priority': 'normal'},
        'default':   {'cores': None, 'threads': os.cpu_count(), 'priority': 'normal'},
    }
    
    # 检测硬件
    hw_type = next((
        k for k, pred in [
            ('tpu', lambda: 'tpu' in os.environ.get('TPU_NAME', '').lower()),
            ('i5-12500h', lambda: '12500' in platform.processor()),
        ] if pred()
    ), 'default')
    
    profile = profiles[hw_type]
    
    # 环境变量
    os.environ['OMP_NUM_THREADS'] = str(profile['threads'] or os.cpu_count())
    os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    # P-Core 锁定 (仅 Windows + 有核心列表)
    lock_result = (
        profile['cores'] and platform.system() == 'Windows' and
        (lambda: (
            psutil.Process(os.getpid()).cpu_affinity(profile['cores']),
            psutil.Process(os.getpid()).nice(psutil.HIGH_PRIORITY_CLASS),
            print(f"🔒 [System] Process locked to P-Cores: {profile['cores']}"),
            print(f"🚀 [System] Priority set to HIGH. E-Cores are banned."),
        ))()
    )
    
    return hw_type, profile

HW_TYPE, HW_PROFILE = bootstrap_hardware()
print(f"{time.strftime('%H:%M:%S')} | 🔥 MODE: FAST | P-Cores Only | Threads: {HW_PROFILE['threads']}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. JAX Imports (Post-Bootstrap)
# ═══════════════════════════════════════════════════════════════════════════════

import jax
import jax.numpy as jnp
import equinox as eqx

from cmsan import CMSAN, data
from cmsan.engine import fit_unified, evaluate_pure, save_checkpoint

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 📦 Immutable Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetMeta(NamedTuple):
    name: str
    subjects: int

class TrainConfig(NamedTuple):
    epochs: int
    batch_size: int
    lr: float
    d_model: int
    slices: int
    save_model: bool
    verbose: bool

class SessionResult(NamedTuple):
    dataset: str
    subject: int
    train_acc: float
    test_acc: float
    duration: float
    params: int

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 📊 Registry Tables (替代 if/else)
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS: Dict[str, DatasetMeta] = {
    'bcic':   DatasetMeta('bcic', 9),
    'bcicha': DatasetMeta('bcicha', 9),
    'mamem':  DatasetMeta('mamem', 11),
}

CONFIG_PRESETS: Dict[str, TrainConfig] = {
    'fast': TrainConfig(
        epochs=100, batch_size=64, lr=1e-3,
        d_model=32, slices=4, save_model=True, verbose=True
    ),
    'paper': TrainConfig(
        epochs=200, batch_size=128, lr=5e-4,
        d_model=64, slices=8, save_model=False, verbose=False
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# 4. 🧮 Pure Functional Primitives
# ═══════════════════════════════════════════════════════════════════════════════

def safe_call(fn: Callable, default=None):
    """安全调用 (替代 try/except)"""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ {e}")
            return default
    return wrapper

def maybe(value, fn: Callable, default=None):
    """Maybe monad (替代 if is not None)"""
    return fn(value) if value is not None else default

def count_params(model) -> int:
    """参数计数"""
    return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))

# ═══════════════════════════════════════════════════════════════════════════════
# 5. 🎯 Core Session Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_session(
    meta: DatasetMeta,
    subject: int,
    cfg: TrainConfig,
    logger
) -> Optional[SessionResult]:
    """
    单会话训练 (纯函数管道)
    """
    start = time.time()
    logger.info(f"📥 Loading {meta.name} Subject {subject}...")
    
    # 1. 数据加载 (safe_call 替代 try/except)
    raw_data = safe_call(data.load_unified, None)(meta.name, subject)
    
    # 2. 训练管道 (maybe 替代 if None)
    def train_pipeline(data_tuple):
        X, y = data_tuple
        
        # 数据准备
        key = jax.random.PRNGKey(42 + subject)
        k1, k2, k3 = jax.random.split(key, 3)
        
        N = X.shape[0]
        perm = jax.random.permutation(k1, N)
        X, y = X[perm], y[perm]
        
        split_idx = int(N * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        # 设备放置
        device = jax.devices()[0]
        X_train = jax.device_put(X_train, device)
        y_train = jax.device_put(y_train, device)
        X_test = jax.device_put(X_test, device)
        y_test = jax.device_put(y_test, device)
        
        # 模型创建
        K = len(np.unique(np.array(y_train)))
        model = CMSAN(k2, C=X_train.shape[1], T=X_train.shape[2], K=K, D=cfg.d_model, S=cfg.slices)
        params = count_params(model)
        
        cfg.verbose and logger.info(f"🧠 Model Params: {params:,}")
        cfg.verbose and logger.info(f"🚀 Compiling & Starting...")
        cfg.verbose and print(f"🚀 Whole-Graph Training: {cfg.epochs} Epochs | Batch: {cfg.batch_size}")
        cfg.verbose and print(f"{'Progress':<12} | {'Elapsed':<10} | {'Core (Type)':<11} | Loss")
        cfg.verbose and print("-" * 60)
        
        # 训练 (调用 engine.py)
        final_model, _ = fit_unified(
            model, X_train, y_train, k3,
            epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
            verbose=cfg.verbose
        )
        jax.block_until_ready(eqx.filter(final_model, eqx.is_array))
        
        # 评估
        train_acc = float(evaluate_pure(final_model, X_train, y_train))
        test_acc = float(evaluate_pure(final_model, X_test, y_test))
        duration = time.time() - start
        
        # 保存模型 (条件执行替代 if)
        cfg.save_model and save_checkpoint(
            final_model, f"checkpoints/{meta.name}_sub{subject:02d}.eqx"
        ) and logger.info(f"💾 Saved: checkpoints/{meta.name}_sub{subject:02d}.eqx")
        
        return SessionResult(meta.name, subject, train_acc, test_acc, duration, params)
    
    return maybe(raw_data, train_pipeline, None)

# ═══════════════════════════════════════════════════════════════════════════════
# 6. 📜 Mode Handlers (派发表替代 if/else)
# ═══════════════════════════════════════════════════════════════════════════════

def run_fast_mode(args, cfg: TrainConfig, logger):
    """FAST: 单被试极速训练"""
    meta = DATASETS[args.dataset]
    
    result = run_session(meta, args.sub, cfg, logger)
    
    # 结果输出
    maybe(result, lambda r: (
        logger.info("=" * 60),
        logger.info(f"✅ Time: {r.duration:.2f}s | Throughput: {int(288 * 0.8 * cfg.epochs / r.duration)} samples/s"),
        logger.info(f"🎓 Train Acc: {r.train_acc:.2%}"),
        logger.info(f"🏆 Test Acc:  {r.test_acc:.2%}"),
        logger.info("=" * 60),
    ))
    
    return result

def run_paper_mode(args, cfg: TrainConfig, logger):
    """PAPER: 全量基准测试"""
    start = time.time()
    
    # 目标数据集 (字典查表替代 if/else)
    targets = {
        True: DATASETS,
        False: {args.dataset: DATASETS[args.dataset]}
    }[args.dataset == 'all']
    
    logger.info(f"📜 PAPER MODE | Targets: {list(targets.keys())}")
    logger.info("=" * 60)
    
    # 生成所有 (dataset, subject) 任务
    tasks = [
        (meta, sub)
        for meta in targets.values()
        for sub in range(1, meta.subjects + 1)
    ]
    
    # map 执行 (替代 for 循环)
    results = tuple(filter(None, map(
        lambda task: run_session(task[0], task[1], cfg, logger),
        tasks
    )))
    
    # 汇总统计 (reduce 替代 for 循环)
    from collections import defaultdict
    grouped = reduce(
        lambda acc, r: (acc[r.dataset].append(r.test_acc), acc)[1],
        results,
        defaultdict(list)
    )
    
    # 打印报告
    total_time = time.time() - start
    logger.info("\n" + "=" * 70)
    logger.info(f"🏁 BENCHMARK REPORT | Time: {total_time/60:.1f} min")
    logger.info("=" * 70)
    logger.info(f"{'Dataset':<12} | {'N':<4} | {'Mean ± Std':<18} | {'Best':<8}")
    logger.info("-" * 50)
    
    # map 打印 (替代 for)
    list(map(
        lambda kv: logger.info(
            f"{kv[0]:<12} | {len(kv[1]):<4} | "
            f"{np.mean(kv[1]):.2%} ± {np.std(kv[1]):.2%} | {max(kv[1]):.2%}"
        ),
        grouped.items()
    ))
    logger.info("=" * 70)
    
    return results

# 模式派发表
MODE_HANDLERS: Dict[str, Callable] = {
    'fast': run_fast_mode,
    'paper': run_paper_mode,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 7. 🎮 Main Entry
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='fast', choices=['fast', 'paper'])
    parser.add_argument('--dataset', default='bcic')
    parser.add_argument('--sub', type=int, default=1)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S', force=True)
    logger = logging.getLogger()
    
    os.makedirs("checkpoints", exist_ok=True)
    
    # 配置 + 模式派发 (字典查表，零 if/else)
    cfg = CONFIG_PRESETS[args.mode]
    handler = MODE_HANDLERS[args.mode]
    
    return handler(args, cfg, logger)

__name__ == "__main__" and main()