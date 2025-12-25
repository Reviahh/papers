"""
CMSAN Main
"""
import os
import platform
import argparse
import time
import logging
from functools import reduce
from collections import defaultdict
import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# 0. P-Core 锁定 (import jax 前)
# ═══════════════════════════════════════════════════════════════════════════════

def lock_p_cores():
    """i5-12500H: 锁定 P-Cores 0-7"""
    platform.system() == 'Windows' and (lambda: (
        __import__('psutil').Process(os.getpid()).cpu_affinity(list(range(8))),
        __import__('psutil').Process(os.getpid()).nice(__import__('psutil').HIGH_PRIORITY_CLASS),
        print(f"🔒 [System] Process locked to P-Cores: [0-7]"),
        print(f"🚀 [System] Priority set to HIGH."),
    ))()

lock_p_cores()
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true'
print(f"{time.strftime('%H:%M:%S')} | 🔥 MODE: FAST | P-Cores Only | Threads: 8")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Imports
# ═══════════════════════════════════════════════════════════════════════════════

import jax
import jax.numpy as jnp
import equinox as eqx

from cmsan import CMSAN, data
from cmsan.engine import fit_unified, evaluate_pure, save_checkpoint
from configs.presets import get_config, DATASETS

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Session Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_session(ds, subject, cfg, logger):
    start = time.time()
    logger.info(f"📥 Loading {ds['name']} Subject {subject}...")
    
    try:
        X, y = data.load_unified(ds['name'], subject)
    except Exception as e:
        logger.warning(f"⚠️ {e}")
        return None
    
    key = jax.random.PRNGKey(42 + subject)
    k1, k2, k3 = jax.random.split(key, 3)
    
    N = X.shape[0]
    perm = jax.random.permutation(k1, N)
    X, y = X[perm], y[perm]
    
    split = int(N * 0.8)
    X_tr, y_tr, X_te, y_te = X[:split], y[:split], X[split:], y[split:]
    
    device = jax.devices()[0]
    X_tr, y_tr = jax.device_put(X_tr, device), jax.device_put(y_tr, device)
    X_te, y_te = jax.device_put(X_te, device), jax.device_put(y_te, device)
    
    K = len(np.unique(np.array(y_tr)))
    model = CMSAN(k2, C=X_tr.shape[1], T=X_tr.shape[2], K=K, D=cfg['d_model'], S=cfg['slices'])
    params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    
    cfg['verbose'] and logger.info(f"🧠 Model Params: {params:,}")
    cfg['verbose'] and logger.info(f"🚀 Compiling & Starting...")
    cfg['verbose'] and print(f"🚀 Whole-Graph Training: {cfg['epochs']} Epochs | Batch: {cfg['batch_size']}")
    cfg['verbose'] and print(f"{'Progress':<12} | {'Elapsed':<10} | {'Core (Type)':<11} | Loss")
    cfg['verbose'] and print("-" * 60)
    
    final, _ = fit_unified(model, X_tr, y_tr, k3, cfg['epochs'], cfg['batch_size'], cfg['lr'], cfg['verbose'])
    jax.block_until_ready(eqx.filter(final, eqx.is_array))
    
    tr_acc = float(evaluate_pure(final, X_tr, y_tr))
    te_acc = float(evaluate_pure(final, X_te, y_te))
    dur = time.time() - start
    
    cfg['save_model'] and save_checkpoint(final, f"checkpoints/{ds['name']}_sub{subject:02d}.eqx")
    
    return {'dataset': ds['name'], 'subject': subject, 'train_acc': tr_acc, 'test_acc': te_acc, 'duration': dur, 'params': params}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Mode Handlers
# ═══════════════════════════════════════════════════════════════════════════════

def run_fast(args, cfg, logger):
    ds = DATASETS[args.dataset]
    r = run_session(ds, args.sub, cfg, logger)
    r and (
        logger.info("=" * 60),
        logger.info(f"✅ Time: {r['duration']:.2f}s | Throughput: {int(ds['subjects'].__len__() * 288 * 0.8 * cfg['epochs'] / r['duration'] / ds['subjects'].__len__())} samples/s"),
        logger.info(f"🎓 Train Acc: {r['train_acc']:.2%}"),
        logger.info(f"🏆 Test Acc:  {r['test_acc']:.2%}"),
        logger.info("=" * 60),
    )
    return r

def run_paper(args, cfg, logger):
    start = time.time()
    targets = DATASETS if args.dataset == 'all' else {args.dataset: DATASETS[args.dataset]}
    
    logger.info(f"📜 PAPER MODE | Targets: {list(targets.keys())}")
    logger.info("=" * 60)
    
    # 从 subjects 列表获取被试编号
    tasks = [(ds, sub) for ds in targets.values() for sub in ds['subjects']]
    results = [r for t in tasks if (r := run_session(t[0], t[1], cfg, logger))]
    
    grouped = reduce(lambda a, r: (a[r['dataset']].append(r['test_acc']), a)[1], results, defaultdict(list))
    
    logger.info("\n" + "=" * 70)
    logger.info(f"🏁 BENCHMARK REPORT | Time: {(time.time()-start)/60:.1f} min")
    logger.info("=" * 70)
    logger.info(f"{'Dataset':<12} | {'N':<4} | {'Mean ± Std':<18} | {'Best':<8}")
    logger.info("-" * 50)
    [logger.info(f"{k:<12} | {len(v):<4} | {np.mean(v):.2%} ± {np.std(v):.2%} | {max(v):.2%}") for k, v in grouped.items()]
    logger.info("=" * 70)
    
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Entry
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='fast', choices=['fast', 'paper'])
    parser.add_argument('--dataset', default='bcic')
    parser.add_argument('--sub', type=int, default=1)
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S', force=True)
    os.makedirs("checkpoints", exist_ok=True)
    
    cfg = get_config(args.mode)
    return {'fast': run_fast, 'paper': run_paper}[args.mode](args, cfg, logging.getLogger())

__name__ == "__main__" and main()