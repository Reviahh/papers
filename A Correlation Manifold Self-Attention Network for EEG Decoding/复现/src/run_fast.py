# run_fast.py
"""
CPU 优化版实验脚本
目标: 1小时内跑完所有数据集
"""

import os
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true'

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
from functools import partial

# 限制 JAX 内存预分配
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'


# ═══════════════════════════════════════════════════════════════════════════
#                       配置
# ═══════════════════════════════════════════════════════════════════════════

CONFIG = {
    'bcic': {
        'folder': 'BCICIV_2a_mat',
        'C': 22, 'T': 438, 'K': 4, 
        'D': 20, 'S': 3,  # 论文设置
        'subjects': list(range(1, 10)),
    },
    'mamem': {
        'folder': 'MAMEM',
        'C': 8, 'T': 125, 'K': 5,
        'D': 15, 'S': 3,
        'subjects': list(range(1, 12)),
    },
    'bcicha': {
        'folder': 'BCIcha',
        'C': 56, 'T': 160, 'K': 2,
        'D': 14, 'S': 3,
        'subjects': [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26],
    },
}

# 训练超参数 (CPU 优化)
TRAIN_CONFIG = {
    'epochs': 50,        # 减少 epochs
    'batch_size': 32,    # 增大 batch (减少迭代)
    'lr': 1e-3,          # 稍大学习率 (配合少 epochs)
    'n_folds': 5,        # 5-fold (论文用10，但5足够)
}


# ═══════════════════════════════════════════════════════════════════════════
#                       数据加载 (同之前)
# ═══════════════════════════════════════════════════════════════════════════

from scipy.io import loadmat
from pathlib import Path
from sklearn.model_selection import KFold

def load_bcic(data_root, subject):
    folder = Path(data_root) / "BCICIV_2a_mat"
    train_data = loadmat(str(folder / f"BCIC_S{subject:02d}_T.mat"))
    test_data = loadmat(str(folder / f"BCIC_S{subject:02d}_E.mat"))
    
    # 处理键名不一致问题
    x_train = train_data.get('x_train', train_data.get('x_test'))
    y_train = train_data.get('y_train', train_data.get('y_test'))
    
    X = np.concatenate([x_train, test_data['x_test']], axis=0).astype(np.float32)
    y = np.concatenate([y_train.flatten(), test_data['y_test'].flatten()]).astype(np.int32)
    
    # 截取时间
    T_target = 438
    T_start = (X.shape[2] - T_target) // 2
    X = X[:, :, T_start:T_start + T_target]
    y = y - y.min()
    return X, y

def load_mamem(data_root, subject):
    file = Path(data_root) / "MAMEM" / f"U{subject:03d}.mat"
    data = loadmat(str(file))
    X = data['x_test'].astype(np.float32)
    y = data['y_test'].flatten().astype(np.int32)
    y = y - y.min()
    return X, y

def load_bcicha(data_root, subject):
    file = Path(data_root) / "BCIcha" / f"Data_S{subject:02d}_Sess.mat"
    data = loadmat(str(file))
    X = data['x_test'].astype(np.float32)
    y = data['y_test'].flatten().astype(np.int32)
    y = y - y.min()
    return X, y

LOADERS = {'bcic': load_bcic, 'mamem': load_mamem, 'bcicha': load_bcicha}

def make_folds(X, y, n_folds=5, seed=42):
    """预生成所有 fold 的索引"""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 标准化
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        
        folds.append((X_train, y_train, X_val, y_val))
    return folds


# ═══════════════════════════════════════════════════════════════════════════
#                       快速训练函数
# ═══════════════════════════════════════════════════════════════════════════

def train_one_fold(fold_data, cfg, seed=0):
    """训练单个 fold (会在子进程中运行)"""
    # 延迟导入，避免多进程问题
    import jax
    import jax.numpy as jnp
    from jax import random
    import equinox as eqx
    import optax
    
    X_train, y_train, X_val, y_val = fold_data
    
    # 转换为 JAX 数组
    X_train = jnp.array(X_train)
    y_train = jnp.array(y_train)
    X_val = jnp.array(X_val)
    y_val = jnp.array(y_val)
    
    # 导入模型
    from cmsan import CMSAN
    from cmsan.train import compute_loss, cross_entropy
    
    # 创建模型
    key = random.key(seed)
    model = CMSAN(
        key, 
        C=cfg['C'], T=cfg['T'], D=cfg['D'], S=cfg['S'], K=cfg['K']
    )
    
    # 优化器
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(TRAIN_CONFIG['lr'], weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # JIT 编译的训练步骤
    @eqx.filter_jit
    def train_step(model, opt_state, X_batch, y_batch):
        loss, grads = eqx.filter_value_and_grad(compute_loss)(model, X_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    @eqx.filter_jit
    def eval_acc(model, X, y):
        from cmsan import batch_predict
        preds = batch_predict(model, X)
        return jnp.mean(preds == y)
    
    # 训练循环
    batch_size = TRAIN_CONFIG['batch_size']
    n_samples = X_train.shape[0]
    n_batches = max(1, n_samples // batch_size)
    
    for epoch in range(TRAIN_CONFIG['epochs']):
        # 打乱
        key, subkey = random.split(key)
        perm = random.permutation(subkey, n_samples)
        X_shuf = X_train[perm]
        y_shuf = y_train[perm]
        
        # 批次训练
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            model, opt_state, loss = train_step(
                model, opt_state, X_shuf[start:end], y_shuf[start:end]
            )
    
    # 评估
    val_acc = float(eval_acc(model, X_val, y_val))
    return val_acc


def train_subject(args):
    """训练单个被试的所有 folds"""
    data_root, dataset, subject = args
    cfg = CONFIG[dataset]
    
    # 加载数据
    X, y = LOADERS[dataset](data_root, subject)
    folds = make_folds(X, y, TRAIN_CONFIG['n_folds'])
    
    # 训练每个 fold
    fold_accs = []
    for fold_idx, fold_data in enumerate(folds):
        acc = train_one_fold(fold_data, cfg, seed=fold_idx)
        fold_accs.append(acc)
    
    mean_acc = np.mean(fold_accs)
    return subject, mean_acc, fold_accs


# ═══════════════════════════════════════════════════════════════════════════
#                       主程序
# ═══════════════════════════════════════════════════════════════════════════

def run_dataset_parallel(data_root, dataset, n_workers=None):
    """并行运行数据集"""
    cfg = CONFIG[dataset]
    subjects = cfg['subjects']
    
    print(f"\n{'='*60}")
    print(f"📊 {dataset.upper()}: {len(subjects)} subjects × {TRAIN_CONFIG['n_folds']} folds")
    print(f"   配置: C={cfg['C']}, T={cfg['T']}, D={cfg['D']}, S={cfg['S']}, K={cfg['K']}")
    print('='*60)
    
    # 准备参数
    args_list = [(data_root, dataset, s) for s in subjects]
    
    results = []
    start = time.time()
    
    # 串行执行 (多进程在某些环境下可能有问题)
    for i, args in enumerate(args_list):
        subject, mean_acc, fold_accs = train_subject(args)
        results.append((subject, mean_acc, fold_accs))
        elapsed = time.time() - start
        eta = elapsed / (i + 1) * (len(subjects) - i - 1)
        print(f"  S{subject:02d}: {mean_acc*100:.2f}% | 进度 {i+1}/{len(subjects)} | ETA {eta/60:.1f}min")
    
    # 汇总
    all_accs = [r[1] for r in results]
    mean = np.mean(all_accs) * 100
    std = np.std(all_accs) * 100
    
    print(f"\n🎯 {dataset.upper()} 结果: {mean:.2f} ± {std:.2f}%")
    print(f"   耗时: {(time.time()-start)/60:.1f} 分钟")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data')
    parser.add_argument('--dataset', default='all', choices=['bcic', 'mamem', 'bcicha', 'all'])
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 CMSAN CPU 快速实验")
    print(f"   Epochs: {TRAIN_CONFIG['epochs']}, Folds: {TRAIN_CONFIG['n_folds']}")
    print(f"   Batch: {TRAIN_CONFIG['batch_size']}, LR: {TRAIN_CONFIG['lr']}")
    print("="*60)
    
    datasets = ['bcic', 'mamem', 'bcicha'] if args.dataset == 'all' else [args.dataset]
    
    all_results = {}
    total_start = time.time()
    
    for ds in datasets:
        all_results[ds] = run_dataset_parallel(args.data, ds)
    
    # 最终汇总
    print("\n" + "="*60)
    print("📋 最终结果对比")
    print("="*60)
    print(f"{'数据集':<10} {'你的结果':<18} {'论文 (CorAtt-OLM)':<18}")
    print("-"*60)
    
    paper = {'bcic': '75.01±2.78', 'mamem': '67.39±3.22', 'bcicha': '78.78±3.40'}
    
    for ds in datasets:
        accs = [r[1] for r in all_results[ds]]
        mean, std = np.mean(accs)*100, np.std(accs)*100
        print(f"{ds:<10} {mean:.2f}±{std:.2f}%        {paper[ds]}%")
    
    print(f"\n⏱️  总耗时: {(time.time()-total_start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()