# src/run_final.py
"""
CMSAN 极速复现脚本 (i5-12500H 优化版)
目标: 单进程+多线程计算，1小时内跑完所有 Benchmarks
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from sklearn.model_selection import KFold

# ─────────────────────────────────────────────────────────────────────────────
# 1. 性能环境配置 (必须在 import jax 之前 !)
# ─────────────────────────────────────────────────────────────────────────────

# 【核心提速】: 允许 Eigen 使用多线程 (利用你的 12核 16线程)
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=true' 

# 关闭内存预分配，防止 Windows 下占用过高
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# 强制使用 CPU
os.environ['JAX_PLATFORMS'] = 'cpu'

# ─────────────────────────────────────────────────────────────────────────────
# 2. JAX & 库导入
# ─────────────────────────────────────────────────────────────────────────────

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler("experiment_results.log", mode='w'), # 结果存文件
        logging.StreamHandler(sys.stdout)                        # 同时打印到屏幕
    ]
)
logger = logging.getLogger()

logger.info("正在初始化 JAX 环境 (利用全核加速)...")

import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax

# 导入你的核心库 (确保 src 在 PYTHONPATH 中，或者此脚本在 src 下运行)
try:
    from cmsan import CMSAN, batch_predict
    from cmsan.train import compute_loss
except ImportError:
    logger.error("❌ 找不到 cmsan 库。请确保你在 src 目录下运行此脚本: python run_final.py")
    sys.exit(1)

logger.info(f"✅ JAX 设备: {jax.devices()[0].device_kind} (核心数已释放)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. 实验配置 (Configuration)
# ─────────────────────────────────────────────────────────────────────────────

# 数据集元数据
DATASET_META = {
    'bcic':   {'C': 22, 'T': 438, 'K': 4, 'D': 20, 'S': 3, 'subjects': range(1, 10)},
    'mamem':  {'C': 8,  'T': 125, 'K': 5, 'D': 15, 'S': 3, 'subjects': range(1, 12)},
    'bcicha': {'C': 56, 'T': 160, 'K': 2, 'D': 14, 'S': 3, 
               'subjects': [2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26]},
}

# 训练超参数 (针对 1 小时完赛调整)
TRAIN_CONFIG = {
    'epochs': 30,       # 足够观察收敛趋势
    'batch_size': 32,   
    'lr': 1e-3,
    'n_folds': 3        # 3折交叉验证 (平衡速度与可信度)
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. 数据加载模块 (适配你的目录结构)
# ─────────────────────────────────────────────────────────────────────────────

def get_data_path(base_dir, mode, dataset_name):
    """根据模式选择数据文件夹"""
    root = Path(base_dir) / "data"
    if mode == 'author':
        target = root / "author_original"
        # 如果作者原文件夹没分那么细，根据你的实际情况修改这里
        # 这里假设你把所有 .mat 按数据集分文件夹放进了 author_original
        # 或者兼容你现在的结构：
        return root # 回退到 data/ 根目录查找
    else:
        return root / "my_custom"

def load_data(data_root, dataset, subject):
    """统一数据加载入口"""
    path = Path(data_root)
    
    try:
        if dataset == 'bcic':
            # 适配 BCICIV_2a_mat 文件夹
            folder = path / "BCICIV_2a_mat"
            if not folder.exists(): folder = path # 尝试直接在 root 找
            
            t = loadmat(str(folder / f"BCIC_S{subject:02d}_T.mat"))
            e = loadmat(str(folder / f"BCIC_S{subject:02d}_E.mat"))
            
            # 拼接 Train 和 Test
            X = np.concatenate([t.get('x_train', t.get('x_test')), e['x_test']], axis=0)
            y = np.concatenate([t.get('y_train', t.get('y_test')).flatten(), e['y_test'].flatten()])
            
            # 裁剪时间窗 (避免内存爆炸)
            T_target = DATASET_META['bcic']['T']
            T_start = (X.shape[2] - T_target) // 2
            X = X[:, :, T_start:T_start+T_target]
            
        elif dataset == 'mamem':
            folder = path / "MAMEM"
            d = loadmat(str(folder / f"U{subject:03d}.mat"))
            X, y = d['x_test'], d['y_test'].flatten()
            
        elif dataset == 'bcicha':
            folder = path / "BCIcha"
            d = loadmat(str(folder / f"Data_S{subject:02d}_Sess.mat"))
            X, y = d['x_test'], d['y_test'].flatten()
            
        return X.astype(np.float32), (y - y.min()).astype(np.int32)
    
    except FileNotFoundError:
        logger.error(f"❌ 数据文件丢失: {dataset} Subject {subject}")
        logger.error(f"   请检查路径: {path}")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. 训练核心 (优化编译版)
# ─────────────────────────────────────────────────────────────────────────────

def make_train_step(optimizer):
    """工厂模式：生成 JIT 编译的训练步"""
    @eqx.filter_jit
    def train_step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    return train_step

@eqx.filter_jit
def evaluate(model, X, y):
    preds = batch_predict(model, X)
    return jnp.mean(preds == y)

def run_subject_cv(data_root, dataset, subject, cfg):
    """单个被试的交叉验证流程"""
    X, y = load_data(data_root, dataset, subject)
    
    # K-Fold 设置
    kf = KFold(n_splits=TRAIN_CONFIG['n_folds'], shuffle=True, random_state=42)
    accs = []

    # 准备 JAX 随机 
    # (注意：在循环外生成 key 避免每次重新初始化)
    key = random.PRNGKey(subject * 999) 

    # 循环 Folds
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        # 1. 数据准备 (Numpy -> JAX Array)
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        # 标准化 (z-score)
        mean = X_tr.mean(axis=(0, 2), keepdims=True)
        std  = X_tr.std(axis=(0, 2), keepdims=True) + 1e-8
        X_tr = jnp.array((X_tr - mean) / std)
        X_val = jnp.array((X_val - mean) / std)
        y_tr, y_val = jnp.array(y_tr), jnp.array(y_val)

        # 2. 模型初始化
        key, m_key = random.split(key)
        model = CMSAN(m_key, C=cfg['C'], T=cfg['T'], D=cfg['D'], S=cfg['S'], K=cfg['K'])
        
        optimizer = optax.adamw(TRAIN_CONFIG['lr'])
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        train_step = make_train_step(optimizer)

        # 3. 训练循环 (最耗时部分)
        n_samples = X_tr.shape[0]
        batch_size = TRAIN_CONFIG['batch_size']
        
        for epoch in range(TRAIN_CONFIG['epochs']):
            # Shuffle
            key, p_key = random.split(key)
            perm = random.permutation(p_key, n_samples)
            X_shuf, y_shuf = X_tr[perm], y_tr[perm]
            
            # Batch Loop
            for i in range(0, n_samples, batch_size):
                end = min(i + batch_size, n_samples)
                model, opt_state, _ = train_step(model, opt_state, X_shuf[i:end], y_shuf[i:end])
        
        # 4. 评估
        acc = float(evaluate(model, X_val, y_val))
        accs.append(acc)
    
    return np.mean(accs)

# ─────────────────────────────────────────────────────────────────────────────
# 6. 主控逻辑
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CMSAN 快速复现脚本")
    parser.add_argument('--dataset', default='all', choices=['bcic', 'mamem', 'bcicha', 'all'])
    parser.add_argument('--mode', default='author', choices=['author', 'my'], help="区分数据来源")
    parser.add_argument('--data_dir', default='data', help="数据根目录") # 默认指向 ./data
    args = parser.parse_args()

    # 确定要跑的任务
    target_datasets = DATASET_META.keys() if args.dataset == 'all' else [args.dataset]
    
    # 打印横幅
    logger.info("="*60)
    logger.info(f"🚀 CMSAN 极速实验 | 模式: {args.mode.upper()}")
    logger.info(f"⚙️  设置: Epochs={TRAIN_CONFIG['epochs']} | Folds={TRAIN_CONFIG['n_folds']}")
    logger.info("="*60)

    total_start = time.time()
    final_report = {}

    for ds_name in target_datasets:
        cfg = DATASET_META[ds_name]
        subs = cfg['subjects']
        
        logger.info(f"\n📊 开始数据集: {ds_name.upper()} (N={len(subs)})")
        logger.info("-" * 40)
        
        ds_accs = []
        ds_start = time.time()
        
        # 逐个跑 Subject (单进程，但内部 JAX 满载多线程)
        for i, sub in enumerate(subs):
            t0 = time.time()
            
            # 核心运行
            acc = run_subject_cv(args.data_dir, ds_name, sub, cfg)
            ds_accs.append(acc)
            
            # 进度条估算
            elapsed = time.time() - ds_start
            avg_time = elapsed / (i + 1)
            remain = avg_time * (len(subs) - i - 1)
            
            logger.info(f"  Subject {sub:02d}: {acc*100:05.2f}% | 耗时 {time.time()-t0:3.0f}s | 剩余约 {remain/60:.1f}m")

        # 数据集汇总
        mean, std = np.mean(ds_accs)*100, np.std(ds_accs)*100
        final_report[ds_name] = f"{mean:.2f} ± {std:.2f}%"
        logger.info(f"🎯 {ds_name.upper()} 完成: {mean:.2f}% (耗时 {(time.time()-ds_start)/60:.1f}m)")

    # 最终大汇总
    logger.info("\n" + "="*60)
    logger.info("📋 最终实验报告")
    logger.info("="*60)
    for k, v in final_report.items():
        logger.info(f"  {k.upper():<10} : {v}")
    logger.info(f"\n⏱️  总耗时: {(time.time()-total_start)/60:.1f} 分钟")

if __name__ == '__main__':
    main()