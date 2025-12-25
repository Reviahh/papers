"""
维度二: 我自己的复现 (My Validation)
═══════════════════════════════════════════════════════════════════════════════

完整实验: 所有被试 × 10-fold CV
使用自己下载的数据复现论文 Table 1 结果

使用:
    python scripts/my_reproduction.py --data data/my_custom --dataset bcic
"""

import sys
from pathlib import Path

# 添加父目录到路径以导入 cmsan
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from tqdm import tqdm

from cmsan import CMSAN, fit, evaluate
from scripts.data_utils.load_data import load_dataset, make_kfold, get_config, DATASET_CONFIG


def run_subject(data_root, dataset, subject, n_folds=10, epochs=100, lr=5e-4):
    """单个被试的 10-fold CV"""
    
    X, y = load_dataset(data_root, dataset, subject)
    config = get_config(dataset)
    
    fold_accs = []
    
    for fold in range(n_folds):
        # 准备数据
        X_train, y_train, X_val, y_val = make_kfold(X, y, n_folds, fold)
        
        # 创建模型
        key = random.key(fold)
        model = CMSAN(
            key,
            C=config['C'],
            T=config['T'],
            D=config['D'],
            S=config['S'],
            K=config['K'],
        )
        
        # 训练 (静默模式)
        trained = fit(
            model,
            (jnp.array(X_train), jnp.array(y_train)),
            (jnp.array(X_val), jnp.array(y_val)),
            epochs=epochs,
            batch_size=16,
            lr=lr,
            verbose=False,
        )
        
        # 评估
        acc = float(evaluate(trained, jnp.array(X_val), jnp.array(y_val)))
        fold_accs.append(acc)
    
    return np.array(fold_accs)


def run_dataset(data_root, dataset, n_folds=10, epochs=100):
    """运行整个数据集的所有被试"""
    
    config = get_config(dataset)
    subjects = config['subjects']
    
    print(f"\n{'='*60}")
    print(f"数据集: {dataset.upper()}")
    print(f"被试数: {len(subjects)}, Folds: {n_folds}")
    print(f"配置: C={config['C']}, T={config['T']}, D={config['D']}, S={config['S']}, K={config['K']}")
    print('='*60)
    
    all_results = []
    
    for subject in tqdm(subjects, desc=dataset):
        fold_accs = run_subject(data_root, dataset, subject, n_folds, epochs)
        mean_acc = fold_accs.mean()
        std_acc = fold_accs.std()
        all_results.append(fold_accs)
        
        print(f"  Subject {subject:2d}: {mean_acc*100:.2f} ± {std_acc*100:.2f}%")
    
    # 汇总
    all_results = np.array(all_results)
    overall_mean = all_results.mean() * 100
    overall_std = all_results.mean(axis=1).std() * 100  # 被试间标准差
    
    print(f"\n📊 {dataset.upper()} 总体结果: {overall_mean:.2f} ± {overall_std:.2f}%")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data', help='数据目录')
    parser.add_argument('--dataset', default='bcic', choices=['bcic', 'mamem', 'bcicha', 'all'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--folds', type=int, default=10)
    args = parser.parse_args()
    
    if args.dataset == 'all':
        datasets = ['bcic', 'mamem', 'bcicha']
    else:
        datasets = [args.dataset]
    
    results = {}
    for ds in datasets:
        results[ds] = run_dataset(args.data, ds, args.folds, args.epochs)
    
    # 最终汇总
    print("\n" + "="*60)
    print("📋 最终结果汇总 (对比论文 Table 1)")
    print("="*60)
    print(f"{'数据集':<10} {'你的结果':<20} {'论文结果':<20}")
    print("-"*60)
    
    paper_results = {
        'bcic': '75.01 ± 2.78',
        'mamem': '67.39 ± 3.22', 
        'bcicha': '78.78 ± 3.40',
    }
    
    for ds in datasets:
        mean = results[ds].mean() * 100
        std = results[ds].mean(axis=1).std() * 100
        print(f"{ds:<10} {mean:.2f} ± {std:.2f}%       {paper_results[ds]}%")