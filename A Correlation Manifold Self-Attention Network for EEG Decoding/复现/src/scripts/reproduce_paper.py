"""
维度一: 作者原文实验复现 (Official Benchmark)
═══════════════════════════════════════════════════════════════════════════════

目的: 使用作者提供的数据和参数，复现论文中的实验结果。
作为"定海神针"，证明代码实现与作者逻辑一致。

使用:
    python scripts/reproduce_paper.py --data data/author_original/eeg_data.npz
"""

import sys
from pathlib import Path

# 添加父目录到路径以导入 cmsan
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import argparse

from cmsan import CMSAN, fit, evaluate

# ═══════════════════════════════════════════════════════════════════════════
#                       论文固定参数 (不可修改)
# ═══════════════════════════════════════════════════════════════════════════

PAPER_CONFIG = {
    'C': 22,        # 通道数
    'T': 438,       # 时间点
    'D': 20,        # 特征维度
    'S': 3,         # 流形段数
    'K': 4,         # 类别数
    'epochs': 100,  # 训练轮数
    'batch_size': 16,
    'lr': 5e-4,
    'seed': 42,
}


# ═══════════════════════════════════════════════════════════════════════════
#                       主程序
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='复现论文实验 (使用作者数据)')
    parser.add_argument('--data', type=str, required=True, 
                       help='作者提供的数据路径 (例如: data/author_original/eeg_data.npz)')
    parser.add_argument('--save', type=str, default='checkpoints/paper_model.pkl', 
                       help='模型保存路径')
    args = parser.parse_args()
    
    print("=" * 60)
    print("维度一: 作者原文实验复现")
    print("=" * 60)
    print("\n使用论文固定参数:")
    for key, value in PAPER_CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # 加载作者提供的数据
    try:
        data = np.load(args.data)
        X_train = jnp.array(data['x_train'])
        y_train = jnp.array(data['y_train'])
        X_val = jnp.array(data['x_val'])
        y_val = jnp.array(data['y_val'])
        print(f"✓ 加载数据: {args.data}")
        print(f"  训练集: {X_train.shape}")
        print(f"  验证集: {X_val.shape}")
    except FileNotFoundError:
        print(f"✗ 数据文件不存在: {args.data}")
        print("\n请先将作者提供的数据放到 data/author_original/ 目录")
        return
    
    # 验证数据维度与论文配置一致
    assert X_train.shape[1] == PAPER_CONFIG['C'], f"通道数不匹配: {X_train.shape[1]} != {PAPER_CONFIG['C']}"
    assert X_train.shape[2] == PAPER_CONFIG['T'], f"时间点不匹配: {X_train.shape[2]} != {PAPER_CONFIG['T']}"
    assert int(y_train.max()) + 1 == PAPER_CONFIG['K'], f"类别数不匹配: {int(y_train.max()) + 1} != {PAPER_CONFIG['K']}"
    
    # 创建模型 (使用固定种子保证可重复性)
    key = random.key(PAPER_CONFIG['seed'])
    model = CMSAN(
        key,
        C=PAPER_CONFIG['C'],
        T=PAPER_CONFIG['T'],
        D=PAPER_CONFIG['D'],
        S=PAPER_CONFIG['S'],
        K=PAPER_CONFIG['K'],
    )
    
    print(f"\n✓ 创建模型 (seed={PAPER_CONFIG['seed']})")
    
    # 训练
    print(f"\n开始训练 (epochs={PAPER_CONFIG['epochs']})...")
    print("-" * 60)
    
    key, subkey = random.split(key)
    trained_model = fit(
        model,
        (X_train, y_train),
        (X_val, y_val),
        epochs=PAPER_CONFIG['epochs'],
        batch_size=PAPER_CONFIG['batch_size'],
        lr=PAPER_CONFIG['lr'],
        key=subkey,
    )
    
    print("-" * 60)
    
    # 最终评估
    final_train = evaluate(trained_model, X_train, y_train)
    final_val = evaluate(trained_model, X_val, y_val)
    print(f"\n最终准确率:")
    print(f"  训练集: {float(final_train):.2%}")
    print(f"  验证集: {float(final_val):.2%}")
    
    # 保存模型
    from cmsan import save_model
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(trained_model, str(save_path))
    print(f"\n✓ 模型已保存: {save_path}")
    
    print("\n" + "=" * 60)
    print("✓ 作者原文实验复现完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
