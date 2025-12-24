"""
CorAtt 主程序

使用:
    python main.py              # 使用假数据测试
    python main.py --data path  # 使用真实数据
"""

import jax
import jax.numpy as jnp
from jax import random
import argparse
import time

from model import init, forward
from train import train, evaluate, save, load


# ═══════════════════════════════════════════════════════════════════════════
#                              配置
# ═══════════════════════════════════════════════════════════════════════════

# BCIC-IV-2a 配置
CFG_BCIC = {
    'C': 22,      # EEG 通道
    'T': 438,     # 时间点
    'D': 20,      # 特征维度
    'S': 3,       # 分段数
    'K': 4,       # 类别数
    'kernel': 25, # 卷积核大小
}

# 轻量配置 (CPU 调试用)
CFG_LIGHT = {
    'C': 8,
    'T': 128,
    'D': 10,
    'S': 2,
    'K': 4,
    'kernel': 11,
}


# ═══════════════════════════════════════════════════════════════════════════
#                              假数据
# ═══════════════════════════════════════════════════════════════════════════

def make_fake_data(key, cfg, n_train=100, n_val=20):
    """生成假数据用于测试"""
    k1, k2, k3, k4 = random.split(key, 4)
    
    xs_train = random.normal(k1, (n_train, cfg['C'], cfg['T']))
    ys_train = random.randint(k2, (n_train,), 0, cfg['K'])
    
    xs_val = random.normal(k3, (n_val, cfg['C'], cfg['T']))
    ys_val = random.randint(k4, (n_val,), 0, cfg['K'])
    
    return (xs_train, ys_train), (xs_val, ys_val)


# ═══════════════════════════════════════════════════════════════════════════
#                              主程序
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='CorAtt Training')
    parser.add_argument('--data', type=str, default=None, help='数据路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch', type=int, default=8, help='批大小 (CPU建议8)')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--light', action='store_true', help='使用轻量配置')
    parser.add_argument('--save', type=str, default='model.pkl', help='保存路径')
    args = parser.parse_args()
    
    # 配置
    cfg = CFG_LIGHT if args.light else CFG_BCIC
    
    print("=" * 60)
    print("CorAtt - 相关矩阵流形注意力网络")
    print("=" * 60)
    print(f"配置: {cfg}")
    print(f"设备: {jax.devices()[0]}")
    print()
    
    # 数据
    key = random.key(42)
    
    if args.data:
        # 加载真实数据
        import numpy as np
        data = np.load(args.data)
        train_data = (jnp.array(data['x_train']), jnp.array(data['y_train']))
        val_data = (jnp.array(data['x_val']), jnp.array(data['y_val']))
        print(f"加载数据: {args.data}")
    else:
        # 使用假数据
        train_data, val_data = make_fake_data(key, cfg)
        print("使用假数据 (测试模式)")
    
    print(f"训练集: {train_data[0].shape}")
    print(f"验证集: {val_data[0].shape}")
    print()
    
    # 初始化模型
    key, subkey = random.split(key)
    θ = init(subkey, cfg)
    
    n_params = sum(x.size for x in jax.tree.leaves(θ))
    print(f"参数量: {n_params:,}")
    
    # 测试前向传播
    print("\n测试前向传播...")
    t0 = time.time()
    x_test = train_data[0][0]
    logits = forward(θ, x_test, cfg)
    print(f"输出形状: {logits.shape}")
    print(f"首次运行: {time.time()-t0:.2f}s (含JIT编译)")
    
    # JIT 后再测一次
    t0 = time.time()
    for _ in range(10):
        _ = forward(θ, x_test, cfg)
    print(f"JIT后平均: {(time.time()-t0)/10*1000:.1f}ms/sample")
    print()
    
    # 训练
    print("开始训练...")
    print("-" * 60)
    
    t_start = time.time()
    θ = train(
        θ, train_data, val_data, cfg,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        save_path=args.save
    )
    
    print("-" * 60)
    print(f"训练完成! 耗时: {time.time()-t_start:.1f}s")
    
    # 最终评估
    final_train = evaluate(θ, train_data[0], train_data[1], cfg)
    final_val = evaluate(θ, val_data[0], val_data[1], cfg)
    print(f"最终准确率 - 训练: {final_train:.2%} | 验证: {final_val:.2%}")
    
    # 保存
    save(θ, args.save)
    print(f"模型已保存: {args.save}")


if __name__ == "__main__":
    main()