"""
CMSAN 主程序 (Equinox)
═══════════════════════════════════════════════════════════════════════════════

Correlation Manifold Self-Attention Network for EEG Decoding
使用 Equinox + Optax 实现完全函数式训练

使用:
    python main.py              # 使用假数据测试
    python main.py --data path  # 使用真实数据
    python main.py --preset bcic  # 使用预设配置
"""

import jax
import jax.numpy as jnp
from jax import random
import argparse
import time

import equinox as eqx

# 使用 CMSAN API
from cmsan import CMSAN, fit, evaluate, batch_forward, save_model
from cmsan.model import PRESETS, create_from_preset


# ═══════════════════════════════════════════════════════════════════════════
#                              假数据
# ═══════════════════════════════════════════════════════════════════════════

def make_fake_data(key, C, T, K, n_train=100, n_val=20):
    """生成假数据用于测试"""
    k1, k2, k3, k4 = random.split(key, 4)
    
    xs_train = random.normal(k1, (n_train, C, T))
    ys_train = random.randint(k2, (n_train,), 0, K)
    
    xs_val = random.normal(k3, (n_val, C, T))
    ys_val = random.randint(k4, (n_val,), 0, K)
    
    return (xs_train, ys_train), (xs_val, ys_val)


# ═══════════════════════════════════════════════════════════════════════════
#                              主程序
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='CMSAN Training')
    parser.add_argument('--data', type=str, default=None, help='数据路径')
    parser.add_argument('--preset', type=str, default='light', 
                       choices=list(PRESETS.keys()), help='预设配置')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch', type=int, default=8, help='批大小')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--save', type=str, default='model.pkl', help='保存路径')
    args = parser.parse_args()
    
    print("═" * 60)
    print("CMSAN - Correlation Manifold Self-Attention Network")
    print("═" * 60)
    
    # 创建模型 (Equinox Module，参数内嵌)
    key = random.key(42)
    key, subkey = random.split(key)
    model = create_from_preset(subkey, args.preset)
    
    print(f"配置: {args.preset}")
    print(f"  C={model.C}, T={model.T}, D={model.D}, S={model.S}, K={model.K}")
    print(f"设备: {jax.devices()[0]}")
    print()
    
    # 统计参数量 (Equinox 方式)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"参数量: {n_params:,}")
    
    # 数据
    if args.data:
        import numpy as np
        data = np.load(args.data)
        train_data = (jnp.array(data['x_train']), jnp.array(data['y_train']))
        val_data = (jnp.array(data['x_val']), jnp.array(data['y_val']))
        print(f"加载数据: {args.data}")
    else:
        key, subkey = random.split(key)
        train_data, val_data = make_fake_data(subkey, model.C, model.T, model.K)
        print("使用假数据 (测试模式)")
    
    print(f"训练集: {train_data[0].shape}")
    print(f"验证集: {val_data[0].shape}")
    print()
    
    # 测试前向传播 (Equinox: 直接调用模型)
    print("测试前向传播...")
    t0 = time.time()
    x_test = train_data[0][0]
    logits = model(x_test)  # Equinox 风格
    print(f"输出形状: {logits.shape}")
    print(f"首次运行: {time.time()-t0:.2f}s (含 JIT 编译)")
    
    # 使用 eqx.filter_jit 测试
    jit_model = eqx.filter_jit(model)
    t0 = time.time()
    # 使用 lax.fori_loop 替代 for 循环
    def bench_body(i, acc):
        return acc + jit_model(x_test).sum()
    _ = jax.lax.fori_loop(0, 10, bench_body, 0.0)
    print(f"JIT 后平均: {(time.time()-t0)/10*1000:.1f}ms/sample")
    print()
    
    # 训练 (Equinox + Optax，完全函数式)
    print("开始训练...")
    print("-" * 60)
    
    t_start = time.time()
    key, subkey = random.split(key)
    
    # fit 函数：带日志输出的训练
    trained_model = fit(
        model,
        train_data,
        val_data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        key=subkey,
    )
    
    print("-" * 60)
    print(f"训练完成! 耗时: {time.time()-t_start:.1f}s")
    
    # 最终评估
    final_train = evaluate(trained_model, train_data[0], train_data[1])
    final_val = evaluate(trained_model, val_data[0], val_data[1])
    print(f"最终准确率 - 训练: {float(final_train):.2%} | 验证: {float(final_val):.2%}")
    
    # 保存 (Equinox 序列化)
    save_model(trained_model, args.save)
    print(f"模型已保存: {args.save}")


if __name__ == "__main__":
    main()