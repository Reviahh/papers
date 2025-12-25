"""
CMSAN 统一入口程序 (Equinox)
═══════════════════════════════════════════════════════════════════════════════

Correlation Manifold Self-Attention Network for EEG Decoding
使用 Equinox + Optax 实现完全函数式训练

三种实验模式:
    1. paper      - 维度一: 作者原文实验 (使用作者数据和参数)
    2. reproduce  - 维度二: 我自己的复现 (使用自己下载的数据)
    3. fast       - 维度三: 框架应用 (CPU优化快速实验)

使用:
    python main.py --mode paper --data data/author_original/eeg_data.npz
    python main.py --mode reproduce --data data/my_custom --dataset bcic
    python main.py --mode fast --data data/my_custom --dataset all
    python main.py  # 使用假数据测试 (默认)
"""

import jax
import jax.numpy as jnp
from jax import random
import argparse
import time
import sys
from pathlib import Path

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
    parser = argparse.ArgumentParser(description='CMSAN 统一入口')
    parser.add_argument('--mode', type=str, default='test', 
                       choices=['test', 'paper', 'reproduce', 'fast'],
                       help='''实验模式:
                       test - 使用假数据测试
                       paper - 维度一: 作者原文实验
                       reproduce - 维度二: 我的复现
                       fast - 维度三: 框架应用''')
    parser.add_argument('--data', type=str, default=None, help='数据路径')
    parser.add_argument('--dataset', type=str, default='bcic',
                       choices=['bcic', 'mamem', 'bcicha', 'all'],
                       help='数据集选择 (用于 reproduce/fast 模式)')
    parser.add_argument('--preset', type=str, default='light', 
                       choices=list(PRESETS.keys()), help='预设配置')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch', type=int, default=8, help='批大小')
    parser.add_argument('--lr', type=float, default=5e-4, help='学习率')
    parser.add_argument('--save', type=str, default='checkpoints/model.pkl', help='保存路径')
    args = parser.parse_args()
    
    print("═" * 70)
    print("CMSAN - Correlation Manifold Self-Attention Network")
    print("═" * 70)
    print(f"\n当前模式: {args.mode}")
    
    # 根据模式选择执行
    if args.mode == 'paper':
        print("\n→ 维度一: 作者原文实验复现")
        print("提示: 建议直接运行 scripts/reproduce_paper.py")
        print(f"命令: python scripts/reproduce_paper.py --data {args.data or 'data/author_original/eeg_data.npz'}")
        if args.data:
            import subprocess
            subprocess.run([sys.executable, 'scripts/reproduce_paper.py', '--data', args.data])
        return
    
    elif args.mode == 'reproduce':
        print("\n→ 维度二: 我自己的复现 (10-fold CV)")
        print("提示: 建议直接运行 scripts/my_reproduction.py")
        print(f"命令: python scripts/my_reproduction.py --data {args.data or 'data/my_custom'} --dataset {args.dataset}")
        if args.data:
            import subprocess
            subprocess.run([sys.executable, 'scripts/my_reproduction.py', 
                          '--data', args.data, '--dataset', args.dataset])
        return
    
    elif args.mode == 'fast':
        print("\n→ 维度三: 框架应用 (CPU优化)")
        print("提示: 建议直接运行 scripts/run_application.py")
        print(f"命令: python scripts/run_application.py --data {args.data or 'data/my_custom'} --dataset {args.dataset}")
        if args.data:
            import subprocess
            subprocess.run([sys.executable, 'scripts/run_application.py',
                          '--data', args.data, '--dataset', args.dataset])
        return
    
    # test 模式: 使用假数据
    print("\n→ 测试模式: 使用假数据")
    print("提示: 这是一个快速测试，用于验证代码是否正常工作")
    
    key = random.key(42)
    
    # 先加载数据，获取维度
    if args.data:
        import numpy as np
        data = np.load(args.data)
        train_data = (jnp.array(data['x_train']), jnp.array(data['y_train']))
        val_data = (jnp.array(data['x_val']), jnp.array(data['y_val']))
        print(f"加载数据: {args.data}")
        
        # 从数据推断维度
        C = train_data[0].shape[1]  # 通道数
        T = train_data[0].shape[2]  # 时间点
        K = int(train_data[1].max()) + 1  # 类别数
        
        # 根据数据维度创建模型
        key, subkey = random.split(key)
        from cmsan import CMSAN
        preset_cfg = PRESETS.get(args.preset, PRESETS['light'])
        model = CMSAN(
            subkey, 
            C=C, 
            T=T, 
            D=preset_cfg.get('D', 20),
            S=preset_cfg.get('S', 3),
            K=K,
        )
        print(f"根据数据自动配置模型")
    else:
        # 使用预设创建模型和假数据
        key, subkey = random.split(key)
        model = create_from_preset(subkey, args.preset)
        key, subkey = random.split(key)
        train_data, val_data = make_fake_data(subkey, model.C, model.T, model.K)
        print(f"使用假数据 (测试模式，preset={args.preset})")
    
    print(f"配置: C={model.C}, T={model.T}, D={model.D}, S={model.S}, K={model.K}")
    print(f"设备: {jax.devices()[0]}")
    print()
    
    # 统计参数量 (Equinox 方式)
    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"参数量: {n_params:,}")
    
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
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(trained_model, str(save_path))
    print(f"模型已保存: {args.save}")
    
    print("\n" + "═" * 70)
    print("✓ 完成")
    print("═" * 70)


if __name__ == "__main__":
    main()
