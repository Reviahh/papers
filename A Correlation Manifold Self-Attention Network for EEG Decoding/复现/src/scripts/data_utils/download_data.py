"""
EEG 数据自动下载与预处理
═══════════════════════════════════════════════════════════════════════════════

使用 MOABB 自动下载 BCI Competition IV 2a 数据集
首次运行会自动下载（约 1.5GB），之后会使用缓存

使用:
    python download_data.py              # 下载并处理所有被试
    python download_data.py --subject 1  # 只处理被试1
"""

import argparse
import numpy as np

def download_bcic_iv_2a(subject_id=1):
    """
    下载 BCI Competition IV 2a 数据集
    
    - 4类运动想象: 左手(0), 右手(1), 双脚(2), 舌头(3)
    - 22 EEG 通道
    - 250 Hz 采样率
    - 每个被试 288 trials
    """
    print("正在加载 MOABB...")
    
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError:
        print("请先安装 moabb: pip install moabb")
        return None
    
    print(f"正在下载/加载 BCI Competition IV 2a 数据集 (被试 {subject_id})...")
    print("(首次运行会自动下载，请稍候...)")
    
    # 加载数据集
    dataset = BNCI2014_001()
    
    # 定义范式: 4类运动想象
    paradigm = MotorImagery(
        n_classes=4,
        fmin=4,    # 带通滤波 4-38 Hz
        fmax=38,
        tmin=0,    # 时间窗口 0-4s
        tmax=4,
        resample=128,  # 降采样到 128 Hz (减少计算)
    )
    
    # 获取数据
    X, y, meta = paradigm.get_data(dataset, subjects=[subject_id])
    
    # 标签转换为 0-3
    unique_labels = np.unique(y)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in y])
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"类别分布: {np.bincount(y)}")
    
    return X.astype(np.float32), y.astype(np.int32)


def prepare_data(X, y, val_ratio=0.2, seed=42):
    """划分训练/验证集"""
    from sklearn.model_selection import train_test_split
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=val_ratio, 
        random_state=seed, 
        stratify=y
    )
    
    # 标准化 (按通道)
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    
    print(f"训练集: {X_train.shape}, 验证集: {X_val.shape}")
    
    return X_train, y_train, X_val, y_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=int, default=1, help='被试编号 1-9')
    parser.add_argument('--output', type=str, default='eeg_data.npz', help='输出文件')
    args = parser.parse_args()
    
    print("=" * 60)
    print("BCI Competition IV 2a 数据下载器")
    print("=" * 60)
    
    # 下载数据
    result = download_bcic_iv_2a(args.subject)
    if result is None:
        return
    
    X, y = result
    
    # 划分数据
    X_train, y_train, X_val, y_val = prepare_data(X, y)
    
    # 保存
    np.savez(args.output,
        x_train=X_train,
        y_train=y_train,
        x_val=X_val,
        y_val=y_val,
    )
    
    print(f"\n✅ 数据已保存到: {args.output}")
    print(f"   训练集: {X_train.shape[0]} 样本")
    print(f"   验证集: {X_val.shape[0]} 样本")
    print(f"   通道数: {X_train.shape[1]}")
    print(f"   时间点: {X_train.shape[2]}")
    print()
    print("使用方法:")
    print(f"   python main.py --data {args.output} --epochs 100")


if __name__ == "__main__":
    main()
