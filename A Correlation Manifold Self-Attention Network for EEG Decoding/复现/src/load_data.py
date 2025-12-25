# load_data.py
"""
作者数据加载器
支持三个数据集: BCICIV_2a (MI), MAMEM (SSVEP), BCIcha (ERN)
"""

import numpy as np
from scipy.io import loadmat
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import KFold

# ═══════════════════════════════════════════════════════════════════════════
#                       数据集配置
# ═══════════════════════════════════════════════════════════════════════════

DATASET_CONFIG = {
    'bcic': {
        'folder': 'BCICIV_2a_mat',
        'C': 22, 'T': 438, 'K': 4, 'D': 20, 'S': 3,
        'subjects': list(range(1, 10)),  # S01-S09
    },
    'mamem': {
        'folder': 'MAMEM', 
        'C': 8, 'T': 125, 'K': 5, 'D': 15, 'S': 3,
        'subjects': list(range(1, 12)),  # U001-U011
    },
    'bcicha': {
        'folder': 'BCIcha',
        'C': 56, 'T': 160, 'K': 2, 'D': 14, 'S': 3,
        'subjects': [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#                       加载函数
# ═══════════════════════════════════════════════════════════════════════════

def load_bcic(data_root: str, subject: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载 BCI Competition IV 2a 数据 (MI)
    
    Args:
        data_root: 数据根目录
        subject: 被试编号 1-9
        
    Returns:
        X: (N, 22, 438) - 截取到论文长度
        y: (N,) - 标签 0-3
    """
    folder = Path(data_root) / "BCICIV_2a_mat"
    
    # 加载训练和测试数据
    train_file = folder / f"BCIC_S{subject:02d}_T.mat"
    test_file = folder / f"BCIC_S{subject:02d}_E.mat"
    
    train_data = loadmat(str(train_file))
    test_data = loadmat(str(test_file))
    
    # 合并数据 (论文用 10-fold CV，所以合并后再划分)
    X = np.concatenate([
        train_data['x_train'],
        test_data['x_test']
    ], axis=0).astype(np.float32)
    
    y = np.concatenate([
        train_data['y_train'].flatten(),
        test_data['y_test'].flatten()
    ]).astype(np.int32)
    
    # 截取时间维度: 562 -> 438 (论文设置)
    # 通常取中间或从头开始
    T_target = 438
    T_start = (X.shape[2] - T_target) // 2  # 居中截取
    X = X[:, :, T_start:T_start + T_target]
    
    # 标签转为 0-indexed
    y = y - y.min()
    
    return X, y


def load_mamem(data_root: str, subject: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载 MAMEM SSVEP 数据
    
    Args:
        data_root: 数据根目录
        subject: 被试编号 1-11
        
    Returns:
        X: (500, 8, 125)
        y: (500,) - 标签 0-4
    """
    folder = Path(data_root) / "MAMEM"
    file = folder / f"U{subject:03d}.mat"
    
    data = loadmat(str(file))
    X = data['x_test'].astype(np.float32)
    y = data['y_test'].flatten().astype(np.int32)
    
    # 标签转为 0-indexed
    y = y - y.min()
    
    return X, y


def load_bcicha(data_root: str, subject: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载 BCI Challenge ERN 数据
    
    Args:
        data_root: 数据根目录
        subject: 被试编号 (2,6,7,11,12,13,14,16,17,18,20,21,22,23,24,26)
        
    Returns:
        X: (340, 56, 160)
        y: (340,) - 标签 0-1
    """
    folder = Path(data_root) / "BCIcha"
    file = folder / f"Data_S{subject:02d}_Sess.mat"
    
    data = loadmat(str(file))
    X = data['x_test'].astype(np.float32)
    y = data['y_test'].flatten().astype(np.int32)
    
    # 标签转为 0-indexed
    y = y - y.min()
    
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
#                       统一接口
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(data_root: str, dataset: str, subject: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    统一加载接口
    
    Args:
        data_root: 数据根目录 (包含 BCICIV_2a_mat, MAMEM, BCIcha 三个文件夹)
        dataset: 'bcic' | 'mamem' | 'bcicha'
        subject: 被试编号
    """
    loaders = {
        'bcic': load_bcic,
        'mamem': load_mamem,
        'bcicha': load_bcicha,
    }
    return loaders[dataset](data_root, subject)


def get_config(dataset: str) -> dict:
    """获取数据集配置"""
    return DATASET_CONFIG[dataset]


# ═══════════════════════════════════════════════════════════════════════════
#                       数据预处理
# ═══════════════════════════════════════════════════════════════════════════

def standardize(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    标准化 (按通道计算均值和标准差)
    
    使用训练集的统计量标准化验证集
    """
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    
    return X_train, X_val


def make_kfold(X: np.ndarray, y: np.ndarray, n_folds: int = 10, fold: int = 0, seed: int = 42):
    """
    K-Fold 划分 (论文使用 10-fold CV)
    
    Args:
        X, y: 完整数据
        n_folds: 折数
        fold: 当前折 (0 到 n_folds-1)
        seed: 随机种子
        
    Returns:
        X_train, y_train, X_val, y_val
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    for i, (train_idx, val_idx) in enumerate(kf.split(X)):
        if i == fold:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 标准化
            X_train, X_val = standardize(X_train, X_val)
            
            return X_train, y_train, X_val, y_val
    
    raise ValueError(f"fold {fold} out of range")


# ═══════════════════════════════════════════════════════════════════════════
#                       便捷函数
# ═══════════════════════════════════════════════════════════════════════════

def prepare_subject(
    data_root: str, 
    dataset: str, 
    subject: int, 
    fold: int = 0,
    n_folds: int = 10,
) -> dict:
    """
    准备单个被试的数据 (用于训练)
    
    Returns:
        dict with keys: x_train, y_train, x_val, y_val, config
    """
    import jax.numpy as jnp
    
    X, y = load_dataset(data_root, dataset, subject)
    X_train, y_train, X_val, y_val = make_kfold(X, y, n_folds, fold)
    
    config = get_config(dataset)
    
    return {
        'x_train': jnp.array(X_train),
        'y_train': jnp.array(y_train),
        'x_val': jnp.array(X_val),
        'y_val': jnp.array(y_val),
        'config': config,
    }


# ═══════════════════════════════════════════════════════════════════════════
#                       测试
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_root = "data"
    
    print("=" * 60)
    print("测试数据加载")
    print("=" * 60)
    
    # 测试三个数据集
    for dataset in ['bcic', 'mamem', 'bcicha']:
        config = get_config(dataset)
        subject = config['subjects'][0]  # 第一个被试
        
        print(f"\n📊 {dataset.upper()}")
        X, y = load_dataset(data_root, dataset, subject)
        print(f"   Subject {subject}: X={X.shape}, y={y.shape}")
        print(f"   类别分布: {np.bincount(y)}")
        print(f"   期望配置: C={config['C']}, T={config['T']}, K={config['K']}")
        
        # 测试 K-Fold
        data = prepare_subject(data_root, dataset, subject, fold=0)
        print(f"   Fold 0: train={data['x_train'].shape}, val={data['x_val'].shape}")