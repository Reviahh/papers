"""
CMSAN Data Adapter (Fixed based on Scan Results)
═══════════════════════════════════════════════════════════════════════════════
修复说明:
1. BCIC: 专门处理 T (Train) 和 E (Eval) 分离的文件结构，自动合并。
2. BCIcha/MAMEM: 确认键名为 x_test/y_test，直接加载。
"""

import numpy as np
import jax.numpy as jnp
from scipy.io import loadmat
from pathlib import Path
import logging

# 设置日志
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# ═══════════════════════════════════════════════════════════════════════════
#                       1. 数据集元数据
# ═══════════════════════════════════════════════════════════════════════════

DATASET_META = {
    'bcic':   {'C': 22, 'T': 438, 'K': 4,  'folder': 'BCICIV_2a_mat'},
    'mamem':  {'C': 8,  'T': 125, 'K': 5,  'folder': 'MAMEM'},
    'bcicha': {'C': 56, 'T': 160, 'K': 2,  'folder': 'BCIcha'},
}

# ═══════════════════════════════════════════════════════════════════════════
#                       2. 辅助函数
# ═══════════════════════════════════════════════════════════════════════════

def normalize(X):
    """Z-Score 标准化"""
    mean = X.mean(axis=(1, 2), keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True) + 1e-8
    return (X - mean) / std

def find_file(base_roots, filename):
    """在多个目录中搜索文件"""
    for root in base_roots:
        path = root / filename
        if path.exists():
            return path
    return None

# ═══════════════════════════════════════════════════════════════════════════
#                       3. 专用加载逻辑
# ═══════════════════════════════════════════════════════════════════════════

def _load_bcic_merged(t_path, e_path):
    """
    BCIC 专用：合并 T (Train) 和 E (Eval) 文件
    """
    X_list, y_list = [], []
    
    # 1. 加载 Training Set
    if t_path and t_path.exists():
        d = loadmat(str(t_path))
        # 扫描结果显示 T 文件里是 x_train
        if 'x_train' in d:
            X_list.append(d['x_train'])
            y_list.append(d['y_train'])
            logger.info(f"   -> Loaded Train: {t_path.name}")
            
    # 2. 加载 Evaluation Set
    if e_path and e_path.exists():
        d = loadmat(str(e_path))
        # 扫描结果显示 E 文件里是 x_test
        if 'x_test' in d:
            X_list.append(d['x_test'])
            y_list.append(d['y_test'])
            logger.info(f"   -> Loaded Eval:  {e_path.name}")
    
    if not X_list:
        raise ValueError("BCIC load failed: No data found in T or E files.")

    # 3. 合并
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0).flatten()
    
    # 4. 裁剪通道 (保留前22个 EEG)
    # 扫描结果显示 shape 是 (288, 22, 562)，已经是 22 通道了，但保险起见
    if X.shape[1] > 22:
        X = X[:, :22, :]
        
    return X, y

def _load_standard(path, key_x='x_test', key_y='y_test'):
    """标准加载 (BCIcha / MAMEM)"""
    d = loadmat(str(path))
    X = d[key_x]
    y = d[key_y].flatten()
    return X, y

# ═══════════════════════════════════════════════════════════════════════════
#                       4. 统一入口
# ═══════════════════════════════════════════════════════════════════════════

def load_unified(dataset_name: str, subject_id: int, data_dir: str = 'data'):
    dataset_name = dataset_name.lower()
    meta = DATASET_META.get(dataset_name)
    
    if not meta:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # 搜索路径
    root = Path(data_dir)
    search_paths = [
        root,
        root / meta['folder'],
        root / "author_original", 
        root / "my_custom"
    ]

    X, y = None, None

    # ─── 分支 1: BCIC (需要找两个文件) ───
    if dataset_name == 'bcic':
        # 构造文件名: BCIC_S01_T.mat 和 BCIC_S01_E.mat
        fname_t = f"BCIC_S{subject_id:02d}_T.mat"
        fname_e = f"BCIC_S{subject_id:02d}_E.mat"
        
        path_t = find_file(search_paths, fname_t)
        path_e = find_file(search_paths, fname_e)
        
        if not path_t and not path_e:
             # 尝试 fallback: A01T.mat (原始格式)
            fname_t_alt = f"A{subject_id:02d}T.mat"
            fname_e_alt = f"A{subject_id:02d}E.mat"
            path_t = find_file(search_paths, fname_t_alt)
            path_e = find_file(search_paths, fname_e_alt)

        if not path_t and not path_e:
            raise FileNotFoundError(f"Missing BCIC files for Subject {subject_id} (searched for {fname_t}/{fname_e})")
            
        X, y = _load_bcic_merged(path_t, path_e)

    # ─── 分支 2: MAMEM ───
    elif dataset_name == 'mamem':
        fname = f"U{subject_id:03d}.mat"
        path = find_file(search_paths, fname)
        if not path:
             raise FileNotFoundError(f"Missing MAMEM file: {fname}")
        logger.info(f"📥 Loading: {path.name}")
        X, y = _load_standard(path, 'x_test', 'y_test')

    # ─── 分支 3: BCIcha ───
    elif dataset_name == 'bcicha':
        fname = f"Data_S{subject_id:02d}_Sess.mat"
        path = find_file(search_paths, fname)
        if not path:
             raise FileNotFoundError(f"Missing BCIcha file: {fname}")
        logger.info(f"📥 Loading: {path.name}")
        X, y = _load_standard(path, 'x_test', 'y_test')

    # ─── 通用预处理 ───
    
    # 1. 裁剪时间窗
    target_T = meta['T']
    current_T = X.shape[2]
    
    if current_T > target_T:
        # 居中裁剪
        start = (current_T - target_T) // 2
        X = X[:, :, start:start+target_T]
    elif current_T < target_T:
        logger.warning(f"⚠️ Padding data: {current_T} -> {target_T}")
        pad_len = target_T - current_T
        X = np.pad(X, ((0,0), (0,0), (0, pad_len)))

    # 2. 标准化
    X = normalize(X)
    
    # 3. 标签从 0 开始
    if y.min() == 1:
        y = y - 1
        
    # 4. 类型转换
    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.int32)