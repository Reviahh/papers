"""
CMSAN Generic Data Adapter (Fixed)
═══════════════════════════════════════════════════════════════════════════════
修复内容:
1. 补回 DATASET_META 变量，解决 ImportError。
2. 专注于 .mat 文件处理。
"""

import logging
import glob
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import numpy as np
import jax.numpy as jnp
from scipy.io import loadmat

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 元数据定义 (保留此变量以兼容 __init__.py 导入)
# ═══════════════════════════════════════════════════════════════════════════════

# 即使我们做通用加载，保留这个字典也有助于快速定位已知数据集的文件夹
DATASET_META = {
    'bcic':       {'folder': 'BCICIV_2a_mat'},
    'bciciv_2a':  {'folder': 'BCICIV_2a_mat'},
    'mamem':      {'folder': 'MAMEM'},
    'bcicha':     {'folder': 'BCIcha'},
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. 通用工具
# ═══════════════════════════════════════════════════════════════════════════════

def normalize(X: np.ndarray) -> np.ndarray:
    """通用 Z-Score 标准化"""
    axes = tuple(range(1, X.ndim))
    mean = X.mean(axis=axes, keepdims=True)
    std = X.std(axis=axes, keepdims=True) + 1e-8
    return (X - mean) / std

def find_dataset_dir(base_name: str, root_dir: Path) -> Path:
    """模糊查找数据集文件夹"""
    # 1. 精确匹配
    target = root_dir / base_name
    if target.exists(): return target
    
    # 2. 查表 (Meta)
    lower_name = base_name.lower()
    if lower_name in DATASET_META:
        folder = DATASET_META[lower_name]['folder']
        target = root_dir / folder
        if target.exists(): return target

    # 3. 模糊匹配 (忽略大小写/下划线)
    clean_name = lower_name.replace('_', '').replace('-', '')
    for d in root_dir.iterdir():
        if not d.is_dir(): continue
        d_clean = d.name.lower().replace('_', '').replace('-', '')
        if clean_name in d_clean or d_clean in clean_name:
            return d
            
    # 找不到就返回 root，假设文件在根目录
    return root_dir

def smart_extract_mat(data_dict: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    智能提取 .mat 内容
    逻辑：最大的数组是 X，第二大(或名字含label)的是 y
    """
    candidates = []
    # 过滤掉 __header__, __version__ 等
    for k, v in data_dict.items():
        if k.startswith('__'): continue
        if isinstance(v, np.ndarray) and v.size > 1:
            candidates.append((k, v))
            
    if len(candidates) < 2:
        # 如果只有一个数组，打印出来看看
        keys = list(data_dict.keys())
        raise ValueError(f"Mat file needs at least 2 arrays (Data & Label). Found: {keys}")
        
    # 按字节大小排序，最大的通常是 EEG 数据
    candidates.sort(key=lambda x: x[1].nbytes, reverse=True)
    
    # 1. 确定 X (最大的)
    X_key, X = candidates[0]
    
    # 2. 确定 y
    y = None
    # 优先找名字像标签的
    for k, v in candidates[1:]:
        name = k.lower()
        if any(tag in name for tag in ['y', 'label', 'class', 'target', 'truth']):
            y = v
            break
            
    # 如果没找到显式名字，就默认取第二大的数组
    if y is None:
        y = candidates[1][1]
        
    logger.info(f"   🔧 Smart Extract: X='{X_key}' {X.shape}, y found.")
    return X, y

# ═══════════════════════════════════════════════════════════════════════════════
# 3. 核心加载逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_shape(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    通用形状修正
    目标: X -> (Batch, Channel, Time), y -> (Batch,)
    """
    # 1. 修正 y (Flatten)
    y = y.flatten()
    # 自动修正 1-based indexing (Matlab 习惯)
    if y.min() == 1:
        y -= 1
        
    # 2. 修正 X
    # 如果是 3D (N, A, B)
    if X.ndim == 3:
        N, A, B = X.shape
        # 启发式转置：通常 Time(T) > Channel(C)
        # 如果第2维比第3维大很多 (例如 A=1000, B=22)，那 A 可能是时间
        # 我们需要 (N, C, T) -> (N, Short, Long)
        if A > B and A > 50:
            logger.info(f"   ⚠️ Auto-Transpose: (N, T, C) {X.shape} -> (N, C, T)")
            X = np.swapaxes(X, 1, 2)
            
    # 如果是 2D (N, T)，扩充为 (N, 1, T)
    elif X.ndim == 2:
        X = X[:, np.newaxis, :]

    return X, y

def load_unified(dataset_name: str, subject_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    统一入口
    """
    # 1. 定位数据根目录
    base_dir = Path(__file__).parent.parent.parent / "data"
    if not base_dir.exists():
        base_dir = Path("data")
        
    # 2. 查找目录
    data_dir = find_dataset_dir(dataset_name, base_dir)
    logger.info(f"   📂 Searching in: {data_dir.name}")

    # 3. 查找被试文件 (.mat)
    # 匹配模式: *1.mat*, *01*.mat
    patterns = [
        f"*{subject_id}.mat",
        f"*{subject_id:02d}*.mat",
        f"*{subject_id}*.mat", # 宽泛匹配
    ]
    
    found_file = None
    all_mat_files = list(data_dir.glob("*.mat"))
    
    # 扫描
    for pat in patterns:
        matches = list(data_dir.glob(pat))
        if matches:
            # 找到最大的那个文件（防止匹配到只有 header 的小文件）
            matches.sort(key=lambda f: f.stat().st_size, reverse=True)
            found_file = matches[0]
            break
            
    if not found_file:
        raise FileNotFoundError(f"No .mat file found for Subject {subject_id} in {data_dir}")

    logger.info(f"   📄 Loading: {found_file.name}")

    # 4. 加载 .mat
    try:
        mat_data = loadmat(str(found_file))
        X, y = smart_extract_mat(mat_data)
    except Exception as e:
        raise RuntimeError(f"Failed to load {found_file.name}: {e}")

    # 5. 后处理
    X, y = ensure_shape(X, y)
    X = normalize(X)
    
    return jnp.array(X, dtype=jnp.float32), jnp.array(y, dtype=jnp.int32)