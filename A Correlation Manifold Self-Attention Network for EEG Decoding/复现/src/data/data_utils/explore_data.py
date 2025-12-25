"""
Dataset Explorer
═══════════════════════════════════════════════════════════════════════════════
功能: 扫描 data 目录，自动识别数据集结构，打印文件清单和 .mat 文件内部信息。
用法: python scripts/data_utils/explore_data.py
"""

import os
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat

# 设置数据根目录 (根据你的截图，脚本默认向上找两级或一级，或者直接指定)
# 假设你在 src/ 目录下运行，数据在 src/../data
BASE_DIR = Path(__file__).parent.parent.parent  # 回退到项目根目录
DATA_DIR = BASE_DIR / "data"

# 如果找不到，尝试当前目录
if not DATA_DIR.exists():
    DATA_DIR = Path("data")

def print_separator(title=""):
    print(f"\n{'='*20} {title} {'='*20}")

def analyze_mat_file(file_path):
    """尝试读取 mat 文件并获取关键信息"""
    try:
        # 只读取元数据，不完全加载数据以加快速度
        mat = loadmat(str(file_path))
        
        info = []
        # 过滤掉 __header__, __version__, __globals__
        keys = [k for k in mat.keys() if not k.startswith('__')]
        
        for k in keys:
            val = mat[k]
            if isinstance(val, np.ndarray):
                info.append(f"{k}: {val.shape} ({val.dtype})")
            else:
                info.append(f"{k}: {type(val).__name__}")
        
        return ", ".join(info)
    except Exception as e:
        return f"读取失败: {str(e)}"

def scan_directory(path):
    """递归扫描目录"""
    path = Path(path)
    if not path.exists():
        print(f"❌ 路径不存在: {path}")
        return

    print(f"📂 正在扫描目录: {path.resolve()}")
    
    # 获取一级子目录
    subdirs = [x for x in path.iterdir() if x.is_dir()]
    files_in_root = [x for x in path.iterdir() if x.is_file()]

    # 1. 打印根目录下的文件 (比如你截图里的 eeg_data.npz)
    if files_in_root:
        print_separator(f"根目录文件 ({len(files_in_root)}个)")
        for f in files_in_root:
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  📄 {f.name:<25} | {size_mb:>6.2f} MB")
            if f.suffix == '.npz':
                try:
                    with np.load(f) as data:
                        print(f"     Keys: {list(data.keys())}")
                        for k in data.keys():
                            print(f"       -> {k}: {data[k].shape}")
                except:
                    pass

    # 2. 遍历子文件夹
    for subdir in subdirs:
        print_separator(f"数据集: {subdir.name}")
        
        mat_files = sorted(list(subdir.glob("*.mat")))
        
        if not mat_files:
            print(f"  (文件夹为空或无 .mat 文件)")
            continue
            
        print(f"  包含 {len(mat_files)} 个 .mat 文件")
        
        # 只详细展示前3个文件作为样本
        for i, f in enumerate(mat_files):
            size_mb = f.stat().st_size / (1024 * 1024)
            
            # 对前3个文件进行深入分析
            if i < 3:
                inner_info = analyze_mat_file(f)
                print(f"  [{i+1}] {f.name:<25} | {size_mb:>6.2f} MB | 内容: {inner_info}")
            elif i == 3:
                print(f"  ... (剩余 {len(mat_files)-3} 个文件格式类似)")
                
        # 统计总大小
        total_size = sum(f.stat().st_size for f in mat_files) / (1024 * 1024)
        print(f"\n  📊 总计: {total_size:.2f} MB")

def main():
    if not DATA_DIR.exists():
        print(f"❌ 找不到数据目录: {DATA_DIR.resolve()}")
        print("请修改脚本中的 DATA_DIR 路径或在正确的位置运行。")
        return

    scan_directory(DATA_DIR)

if __name__ == "__main__":
    main()