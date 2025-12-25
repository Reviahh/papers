# explore_data.py
"""探索作者数据格式"""
import os
import numpy as np
from pathlib import Path

def explore_folder(folder):
    """探索单个文件夹"""
    print(f"\n{'='*60}")
    print(f"📁 {folder}")
    print('='*60)
    
    for f in sorted(Path(folder).iterdir()):
        print(f"\n  📄 {f.name}")
        
        try:
            if f.suffix == '.mat':
                from scipy.io import loadmat
                data = loadmat(str(f))
                keys = [k for k in data.keys() if not k.startswith('__')]
                for k in keys:
                    v = data[k]
                    if isinstance(v, np.ndarray):
                        print(f"      {k}: shape={v.shape}, dtype={v.dtype}")
                    else:
                        print(f"      {k}: {type(v)}")
                        
            elif f.suffix == '.npz':
                data = np.load(str(f), allow_pickle=True)
                for k in data.keys():
                    v = data[k]
                    print(f"      {k}: shape={v.shape}, dtype={v.dtype}")
                    
            elif f.suffix == '.npy':
                data = np.load(str(f), allow_pickle=True)
                print(f"      shape={data.shape}, dtype={data.dtype}")
                
            elif f.is_dir():
                # 子文件夹，列出内容
                subfiles = list(f.iterdir())[:5]
                print(f"      (文件夹, 包含 {len(list(f.iterdir()))} 个文件)")
                for sf in subfiles:
                    print(f"        - {sf.name}")
                if len(list(f.iterdir())) > 5:
                    print(f"        ...")
                    
        except Exception as e:
            print(f"      ❌ 读取失败: {e}")

# 探索三个数据集
data_root = "data"  # 如果不对，改成你的路径

for dataset in ["BCICIV_2a_mat", "MAMEM", "BCIcha"]:
    folder = os.path.join(data_root, dataset)
    if os.path.exists(folder):
        explore_folder(folder)
    else:
        print(f"❌ 找不到: {folder}")