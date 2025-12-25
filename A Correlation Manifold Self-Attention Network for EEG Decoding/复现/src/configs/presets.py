"""
CMSAN Configuration Presets
"""

# ═══════════════════════════════════════════════════════════════════════════════
# 训练配置
# ═══════════════════════════════════════════════════════════════════════════════

# 【FAST 模式: i5-12500H 本地开发】
FAST = {
    'd_model': 32,
    'slices': 4,
    'lr': 1.5e-3,
    'epochs': 100,
    'batch_size': 64,
    'save_model': True,
    'verbose': True,
}

# 【PAPER 模式: TPU 刷榜】
PAPER = {
    'd_model': 32,
    'slices': 4,
    'lr': 1e-3,
    'epochs': 200,
    'batch_size': 128,
    'save_model': False,
    'verbose': False,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 数据集配置
# ═══════════════════════════════════════════════════════════════════════════════

DATASETS = {
    # BCI Competition IV 2a
    # 9被试, 4类MI, shape: (288, 22, 562) per file
    'bcic': {
        'name': 'bcic',
        'subjects': list(range(1, 10)),  # [1,2,3,4,5,6,7,8,9]
        'channels': 22,
        'timepoints': 562,
        'classes': 4,
    },
    
    # BCI Challenge
    # 16被试 (编号不连续), 2类, shape: (340, 56, 160)
    'bcicha': {
        'name': 'bcicha',
        'subjects': [2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26],  # ← 按实际文件填
        'channels': 56,
        'timepoints': 160,
        'classes': 2,
    },
    
    # MAMEM SSVEP
    # 11被试, 5类, shape: (500, 8, 125)
    'mamem': {
        'name': 'mamem',
        'subjects': list(range(1, 12)),  # [1,2,...,11]
        'channels': 8,
        'timepoints': 125,
        'classes': 5,
    },
    
    # 作者原始数据
    # 单文件, 4类, shape: (576, 22, 513)
    'author': {
        'name': 'author',
        'subjects': [1],
        'channels': 22,
        'timepoints': 513,
        'classes': 4,
    },
}

def get_config(mode):
    return PAPER if mode == 'paper' else FAST