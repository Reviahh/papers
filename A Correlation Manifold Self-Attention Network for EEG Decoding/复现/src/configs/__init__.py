"""
Global Configuration (Unified)
"""

# 1. 已知数据集的物理参数 (Metadata)
# 如果扫描到的数据集在这里面，就用这些参数；如果不在，就需要手动补充或给默认值
KNOWN_META = {
    'bcic':      {'C': 22, 'T': 438, 'K': 4},
    'physionet': {'C': 64, 'T': 640, 'K': 4},
}

# 2. 默认训练配置
DEFAULT_CONFIG = {
    'batch_size': 32,
    'epochs': 200,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'model': {
        'fem': 'conv', 'mmm': 'corr', 'hom': 'olm',
        'att': 'manifold', 'prj': 'tangent', 'cls': 'linear',
        'kernel': 25, 'D': 20, 'S': 3,
    }
}

def get_config(dataset_name: str) -> dict:
    """获取配置，动态注入参数"""
    config = DEFAULT_CONFIG.copy()
    config['dataset'] = dataset_name
    
    # 尝试查找已知的物理维度
    if dataset_name in KNOWN_META:
        meta = KNOWN_META[dataset_name]
    else:
        # 如果是扫描出来的新数据集，但没配置过 metadata
        print(f"⚠️ Warning: Dataset '{dataset_name}' dimensions unknown. Using BCIC defaults.")
        meta = KNOWN_META['bcic'] # 默认回退，或者抛出异常让你去补全

    # 注入 C, T, K 到 model 配置中 (关键！)
    config['model'].update(meta)
    
    return config