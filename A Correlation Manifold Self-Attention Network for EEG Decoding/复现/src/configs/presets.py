"""
CMSAN Configuration Presets
═══════════════════════════════════════════════════════════════════════════════
"""

# 【FAST 模式: 本地开发 & 产出模型】
# 场景: 在你的 i5-12500H 上跑，用来调试或生成给别人用的 .eqx 文件。
FAST = {
    'd_model': 24,       
    'slices': 3,
    'lr': 1.5e-3,       
    'epochs': 100,       
    'batch_size': 64,
    'save_model': True,  # 🔥 重点: FAST 模式下保存模型
}

# 【PAPER 模式: 论文刷榜 & TPU】
# 场景: 在 Colab/TPU 上跑，追求极致分数，跑完所有数据集。
# 默认不保存模型（因为跑 3 个数据集几十个被试，存模型太占空间且拖慢 IO）。
PAPER = {
    'd_model': 32,       # 论文级大参数
    'slices': 3,
    'lr': 1e-3,
    'epochs': 200,       # 跑满收敛
    'batch_size': 128,   # TPU 大显存优势
    'save_model': False, # 🔥 重点: PAPER 模式只看指标，不存文件
}

def get_config(mode):
    return PAPER if mode == 'paper' else FAST