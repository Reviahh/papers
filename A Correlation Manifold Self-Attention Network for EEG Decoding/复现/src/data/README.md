# 数据目录说明

本目录用于存放所有实验数据，按来源和状态分类。

## 子目录结构

### author_original/
存放作者提供的原始数据文件（如 `.npz` 文件）。

**用途**: 维度一实验（作者原文复现）

**示例**:
```
author_original/
└── eeg_data.npz
```

### my_custom/
存放自己下载和处理的数据。

**用途**: 维度二和维度三实验（我的复现和框架应用）

**示例**:
```
my_custom/
├── BCICIV_2a_mat/
├── MAMEM/
└── BCIcha/
```

### raw/
存放未经处理的原始数据。

**用途**: 数据预处理和探索

## 注意事项

1. 数据文件不会被提交到 Git（已在 .gitignore 中配置）
2. 请根据需要自行下载或获取数据
3. 可使用 `scripts/data_utils/download_data.py` 下载 BCI Competition IV 2a 数据
