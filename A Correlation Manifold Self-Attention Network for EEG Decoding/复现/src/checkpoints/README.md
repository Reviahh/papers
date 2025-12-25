# Checkpoints 目录

本目录用于存放训练好的模型权重文件。

## 文件组织

建议按实验维度命名：

```
checkpoints/
├── paper_model.pkl          # 维度一: 作者原文实验
├── my_bcic_model.pkl        # 维度二: 我的 BCIC 复现
├── my_mamem_model.pkl       # 维度二: 我的 MAMEM 复现
├── fast_bcic_model.pkl      # 维度三: 快速实验
└── ...
```

## 注意事项

1. 模型文件不会被提交到 Git（已在 .gitignore 中配置）
2. 使用 `cmsan.save_model()` 和 `cmsan.load_model()` 保存和加载模型
3. 建议定期备份重要的模型文件
