# 迁移指南

如果你之前使用了旧版本的 CMSAN 项目，本指南帮助你迁移到新的目录结构。

---

## 主要变化

### 1. 文件移动

| 旧位置 | 新位置 |
|--------|--------|
| `src/download_data.py` | `src/scripts/data_utils/download_data.py` |
| `src/load_data.py` | `src/scripts/data_utils/load_data.py` |
| `src/explore_data.py` | `src/scripts/data_utils/explore_data.py` |
| `src/run_experiment.py` | `src/scripts/my_reproduction.py` |
| `src/run_fast.py` | `src/scripts/run_application.py` |
| `src/cmsan/train.py` | `src/cmsan/train_engine.py` |

### 2. 新增文件

- `src/scripts/reproduce_paper.py` - 维度一：作者原文实验
- `src/configs/paper_config.yaml` - 论文固定参数
- `src/configs/custom_config.yaml` - 自定义参数
- `src/USAGE_GUIDE.md` - 详细使用指南

### 3. 新增目录

```
src/
├── data/                    # 数据存放区
│   ├── author_original/     # 作者数据
│   ├── my_custom/           # 自己的数据
│   └── raw/                 # 原始数据
├── checkpoints/             # 模型权重
└── configs/                 # 配置文件
```

---

## 迁移步骤

### 步骤 1: 更新代码

```bash
cd src/
git pull origin main  # 或你的分支名
```

### 步骤 2: 迁移数据文件

如果你之前在 `src/` 目录下有 `.npz` 或 `.mat` 文件：

```bash
# 作者提供的数据
mv *.npz data/author_original/

# 自己下载的数据集文件夹
mv BCICIV_2a_mat/ data/my_custom/
mv MAMEM/ data/my_custom/
mv BCIcha/ data/my_custom/
```

### 步骤 3: 迁移模型文件

```bash
# 移动 .pkl 模型文件
mv *.pkl checkpoints/
```

### 步骤 4: 更新导入语句

如果你的自定义脚本导入了被移动的模块：

**旧代码**:
```python
from load_data import load_dataset
```

**新代码**:
```python
from scripts.data_utils.load_data import load_dataset
```

或者在脚本开头添加路径：
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_utils.load_data import load_dataset
```

### 步骤 5: 更新命令行调用

**旧命令**:
```bash
python run_experiment.py --data data --dataset bcic
```

**新命令**:
```bash
# 方法一: 直接调用
python scripts/my_reproduction.py --data data/my_custom --dataset bcic

# 方法二: 通过主入口
python main.py --mode reproduce --data data/my_custom --dataset bcic
```

---

## 导入路径对照表

### 数据加载模块

**旧**:
```python
from load_data import load_dataset, get_config
```

**新**:
```python
from scripts.data_utils.load_data import load_dataset, get_config
```

### CMSAN 训练模块

**旧**:
```python
from cmsan.train import train, fit
```

**新**:
```python
from cmsan.train_engine import train, fit
# 或者直接从 cmsan 导入
from cmsan import train, fit
```

---

## 常见问题

### Q: 我的旧脚本还能用吗？

**A**: 大部分可以，但需要：
1. 更新导入路径
2. 调整数据文件路径
3. 如果使用了 `cmsan.train` 模块，改为 `cmsan.train_engine` 或直接从 `cmsan` 导入

### Q: 我在根目录下的数据文件会被删除吗？

**A**: 不会。但建议你手动移动到 `data/` 目录下的相应子目录，以保持项目整洁。

### Q: 旧的 main.py 还能用吗？

**A**: 新的 `main.py` 功能更强大，支持三种模式：
```bash
# 测试模式（旧 main.py 的默认行为）
python main.py

# 作者原文实验
python main.py --mode paper --data data/author_original/eeg_data.npz

# 我的复现
python main.py --mode reproduce --data data/my_custom --dataset bcic

# 快速实验
python main.py --mode fast --data data/my_custom --dataset all
```

### Q: 为什么要进行这次重构？

**A**: 
1. **清晰区分**: 代码、数据、配置、权重分离，避免混乱
2. **三维度实验**: 明确区分作者原证、自我复现、扩展应用
3. **易于管理**: 新增实验只需添加脚本，不影响核心库
4. **专业规范**: 符合学术项目的最佳实践

---

## 示例：迁移一个自定义脚本

**旧脚本** (`my_experiment.py`):
```python
from load_data import load_dataset, get_config
from cmsan import CMSAN, fit

data_root = "data"
dataset = "bcic"
config = get_config(dataset)

X, y = load_dataset(data_root, dataset, subject=1)
model = CMSAN(key, C=config['C'], T=config['T'], ...)
```

**新脚本** (`scripts/my_experiment.py`):
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.data_utils.load_data import load_dataset, get_config
from cmsan import CMSAN, fit

data_root = "data/my_custom"  # 更新路径
dataset = "bcic"
config = get_config(dataset)

X, y = load_dataset(data_root, dataset, subject=1)
model = CMSAN(key, C=config['C'], T=config['T'], ...)
```

---

## 需要帮助？

如果在迁移过程中遇到问题：
1. 查看 `USAGE_GUIDE.md` 了解新用法
2. 查看 `README.md` 了解新结构
3. 提交 GitHub Issue

---

**祝迁移顺利！** 🎉
