# papers

本项目使用 Python 虚拟环境进行开发，以确保依赖隔离和版本一致性。

## 环境要求
- Windows 系统
- 已安装 Python Launcher (`py` 命令)
- 本项目建议使用 Python 版本：3.11 或 3.13

## 虚拟环境配置 (核心步骤)

### 1. 找到/创建虚拟环境
如果你已经有现成的环境，请直接跳到第 2 步。如果没有，请在根目录运行：
```bash
# 使用 Python 3.11 创建
py -3.11 -m venv .venv

```

### 2. 在 VS Code 中关联环境 (重要)

由于本项目的环境可能位于非标准路径或初次加载，请手动关联：

1. 按下 `Ctrl + Shift + P` 打开命令面板。
2. 输入并选择 `Python: Select Interpreter`。
3. 选择 `Enter interpreter path...` -> `Find...`。
4. 浏览并选中该路径：`.venv/Scripts/python.exe`。

### 3. 激活环境与安装依赖

打开一个新的终端（Terminal），你应该能看到命令行开头有 `(.venv)` 标志。
然后运行以下命令安装所需库：

```bash
pip install -r requirements.txt

```

## 常用命令对照表

* **查看已安装 Python 列表**: `py -0`
* **运行脚本**: `python main.py`
* **导出依赖**: `pip freeze > requirements.txt`

---

*注：本项目已配置 `.vscode/settings.json`，通常下次打开此文件夹时会自动识别环境。*

```

---

### 额外的小提示：
1. **requirements.txt**：如果你昨天装了很多包，记得在激活环境后运行 `pip freeze > requirements.txt`。这样下次你如果不小心把环境删了，只需要运行 `pip install -r requirements.txt` 就能一键恢复所有包。
2. **.gitignore**：如果你使用 Git，记得在 `.gitignore` 文件里加上 `.venv/`，不要把几百兆的环境文件夹上传到代码库。

```