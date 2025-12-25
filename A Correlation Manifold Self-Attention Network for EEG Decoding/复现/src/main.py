#!/usr/bin/env python3
import sys
import os
import jax
from pathlib import Path
from rich.console import Console
from rich.panel import Panel

# 相对导入
try:
    from configs import get_config
    from data.data_utils import explore_data 
except ImportError as e:
    sys.exit(f"❌ Import Error: {e}")

try:
    import questionary
except ImportError:
    sys.exit("Install deps: pip install questionary rich")

# === 动态扫描工具 ===
def scan_available_datasets():
    """扫描 ../data 目录下的子文件夹作为数据集选项"""
    # 假设 main.py 在 src/ 下，数据在 src/../data
    base_dir = Path(__file__).parent.parent / "data"
    if not base_dir.exists():
        # 如果找不到，尝试当前目录下的 data
        base_dir = Path("data")
    
    if not base_dir.exists():
        return []

    # 只要是文件夹，就认为是数据集
    return [d.name for d in base_dir.iterdir() if d.is_dir()]

# === 核心流程 ===

def run_training_mode():
    console = Console()
    
    # 1. 动态获取数据集列表
    datasets = scan_available_datasets()
    
    if not datasets:
        console.print("[red]❌ No datasets found in 'data/' folder![/red]")
        return

    ds_name = questionary.select(
        "📚 Select Dataset (Scanned from disk):",
        choices=datasets, # <--- 这里的选项现在是活的了
    ).ask()
    
    if not ds_name: return

    # 2. 选被试
    subj_input = questionary.text("👤 Subject ID:", default="1").ask()
    if not subj_input: return
    subject = int(subj_input)

    # 3. 加载配置
    config = get_config(ds_name)
    config['name'] = f"{ds_name}_S{subject}"
    
    console.print(f"\n🚀 Launching: [bold cyan]{ds_name}[/bold cyan] | Subject {subject}")
    
    # 4. 导入与运行
    from cmsan import train_session, load_unified
    
    print("📥 Loading Data...")
    try:
        X, y = load_unified(ds_name, subject)
        
        # 简单 Split
        key = jax.random.PRNGKey(42)
        k_run, k_model = jax.random.split(key)
        perm = jax.random.permutation(k_run, len(X))
        X, y = X[perm], y[perm]
        split = int(len(X) * 0.8)
        
        train_session(
            X_train=X[:split], y_train=y[:split],
            X_test=X[split:], y_test=y[split:],
            config=config,
            key=k_model
        )
    except Exception as e:
        console.print(f"[bold red]❌ Runtime Error:[/bold red] {e}")
        # 这里你可以选择打印 traceback
        import traceback
        traceback.print_exc()

# ... (inspect_mode 和 main 函数保持不变) ...

def main():
    # ... (同上一个版本) ...
    run_training_mode() # 简化演示

if __name__ == '__main__':
    main()