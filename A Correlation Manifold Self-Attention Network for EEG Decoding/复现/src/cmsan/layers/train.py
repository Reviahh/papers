r"""
CorAtt 函数式训练脚本
═══════════════════════════════════════════════════════════════════════════════

## 训练工作流数学描述

### 1. 训练目标
给定数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$，最小化经验风险：

$$
\min_{\theta} \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)
$$

其中：
- $f_\theta$: CorAtt 模型（参见 pipe.py）
- $\ell$: 交叉熵损失 $\ell(\hat{y}, y) = -\log(\text{softmax}(\hat{y})_y)$

### 2. Adam 优化器

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= m_t / (1 - \beta_1^t) \\
\hat{v}_t &= v_t / (1 - \beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta_t \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
\end{aligned}
$$

### 3. 余弦学习率衰减

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

### 4. 函数式训练范式

训练循环使用 `jax.lax.scan` 实现，消除命令式 for 循环：

$$
(\theta_T, s_T) = \text{scan}(f_{\text{step}}, (\theta_0, s_0), \{(x_b, y_b)\}_{b=1}^B)
$$

用法:
    python train.py --config default
    python train.py --config ablation_euclidean
"""

import jax
import jax.numpy as jnp
from jax import random, jit, lax
from typing import Tuple, Dict, Any, NamedTuple, Callable
from functools import partial
import pickle
import argparse

from .pipe import build
from optim import cosine_lr


# ═══════════════════════════════════════════════════════════════════════════
#                       类型定义
# ═══════════════════════════════════════════════════════════════════════════

class TrainState(NamedTuple):
    """训练状态（不可变）"""
    θ: Dict[str, Any]      # 模型参数
    opt_state: Dict        # 优化器状态
    key: jax.Array         # 随机数密钥
    step: int              # 当前步数


class TrainConfig(NamedTuple):
    """训练配置"""
    epochs: int
    batch_size: int
    lr_max: float
    lr_min: float = 1e-6
    log_every: int = 5


# ═══════════════════════════════════════════════════════════════════════════
#                       函数式训练核心
# ═══════════════════════════════════════════════════════════════════════════

def make_step_fn(pipe, total_steps: int, lr_max: float, lr_min: float = 1e-6):
    """
    创建单步训练函数
    
    数学:
        $$g = \nabla_\theta \mathcal{L}(\theta; x, y)$$
        $$\theta' = \text{Adam}(\theta, g, \eta_t)$$
    
    返回: (state, batch) → (state', metrics)
    """
    @jit
    def step_fn(state: TrainState, batch: Tuple[jax.Array, jax.Array]) -> Tuple[TrainState, Dict]:
        xs, ys = batch
        
        # 学习率调度: η_t = η_min + 0.5(η_max - η_min)(1 + cos(πt/T))
        lr_t = cosine_lr(state.step, total_steps, lr_max, lr_min)
        
        # 计算损失和梯度: ∇_θ L(θ; x, y)
        loss, grads = jax.value_and_grad(pipe.loss_batch)(state.θ, xs, ys)
        
        # Adam 更新
        θ_new, opt_state_new = pipe.opt_fn(state.θ, state.opt_state, grads, lr=lr_t)
        
        # 返回新状态
        new_state = TrainState(
            θ=θ_new,
            opt_state=opt_state_new,
            key=state.key,
            step=state.step + 1
        )
        
        metrics = {'loss': loss, 'lr': lr_t}
        
        return new_state, metrics
    
    return step_fn


def make_epoch_fn(step_fn, batch_size: int, N: int):
    """
    创建 epoch 函数 (使用 lax.scan，无 for 循环)
    
    数学:
        $$(s_B, \{m_b\}_{b=1}^B) = \text{scan}(f_{\text{step}}, s_0, \{B_b\}_{b=1}^B)$$
    
    其中 $B_b = (x_{[b \cdot n : (b+1) \cdot n]}, y_{[b \cdot n : (b+1) \cdot n]})$
    """
    n_batches = N // batch_size
    
    def epoch_fn(state: TrainState, xs: jax.Array, ys: jax.Array) -> Tuple[TrainState, Dict]:
        # 打乱数据
        key, subkey = random.split(state.key)
        perm = random.permutation(subkey, N)
        xs_shuf = xs[perm]
        ys_shuf = ys[perm]
        
        # 重塑为批次: (N, ...) → (n_batches, batch_size, ...)
        xs_batched = xs_shuf[:n_batches * batch_size].reshape(n_batches, batch_size, *xs.shape[1:])
        ys_batched = ys_shuf[:n_batches * batch_size].reshape(n_batches, batch_size)
        
        # 更新状态中的 key
        state = state._replace(key=key)
        
        # 使用 lax.scan 遍历所有批次
        # scan: (carry, inputs) → (carry', outputs)
        final_state, all_metrics = lax.scan(
            step_fn,
            state,
            (xs_batched, ys_batched)
        )
        
        # 聚合指标
        epoch_metrics = {
            'loss': jnp.mean(all_metrics['loss']),
            'lr': all_metrics['lr'][-1],
        }
        
        return final_state, epoch_metrics
    
    return epoch_fn


def make_train_fn(pipe, train_cfg: TrainConfig, N: int):
    """
    创建完整训练函数 (全函数式)
    
    训练流程:
        $$\theta^* = \arg\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}}[\ell(f_\theta(x), y)]$$
    
    使用 lax.scan 实现 epoch 循环，消除所有命令式循环
    """
    steps_per_epoch = N // train_cfg.batch_size
    total_steps = train_cfg.epochs * steps_per_epoch
    
    step_fn = make_step_fn(pipe, total_steps, train_cfg.lr_max, train_cfg.lr_min)
    epoch_fn = make_epoch_fn(step_fn, train_cfg.batch_size, N)
    
    def train_fn(θ, xs, ys, key):
        """
        Args:
            θ: 初始参数
            xs: 训练数据 (N, C, T)
            ys: 标签 (N,)
            key: 随机数密钥
            
        Returns:
            θ: 训练后的参数
            history: 训练历史
        """
        # 初始化状态
        state = TrainState(
            θ=θ,
            opt_state=pipe.opt_init(θ),
            key=key,
            step=0
        )
        
        # 使用 lax.fori_loop 实现 epoch 循环
        def epoch_body(epoch: int, carry: Tuple):
            state, history = carry
            state, metrics = epoch_fn(state, xs, ys)
            # 追加历史记录
            history = jax.tree.map(
                lambda h, m: h.at[epoch].set(m),
                history, metrics
            )
            return (state, history)
        
        # 初始化历史记录
        history = {
            'loss': jnp.zeros(train_cfg.epochs),
            'lr': jnp.zeros(train_cfg.epochs),
        }
        
        # 执行所有 epochs
        final_state, final_history = lax.fori_loop(
            0, train_cfg.epochs,
            epoch_body,
            (state, history)
        )
        
        return final_state.θ, final_history
    
    return train_fn


# ═══════════════════════════════════════════════════════════════════════════
#                       带日志的训练函数
# ═══════════════════════════════════════════════════════════════════════════

def train_with_logging(pipe, θ, train_data, val_data, cfg: TrainConfig):
    """
    带日志输出的训练函数 (外层 Python 循环用于日志，内层使用 JAX)
    
    这是 `make_train_fn` 的用户友好版本，支持：
    - 验证集评估
    - 日志输出
    - 提前停止（可扩展）
    
    数学:
        每个 epoch 执行:
        $$\theta_{e+1}, m_e = \text{epoch\_fn}(\theta_e, \mathcal{D}_{\text{train}})$$
        
        评估:
        $$\text{acc} = \frac{1}{|\mathcal{D}|}\sum_{(x,y) \in \mathcal{D}} \mathbb{1}[\arg\max f_\theta(x) = y]$$
    """
    xs, ys = train_data
    N = xs.shape[0]
    steps_per_epoch = N // cfg.batch_size
    total_steps = cfg.epochs * steps_per_epoch
    
    # 创建函数
    step_fn = make_step_fn(pipe, total_steps, cfg.lr_max, cfg.lr_min)
    epoch_fn = jit(make_epoch_fn(step_fn, cfg.batch_size, N))
    
    # 初始化
    state = TrainState(
        θ=θ,
        opt_state=pipe.opt_init(θ),
        key=random.key(0),
        step=0
    )
    
    # 训练循环 (使用 lax.fori_loop 或 scan 可完全函数化，这里保留 for 用于日志)
    # 注意: 核心计算已函数化，for 仅用于副作用（打印日志）
    history = []
    
    # 使用 fold 思想：将训练视为状态变换的组合
    # train = epoch ∘ epoch ∘ ... ∘ epoch (epochs 次)
    def log_epoch(epoch: int, state: TrainState, metrics: Dict):
        """日志记录（纯副作用）"""
        if epoch % cfg.log_every == 0:
            train_acc = pipe.accuracy(state.θ, xs, ys)
            val_acc = pipe.accuracy(state.θ, val_data[0], val_data[1]) if val_data else 0
            print(f"Epoch {epoch:3d} | Loss {metrics['loss']:.4f} | "
                  f"LR {metrics['lr']:.2e} | Train {train_acc:.2%} | Val {val_acc:.2%}")
        return {'epoch': epoch, **metrics}
    
    # 执行训练 - 使用 reduce 模式
    def train_one_epoch(state_acc: Tuple[TrainState, list], epoch: int) -> Tuple[TrainState, list]:
        state, acc = state_acc
        state, metrics = epoch_fn(state, xs, ys)
        record = log_epoch(epoch, state, metrics)
        return (state, acc + [record])
    
    # 使用 Python reduce (等价于 lax.fori_loop，但支持日志)
    from functools import reduce
    epochs_range = range(cfg.epochs)
    (final_state, history) = reduce(train_one_epoch, epochs_range, (state, []))
    
    return final_state.θ, history


# ═══════════════════════════════════════════════════════════════════════════
#                       配置
# ═══════════════════════════════════════════════════════════════════════════

CONFIGS = {
    'default': {
        'fem': 'conv', 'mmm': 'corr', 'hom': 'olm',
        'att': 'manifold', 'prj': 'tangent', 'cls': 'linear',
    },
    'ablation_euclidean': {
        'fem': 'conv', 'mmm': 'corr', 'hom': 'olm',
        'att': 'euclidean', 'prj': 'tangent', 'cls': 'linear',
    },
    'ablation_no_hom': {
        'fem': 'conv', 'mmm': 'corr', 'hom': 'identity',
        'att': 'manifold', 'prj': 'tangent', 'cls': 'linear',
    },
    'light': {
        'fem': 'linear', 'mmm': 'corr', 'hom': 'olm',
        'att': 'manifold', 'prj': 'tangent', 'cls': 'linear',
    },
}

DATA_CFG = {
    'bcic': {'C': 22, 'T': 438, 'D': 20, 'S': 3, 'K': 4},
    'test': {'C': 8, 'T': 128, 'D': 10, 'S': 2, 'K': 4},
}


# ═══════════════════════════════════════════════════════════════════════════
#                       主程序
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='default', choices=CONFIGS.keys())
    parser.add_argument('--data', default='test', choices=DATA_CFG.keys())
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    args = parser.parse_args()
    
    # 构建 Pipeline
    cfg = CONFIGS[args.config]
    data_cfg = DATA_CFG[args.data]
    
    print("═" * 60)
    print("CorAtt 函数式训练")
    print("═" * 60)
    print(f"Config: {args.config}")
    print(f"Data: {args.data} {data_cfg}")
    print(f"Pipeline: {' → '.join(cfg.values())}")
    print()
    
    pipe = build(cfg)
    
    # 初始化
    key = random.key(42)
    k1, k2, k3 = random.split(key, 3)
    
    θ = pipe.init(k1, **data_cfg)
    
    # 假数据
    xs = random.normal(k2, (64, data_cfg['C'], data_cfg['T']))
    ys = random.randint(k3, (64,), 0, data_cfg['K'])
    
    # 训练配置
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        lr_max=args.lr,
    )
    
    # 训练
    θ, history = train_with_logging(
        pipe, θ,
        train_data=(xs[:48], ys[:48]),
        val_data=(xs[48:], ys[48:]),
        cfg=train_cfg,
    )
    
    # 保存
    with open('model.pkl', 'wb') as f:
        pickle.dump(jax.tree.map(lambda x: x.tolist(), θ), f)
    print("\n✓ Saved to model.pkl")


if __name__ == '__main__':
    main()