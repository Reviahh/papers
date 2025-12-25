r"""
CMSAN 训练器 (Equinox)
═══════════════════════════════════════════════════════════════════════════════

使用 Equinox + Optax 实现完全函数式训练，无 for/while/if-else。

## 依赖

    pip install equinox optax

## Equinox 训练范式

Equinox 模型是不可变的 PyTree，训练通过返回新模型实现：

$$
\text{model}_{t+1} = \text{update}(\text{model}_t, \nabla_\theta \mathcal{L})
$$

## 核心函数

- `eqx.filter_grad`: 仅对数组求梯度
- `eqx.filter_jit`: 自动处理静态/动态分离
- `eqx.apply_updates`: 应用梯度更新
- `jax.lax.scan`: 替代 for 循环
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Tuple, Optional, NamedTuple
import pickle

import equinox as eqx
import optax

from .model import CMSAN, batch_forward, batch_predict


# ═══════════════════════════════════════════════════════════════════════════
#                       训练状态 (不可变 NamedTuple)
# ═══════════════════════════════════════════════════════════════════════════

class TrainState(NamedTuple):
    r"""
    不可变训练状态
    
    $$
    \text{State}_t = (\text{model}_t, \text{opt\_state}_t, t, \text{key}_t)
    $$
    """
    model: CMSAN
    opt_state: optax.OptState
    step: int
    key: jax.Array


# ═══════════════════════════════════════════════════════════════════════════
#                       损失函数
# ═══════════════════════════════════════════════════════════════════════════

def cross_entropy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    r"""
    交叉熵损失
    
    $$
    \mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \log \hat{y}_{i, y_i}
    $$
    """
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))


def compute_loss(model: CMSAN, xs: jax.Array, ys: jax.Array) -> jax.Array:
    """计算批量损失"""
    logits = batch_forward(model, xs)
    return cross_entropy(logits, ys)


# ═══════════════════════════════════════════════════════════════════════════
#                       优化器构造
# ═══════════════════════════════════════════════════════════════════════════

def make_optimizer(
    lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    r"""
    创建优化器
    
    $$
    \eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))
    $$
    """
    # 学习率调度
    schedule = (
        optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=lr * 0.01,
        )
        if warmup_steps > 0
        else optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps, alpha=0.01)
    )
    
    # 优化器链
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(schedule, weight_decay=weight_decay),
    )


# ═══════════════════════════════════════════════════════════════════════════
#                       训练步骤 (纯函数)
# ═══════════════════════════════════════════════════════════════════════════

def make_step_fn(optimizer: optax.GradientTransformation):
    r"""
    创建单步训练函数
    
    $$
    (\theta_{t+1}, s_{t+1}) = \text{step}(\theta_t, s_t, x, y)
    $$
    """
    
    @eqx.filter_jit
    def step(state: TrainState, batch: Tuple[jax.Array, jax.Array]) -> Tuple[TrainState, jax.Array]:
        xs, ys = batch
        
        # eqx.filter_grad 只对数组求梯度，忽略静态字段
        loss, grads = eqx.filter_value_and_grad(compute_loss)(state.model, xs, ys)
        
        # optax 更新
        updates, new_opt_state = optimizer.update(
            grads, state.opt_state, eqx.filter(state.model, eqx.is_array)
        )
        
        # eqx.apply_updates 更新模型
        new_model = eqx.apply_updates(state.model, updates)
        
        new_state = TrainState(
            model=new_model,
            opt_state=new_opt_state,
            step=state.step + 1,
            key=state.key,
        )
        
        return new_state, loss
    
    return step


# ═══════════════════════════════════════════════════════════════════════════
#                       Epoch 函数 (使用 lax.scan)
# ═══════════════════════════════════════════════════════════════════════════

def make_epoch_fn(step_fn, batch_size: int, N: int):
    r"""
    创建 epoch 函数 (完全函数式，使用 lax.scan)
    
    $$
    \text{State}_{T} = \text{scan}(\text{step}, \text{State}_0, \text{Batches})
    $$
    """
    n_batches = N // batch_size
    
    @eqx.filter_jit
    def epoch_fn(state: TrainState, xs: jax.Array, ys: jax.Array) -> Tuple[TrainState, jax.Array]:
        # 打乱数据
        key, subkey = random.split(state.key)
        perm = random.permutation(subkey, N)
        xs_shuf = xs[perm]
        ys_shuf = ys[perm]
        state = state._replace(key=key)
        
        # 批次化 (reshape 替代 for 循环)
        xs_batched = xs_shuf[:n_batches * batch_size].reshape(n_batches, batch_size, *xs.shape[1:])
        ys_batched = ys_shuf[:n_batches * batch_size].reshape(n_batches, batch_size)
        
        # lax.scan 替代 for 循环
        final_state, losses = lax.scan(step_fn, state, (xs_batched, ys_batched))
        
        return final_state, jnp.mean(losses)
    
    return epoch_fn


# ═══════════════════════════════════════════════════════════════════════════
#                       完整训练 (使用 lax.fori_loop)
# ═══════════════════════════════════════════════════════════════════════════

def train(
    model: CMSAN,
    xs_train: jax.Array,
    ys_train: jax.Array,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 5e-4,
    warmup_steps: int = 0,
    weight_decay: float = 0.0,
    key: Optional[jax.Array] = None,
) -> Tuple[CMSAN, jax.Array]:
    r"""
    完整训练函数
    
    $$
    \text{model}^* = \arg\min_\theta \mathbb{E}_{(x,y)} \left[ \mathcal{L}(f_\theta(x), y) \right]
    $$
    
    Args:
        model: CMSAN 模型
        xs_train: 训练数据 (N, C, T)
        ys_train: 标签 (N,)
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        warmup_steps: 预热步数
        weight_decay: 权重衰减
        key: 随机密钥
        
    Returns:
        (trained_model, losses)
    """
    N = xs_train.shape[0]
    steps_per_epoch = N // batch_size
    total_steps = epochs * steps_per_epoch
    
    # 创建优化器
    optimizer = make_optimizer(lr, total_steps, warmup_steps, weight_decay)
    
    # 初始状态
    key = key if key is not None else random.key(0)
    state = TrainState(
        model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_array)),
        step=0,
        key=key,
    )
    
    # 创建函数
    step_fn = make_step_fn(optimizer)
    epoch_fn = make_epoch_fn(step_fn, batch_size, N)
    
    # 使用 lax.fori_loop 替代 for 循环
    def body_fn(i, carry):
        state, losses = carry
        new_state, loss = epoch_fn(state, xs_train, ys_train)
        losses = losses.at[i].set(loss)
        return new_state, losses
    
    losses = jnp.zeros(epochs)
    final_state, losses = lax.fori_loop(0, epochs, body_fn, (state, losses))
    
    return final_state.model, losses


# ═══════════════════════════════════════════════════════════════════════════
#                       评估函数
# ═══════════════════════════════════════════════════════════════════════════

@eqx.filter_jit
def evaluate(model: CMSAN, xs: jax.Array, ys: jax.Array) -> jax.Array:
    """评估准确率"""
    preds = batch_predict(model, xs)
    return jnp.mean(preds == ys)


# ═══════════════════════════════════════════════════════════════════════════
#                       保存/加载 (Equinox 序列化)
# ═══════════════════════════════════════════════════════════════════════════

def save_model(model: CMSAN, path: str):
    """保存模型 (使用 eqx.tree_serialise_leaves)"""
    with open(path, 'wb') as f:
        eqx.tree_serialise_leaves(f, model)


def load_model(path: str, model_template: CMSAN) -> CMSAN:
    """加载模型 (需要模板)"""
    with open(path, 'rb') as f:
        return eqx.tree_deserialise_leaves(f, model_template)


# ═══════════════════════════════════════════════════════════════════════════
#                       带回调的训练 (可选打印)
# ═══════════════════════════════════════════════════════════════════════════

def fit(
    model: CMSAN,
    train_data: Tuple[jax.Array, jax.Array],
    val_data: Optional[Tuple[jax.Array, jax.Array]] = None,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 5e-4,
    warmup_steps: int = 0,
    weight_decay: float = 0.0,
    verbose: bool = True,
    log_every: int = 5,
    key: Optional[jax.Array] = None,
) -> CMSAN:
    """
    训练模型 (带日志输出)
    
    注意：此函数使用 Python for 循环以支持日志输出。
    如需完全函数式，请使用 train() 函数。
    """
    xs_train, ys_train = train_data
    N = xs_train.shape[0]
    steps_per_epoch = N // batch_size
    total_steps = epochs * steps_per_epoch
    
    # 创建优化器
    optimizer = make_optimizer(lr, total_steps, warmup_steps, weight_decay)
    
    # 初始状态
    key = key if key is not None else random.key(0)
    state = TrainState(
        model=model,
        opt_state=optimizer.init(eqx.filter(model, eqx.is_array)),
        step=0,
        key=key,
    )
    
    # 创建函数
    step_fn = make_step_fn(optimizer)
    epoch_fn = make_epoch_fn(step_fn, batch_size, N)
    
    # 训练循环 (这里用 Python for 以支持 print)
    # 如需完全函数式，使用 train() 函数
    epoch_range = range(epochs)
    
    # 使用 reduce 风格（虽然这里为了日志还是用 for）
    def train_epoch(state, epoch):
        # 1. 纯计算部分：交给已经 JIT 的 epoch_fn，这是 CPU 跑得快的关键
        new_state, loss = epoch_fn(state, xs_train, ys_train)
        
        # 2. 交互与副作用部分：回归原生 Python if
        # 这里不在 JIT 内部，所以可以自由使用 print, float(), if 等
        if verbose and (epoch % log_every == 0):
            # 将 JAX 数组转为 Python 标量以供格式化打印
            current_loss = float(loss) 
            
            # 计算准确率 (evaluate 本身也是带 jit 的)
            train_acc = float(evaluate(new_state.model, xs_train, ys_train))
            
            log = f"Epoch {epoch:3d} | Loss {current_loss:.4f} | Train {train_acc:.2%}"
            
            if val_data is not None:
                val_acc = float(evaluate(new_state.model, val_data[0], val_data[1]))
                log += f" | Val {val_acc:.2%}"
            
            print(log)
        
        return new_state

    # 外层循环：保持 Python for，方便随时中断和观察
    for epoch in range(epochs):
        state = train_epoch(state, epoch)
    
    return state.model
