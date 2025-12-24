r"""
CorAtt 函数式训练逻辑
═══════════════════════════════════════════════════════════════════════════════

## 数学基础

### 损失函数
交叉熵损失:
$$
\mathcal{L}(\theta; x, y) = -\log\left(\frac{e^{z_y}}{\sum_{k=1}^K e^{z_k}}\right) = -z_y + \log\sum_{k=1}^K e^{z_k}
$$

其中 $z = f_\theta(x) \in \mathbb{R}^K$ 是模型输出的 logits。

### 批量损失
$$
\mathcal{L}_{\text{batch}}(\theta; \mathcal{B}) = \frac{1}{|\mathcal{B}|}\sum_{(x,y) \in \mathcal{B}} \mathcal{L}(\theta; x, y)
$$

### 函数式训练范式

使用高阶函数和 `jax.lax` 原语实现无副作用的训练循环:

1. **单步更新** (纯函数):
   $$f_{\text{step}}: (\theta, s, \mathcal{B}) \mapsto (\theta', s', m)$$

2. **Epoch** (使用 scan):
   $$\text{epoch} = \text{fold}(f_{\text{step}}, \text{batches})$$

3. **训练** (组合):
   $$\text{train} = \text{fold}(\text{epoch}, \text{epochs})$$

### Adam 优化器状态转移

$$
s_{t+1} = \begin{pmatrix} m_t \\ v_t \\ t+1 \end{pmatrix} \quad \text{where} \quad
\begin{cases}
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
\end{cases}
$$
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, lax
from typing import Tuple, Dict, Any, NamedTuple, Callable
from functools import partial, reduce
import pickle

from model import forward, forward_batch, predict_batch
from optim import init_adam, adam, cosine_decay


# ═══════════════════════════════════════════════════════════════════════════
#                       类型定义
# ═══════════════════════════════════════════════════════════════════════════

class TrainState(NamedTuple):
    """
    不可变训练状态
    
    数学表示: $s = (\theta, \text{opt\_state}, \text{key}, t)$
    """
    θ: Dict[str, Any]      # 模型参数 θ
    opt_state: Dict        # Adam 状态 (m, v, t)
    key: jax.Array         # PRNG 密钥
    step: int              # 全局步数 t


# ═══════════════════════════════════════════════════════════════════════════
#                       损失函数 (纯函数)
# ═══════════════════════════════════════════════════════════════════════════

def cross_entropy(logits: jax.Array, label: int) -> jax.Array:
    """
    单样本交叉熵
    
    $$\ell(z, y) = -\log(\text{softmax}(z)_y) = -z_y + \log\sum_k e^{z_k}$$
    
    Args:
        logits: (K,) 模型输出
        label: 整数标签 y ∈ {0, ..., K-1}
        
    Returns:
        标量损失值
    """
    log_probs = jax.nn.log_softmax(logits)
    return -log_probs[label]


def loss_single(θ: Dict, x: jax.Array, y: int, cfg: Dict) -> jax.Array:
    """
    单样本损失
    
    $$\mathcal{L}(\theta; x, y) = \ell(f_\theta(x), y)$$
    """
    logits = forward(θ, x, cfg)
    return cross_entropy(logits, y)


def loss_batch(θ: Dict, xs: jax.Array, ys: jax.Array, cfg: Dict) -> jax.Array:
    """
    批量损失 (使用 vmap 向量化)
    
    $$\mathcal{L}_{\mathcal{B}}(\theta) = \frac{1}{B}\sum_{i=1}^B \mathcal{L}(\theta; x_i, y_i)$$
    """
    # vmap 将单样本函数提升为批量函数
    losses = vmap(loss_single, in_axes=(None, 0, 0, None))(θ, xs, ys, cfg)
    return jnp.mean(losses)


# ═══════════════════════════════════════════════════════════════════════════
#                       训练步 (纯函数)
# ═══════════════════════════════════════════════════════════════════════════

def make_train_step(cfg: Dict, total_steps: int, lr_init: float):
    """
    创建 JIT 编译的训练步函数
    
    数学:
        $$f_{\text{step}}: (s, \mathcal{B}) \mapsto (s', \text{loss})$$
        
    其中:
        $$s' = (\\theta - \\eta_t \cdot \text{Adam}(g_t), s_{\text{opt}}', \text{key}, t+1)$$
        $$g_t = \\nabla_\\theta \mathcal{L}_{\mathcal{B}}(\\theta)$$
    """
    @partial(jit, static_argnums=(3,))
    def train_step(state: TrainState, xs: jax.Array, ys: jax.Array, 
                   cfg_static: Dict) -> Tuple[TrainState, jax.Array]:
        """
        单步训练 (纯函数，无副作用)
        
        Args:
            state: 当前训练状态
            xs: 批量输入 (B, C, T)
            ys: 批量标签 (B,)
            cfg_static: 静态配置
            
        Returns:
            new_state: 更新后的状态
            loss: 当前批次损失
        """
        # 余弦学习率衰减
        lr_t = cosine_decay(state.step, total_steps, lr_init)
        
        # 计算损失和梯度
        loss_val, grads = jax.value_and_grad(loss_batch)(
            state.θ, xs, ys, cfg_static
        )
        
        # Adam 更新
        θ_new, opt_state_new = adam(state.θ, state.opt_state, grads, lr=lr_t)
        
        # 返回新状态
        new_state = TrainState(
            θ=θ_new,
            opt_state=opt_state_new,
            key=state.key,
            step=state.step + 1
        )
        
        return new_state, loss_val
    
    return lambda state, xs, ys: train_step(state, xs, ys, cfg)


# ═══════════════════════════════════════════════════════════════════════════
#                       评估 (纯函数)
# ═══════════════════════════════════════════════════════════════════════════

@partial(jit, static_argnums=(2,))
def accuracy(θ: Dict, xs: jax.Array, cfg: Dict, ys: jax.Array) -> jax.Array:
    """
    准确率 (JIT 编译)
    
    $$\text{acc} = \frac{1}{N}\sum_{i=1}^N \mathbb{1}[\hat{y}_i = y_i]$$
    
    其中 $\hat{y}_i = \arg\max_k f_\theta(x_i)_k$
    """
    preds = predict_batch(θ, xs, cfg)
    return jnp.mean(preds == ys)


def evaluate(θ: Dict, xs: jax.Array, ys: jax.Array, cfg: Dict, 
             batch_size: int = 32) -> jax.Array:
    """
    分批评估 (避免内存溢出，使用 reduce)
    
    $$\text{acc} = \frac{1}{N}\sum_{i=1}^N \mathbb{1}[\hat{y}_i = y_i]$$
    """
    N = xs.shape[0]
    n_batches = (N + batch_size - 1) // batch_size
    
    # 使用 reduce 替代 for 循环
    def accumulate(acc: Tuple[int, int], batch_idx: int) -> Tuple[int, int]:
        start = batch_idx * batch_size
        end = jnp.minimum(start + batch_size, N)
        batch_xs = lax.dynamic_slice(xs, (start, 0, 0), (end - start, xs.shape[1], xs.shape[2]))
        batch_ys = lax.dynamic_slice(ys, (start,), (end - start,))
        preds = predict_batch(θ, batch_xs, cfg)
        correct, total = acc
        return (correct + jnp.sum(preds == batch_ys), total + (end - start))
    
    # 对于小数据集直接计算
    if N <= batch_size:
        return accuracy(θ, xs, cfg, ys)
    
    # 分批累积
    correct, total = reduce(accumulate, range(n_batches), (0, 0))
    return correct / total


# ═══════════════════════════════════════════════════════════════════════════
#                       数据工具 (纯函数)
# ═══════════════════════════════════════════════════════════════════════════

def shuffle(key: jax.Array, xs: jax.Array, ys: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    打乱数据 (纯函数)
    
    使用 Fisher-Yates 置换: $\sigma \sim \text{Uniform}(S_N)$
    """
    N = xs.shape[0]
    perm = random.permutation(key, N)
    return xs[perm], ys[perm]


def make_batches(xs: jax.Array, ys: jax.Array, batch_size: int) -> Tuple[jax.Array, jax.Array]:
    """
    创建批次数组 (纯函数)
    
    将 $(N, ...) \mapsto (B, n, ...)$ 其中 $B = N / n$
    
    Returns:
        xs_batched: (n_batches, batch_size, C, T)
        ys_batched: (n_batches, batch_size)
    """
    N = xs.shape[0]
    n_batches = N // batch_size
    xs_batched = xs[:n_batches * batch_size].reshape(n_batches, batch_size, *xs.shape[1:])
    ys_batched = ys[:n_batches * batch_size].reshape(n_batches, batch_size)
    return xs_batched, ys_batched


# ═══════════════════════════════════════════════════════════════════════════
#                       函数式 Epoch (使用 lax.scan)
# ═══════════════════════════════════════════════════════════════════════════

def make_epoch_fn(train_step_fn: Callable, batch_size: int, N: int):
    """
    创建 epoch 函数 (使用 lax.scan 消除 for 循环)
    
    数学:
        $$\text{epoch}: (s, \mathcal{D}) \mapsto (s', \bar{L})$$
        
    实现:
        $$(s_B, \{L_b\}_{b=1}^B) = \text{scan}(f_{\text{step}}, s_0, \{B_b\}_{b=1}^B)$$
        $$\bar{L} = \frac{1}{B}\sum_{b=1}^B L_b$$
    """
    def epoch_fn(state: TrainState, xs: jax.Array, ys: jax.Array) -> Tuple[TrainState, float]:
        """
        执行一个 epoch
        
        Args:
            state: 训练状态
            xs: 训练数据 (N, C, T)
            ys: 标签 (N,)
            
        Returns:
            new_state: 更新后的状态
            avg_loss: 平均损失
        """
        # 打乱数据
        key, subkey = random.split(state.key)
        xs_shuf, ys_shuf = shuffle(subkey, xs, ys)
        state = state._replace(key=key)
        
        # 创建批次
        xs_batched, ys_batched = make_batches(xs_shuf, ys_shuf, batch_size)
        
        # 定义 scan 的 body 函数
        def scan_body(carry: TrainState, batch: Tuple[jax.Array, jax.Array]) -> Tuple[TrainState, jax.Array]:
            xs_b, ys_b = batch
            new_carry, loss = train_step_fn(carry, xs_b, ys_b)
            return new_carry, loss
        
        # 使用 lax.scan 遍历所有批次
        final_state, losses = lax.scan(
            scan_body,
            state,
            (xs_batched, ys_batched)
        )
        
        avg_loss = jnp.mean(losses)
        return final_state, avg_loss
    
    return epoch_fn


# ═══════════════════════════════════════════════════════════════════════════
#                       完整训练 (函数式)
# ═══════════════════════════════════════════════════════════════════════════

def train(θ: Dict, train_data: Tuple, val_data: Tuple, cfg: Dict,
          epochs: int = 100, batch_size: int = 16, lr: float = 5e-4,
          verbose: bool = True, save_path: str = None) -> Dict:
    """
    完整训练流程 (函数式核心 + 可选日志)
    
    ## 训练算法
    
    输入: 初始参数 $\\theta_0$, 数据集 $\mathcal{D}$, 超参数
    
    For $e = 1, ..., E$:
    1. 打乱数据: $\mathcal{D}_e = \sigma_e(\mathcal{D})$
    2. 划分批次: $\{B_1, ..., B_M\} = \text{partition}(\mathcal{D}_e, n)$
    3. 对每个批次 $B_m$:
       - $g_m = \\nabla_\\theta \mathcal{L}_{B_m}(\\theta)$
       - $\\theta \leftarrow \text{Adam}(\\theta, g_m, \\eta_t)$
    
    输出: 优化后的参数 $\\theta^*$
    
    Args:
        θ: 初始参数
        train_data: (xs, ys) 训练数据
        val_data: (xs, ys) 验证数据
        cfg: 模型配置
        epochs: 训练轮数
        batch_size: 批大小
        lr: 初始学习率
        verbose: 是否打印日志
        save_path: 保存路径
    
    Returns:
        θ: 训练后的参数
    """
    xs_train, ys_train = train_data
    N = xs_train.shape[0]
    steps_per_epoch = N // batch_size
    total_steps = epochs * steps_per_epoch
    
    # 初始化状态
    state = TrainState(
        θ=θ,
        opt_state=init_adam(θ),
        key=random.key(0),
        step=0
    )
    
    # 创建训练函数
    train_step_fn = make_train_step(cfg, total_steps, lr)
    epoch_fn = make_epoch_fn(train_step_fn, batch_size, N)
    
    # JIT 编译 epoch
    epoch_fn_jit = jit(epoch_fn)
    
    best_acc = 0.0
    
    # 使用 reduce 实现训练循环
    def train_epoch(acc: Tuple[TrainState, float], epoch: int) -> Tuple[TrainState, float]:
        state, best_acc = acc
        
        # 执行 epoch
        state, epoch_loss = epoch_fn_jit(state, xs_train, ys_train)
        
        # 日志和评估 (副作用)
        if verbose and epoch % 5 == 0:
            train_acc = evaluate(state.θ, xs_train, ys_train, cfg)
            log = f"Epoch {epoch:3d} | Loss: {epoch_loss:.4f} | Train: {train_acc:.2%}"
            
            if val_data is not None:
                xs_val, ys_val = val_data
                val_acc = evaluate(state.θ, xs_val, ys_val, cfg)
                log += f" | Val: {val_acc:.2%}"
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    if save_path:
                        save(state.θ, save_path)
                        log += " *"
            
            print(log)
        
        return (state, best_acc)
    
    # 执行所有 epochs
    (final_state, _) = reduce(train_epoch, range(epochs), (state, best_acc))
    
    return final_state.θ


# ═══════════════════════════════════════════════════════════════════════════
#                       保存/加载 (纯函数)
# ═══════════════════════════════════════════════════════════════════════════

def save(θ: Dict, path: str) -> None:
    """
    保存参数
    
    将 PyTree 转换为可序列化格式
    """
    data = jax.tree.map(lambda x: x.tolist(), θ)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load(path: str) -> Dict:
    """
    加载参数
    
    从文件恢复 PyTree
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return jax.tree.map(jnp.array, data)