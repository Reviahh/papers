r"""
CorAtt 模型 - 相关矩阵流形上的注意力网络
═══════════════════════════════════════════════════════════════════════════════

## 完整前向传播公式

### Step 1: 特征提取
$$
h = \sigma(W_t * \sigma(W_s \cdot x + b_s) + b_t)
$$

### Step 2: 流形映射
$$
h \to [h_1, \ldots, h_S], \quad C_i = D_i^{-1/2} (h_i h_i^\top / T_s) D_i^{-1/2}
$$

### Step 3: 李群同态
$$
\text{hom}(C; A) = \text{Expo}(\text{Off}(M^\top \cdot \text{Logo}(C) \cdot M))
$$
$$
Q, K, V = \text{hom}(C; A_Q), \text{hom}(C; A_K), \text{hom}(C; A_V)
$$

### Step 4: 流形注意力
$$
d_{ij} = \|\text{Logo}(Q_i) - \text{Logo}(K_j)\|_F
$$
$$
\alpha_{ij} = \text{softmax}_j\left(\frac{1}{1 + \log(1 + d_{ij})}\right)
$$
$$
R_i = \text{Expo}\left(\sum_j \alpha_{ij} \cdot \text{Logo}(V_j)\right)
$$

### Step 5: 切空间投影
$$
f = \text{concat}(\text{tril}(\text{Logo}(R_1)), \ldots, \text{tril}(\text{Logo}(R_S)))
$$

### Step 6: 分类
$$
\hat{y} = \text{softmax}(W_c \cdot f + b_c)
$$
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from functools import partial

from manifold import corr, logo, expo, dist, wfm, cayley, vec_tril, tril_dim, off


# ═══════════════════════════════════════════════════════════════════════════
#                              参数初始化
# ═══════════════════════════════════════════════════════════════════════════

def init(key, cfg):
    """
    初始化参数 θ
    
    cfg: dict with keys
        - C: 输入通道数 (EEG电极)
        - T: 时间步数
        - D: 特征维度
        - S: 分段数
        - K: 类别数
        - kernel: 时间卷积核大小
    """
    C, D, S, K = cfg['C'], cfg['D'], cfg['S'], cfg['K']
    kernel = cfg.get('kernel', 25)
    
    keys = random.split(key, 6)
    
    # 特征维度
    feat_dim = tril_dim(D) * S
    
    # He 初始化
    he_spatial = jnp.sqrt(2.0 / C)
    he_temporal = jnp.sqrt(2.0 / (D * kernel))
    he_fc = jnp.sqrt(2.0 / feat_dim)
    
    return {
        # 空间卷积: (C,) → (D,)
        'Ws': random.normal(keys[0], (D, C)) * he_spatial,
        'bs': jnp.zeros(D),
        
        # 时间卷积: (D, T) → (D, T)
        'Wt': random.normal(keys[1], (D, kernel)) * he_temporal,
        'bt': jnp.zeros(D),
        
        # 注意力: 李群同态参数
        'Aq': random.normal(keys[2], (D, D)) * 0.01,
        'Ak': random.normal(keys[3], (D, D)) * 0.01,
        'Av': random.normal(keys[4], (D, D)) * 0.01,
        
        # 分类器
        'Wc': random.normal(keys[5], (K, feat_dim)) * he_fc,
        'bc': jnp.zeros(K),
    }


# ═══════════════════════════════════════════════════════════════════════════
#                              核心层
# ═══════════════════════════════════════════════════════════════════════════

def spatial_conv(x, W, b):
    """
    空间卷积 (跨通道线性组合)
    x: (C, T) → (D, T)
    """
    return W @ x + b[:, None]


def temporal_conv(x, W, b, stride=1):
    """
    时间卷积 (逐通道 1D 卷积)
    x: (D, T) → (D, T)
    W: (D, kernel)
    """
    D, T = x.shape
    kernel = W.shape[1]
    pad = kernel // 2
    
    # 手动 padding
    x_pad = jnp.pad(x, ((0, 0), (pad, pad)), mode='edge')
    
    # 滑动窗口卷积 (用 vmap 实现)
    def conv_at(t):
        window = x_pad[:, t:t+kernel]  # (D, kernel)
        return jnp.sum(W * window, axis=1)  # (D,)
    
    out = vmap(conv_at)(jnp.arange(T))  # (T, D)
    return out.T + b[:, None]  # (D, T)


def hom(C, A):
    """
    李群同态
    hom(C) = Expo(Off(Mᵀ Logo(C) M))
    
    M = Cayley(A) 是正交矩阵
    """
    M = cayley(A)
    L = logo(C)
    return expo(off(M.T @ L @ M))


def attention_scores(Qs, Ks):
    """
    计算注意力分数矩阵
    
    Qs, Ks: (S, D, D)
    returns: (S, S) 分数矩阵
    
    sᵢⱼ = 1 / (1 + log(1 + d(Qᵢ, Kⱼ)))
    """
    S = Qs.shape[0]
    
    # 双重 vmap 计算所有 pair 的距离
    def dist_row(Qi):
        return vmap(lambda Kj: dist(Qi, Kj))(Ks)
    
    dists = vmap(dist_row)(Qs)  # (S, S)
    
    # 转换为分数
    scores = 1.0 / (1.0 + jnp.log(1.0 + dists))
    
    return scores


def attention(Cs, θ):
    """
    相关矩阵注意力 (向量化版本)
    
    Cs: (S, D, D) 相关矩阵
    returns: (S, D, D) 精炼后的相关矩阵
    """
    # 批量李群同态
    Qs = vmap(hom, in_axes=(0, None))(Cs, θ['Aq'])
    Ks = vmap(hom, in_axes=(0, None))(Cs, θ['Ak'])
    Vs = vmap(hom, in_axes=(0, None))(Cs, θ['Av'])
    
    # 注意力分数
    scores = attention_scores(Qs, Ks)  # (S, S)
    
    # Softmax
    weights = jax.nn.softmax(scores, axis=1)  # (S, S)
    
    # 对每个位置计算 WFM
    def compute_wfm_i(ws_i):
        return wfm(ws_i, Vs)
    
    Rs = vmap(compute_wfm_i)(weights)  # (S, D, D)
    
    return Rs


# ═══════════════════════════════════════════════════════════════════════════
#                              前向传播
# ═══════════════════════════════════════════════════════════════════════════

def forward(θ, x, cfg):
    """
    前向传播
    
    θ: 参数字典
    x: (C, T) EEG 信号
    cfg: 配置
    
    returns: (K,) logits
    """
    S = cfg['S']
    
    # ─── 1. 特征提取 ───
    h = spatial_conv(x, θ['Ws'], θ['bs'])      # (D, T)
    h = jax.nn.elu(h)
    h = temporal_conv(h, θ['Wt'], θ['bt'])     # (D, T)
    h = jax.nn.elu(h)
    
    # ─── 2. 分段 + 相关矩阵 ───
    T = h.shape[1]
    seg_len = T // S
    
    # 切分并计算相关矩阵
    def segment_to_corr(i):
        seg = jax.lax.dynamic_slice(h, (0, i * seg_len), (h.shape[0], seg_len))
        return corr(seg)
    
    Cs = vmap(segment_to_corr)(jnp.arange(S))  # (S, D, D)
    
    # ─── 3. 注意力 ───
    Rs = attention(Cs, θ)  # (S, D, D)
    
    # ─── 4. 切空间投影 ───
    log_Rs = vmap(logo)(Rs)          # (S, D, D)
    vecs = vmap(vec_tril)(log_Rs)    # (S, D*(D-1)/2)
    features = vecs.flatten()        # (S * D*(D-1)/2,)
    
    # ─── 5. 分类 ───
    logits = θ['Wc'] @ features + θ['bc']
    
    return logits


# ═══════════════════════════════════════════════════════════════════════════
#                              批量版本
# ═══════════════════════════════════════════════════════════════════════════

def forward_batch(θ, xs, cfg):
    """批量前向"""
    return vmap(forward, in_axes=(None, 0, None))(θ, xs, cfg)


def predict(θ, x, cfg):
    """预测类别"""
    return jnp.argmax(forward(θ, x, cfg))


def predict_batch(θ, xs, cfg):
    """批量预测"""
    return vmap(predict, in_axes=(None, 0, None))(θ, xs, cfg)