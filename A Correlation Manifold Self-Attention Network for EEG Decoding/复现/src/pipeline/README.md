## CorAtt: Correlation Manifold Self-Attention Network

### 0. 数学符号表 (Mathematical Notation)

| 符号 | 含义 | 维度 |
|------|------|------|
| $x$ | EEG 输入信号 | $\mathbb{R}^{C \times T}$ |
| $h$ | 特征表示 | $\mathbb{R}^{D \times T}$ |
| $C_i$ | 第 $i$ 段的相关矩阵 | $\text{Corr}^{++}_D$ |
| $Q, K, V$ | 查询、键、值矩阵 | $(\text{Corr}^{++}_D)^S$ |
| $R_i$ | 注意力聚合结果 | $\text{Corr}^{++}_D$ |
| $f$ | 展平特征向量 | $\mathbb{R}^{S \cdot D(D-1)/2}$ |
| $\hat{y}$ | 预测概率 | $\Delta^{K-1}$ |

---

### 1. CorAtt 数学工作流 (Mathematical Workflow)

#### 完整前向传播公式

**Step 1: 特征提取 (FEM)**

$$
h = \sigma(W_t * \sigma(W_s \cdot x + b_s) + b_t)
$$

其中：
- $W_s \in \mathbb{R}^{D \times C}$: 空间卷积权重
- $W_t \in \mathbb{R}^{D \times k}$: 时间卷积核
- $\sigma$: ELU 激活函数

**Step 2: 流形映射 (MMM)**

$$
C_i = D^{-1/2} P_i D^{-1/2}, \quad P_i = \frac{1}{T_s - 1} h_i h_i^\top
$$

其中 $h = [h_1, ..., h_S]$ 是时间分段，$D = \text{diag}(\sqrt{\text{diag}(P_i)})$

**Step 3: 李群同态 (Hom)**

$$
\text{hom}(C; A) = \text{Expo}(\text{Off}(M^\top \cdot \text{Logo}(C) \cdot M))
$$

其中：
- $M = \text{Cayley}(A) = (I - S)(I + S)^{-1}, \quad S = A - A^\top$
- $\text{Logo}(C) = \text{Off}(\log C)$
- $\text{Expo}(S) = \exp(S + D^\circ)$, $D^\circ$ 由固定点迭代求解

**Step 4: 流形注意力 (Att)**

$$
\begin{aligned}
d_{ij} &= \|\text{Logo}(Q_i) - \text{Logo}(K_j)\|_F \\
s_{ij} &= \frac{1}{1 + \log(1 + d_{ij})} \\
\alpha_{ij} &= \text{softmax}_j(s_{ij}) \\
R_i &= \text{Expo}\left(\sum_j \alpha_{ij} \cdot \text{Logo}(V_j)\right)
\end{aligned}
$$

**Step 5: 切空间投影 (Proj)**

$$
v_i = \text{tril}(\text{Logo}(R_i)), \quad f = [v_1; \ldots; v_S]
$$

**Step 6: 分类 (Cls)**

$$
\hat{y} = \text{softmax}(W_c \cdot f + b_c)
$$

---

#### 流程图

```mermaid
graph TD
    %% 样式定义
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef process fill:#f3f4f6,stroke:#374151,stroke-width:1px;
    classDef result fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef loss fill:#ffebee,stroke:#c62828,stroke-width:2px,stroke-dasharray: 5 5;

    %% 节点定义
    Start(Input: x ∈ ℝᶜˣᵀ):::input

    %% Step 1
    subgraph S1 ["Step 1: Feature Extraction (FEM)"]
        N1["h = σ(Wₜ * σ(Wₛ · x + bₛ) + bₜ)\nOutput: h ∈ ℝᴰˣᵀ"]:::process
    end

    %% Step 2
    subgraph S2 ["Step 2: Manifold Map (MMM)"]
        N2["h → [h₁, ..., hₛ] (Split)\nCᵢ = D⁻½ · (hᵢhᵢᵀ/T) · D⁻½\nOutput: C ∈ (Corr⁺⁺)ˢ"]:::process
    end

    %% Step 3
    subgraph S3 ["Step 3: Lie Group Homomorphism (Hom)"]
        N3["M = (I - A + Aᵀ)(I + A - Aᵀ)⁻¹ (Cayley)\nQ, K, V = Expo(Off(Mᵀ · Logo(C) · M))\nLogo(C) = Off(log(C))"]:::process
    end

    %% Step 4
    subgraph S4 ["Step 4: Manifold Attention (Att)"]
        N4["dᵢⱼ = ‖Logo(Qᵢ) - Logo(Kⱼ)‖F\nαᵢⱼ = softmax(1 / (1 + log(1 + dᵢⱼ)))\nRᵢ = Expo(Σ αᵢⱼ · Logo(Vⱼ))"]:::process
    end

    %% Step 5
    subgraph S5 ["Step 5: Tangent Projection (Proj)"]
        N5["vᵢ = tril(Logo(Rᵢ))\nf = [v₁; ...; vₛ]\nOutput: f ∈ ℝˢ·ᴰ⁽ᴰ⁻¹⁾/²"]:::process
    end

    %% Step 6
    subgraph S6 ["Step 6: Classification (Cls)"]
        N6["ŷ = softmax(Wc · f + bc)\nOutput: ŷ ∈ Δᴷ⁻¹"]:::result
    end

    Loss("Loss = -Σ yₖ log(ŷₖ)"):::loss

    %% 连接
    Start --> N1
    N1 --> N2
    N2 --> N3
    N3 --> N4
    N4 --> N5
    N5 --> N6
    N6 -.-> Loss

```

---

### 2. 训练工作流 (Training Workflow)

#### 数学描述

**目标函数**

$$
\min_{\theta} \mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)
$$

其中交叉熵损失：

$$
\ell(\hat{y}, y) = -\log(\hat{y}_y) = -z_y + \log\sum_{k=1}^K e^{z_k}
$$

**Adam 优化器**

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) g_t \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \\
\hat{m}_t &= m_t / (1 - \beta_1^t) \\
\hat{v}_t &= v_t / (1 - \beta_2^t) \\
\theta_t &= \theta_{t-1} - \eta_t \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)
\end{aligned}
$$

**余弦学习率衰减**

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{\pi t}{T}\right)\right)
$$

#### 函数式训练范式

训练循环完全函数化，使用 `jax.lax.scan` 替代命令式 for 循环：

$$
(\theta_T, s_T) = \text{scan}(f_{\text{step}}, (\theta_0, s_0), \{(x_b, y_b)\}_{b=1}^B)
$$

其中单步更新函数：

$$
f_{\text{step}}(s, \mathcal{B}) = (s', \text{loss})
$$

---

### 2. 可插拔 Pipeline 设计 (Pluggable Architecture)

这个图结合了横向的 Pipeline 流程和树状的文件结构。为了保持 README 的整洁，我将“模块流”和“代码结构”分为了左右（或上下）逻辑清晰的部分。

```mermaid
graph TB
    %% 样式定义
    classDef module fill:#fff,stroke:#333,stroke-width:2px;
    classDef options fill:#f9f9f9,stroke:#999,stroke-width:1px,stroke-dasharray: 5 5;
    classDef file fill:#e3f2fd,stroke:#1565c0,stroke-width:1px;

    %% 主标题
    Title[==== CorAtt Pluggable Pipeline ====]:::module
    
    %% 流程部分
    subgraph Pipeline [Data Flow Pipeline]
        direction LR
        In((Input)) --> M1[FEM]:::module
        M1 --> M2[MMM]:::module
        M2 --> M3[HOM]:::module
        M3 --> M4[ATT]:::module
        M4 --> M5[PRJ]:::module
        M5 --> M6[CLS]:::module
        M6 --> Out((Output))
    end

    %% 模块选项部分 (使用子图关联)
    subgraph Options [Extensible Modules]
        direction TB
        O1["<b>FEM</b><br>conv, lstm<br>transformer"]:::options
        O2["<b>MMM</b><br>corr, cov<br>gram"]:::options
        O3["<b>HOM</b><br>olm, lsm<br>cayley"]:::options
        O4["<b>ATT</b><br>self, cross<br>sparse"]:::options
        O5["<b>PRJ</b><br>logo, identity<br>log_star"]:::options
        O6["<b>CLS</b><br>linear, mlp<br>svm"]:::options
    end

    %% 强制布局对齐 (通过隐藏线)
    M1 -.- O1
    M2 -.- O2
    M3 -.- O3
    M4 -.- O4
    M5 -.- O5
    M6 -.- O6

    %% 代码结构部分
    subgraph CodeStruct [Project Structure]
        Root[pipeline/]:::file
        Root --> F1[ops.py <br><i>Foundations</i>]:::file
        Root --> F2[manifold.py <br><i>Manifold Ops</i>]:::file
        Root --> F3[pipe.py <br><i>Assembly</i>]:::file
        
        %% 映射具体实现
        F4[fem.py]:::file
        F5[mmm.py]:::file
        F6[hom.py]:::file
        F7[att.py]:::file
        F8[prj.py]:::file
        F9[cls.py]:::file

        Root --> F4 & F5 & F6 & F7 & F8 & F9
    end

    %% 整体布局连接
    Pipeline ~~~ Options
    Options ~~~ CodeStruct

```

```mermaid
graph TD
    %% ================= 样式定义 (Style Definitions) =================
    %% 核心模块样式
    classDef core fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,rx:5,ry:5;
    %% 选项说明样式
    classDef option fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 5 5,color:#616161;
    %% 文件节点样式
    classDef file fill:#fff3e0,stroke:#ef6c00,stroke-width:1px,rx:0,ry:0;
    %% 容器样式
    classDef container fill:#ffffff,stroke:#333,stroke-width:2px;

    %% ================= 第一部分：Pipeline 流程与选项 =================
    subgraph Logic ["🏗️ 可插拔 Pipeline 设计 (Pluggable Architecture)"]
        direction TB
        
        %% 数据流向 (水平)
        subgraph Flow ["数据流 (Data Flow)"]
            direction LR
            Input((x)) --> FEM:::core
            FEM --> MMM:::core
            MMM --> HOM:::core
            HOM --> ATT:::core
            ATT --> PRJ:::core
            PRJ --> CLS:::core
            CLS --> Output((ŷ))
        end

        %% 可替换选项 (垂直挂载)
        %% 使用 Unicode 列表符，避免 HTML 标签
        Opt_FEM["可替换:\n• conv\n• lstm\n• tfm"]:::option
        Opt_MMM["可替换:\n• corr\n• cov\n• gram"]:::option
        Opt_HOM["可替换:\n• olm\n• lsm\n• bw"]:::option
        Opt_ATT["可替换:\n• self\n• cross\n• sparse"]:::option
        Opt_PRJ["可替换:\n• logo\n• log_star\n• identity"]:::option
        Opt_CLS["可替换:\n• linear\n• mlp\n• svm"]:::option

        %% 连接 模块-选项
        FEM -.- Opt_FEM
        MMM -.- Opt_MMM
        HOM -.- Opt_HOM
        ATT -.- Opt_ATT
        PRJ -.- Opt_PRJ
        CLS -.- Opt_CLS
    end

    %% ================= 第二部分：代码组织结构 =================
    subgraph Files ["📂 代码组织 (pipeline/)"]
        direction TB
        
        Root[pipeline/]:::file
        
        %% 基础层
        Root --> Ops["ops.py\n(基础算子)"]:::file
        Root --> Manifold["manifold.py\n(流形运算)"]:::file
        Root --> Pipe["pipe.py\n(管道组装)"]:::file
        
        %% 实现层 (对应上面的模块)
        Root --> F_FEM["fem.py"]:::file
        Root --> F_MMM["mmm.py"]:::file
        Root --> F_HOM["hom.py"]:::file
        Root --> F_ATT["att.py"]:::file
        Root --> F_PRJ["prj.py"]:::file
        Root --> F_CLS["cls.py"]:::file
        
        %% 辅助层
        Root --> Loss["loss.py"]:::file
        Root --> Optim["optim.py"]:::file
    end

    %% ================= 视觉对齐 =================
    %% 让代码结构图位于逻辑图下方
    Logic ~~~ Files

```