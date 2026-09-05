# 基于深度学习的屏幕渲染异常检测

本项目实现了一套完整的屏幕缺陷检测系统，涵盖从像素级语义分割到多类别快速分类的完整技术栈。

---

## 目录

- [项目结构](#项目结构)
- [核心网络架构](#核心网络架构)
  - [UNet & UNetEncoder](#unet--unetencoder)
  - [UnetDefeat (元任务模型)](#unetdefeat-元任务模型)
  - [Gate (逻辑门网络)](#gate-逻辑门网络)
  - [ClassifyNet (多分类网络)](#classifynet-多分类网络)
  - [ResNet50 (蒸馏教师模型)](#resnet50-蒸馏教师模型)
  - [LightRoutedNet (轻量路由网络)](#lightroutednet-轻量路由网络)
- [损失函数](#损失函数)
- [训练范式](#训练范式)
  - [元任务训练 (Meta-Tasking)](#元任务训练-meta-tasking)
  - [知识蒸馏 (Knowledge Distillation)](#知识蒸馏-knowledge-distillation)
  - [强化学习监督微调 (PPO)](#强化学习监督微调-ppo)
  - [轻量路由训练 (LightRoutedNet)](#轻量路由训练-lightroutednet)
- [数据集生成](#数据集生成)
- [数据集结构](#数据集结构)
- [快速开始](#快速开始)

---

## 项目结构

```
ScreenAnomalyDetection/
├── dataset/                          # 数据集目录
│   ├── images/screen_images/          # 正常屏幕截图 (38 images)
│   ├── masks/defeat_items{1-4}/       # 4类缺陷遮罩模板
│   └── mixtures/                     # 混合缺陷样本图
├── model/                            # 网络模型定义
│   ├── UNet.py                       # UNet分割网络 + UNetEncoder编码器
│   ├── DefeatNet.py                  # UnetDefeat + Gate + FocalLoss
│   ├── ClassifyNet.py                # 多分类网络 (UNetEncoder + SPP)
│   ├── ResNet.py                     # ResNet50 (知识蒸馏教师)
│   └── LightRoutedNet.py             # 轻量级路由网络
├── dataset_utils/                    # 数据集工具包
│   ├── dataset_generator.py           # SplitDataset / MixtureSplitDataset
│   ├── dataset_utils.py               # 图像增强与叠加工具
│   ├── unified_dataset.py             # 统一数据集类
│   └── rl_utils.py                   # 强化学习工具
├── dataset_config.py                 # 统一路径配置
├── DefeatMain.py                     # 主程序 (整屏推理入口)
├── experiment_light_routed.py         # LightRoutedNet 实验脚本
├── experiment_complete.py            # 完整对比实验脚本
├── benchmark_light.py                 # 轻量模型基准测试
├── migrate_dataset.py                # 数据集迁移工具
├── net_unet_workbench.ipynb          # 元任务模型训练 Notebook
├── net_distill_workbench.ipynb       # 知识蒸馏训练 Notebook
├── net_classify_workbench.ipynb      # 多分类训练 Notebook
├── ppo_classify_workbench.ipynb      # PPO 强化学习微调 Notebook
├── net_light_routed_workbench.ipynb   # LightRoutedNet 训练 Notebook
├── experiment_summary.ipynb           # 实验结果分析 Notebook
├── requirements.txt                   # Python 依赖
└── README.md                         # 本文档
```

---

## 核心网络架构

### UNet & UNetEncoder

UNet 是本项目的核心骨干网络，采用经典的编码器-解码器（Encoder-Decoder）结构，包含 5 层卷积块。

**结构图示：**

```
输入图像 (3×H×W)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  编码器 (Encoder)                                       │
│                                                         │
│  Conv1: 3→32    ──────────────────┐                     │
│      ▼                            │                     │
│  MaxPool ↓                        │                     │
│      ▼                            │                     │
│  Conv2: 32→64   ──────────────────│─────────────────┐   │
│      ▼                            │                 │   │
│  MaxPool ↓                        │                 │   │
│      ▼                            │                 │   │
│  Conv3: 64→128  ──────────────────│─────────────────│─┐ │
│      ▼                            │                 │ │ │
│  MaxPool ↓                        │                 │ │ │
│      ▼                            │                 │ │ │
│  Conv4: 128→256 ──────────────────│─────────────────│─│─┤
│      ▼                            │                 │ │ │
│  MaxPool ↓                        │                 │ │ │
│      ▼                            │                 │ │ │
│  Conv5: 256→512 (瓶颈层)         │                 │ │ │
│      = dfeature =                 │                 │ │ │
│      = (B,512,H/32,W/32) =       │                 │ │ │
└──────┼────────────────────────────┴─────────────────┴─┴─┘
       │
       │  UNetEncoder 输出: (B,512,H/32,W/32)
       │  UNet 继续解码...
       ▼
┌─────────────────────────────────────────────────────────┐
│  解码器 (Decoder)                                       │
│                                                         │
│  Up4: 512→256 + concat(x4) → Conv: 512→256            │
│      ▼                                                  │
│  Up3: 256→128 + concat(x3) → Conv: 256→128            │
│      ▼                                                  │
│  Up2: 128→64  + concat(x2) → Conv: 128→64             │
│      ▼                                                  │
│  Up1: 64→32   + concat(x1) → Conv: 64→32              │
│      ▼                                                  │
│  Conv_1x1: 32→1  (输出掩码)                            │
└─────────────────────────────────────────────────────────┘
    │
    ▼
输出掩码 (1×H×W)
```

**核心组件：**

| 组件 | 说明 |
|------|------|
| `LightConvBlock` | 两层 3×3 卷积 + BatchNorm + ReLU |
| `LightUpConv` | 双线性上采样 2× + 3×3 卷积 + BatchNorm + ReLU |
| `UNetEncoder` | 仅保留编码器部分，输出 512 通道特征图，供 Gate/ClassifyNet/LightRoutedNet 使用 |
| `dfeature` 属性 | 前向传播时保存瓶颈层输出 (B,512,H/32,W/32) |

**参数量估算：**
- 编码器: ~0.5M 参数
- 完整 UNet: ~4.5M 参数

---

### UnetDefeat (元任务模型)

`UnetDefeat` 是元任务训练模式下的完整模型，将 `UNet` 封装为一个可端到端训练的分割网络。

```
UnetDefeat.forward(defeat_image)
    │
    ▼
UNet(defeat_image) ──────┐
    │                     │
    ├─→ 输出掩码 y_hat    │  ← 1×H×W 像素级掩码
    │   (B,1,H,W)         │
    │                     │
    └─→ get_feature()    │  ← 保存到 dfeature
        (B,512,H/32,W/32) │
                          │
                          ▼
                    返回: (y_hat, dfeature)
```

**损失函数：** 使用 `FocalLoss`，详情见[损失函数](#损失函数)章节。

---

### Gate (逻辑门网络)

Gate 是元任务模型的第二部分，将 UNet 编码器输出的深层语义特征转换为二分类逻辑判断。

```
dfeature: (B,512,H/32,W/32)
    │
    ▼
AdaptiveAvgPool2d((1,1)) ──→ (B,512,1,1)
    │
    ▼
Flatten ──────────────────→ (B,512)
    │
    ▼
Linear(512→1024) + ReLU
    │
    ▼
Linear(1024→1024) + ReLU
    │
    ▼
Linear(1024→512) + ReLU
    │
    ▼
Linear(512→128) + ReLU ──→ 中间特征 midd_f
    │
    ▼
Linear(128→1) + Sigmoid ──→ 逻辑概率 p_hat (B,1)
```

**设计目的：** 编码器特征（512通道）维度高、语义抽象，直接分类效果差。Gate 通过多层全连接层将高维特征压缩为低维逻辑空间，再输出二分类概率。

**参数量：** ~0.8M 参数

---

### ClassifyNet (多分类网络)

ClassifyNet 是路由模式下的多分类网络，使用 UNetEncoder 作为特征提取器，并引入 SPP（Spatial Pyramid Pooling）多尺度池化来增强空间感知能力。

```
输入图像 ──→ UNetEncoder ──→ (B,512,8,8)
                                 │
              ┌──────────────────┴──────────────────┐
              ▼                                      ▼
    AdaptiveAvgPool2d((1,1))              AdaptiveAvgPool2d((2,2))
              │                                      │
              ▼                                      ▼
        flatten → (512)                   flatten → (512×4)
              │                                      │
              └──────────────────┬──────────────────┘
                                  ▼
                        AdaptiveAvgPool2d((4,4))
                                  │
                                  ▼
                          flatten → (512×16)
                                  │
                                  ▼
                        Concat ──→ (512×21) = 10752
                                  │
                                  ▼
                          Linear(10752→512) + ReLU
                                  │
                                  ▼
                          Linear(512→128) + ReLU
                                  │
                                  ▼
                          Linear(128→5) ──→ 5类输出 logits
                          (Normal, Defect-1~4)
```

**SPP 的作用：** 标准分类器对输入图像尺寸有严格限制。SPP 将不同尺寸的特征图聚合为固定维度的多尺度表征，增强模型对目标大小变化的鲁棒性。

**参数量：** ~5.5M 参数

---

### ResNet50 (蒸馏教师模型)

ResNet50 是知识蒸馏中的"教师模型"，用于将元任务模型（UNet+Gate）的知识迁移到更轻量的 ResNet50 学生模型中。

```
输入图像 (3×H×W)
    │
    ▼
Conv7×7 (stride=2) → BN → ReLU → MaxPool3×3 (stride=2)
    │
    ▼
Layer1: Bottleneck×3  (64 通道, 输出 H/4 × W/4)
    │
    ▼
Layer2: Bottleneck×4  (128 通道, stride=2, 输出 H/8 × W/8)
    │
    ▼
Layer3: Bottleneck×6  (256 通道, stride=2, 输出 H/16 × W/16)
    │
    ▼
Layer4: Bottleneck×3  (512 通道, stride=2, 输出 H/32 × W/32)
    │
    ▼
AdaptiveAvgPool2d((1,1)) → Flatten → Linear(2048→num_classes)
```

**Bottleneck 结构：** `Conv1×1(降维) → Conv3×3 → Conv1×1(升维) + 残差连接`

**配置：** ResNet50 = [3, 4, 6, 3] 个 Bottleneck block

**参数量：** ~23.5M 参数（教师）

---

### LightRoutedNet (轻量路由网络)

LightRoutedNet 是最新设计的轻量级路由网络，核心思想是**复用冻结的 UNet 编码器作为特征提取器，仅训练极轻量的分类/路由头**，实现极速推理。

```
输入图像 ──→ UNetEncoder(冻结) ──→ (B,512,H/32,W/32)
                                           │
                    ┌───────────────────────┴───────────────────────┐
                    ▼                                               ▼
          LightRouteHead (路由头)                        LightClassifier (分类头)
          输出K类存在概率                                输出多分类logits
          (B,K) sigmoid                                  (B,K+1) softmax
                    │                                               │
                    └───────────────────────┬───────────────────────┘
                                            ▼
                                      融合预测
                                            │
                                            ▼
                                      输出 (B,K+1)
```

**LightRouteHead (仅 ~0.05M 参数)：**
```
(512,H/32,W/32)
    │
    ▼
AdaptiveAvgPool2d((1,1)) → Flatten → (512)
    │
    ▼
Linear(512→64) + ReLU
    │
    ▼
Linear(64→K) ──→ Sigmoid ──→ (B,K) 每类存在概率
```

**LightClassifier (仅 ~0.1M 参数)：**
```
(512,H/32,W/32)
    │
    ▼
AdaptiveAvgPool2d((1,1)) → Flatten → (512)
    │
    ▼
Linear(512→128) + ReLU + Dropout(0.2)
    │
    ▼
Linear(128→K+1) ──→ (B,K+1) 多分类 logits
```

**总参数量：** 4.91M（编码器 4.71M 冻结 + 路由头 0.20M 可训练）

**推理融合策略：**

```python
# 分类概率 × 路由概率 → 融合
defect_probs = class_probs[:, 1:]          # (B,K) 缺陷类概率
fused_probs = defect_probs * route_probs     # (B,K) 路由加权
predictions = concat([normal_score, fused_probs])
predictions = predictions / predictions.sum() # 归一化
```

---

## 损失函数

### Focal Loss

用于 UnetDefeat 的像素级缺陷分割，解决正负样本严重不平衡的问题。

$$\mathcal{L}_{focal} = -\alpha \cdot (1 - p_t)^\gamma \cdot \log(p_t)$$

其中 $p_t$ 是模型对真实标签的预测概率：

- 当样本为正类（缺陷像素）时：$p_t = p$
- 当样本为负类（正常像素）时：$p_t = 1 - p$

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| $\alpha$ | 0.25 | 正样本权重系数 |
| $\gamma$ | 2.0 | 难易样本调节因子。$\gamma$ 越大，越关注困难样本 |

**代码实现（`DefeatNet.py`）：**

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()
```

**设计动机：** 屏幕图像中正常区域远多于缺陷区域。标准 BCE Loss 会被大量负样本主导，Focal Loss 通过 $(1-p_t)^\gamma$ 降低易分类样本的权重，让模型更关注难以分类的缺陷边界像素。

### Cross Entropy Loss

用于分类任务（Gate 二分类、ClassifyNet 多分类、LightRoutedNet 多分类）：

$$\mathcal{L}_{CE} = -\sum_{i} y_i \cdot \log(\hat{y}_i)$$

PyTorch 实现：`nn.CrossEntropyLoss()`（内部包含 LogSoftmax + NLLLoss）

### KL Divergence（蒸馏损失）

用于知识蒸馏，让 ResNet50 学生模型学习 UNet+Gate 教师模型的软标签分布：

$$\mathcal{L}_{KD} = T^2 \cdot \text{KL}\left( \text{Softmax}\left(\frac{z_s}{T}\right) \| \text{Softmax}\left(\frac{z_t}{T}\right) \right)$$

其中 $z_s$ 和 $z_t$ 分别是学生和教师的 logits，$T$ 是温度参数（默认 $T=3.0$）。

**代码实现（`LightRoutedNet.py`）：**

```python
def distill_loss(self, student_logits, teacher_logits):
    T = self.temperature
    soft_student = F.log_softmax(student_logits / T, dim=-1)
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)
    return F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
```

**设计动机：** 硬标签（0/1）只告诉模型"正确答案"，软标签包含类别间的相似性信息。例如：一张图既像 Defect-1 又像 Defect-2，这种相似性关系通过软标签传递给学生模型。

---

## 训练范式

### 元任务训练 (Meta-Tasking)

元任务是本项目的核心训练范式，采用两阶段训练策略：**分割引导逻辑**。

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: 训练 UNet (像素级分割)                            │
│                                                             │
│  defeat_image ──→ UNet ──→ y_hat (掩码)                     │
│                          │                                  │
│  real_mask ←─────────────┘                                  │
│       │                                                     │
│       ▼                                                     │
│  FocalLoss(y_hat, real_mask) ──→ 反向传播更新 UNet           │
│                                                             │
│  目标: 让 UNet 学会精确标定缺陷像素位置                       │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼ dfeature (B,512,H/32,W/32)
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: 训练 Gate (逻辑判断)                               │
│                                                             │
│  dfeature ──→ Gate ──→ p_hat (逻辑概率)                     │
│                      │                                      │
│  real_label ←────────┘                                      │
│       │                                                     │
│       ▼                                                     │
│  BCE(p_hat, real_label) ──→ 反向传播更新 Gate               │
│                                                             │
│  目标: 让 Gate 学会从深层语义特征判断是否存在缺陷             │
└─────────────────────────────────────────────────────────────┘
```

**工作流程：**
1. UNet 接收缺陷图像，输出像素级掩码 $y_{hat}$，同时提取瓶颈层特征 $d_{feature}$
2. Gate 接收 $d_{feature}$，输出二分类逻辑判断 $p_{hat}$
3. 两个损失函数分别优化 UNet（分割）和 Gate（分类）

**设计动机：** 将"缺陷检测"分解为"哪里有缺陷"（分割）和"是否有缺陷"（分类）两个子任务。UNet 负责精确的像素级定位，Gate 负责基于全局语义的逻辑判断，两者协同工作。

---

### 知识蒸馏 (Knowledge Distillation)

知识蒸馏将元任务模型（教师）的知识迁移到 ResNet50（学生），大幅提升推理速度。

```
┌─────────────────────────────────────────────────────────────┐
│  教师模型 (UNet + Gate, ~5.3M params, 慢但准)              │
│                                                             │
│  defeat_image ──→ UNet ──→ dfeature ──→ Gate ──→ p_teacher │
└─────────────────────────────────────────────────────────────┘
                          │ 软标签
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  学生模型 (ResNet50, ~23.5M params, 快)                     │
│                                                             │
│  defeat_image ──→ ResNet50 ──→ p_student                   │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  蒸馏损失: KL(Softmax(p_student/T), Softmax(p_teacher/T))   │
│  × T²                                                        │
│  + 硬标签损失: CE(p_student, hard_label)                    │
│                                                             │
│  最终损失 = (1-α)·CE + α·T²·KL                              │
│  其中 α=0.7, T=3.0                                          │
└─────────────────────────────────────────────────────────────┘
```

**为什么有效：** ResNet50 如果从头训练，效果远不如蒸馏后的模型。教师模型的软标签提供了类别间的相对关系（如"85% Defect-1 + 15% Normal"），学生从中学习到了比硬标签更丰富的知识。

---

### 强化学习监督微调 (PPO)

使用 PPO（Proximal Policy Optimization）算法对 ClassifyNet 进行监督微调，使其能够利用元任务模型的反馈来改善多分类性能。

```
┌─────────────────────────────────────────────────────────────┐
│  PPO 强化学习框架                                            │
│                                                             │
│  Actor (ClassifyNet)                                        │
│    输入: 缺陷图像                                            │
│    输出: 5类动作 logits                                      │
│    ├── 正常                                                  │
│    ├── Defect-1                                              │
│    ├── Defect-2                                              │
│    ├── Defect-3                                              │
│    └── Defect-4                                              │
│                                                             │
│  Critic (Value Network)                                      │
│    估计当前状态的价值 V(s)                                   │
│                                                             │
│  奖励函数:                                                   │
│    r = 1.0  若 pred == true_class (正确分类)                 │
│    r = -1.0 若 pred != true_class (错误分类)                 │
│    r = +0.5  若 Gate(p > 0.5) 与 ClassifyNet 一致          │
│    r = -0.5  若 Gate(p > 0.5) 与 ClassifyNet 不一致        │
│                                                             │
│  PPO 裁剪目标:                                               │
│    L^CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ),1-ε,1+ε)·A_t)]│
└─────────────────────────────────────────────────────────────┘
```

**设计动机：** 直接用交叉熵训练 ClassifyNet 容易导致特征混叠（所有缺陷类特征混淆）。通过 PPO 引入元任务模型的逻辑判断作为奖励信号，引导 ClassifyNet 学习更具有区分性的特征表示。

---

### 轻量路由训练 (LightRoutedNet)

轻量路由训练仅微调路由头和分类头，冻结 UNet 编码器，大幅减少可训练参数。

```
┌─────────────────────────────────────────────────────────────┐
│  UNetEncoder (冻结, ~4.71M params)                          │
│    输入: defeat_image ──→ 输出: dfeature (B,512,H/32,W/32)  │
└─────────────────────────────────────────────────────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
           LightRouteHead    LightClassifier
           (可训练, 0.05M)   (可训练, 0.15M)
                    │             │
                    ▼             ▼
           route_probs (K,)  class_logits (K+1,)
                    │             │
                    └──────┬──────┘
                           ▼
                    融合 + CrossEntropyLoss
```

**训练配置：**
- 优化器: Adam (lr=1e-3)
- 学习率调度: CosineAnnealingLR
- 损失: CrossEntropyLoss（直接多分类）
- 冻结策略: 整个 UNetEncoder 梯度关闭

---

## 数据集生成

### 样本合成原理

数据通过**遮罩叠加**方式合成：

```
正常屏幕截图               缺陷遮罩 (白色背景)
(screen_image)            (defeat_mask)
      │                         │
      │    255 - mask ──→ 反相遮罩
      │         │
      │         ▼
      │    screen_image × (反相遮罩/255)
      │         │
      └─────────┴──→ 叠加 ──→ defeat_image
```

其中 `255 - mask` 是为了**反相**：遮罩中白色区域（缺陷）变为黑色，在叠加时保留原屏幕内容。

### 数据增强

每次采样时对遮罩进行随机变换，增加样本多样性：

| 变换 | 说明 |
|------|------|
| 水平翻转 | 左右镜像 |
| 垂直翻转 | 上下镜像 |
| 旋转 90° | 顺时针旋转 |
| 旋转 180° | 半圈旋转 |
| 不变换 | 保留原图 |

### 数据集类型

| 类 | 说明 |
|----|------|
| `SplitDataset` | 单类缺陷数据集（Normal + 1类缺陷），用于元任务训练 |
| `MixtureSplitDataset` | 多类混合数据集（Normal + 4类缺陷），用于路由训练 |

---

## 数据集结构

```
dataset/
├── images/
│   └── screen_images/           # 正常屏幕截图 (38张, 统一尺寸)
│       ├── Screenshot_xxx1.jpg
│       ├── Screenshot_xxx2.jpg
│       └── ...
├── masks/
│   ├── defeat_items1/           # 第1类缺陷遮罩 (4张)
│   │   ├── defeat1.jpg          # 白色背景 + 缺陷图案
│   │   ├── defeat2.jpg
│   │   ├── defeat3.jpg
│   │   └── defeat8.jpg
│   ├── defeat_items2/           # 第2类缺陷遮罩 (4张)
│   ├── defeat_items3/           # 第3类缺陷遮罩 (5张)
│   └── defeat_items4/           # 第4类缺陷遮罩 (3张)
│       ├── defeat9.jpg
│       ├── defeat10.jpg
│       └── defeat13.jpg
└── mixtures/                    # 预混合缺陷样本图
    ├── mixture1.png
    └── ...
```

**路径配置统一由 `dataset_config.py` 管理：**

```python
from dataset_config import (
    SCREEN_IMAGES_DIR,   # dataset/images/screen_images
    MASK_CLASSES,        # {1: defeat_items1, 2: defeat_items2, ...}
    MIXTURES_DIR,        # dataset/mixtures
    get_all_mask_dirs,   # 获取所有遮罩目录
    get_mask_dirs_for_classes  # 获取指定类别的遮罩目录
)
```

---

## 快速开始

### 1. 环境安装

```bash
# 克隆仓库
git clone https://gitee.com/g00db0y2025/ScreenAnomalyDetection.git
cd ScreenAnomalyDetection
git checkout develop

# 安装依赖
pip install -r requirements.txt
```

### 2. 训练元任务模型

```bash
# 打开 Notebook
jupyter notebook net_unet_workbench.ipynb

# 或直接运行主程序
python DefeatMain.py
```

### 3. 训练 LightRoutedNet（推荐，更快）

```bash
python experiment_light_routed.py
```

结果自动保存在 `experiment_results/` 目录下。
