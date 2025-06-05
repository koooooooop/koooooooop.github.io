# BirdCLEF+ 2025 技术研究报告

## 1. 比赛概况与基线方案

BirdCLEF (Bird Sound Classification and Detection Challenge) 系列竞赛旨在推动鸟类鸣声自动识别技术的发展，以应对生物多样性监测的需求。BirdCLEF+ 2025 竞赛延续了这一目标，要求参赛者对音频中可能出现的 182 种鸟类鸣声进行多标签预测。竞赛规则通常允许使用公开的外部数据集，如 Xeno-canto 鸟声库 1，以及预训练模型，这为利用大规模数据和先进模型提供了便利。

本报告采用 BirdCLEF 2024 第四名"Cerberus"团队的解决方案作为基线 4。该方案的核心思想是集成学习，结合了多种基于梅尔频谱图（log-mel spectrogram）的卷积神经网络（CNN）模型（如 SEResNeXt-26-TS, RexNet-150, Inception-Next-Nano）和一个基于原始波形的 CNN 模型（tf_efficientnet_b0_ns）4。通过对这些不同模型的预测结果进行加权融合，显著提升了整体性能。

在数据处理方面，Cerberus 团队利用了 BirdCLEF 2021–2023 年的数据进行预训练，并以 BirdCLEF 2024 数据集作为主训练集 4。此外，他们还从外部数据源收集了背景噪音音频，用于数据增强，以提高模型在真实嘈杂环境中的鲁棒性 4。

训练时，通常截取每段音频的前 5–20 秒作为样本，例如，模型 A（基于梅尔谱图的 CNN）在训练时随机截取 15–20 秒，验证时使用前 5 秒 4。梅尔频谱图的生成参数包括 n_mels=128, n_fft=2048, f_min=0, f_max=16000, hop_length=627 等 4。

考虑到竞赛对推理时间的严格限制（例如，BirdCLEF 2024 要求在 120 分钟内完成 CPU 推理 5），基线方案采用了多种推理加速技术。其中包括使用 OpenVINO 对模型进行优化以加快推理速度，以及应用测试时增强（Test Time Augmentation, TTA）策略 4。TTA 通过对输入数据进行多种变换（如对梅尔谱图进行小幅度的时移）并平均预测结果，来提高预测的稳定性和准确性 4。

此外，后处理步骤，如对模型输出的概率得分进行加权、几何平均、平滑处理和阈值截断，也对最终性能的提升起到了积极作用 4。

下图展示了 BirdCLEF 2024 第四名方案中不同模型及其融合策略对最终 AUC（Area Under the ROC Curve）分数的影响。可以看出，从单个模型到多模型加权平均，再到结合几何平均、平滑和阈值截断等后处理步骤，AUC 分数逐步提升，凸显了集成学习和精细后处理在竞赛中的重要性 4。

| 模型/策略 | 公开榜 AUC (+TTA) | 私有榜 AUC (+TTA) |
|-----------|------------------|------------------|
| Melspec Model A (rexnet_150) | 0.690 | 0.649 |
| Melspec Model A (seresnext26ts) | 0.693 | 0.651 |
| Raw signal Model C (tf_efficientnet_b0_ns) | 0.691 | 0.636 |
| Weighted Mean | 0.731 | 0.676 |
| Weighted Mean + Geometric Mean | 0.732 | 0.677 |
| Weighted Mean + Geometric Mean + Smoothing | 0.741 | 0.685 |
| Weighted Mean + Geometric Mean + Smoothing + Cut-off (Final Submission) | 0.7469 | 0.6877 |

*表格数据来源: 4*

该基线方案为 BirdCLEF+ 2025 提供了一个坚实的起点，其在数据使用、模型选择、集成策略和推理优化方面的经验值得借鉴。

## 2. 数据增强方法

鸟类音频数据常面临信噪比低、目标声音稀疏、类别不平衡以及环境音干扰等挑战 7。有效的数据增强技术能够显著扩充训练数据集的多样性，从而提升模型的鲁棒性和泛化能力。数据增强可以在梅尔频谱图层面或原始波形层面进行。

### 2.1 梅尔频谱图层面增强

#### SpecAugment

SpecAugment 是一种在梅尔频谱图上直接进行操作的增强方法，它包括时间扭曲（Time Warping）、频率掩码（Frequency Masking）和时间掩码（Time Masking）9。频率掩码随机选择一定宽度的连续频率通道并将其置零；时间掩码则随机选择一定长度的连续时间帧并将其置零 9。时间扭曲在实践中对分类任务贡献较小，常被省略 9。这种方法迫使模型学习更具鲁棒性的特征，减少对特定频率或时间信息的依赖。在 BirdCLEF 竞赛中，一些方案采用了类似思想，如对频谱图进行随机翻转或掩码 10。

**实现建议：** 时间掩码的长度 T 和频率掩码的宽度 F 可以从一个预设范围内的均匀分布中随机采样。例如，时间掩码长度可在 10~50 帧之间随机选择，频率掩码宽度可在若干 Mel 维度随机选择。可以多次应用掩码操作。

```python
# Conceptual pseudo-code for SpecAugment using torchaudio
# import torchaudio.transforms as T
# spectrogram = # your log-mel spectrogram tensor
# freq_mask_param = 80 # Max width of frequency mask
# time_mask_param = 80 # Max width of time mask
# # Apply frequency masking
# freq_mask = T.FrequencyMasking(freq_mask_param=freq_mask_param)
# spectrogram = freq_mask(spectrogram)
# # Apply time masking
# time_mask = T.TimeMasking(time_mask_param=time_mask_param)
# spectrogram = time_mask(spectrogram)
```

#### Mixup/CutMix 系列方法

**Mixup：** 该方法通过对两个随机选取的样本（xi​,xj​）及其标签（yi​,yj​）进行线性插值来生成新的训练样本（x~,y~​）11。公式为 x~=λxi​+(1−λ)xj​ 和 y~​=λyi​+(1−λ)yj​，其中混合系数 λ 从 Beta 分布（通常 α=β）中采样，α 常取 0.2 到 0.4 11。Mixup 可以应用于原始波形或梅尔频谱图 4。基线方案在梅尔谱图上使用了 Mixup，有助于提高模型的泛化能力 4。

**CutMix：** CutMix 则是将一个样本中的随机区域块移除，并用从另一个样本中随机选取的区域块填充该移除区域 12。标签也按混合区域的比例进行插值。在音频领域，这类似于将不同音频片段叠加或拼接。

**SpecMix：** SpecMix 是 CutMix 针对音频频谱图的改进版本，它通过特定的掩码策略（如频带掩码和时域掩码）混合两个频谱图，旨在更好地保留时频相关性 15。SpecMix 定义了频率和时间掩码的起始位置和宽度（由用户定义的参数 γ 控制），最多可应用三个频带和三个时域掩码 15。

**实现建议：** 对于 Mixup，α 参数通常设置为 0.4 左右。对于 CutMix/SpecMix，需要确定剪切和粘贴区域的大小和位置。这些增强方法可以在批次级别随机应用。

#### 其他谱图操作

**随机遮挡 (Cutout/Random Erasing)：** 在频谱图上随机选择一个或多个矩形区域并将其置零，类似于 SpecAugment 的掩码，但通常区域大小固定或从一定范围随机选择 15。

**CoarseDropout：** 与随机遮挡类似，随机丢弃频谱图中的一些块。

**频谱均值/方差归一化 (CMVN)：** 对每个频谱图或一个批次内的频谱图进行均值和方差归一化，有助于稳定训练。

### 2.2 原始波形层面增强

#### 噪声注入与背景音混合

向原始音频中添加各种类型的噪声是一种简单有效的方法，可以模拟真实世界的录音条件。基线方案从外部数据集中收集背景噪声片段，并将其叠加到训练音频中 4。常用的噪声类型包括高斯白噪声、粉红噪声等 4。

环境音混合则更进一步，将目标鸟鸣与真实的环境录音（如森林、城市背景声）混合。这有助于模型学习在复杂声学场景中区分目标信号。DCASE（Detection and Classification of Acoustic Scenes and Events）等竞赛中常使用此类增强 17。

**实现建议：** 噪声叠加的强度可以通过信噪比（SNR）控制，例如在 10 dB 到 20 dB 之间随机选择。可以使用 librosa 或 audiomentations 等库实现。

```python
# Conceptual pseudo-code for adding background noise
# import numpy as np
# import librosa
# from audiomentations import AddBackgroundNoise
# # Load bird sound and background noise
# bird_sound, sr_bird = librosa.load("bird_call.wav", sr=None)
# background_noise, sr_noise = librosa.load("background.wav", sr=sr_bird) # Ensure same sr
# # Using audiomentations
# augment = AddBackgroundNoise(sounds_path="path/to/background_noises/", min_snr_in_db=10.0, max_snr_in_db=20.0, p=1.0)
# augmented_sound = augment(samples=bird_sound, sample_rate=sr_bird)
```

#### 增益调整与时频变换

**增益调整 (Gain)：** 随机改变音频的整体音量，例如在 ±6 dB 范围内调整 4。这模拟了不同录音设备或录音距离导致的音量变化。

**时间拉伸与音高变换 (Time Stretching and Pitch Shifting)：**

- **时间拉伸：** 在不改变音高的情况下改变音频的播放速度和时长。
- **音高变换：** 在不改变音频时长的情况下改变音频的音高。

这些变换可以模拟鸟鸣的自然变异或不同个体间的差异。librosa.effects.time_stretch 和 librosa.effects.pitch_shift 是常用的实现工具 20。例如，可以将时间拉伸因子设为均值为 1、标准差为 0.05 的高斯分布，音高偏移设为均值为 0、标准差为 25 音分（cents）的高斯分布 20。

**实现建议：** 时间拉伸因子通常在 0.8 到 1.2 之间，音高变换的半音数（n_steps）可以在 -2 到 2 之间。需要注意过度变换可能导致音频失真。

**随机时移 (Random Time Shift)：** 将音频信号在时间轴上随机平移，模拟鸟鸣开始时间的随机性。

### 2.3 通用策略

在实践中，通常会组合使用多种数据增强方法，并在每个训练批次中随机应用其中的一种或几种。增强方法的超参数（如 Mixup 的 α 值、SpecAugment 的掩码大小、噪声的 SNR 范围）应通过在验证集上的实验来仔细调整。一些研究表明，针对动物音频分类，组合多种增强策略（包括原始音频和频谱图层面的增强）通常比单一方法效果更好 8。

## 3. 模型结构改进

基于梅尔频谱图的鸟类声纹识别任务中，深度学习模型结构不断演进。近年来，Transformer 及其变体在音频处理领域展现出强大潜力，同时也涌现出优秀的 CNN 架构和混合模型。

### 3.1 AST (Audio Spectrogram Transformer)

AST 是一种完全基于 Transformer 架构的音频分类模型，它不依赖卷积层进行特征提取 26。其核心思想是将输入的梅尔频谱图分割成一系列不重叠或部分重叠的 16x16 的块（patches），并将这些块线性投影为 patch embeddings 27。与 ViT (Vision Transformer) 类似，AST 在 patch embeddings 序列前加入一个可学习的 [CLS] 分类令牌，并为每个 patch embedding 添加位置编码（positional embedding）以保留时频位置信息 27。整个序列随后被送入标准的 Transformer 编码器（通常包含 12 个注意力头和 12 层）27。最终，[CLS] 令牌对应的输出向量用于分类。

**优势：** 自注意力机制使 AST 能够捕捉频谱图上全局的时频上下文依赖关系，对于分析具有长时程结构或复杂谐波结构的鸟鸣声可能特别有效。AST 在多个音频分类基准任务上取得了领先成果，例如在 AudioSet 上的平均精度均值（mAP）达到 0.485，ESC-50 环境声分类准确率达到 95.6% 26。

**预训练：** AST 的一个关键优势在于可以利用在 ImageNet 上预训练的 ViT 模型权重进行初始化，通过调整输入层和位置编码来适应音频频谱图 26。这种跨模态迁移学习有助于模型在音频数据相对不足的情况下学习到有用的特征。

**挑战：** 纯 Transformer 模型通常参数量较大，计算密集，需要大量的训练数据或有效的预训练才能达到最佳性能 27。对于 BirdCLEF 竞赛中有限的计算资源和时间，直接从头训练大型 AST 可能具有挑战性。

**实现参考：** 官方实现和 Colab 演示可见于 https://github.com/YuanGongND/ast 28。

### 3.2 HTS-AT (Hierarchical Token-Semantic Audio Transformer)

HTS-AT 针对 AST 计算量大、训练时间长的问题进行了改进，引入了层级结构（Hierarchical Structure）和窗口注意力（Window Attention）机制，显著降低了模型规模和训练成本 29。其设计灵感部分来源于 Swin Transformer。

**层级结构与 Patch-Merge：** HTS-AT 首先将梅尔频谱图分割成 patch，然后通过多阶段的 Transformer 编码器。在每个阶段的末尾，使用 Patch-Merge 层来减少序列长度（token 数量）并增加特征维度，形成金字塔式的特征表示 29。这种方式逐步缩小感受野，同时减少了后续层的计算量。

**窗口注意力：** 在每个 Transformer 块内部，HTS-AT 采用窗口多头自注意力（Windowed Multi-Head Self-Attention, W-MSA）或移位窗口多头自注意力（Shifted Window MSA, SW-MSA），将自注意力的计算限制在局部窗口内，而非全局范围，从而大幅降低了计算复杂度 29。

**Token-Semantic 模块：** 为了进行声音事件检测（定位事件的起止时间），HTS-AT 在最后一个 Transformer 块之后增加了一个 Token-Semantic CNN 层，将输出的 token 序列映射为类别激活图（class activation map），指示每个类别在不同时间帧的出现概率 29。

**优势：** HTS-AT 在 AudioSet 和 ESC-50 等数据集上取得了与 AST 相当甚至更好的性能，但其参数量仅为 AST 的约 35%（31M vs 87M），训练时间也大幅缩短（例如在 AudioSet 上约 80 小时 vs 600 小时）29。这种轻量化和高效性使其更适合 BirdCLEF 竞赛的实际约束。

**实现参考：** 官方代码库位于 https://github.com/RetroCirce/HTS-Audio-Transformer 32。

### 3.3 PANNs (Pretrained Audio Neural Networks)

PANNs 是一系列在超大规模音频数据集 AudioSet（包含约 200 万段音频，527 个类别）上预训练的 CNN 模型 34。这些模型为各种下游音频任务提供了强大的预训练权重。

**架构多样性：** PANNs 系列包含了多种 CNN 架构，如不同深度的 CNN（CNN6, CNN10, CNN14），ResNet 变体（ResNet22, ResNet38, ResNet54），MobileNet 变体，以及直接处理原始波形的一维 CNN 34。

**Wavegram-Logmel-CNN：** 这是 PANNs 中一个具有代表性的架构，它创新地同时使用对数梅尔频谱图和从原始波形中学习到的表示（Wavegram）作为双输入 34。Wavegram 本身由一个一维 CNN 从原始波形中提取，旨在学习一种类似频谱图但数据驱动的时频表示。这两种特征随后被融合并送入后续的 CNN 分类网络。

**性能：** PANNs 的最佳模型在 AudioSet 标注任务上取得了 0.439 的 mAP，超越了当时的最佳系统 34。由于其在大规模多样化音频数据上的预训练，PANNs 在迁移到其他音频识别任务（如声学场景分类、音乐分类、语音情感识别）时表现出色，常作为强基线或用于特征提取 34。

**应用：** 对于 BirdCLEF，可以直接加载 PANNs 的预训练权重，并在竞赛数据上进行微调。也可以借鉴其多模态输入（波形+频谱图）的思想，设计自己的模型。

**实现参考：** 相关代码和预训练模型可在 https://github.com/qiuqiangkong/audioset_tagging_cnn (原始) 和一些 forks 如 https://github.com/bakhtos/PANNs 37 找到。

### 3.4 混合 CNN-Transformer 架构

为了结合 CNN 在提取局部特征方面的优势和 Transformer 在建模全局依赖方面的能力，混合架构应运而生。

#### FAST (Fast Audio Spectrogram Transformer)

FAST 借鉴了 MobileViT 的设计理念，将轻量级的 CNN (如 MobileNetV2 中的模块) 用于高效的局部特征提取，然后将提取的特征块（patches）送入 Transformer 模块进行全局上下文建模 39。FAST 还引入了 Lipschitz 连续注意力机制以增强训练稳定性和加速收敛 40。

**优势：** FAST 旨在实现性能和计算效率之间的平衡，特别适合资源受限的场景。报告称其在 ADIMA 和 AudioSet 数据集上取得了 SOTA 性能，且参数量远少于某些现有模型（例如，比 AST 少高达 150 倍的参数）39。

#### 其他混合思路

还可以在 CNN 提取的特征图之上应用多尺度 Transformer，或者在 CNN 的不同阶段嵌入轻量化的注意力模块（如 Linformer, Performer），或者设计时频卷积与 Transformer 模块交替的结构 42。例如，Vindas 等人提出了一种模型，使用 CNN-Transformer 处理原始信号，使用 2D CNN 处理时频表示（TFR），然后进行后期融合 42。

**适用性：** 这类混合模型能够有效捕捉音频频谱图中的局部纹理细节（通过 CNN）和长时程的全局模式（通过 Transformer），是解决 BirdCLEF 这类大规模多类别分类任务的有力候选。

### 3.5 模型选择建议

在 BirdCLEF 竞赛中，模型选择需权衡性能、计算资源和推理时间限制：

- 如果计算资源充足，且有大规模预训练数据，AST 或其改进版本 HTS-AT 是强有力的选择，因其强大的全局建模能力。
- PANNs 提供的预训练权重是一个宝贵的起点，特别是其 Wavegram-Logmel-CNN 结构，为多模态输入提供了思路。
- 混合 CNN-Transformer 架构如 FAST，因其在效率和性能上的平衡，特别值得关注，尤其是在有严格推理时间限制的情况下。
- 建议在验证集上对不同类型的模型（纯 CNN、纯 Transformer、混合模型）进行细致比较，评估其 AUC、参数量和推理速度，为最终的模型集成提供具有结构多样性的候选模型。

## 4. 伪标签策略

伪标签（Pseudo-Labeling）是一种有效的半监督学习技术，它利用模型自身对未标注数据的预测来生成"伪"标签，然后将这些伪标签数据加入训练集，以期提升模型性能和泛化能力，尤其在标注数据有限而未标注数据充足的情况下 44。在 BirdCLEF 竞赛中，通常会提供额外的无标签声景数据（unlabeled soundscapes），并且允许使用外部公开数据（如 Xeno-canto），这为伪标签策略的应用提供了良好基础 2。

### 4.1 高置信度样本筛选 (High-Confidence Thresholding)

这是最常用的伪标签策略。首先，使用在有标签数据上训练好的模型对未标注音频进行预测。然后，仅选择那些模型预测置信度高于某一预设阈值（例如 0.9 或 0.95）的样本及其对应的预测类别作为伪标签数据 44。这些高置信度的伪标签样本被认为具有较高的标签质量，加入训练集后有助于模型学习。

**阈值选择：** 阈值的设定至关重要。过低的阈值可能引入大量噪声标签，损害模型性能；过高的阈值则可能导致伪标签数量过少，学习效果不明显。FixMatch 等研究发现，0.95 的阈值在某些任务上效果较好 46。在 BirdCLEF 竞赛中，有参赛者使用 0.5 作为阈值，结合外部模型（如 Google 的鸟声分类器）的预测进行筛选 10。阈值可以根据验证集表现进行调整，也可以随着训练的进行动态调整（例如，在模型性能提升后逐渐降低阈值以纳入更多样本）。

**迭代训练：** 伪标签过程通常是迭代的。模型在加入伪标签数据后重新训练，更新后的模型可以用来生成新一轮（可能质量更高或数量更多）的伪标签，如此反复 2。

### 4.2 软伪标签与温度缩放 (Soft Pseudo-Labels & Temperature Scaling)

与使用"硬"的 one-hot 伪标签不同，软伪标签直接使用模型输出的概率分布作为目标。这种方法保留了模型的不确定性信息，可能比硬标签更鲁棒。

温度缩放（Temperature Scaling）可以用于调整模型输出概率分布的"尖锐度"。较高的温度会使概率分布更平滑，较低的温度则使其更尖锐。在生成伪标签时，可以使用较低的温度来强化高置信度的预测。

### 4.3 利用外部数据源进行伪标签

BirdCLEF 竞赛允许使用公开的外部鸟类音频数据库，如 Xeno-canto 1。这些数据库包含大量未针对当前竞赛任务进行标注的录音。

**策略：** 首先，可以根据地理位置、物种列表等信息对外部数据进行初步筛选，以提高包含目标鸟种鸣声的概率。然后，使用当前在有标签数据上训练的模型为这些外部音频打上伪标签。高质量的伪标签样本随后可以加入训练集，用于模型的再训练或微调。一些 BirdCLEF 团队明确提到了使用 Xeno-Canto 数据进行预训练或提取样本，但对其直接用于伪标签的效果评价不一，强调了数据质量和筛选的重要性 2。

### 4.4 一致性正则化 (Consistency Regularization) - MixMatch/FixMatch 思想

如 FixMatch 46 和 MixMatch 47 等方法的核心思想是，模型对于同一未标注样本的不同增强版本，其预测应当保持一致。

#### FixMatch 流程

1. 对一个未标注样本，首先应用弱增强 (Weak Augmentation)，如简单的翻转、移位。
2. 将弱增强后的样本输入模型，得到预测概率。如果最高概率超过预设阈值 τ，则将该预测类别作为伪标签。
3. 对同一个未标注样本，应用强增强 (Strong Augmentation)，如 RandAugment 48、Cutout，或针对音频的特定强增强（如 SpecAugment 的较强参数、FilterAugment 49）。
4. 将强增强后的样本输入模型，得到其预测概率。
5. 计算强增强样本预测与伪标签之间的损失（如交叉熵），并将其作为无监督损失项加入总损失中进行优化。

**弱/强增强策略：** 对于音频频谱图，弱增强可以是简单的水平翻转、小幅时移频移；强增强可以是 SpecAugment、Cutout、FilterAugment（改变不同频段的增益）49 或 Mixup 等。关键在于强增强应比弱增强引入更大的扰动，但又不至于完全破坏信号内容。

### 4.5 BirdCLEF 竞赛背景下的实践考量与风险

**领域漂移 (Domain Shift)：** 训练数据（通常是 Xeno-canto 等来源的较为清晰的录音）与测试数据（通常是连续的、充满噪声的野外声景录音）之间存在显著的领域差异 2。伪标签是应对领域漂移的关键策略之一，通过利用与测试数据同源的无标签声景数据进行伪标签，可以帮助模型适应测试领域。

**噪声与非目标物种：** 测试声景中可能包含大量噪声、非鸟鸣声，甚至非目标鸟种的鸣声。直接对所有未标注声景进行伪标签风险较高，容易引入错误标签。

**确认偏差 (Confirmation Bias)：** 模型可能会不断强化自己最初的（可能是错误的）预测，导致性能下降 44。

### 4.6 稳妥策略

- **谨慎选择伪标签来源：** 优先使用与测试集同源的无标签声景数据进行伪标签。对于外部数据（如 Xeno-canto），需仔细筛选，例如使用元数据（地理位置、录音质量评级 3）或辅助模型（如 Google Bird Vocalization Classifier 6）进行预过滤。

- **迭代和逐步扩充：** 采用迭代伪标签，并在早期迭代中使用较高的置信度阈值，随着模型性能提升逐步调整阈值或增加伪标签数量。

- **结合有标签数据：** 始终确保有高质量标注数据在训练中的主导地位，伪标签数据作为补充。

- **多模型伪标签：** 如果训练了多个不同模型，可以考虑使用这些模型的集成预测或高一致性预测来生成更可靠的伪标签。

- **人工校验：** 对于少量特别不确定的或对模型影响大的伪标签样本，可以考虑进行小范围的人工校验（如果时间和资源允许）50。

总结而言，伪标签是提升 BirdCLEF 竞赛成绩的有力工具，但其实施需要精心设计，特别是要关注数据来源的质量、置信度阈值的选择、迭代策略以及如何缓解领域漂移和确认偏差带来的负面影响。

## 5. 模型融合策略

模型融合（Ensemble Learning）是将多个独立训练的模型的预测结果进行组合，以期获得比任何单一模型更好的性能、更高的鲁棒性和更低的预测方差的常用技术 51。在 BirdCLEF 这类复杂的分类竞赛中，模型融合几乎是所有顶级方案的标配 2。

### 5.1 横向结构融合 (Ensembling Diverse Architectures)

这是最常见的融合方式，即训练多个具有不同网络结构的模型，然后将其预测结果进行组合。例如，可以融合多种 CNN 模型（如基线中使用的 SEResNeXt, RexNet, Inception-Next-Nano 4）、Transformer 模型（如 AST, HTS-AT）以及混合 CNN-Transformer 模型（如 FAST）。

**多样性是关键：** 模型结构的多样性（如 CNN 捕捉局部特征，Transformer 捕捉全局依赖）、训练数据或数据子集的不同、不同数据增强策略的应用，都有助于提升融合效果 54。不同模型在决策边界上的差异越大，融合后性能提升的潜力也越大。

**权重确定：** 最简单的组合方式是平均投票或平均概率。更常见的是加权平均，其中权重通常根据各模型在验证集上的性能（如 AUC）来确定或优化 4。Cerberus 团队的方案就采用了加权平均，并提到他们没有花过多时间调整权重以避免过拟合 4。

### 5.2 输入模态融合 (Ensembling Models with Different Input Modalities)

利用不同类型的输入特征训练模型并进行融合。在鸟声识别中，常见的输入模态包括：

- **梅尔频谱图 (Log-Mel Spectrogram)：** 大多数 BirdCLEF 方案的核心输入 2。
- **原始波形 (Raw Waveform)：** 直接使用一维时域信号作为输入，通常由 1D CNN 或特定波形模型（如 PANNs 中的 Wavegram 部分 34，或 Aves 2）处理。Cerberus 团队就包含了一个原始波形模型 (Model C: tf_efficientnet_b0_ns) 4。
- **其他声学特征：** 如 MFCC (Mel-Frequency Cepstral Coefficients)、CQT (Constant-Q Transform) 等，虽然在近年来不如梅尔频谱图主流，但仍可作为补充。

**融合方式：** 可以独立训练基于不同模态的模型，然后对它们的预测概率进行后期融合（Late Fusion）42。或者，设计一个多分支模型，在模型内部的不同路径处理不同模态的输入，并在网络的较深层进行特征融合（Early or Intermediate Fusion）。PANNs 的 Wavegram-Logmel-CNN 就是一个内部融合的例子 34。

**优势：** 不同模态的特征可能包含互补的信息。例如，频谱图擅长表达频率结构，而波形可能保留更精细的时间信息或相位信息。融合这些模型有助于捕捉更全面的声学特性。

### 5.3 Snapshot Ensemble

这是一种在单次模型训练过程中生成多个模型进行集成的方法 57。它通过使用带有周期性重启（Cyclical Learning Rates with Restarts）的学习率调度策略（如余弦退火后突然将学习率重置为一个较高值）实现。

**原理：** 在学习率下降到局部最小值附近时保存模型权重（即一个"快照"），然后提高学习率使模型跳出当前局部最小值，去探索解空间中的其他区域，并收敛到另一个局部最小值，再次保存快照。重复此过程 M 次，即可得到 M 个不同的模型。

**优势：** 相比于独立训练 M 个模型，Snapshot Ensemble 的训练成本大大降低，几乎与训练单个模型的成本相当，但能提供集成带来的性能提升和鲁棒性。这对于计算资源或时间受限的竞赛场景尤其有价值。

### 5.4 K 折交叉验证融合 (K-Fold Cross-Validation Ensembling)

对同一模型结构，使用 K 折交叉验证的方式进行训练。即将训练数据划分为 K 个互不相交的子集，每次使用 K-1 个子集进行训练，剩下的 1 个子集进行验证。这样可以得到 K 个在不同数据子集上训练出来的模型。

**预测：** 在推理阶段，对测试样本的预测结果是这 K 个模型的预测的平均值或加权平均值。

**优势：** 这种方法使得每个数据样本都有机会作为验证集的一部分，从而使得模型对数据划分不那么敏感，融合结果通常更为稳健。它也有助于更好地估计模型的泛化性能。

### 5.5 后处理融合与技巧 (Post-Processing and Blending Techniques)

**几何平均 vs. 算术平均：** 除了加权算术平均，还可以使用加权几何平均。几何平均对极端值（非常高或非常低的预测概率）的抑制作用更强。Cerberus 团队在加权平均的基础上加入了原始波形模型和梅尔谱图模型的几何平均项，并观察到轻微的性能提升 4。

**排序平均 (Rank Averaging)：** 将每个模型的预测概率转换为排序，然后对排序进行平均，再将平均排序转换回概率。这种方法对模型的校准程度不敏感。

**堆叠 (Stacking) / 混合 (Blending)：** 更高级的融合方法。

- **Blending：** 将训练集分成两部分，一部分用于训练基础模型，另一部分（留出集）用于训练一个"元学习器"（meta-learner）。基础模型在留出集上进行预测，这些预测结果作为元学习器的输入特征。
- **Stacking：** 类似于 Blending，但使用 K 折交叉验证来生成基础模型的"out-of-fold"预测作为元学习器的训练数据。元学习器（如逻辑回归、梯度提升树或简单神经网络）学习如何最优地组合基础模型的预测 51。

**阈值优化与平滑：** 对融合后的概率进行阈值截断（如去除低于某个极小概率的预测），或应用时间平滑（如对相邻时间帧的预测进行滑动平均），可以进一步优化最终提交结果，去除噪声预测并使结果更连续 2。Cerberus 团队使用了平滑和截断（cut-off）策略，例如，如果一个鸟种在4分钟音频的所有48个分析窗口中的置信度都低于0.10，则将其概率减半 4。

### 5.6 实施建议

- **从简单开始：** 首先尝试简单的加权平均，根据验证集性能确定权重。
- **确保多样性：** 融合成功的关键在于基础模型的多样性。尝试融合不同架构（CNN, Transformer, Hybrid）、不同输入（频谱图, 波形）、不同预训练策略或不同重要超参数设置的模型。
- **验证驱动：** 所有融合策略和权重都应在可靠的验证集上进行评估和优化，避免过拟合到特定模型或融合方案。

通过上述各环节的精心设计和优化，并结合前述的数据增强、模型改进、伪标签和模型融合策略，可以构建一个在 BirdCLEF+ 2025 竞赛中具有强大竞争力的解决方案。每个模块和参数的选择都应通过在可靠的本地验证集上进行充分实验来指导，最终目标是实现 AUC 分数和推理效率的最佳平衡。

## 6. 训练与推理流程示例

### 6.1 简化训练循环

```python
# Conceptual Pseudo-code for a simplified training loop
# import torch
# import torch.optim as optim
# from torch.utils.data import DataLoader
# # Assume model, train_dataset, val_dataset, loss_fn are defined
# # model = YourBirdModel()
# # optimizer = optim.AdamW(model.parameters(), lr=1e-3)
# # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
# # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# # val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# for epoch in range(num_epochs):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         # data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = loss_fn(output, target)
#         loss.backward()
#         optimizer.step()
#     scheduler.step()

#     model.eval()
#     val_loss = 0
#     all_preds = []
#     all_targets = []
#     with torch.no_grad():
#         for data, target in val_loader:
#             # data, target = data.to(device), target.to(device)
#             output = model(data)
#             val_loss += loss_fn(output, target, reduction='sum').item()
#             all_preds.append(output.cpu())
#             all_targets.append(target.cpu())
#     # val_auc = calculate_auc(torch.cat(all_preds), torch.cat(all_targets))
#     # Save best model based on val_auc
```

### 6.2 迭代伪标签示例

```python
# Conceptual Pseudo-code for Iterative Pseudo-Labeling
# model = initial_model_trained_on_labeled_data
# for iteration in range(num_iterations):
#   predictions_on_unlabeled = model.predict_proba(unlabeled_data_loader)
#   # pseudo_labels_indices, pseudo_labels_targets = select_high_confidence(predictions_on_unlabeled, threshold=0.95)
#   # pseudo_labeled_samples = get_samples_by_indices(unlabeled_data, pseudo_labels_indices)
#   # combined_train_dataset = LabeledDataset(labeled_data_samples, labeled_data_targets) + \
#   #                          PseudoLabeledDataset(pseudo_labeled_samples, pseudo_labels_targets)
#   # combined_train_loader = DataLoader(combined_train_dataset, batch_size=64, shuffle=True)
#   # model = retrain_model(model, combined_train_loader, num_epochs_retrain)
#   # # Optionally, evaluate model and adjust threshold
#   # current_auc = evaluate_model(model, val_loader)
#   # if current_auc > best_auc_so_far:
#   #    save_model(model)
#   #    # threshold = adjust_threshold(threshold, current_auc) # e.g., slightly lower if confident
```

## 7. 结论与展望

本报告围绕 BirdCLEF+ 2025 竞赛任务，以 BirdCLEF 2024 第四名方案为基线，深入探讨了数据增强、模型结构改进、伪标签策略以及模型融合四大关键技术方向的最新进展与实用方法。

### 7.1 核心建议总结

**坚实基线与迭代优化：** 以 Cerberus 4 等优秀历史方案为起点，理解其核心组件（如多模型、多模态输入、精细后处理）的有效性，并在此基础上进行模块化改进和迭代。

**数据为王，增强先行：**

- **组合增强：** 综合运用梅尔频谱图层面（SpecAugment 9, Mixup/CutMix 11, 噪声注入 4）和原始波形层面（背景音混合 4, 增益调整 4, 时频变换 20）的增强方法。针对鸟鸣特性（如频率范围、持续时间）仔细选择增强参数，并随机组合应用，以最大化数据多样性。

**模型创新与效率并重：**

- **探索新架构：** 关注基于 Transformer 的模型如 AST 26、HTS-AT 29（因其高效性更佳），以及混合 CNN-Transformer 架构如 FAST 39，它们在捕捉全局上下文和平衡效率方面具有优势。
- **善用预训练：** 充分利用 ImageNet 预训练权重（对于频谱图模型）和大规模音频预训练模型（如 PANNs 34）进行迁移学习和微调。

**挖掘无标签数据潜力：**

- **稳健伪标签：** 实施迭代的伪标签策略，优先使用与测试集同源的无标签声景数据 2。结合高置信度筛选 44 和一致性正则化思想（如 FixMatch 46），谨慎处理外部数据（如 Xeno-canto 2），注意数据清洗和质量控制，以缓解领域漂移和确认偏差。

**集成致胜，多样融合：**

- **多维度多样性：** 融合模型时，追求架构（CNN, Transformer, Hybrid）、输入模态（频谱图, 波形 2）、训练数据子集（K-折交叉验证模型）和关键超参数的多样性。
- **智能融合策略：** 从加权平均 4 开始，逐步探索几何平均、Snapshot Ensemble 57，甚至更复杂的 Stacking/Blending（若计算资源允许）。精细的后处理（如时间平滑、阈值优化 4）同样重要。

**全局优化与严格验证：**

- **端到端效率：** 整个流程，从数据加载、预处理、增强到模型训练、推理和后处理，都需兼顾效果与效率，特别是推理速度以满足竞赛限制 4。
- **可靠验证：** 建立一个能够准确模拟测试集条件（尤其是噪声、多标签、领域差异）的本地验证方案至关重要 4。所有技术和参数的选择都应基于此验证集上的表现。

### 7.2 未来展望

鸟类鸣声自动识别领域正经历快速发展，未来值得关注的方向包括：

- **自监督与无监督学习：** 进一步发展适用于生物声学的自监督学习方法 69，以更有效地利用海量无标签音频数据，减少对精细标注的依赖。

- **极端环境下的鲁棒性：** 研发更能抵抗强噪声干扰、处理高度重叠发声事件的模型。

- **小样本与稀有物种识别：** 改进小样本学习（Few-Shot Learning）技术，以有效识别数据稀缺的鸟类物种。

- **多模态信息融合：** 更深度地融合音频信号与地理位置、时间、天气等环境元数据，甚至结合视觉信息，以提高识别的准确性和情境感知能力。

- **可解释性与可信度：** 发展模型可解释性方法，理解模型决策依据，并提供预测结果的可信度评估，这对于实际生态监测应用至关重要。

- **音频-语言基础模型：** 探索如 NatureLM-audio 66 这样的大型音频-语言基础模型在生物声学领域的潜力，它们可能通过学习丰富的声学和语义关联，为鸟声识别乃至更广泛的生物声学分析任务（如行为分类、种群估计）带来突破。

- **边缘计算与实时监测：** 优化模型使其能高效部署在边缘设备上，实现低延迟、低功耗的实时鸟类监测。

- **伦理与应用：** 持续关注技术应用的伦理问题，确保其负责任地应用于鸟类保护和生物多样性研究。

通过系统性地应用本报告中探讨的技术，并关注领域内的前沿进展，参赛者有望在 BirdCLEF+ 2025 竞赛中取得优异成绩，并为推动生物声学监测技术的发展做出贡献。

## 参考文献

1. Raza, A., Ullah, N., AlSalman, H., AlQahtani, S. A., & Kim, K. H. (2023). A Hybrid Deep Transfer Learning Model for Bird Sound Classification. EasyChair Preprint no. 77kN.

2. Kaggle. (2024). BirdCLEF: Summary of Techniques from Past Top Solutions (2024). Kaggle Competitions Discussion.

3. Le Bienvenu, T., Gasc, A., Pavoine, S., & Grandcolas, P. (2023). Automatic labelling of Xeno-Canto bird sound recordings for bioacoustic studies. arXiv:2302.07560.

4. Kaggle. (2024). BirdCLEF 2024 4th place solution "Cerberus" details. Kaggle Competitions Discussion.

5. Lasseck, M. (2024). The BirdCLEF 2024 KAGGLE competition: Making AI work for field ecology. CEUR Workshop Proceedings, Vol-3740, paper-199.

6. Bhatia, K., Pal, A., & Shah, A. (2024). Transfer Learning with Pseudo Multi-Label Birdcall Classification for DSGT BirdCLEF 2024. arXiv:2407.06291.

7. Alharbi, F., Alsaedi, N., Aljohani, M., Alrshoudi, F., & Gojobori, T. (2025). Advanced Framework for Animal Sound Classification With Features Optimization. KAUST Repository.

8. Wilhelm, N., Kahl, S., Eibl, M., & Wilhelm, A. (2024). Improving Learning-Based Birdsong Classification by Utilizing Combined Audio Augmentation Strategies. ResearchGate.

9. Wang, Y., Zou, Y., Cheng, G., & Yu, D. (2021). SpecAugment++: A Hidden Space Data Augmentation Method for Acoustic Scene Classification. arXiv:2103.16858.

10. Kaggle User "optimo". (2024). BirdCLEF2024 1st place solution. Kaggle Competitions Discussion.

11. Tran, V. T., Nguyen, T. P. N., Le, T., & Nguyen, L. (2025). Lungmix: A Mixup-Based Strategy for Generalization in Respiratory Sound Classification. arXiv:2501.00064.

12. Eldele, E., Chen, W., Liu, C., Wu, M., Kresta, M., Volkovs, M., & R Međutim, M. (2023). Empirical Study of Mix-based Data Augmentation Methods in Physiological Time Series Data. arXiv:2309.09970.

*[省略其他参考文献以节省空间，完整列表包含79篇文献]*
