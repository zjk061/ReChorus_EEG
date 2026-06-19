# v1 模型设计可选方案及注意事项

本文档记录针对新的严格前置 CTR 数据组织方案，后续模型结构如何调整以适配 `EEGsvRec_eeg_strict_prectr` 派生数据集的设计建议。

当前新数据方案已经不再把当前样本观看后才能获得的 EEG、问卷评分、播放比例、观看时长作为当前输入，而是将用户当前样本之前的交互记录组织为历史序列：

```text
history_item_id
history_eeg_310
history_interest
history_immersion
history_valence
history_arousal
history_length
```

因此，后续模型需要从“当前 EEG 编码”转向“历史序列编码”。

## 1. 当前模型必须调整的原因

当前初始版 `CTRLightGCNCTR` 的 forward 强依赖：

```text
feed_dict['c_EEG_data_310_f']
```

并且会使用当前样本中的部分情境特征：

```text
c_interest_f
c_immersion_f
c_valence_f
c_arousal_f
c_playrate_f
c_video_type_c
```

但是在新的严格前置 CTR 数据集 `EEGsvRec_eeg_strict_prectr` 中，当前后验特征已经被移除，替换为历史序列字段：

```text
history_item_id
history_eeg_310
history_interest
history_immersion
history_valence
history_arousal
history_length
```

因此，当前 `CTRLightGCNCTR` 不能直接用于新数据集。新的模型结构需要把历史序列作为主要动态用户状态来源。

## 2. 推荐总体架构

新模型建议拆成五个主要分支：

```text
当前用户/物品 ID 分支
当前用户元特征分支
当前物品元特征分支
当前 item 上下文分支
历史交互序列分支
```

最终融合后预测 CTR：

```text
user_id/item_id 表示
+ user_meta 表示
+ item_meta 表示
+ c_video_type_c 表示
+ history sequence 表示
-> fusion
-> predictor
-> pCTR
```

这样可以让模型同时利用：

- 当前用户和当前候选 item 的 ID 信息。
- 用户静态属性。
- 当前候选 item 的静态内容属性。
- 当前候选 item 的推荐前可知类型信息。
- 用户在当前样本之前的 EEG 和情绪历史状态。

## 3. 历史序列建模方案 A：Mean Pooling 基线

第一种方案是对历史序列做简单编码后平均池化。

每条历史记录包含：

```text
history_item_id
history_eeg_310
history_interest
history_immersion
history_valence
history_arousal
```

处理方式：

```text
history_item_id -> item embedding
history_eeg_310 -> EEG encoder -> eeg embedding
四类情绪评分 -> Linear/MLP -> emotion embedding
拼接或相加 -> history step embedding
所有历史 step embedding -> masked mean pooling
```

输出得到一个固定长度的历史表示：

```text
history_repr: [batch_size, hidden_dim]
```

### 3.1 优点

- 实现最简单。
- 计算开销小。
- 适合作为第一个可运行 baseline。
- 方便验证新数据读取、历史 padding、mask、batch 组织是否正确。
- 对小数据集相对稳，不容易因为模型太复杂而过拟合。

### 3.2 缺点

- 丢失历史顺序。
- 无法区分近期历史和远期历史。
- 难以建模历史 item 与当前候选 item 的相关性。
- 表达能力较弱。
- 无法充分利用 EEG 和情绪状态的时间动态变化。

### 3.3 适用原因

该方案适合作为严格前置 CTR 的第一版模型，先打通完整训练流程。它的目标不是取得最优效果，而是验证新数据方案和训练链路是否正确。

## 4. 历史序列建模方案 B：GRU/LSTM 序列编码

第二种方案是用 GRU 或 LSTM 编码历史 step embedding。

处理流程：

```text
每条历史记录 -> history step embedding
history step embedding 序列 -> GRU/LSTM
取最后有效 hidden state -> history representation
```

其中 `history_length` 可以用于获得每条样本的最后有效 hidden state，避免 padding 部分影响表示。

### 4.1 优点

- 能建模历史顺序。
- 比 Transformer 更省显存。
- 对小数据集更稳，参数量相对可控。
- 可以自然利用 `history_length` 处理变长序列。
- 适合捕捉用户状态随历史观看行为逐步变化的趋势。

### 4.2 缺点

- 长序列建模能力有限。
- 串行结构训练速度相对慢。
- 难以直接建模当前候选 item 对不同历史 item 的注意力。
- 对很长历史序列，早期信息可能被弱化。

### 4.3 适用原因

当前数据量较小，GRU 是比较稳妥的第一版序列模型。如果 mean pooling baseline 跑通后希望加入顺序建模，GRU/LSTM 是比 Transformer 更保守的升级路线。

## 5. 历史序列建模方案 C：Transformer Encoder

第三种方案是用 Transformer Encoder 编码历史交互序列。

处理流程：

```text
每条历史记录 -> history step embedding
加入位置编码
TransformerEncoder
使用 padding mask 忽略补齐位置
pooling 或 CLS token -> history representation
```

### 5.1 优点

- 能建模任意历史位置之间的关系。
- 更适合捕捉 EEG 和情绪状态随历史交互变化的复杂动态。
- 表达能力强。
- 后续容易扩展为多模态序列建模。
- 与“让模型自己从原始历史序列提取特征”的设计目标一致。

### 5.2 缺点

- 参数更多，小数据集上更容易过拟合。
- 计算量和显存开销更高。
- 对 `max_history_len`、dropout、weight decay、embedding 维度等超参数更敏感。
- 如果历史序列过长，训练成本会明显增加。
- 需要严格处理 padding mask，否则 padding 会污染注意力结果。

### 5.3 适用原因

Transformer 适合作为后续主模型方向，但不建议作为第一步直接实现。更稳妥的路线是先跑通 mean pooling 或 GRU baseline，再尝试 Transformer。

## 6. 历史序列建模方案 D：候选 item 感知的 DIN 风格注意力

第四种方案是让当前候选 item 作为 query，对用户历史序列做 attention。

其核心思想不是把历史编码成一个与当前 item 无关的固定向量，而是根据当前候选 item 动态选择最相关的历史交互。

处理流程：

```text
当前 item embedding = query
历史 step embeddings = keys/values
attention(query, history)
得到与当前 item 最相关的历史表示
```

### 6.1 优点

- 非常适合 CTR 推荐任务。
- 能回答“用户过去哪些 item 或状态对当前候选 item 最相关”。
- 比完整 Transformer 更轻量。
- 可解释性更强，可以查看历史 attention 权重。
- 能显式建模当前候选 item 与历史 item/EEG/情绪状态之间的匹配关系。

### 6.2 缺点

- 主要建模当前 item 与历史 item 的相关性，对历史内部时序关系建模较弱。
- 需要设计 query/key/value 的维度和融合方式。
- 如果当前 item embedding 或 item_meta 表示质量不高，attention 效果会受影响。
- 如果历史很短，attention 的优势不明显。

### 6.3 适用原因

该方案非常适合当前任务，因为 CTR 推荐本质上是候选 item 条件下的历史兴趣匹配。相比 Transformer，它更轻量，也更贴近推荐系统经典做法。

## 7. 推荐推进路线

建议分三阶段推进。

### 7.1 第一阶段：StrictPreCTR-MEAN baseline

目标是先跑通新数据集和训练流程。

使用特征：

```text
user_id
item_id
user_meta
item_meta
c_video_type_c
history_item_id
history_eeg_310
history_interest
history_immersion
history_valence
history_arousal
```

历史序列只做 masked mean pooling。

特点：

- 简单稳定。
- 便于排查数据和维度问题。
- 可作为后续复杂模型的对照组。

### 7.2 第二阶段：StrictPreCTR-DIN attention

在 mean pooling baseline 基础上，引入候选 item 感知的历史注意力。

特点：

- 更符合推荐任务。
- 轻量，适合当前小数据集。
- 可以解释当前 item 关注了哪些历史交互。

该方案建议作为优先主模型。

### 7.3 第三阶段：StrictPreCTR-Transformer

在数据、baseline、attention 模型都稳定后，再尝试 Transformer。

特点：

- 表达能力最强。
- 适合建模 EEG 和情绪状态动态。
- 但需要更强正则化和更小规模设计。

## 8. 各类特征编码建议

### 8.1 user_id 和 item_id

可以继续保留 LightGCN，但需要谨慎。

如果 train/dev/test 是按用户划分的，那么 dev/test 用户没有参与训练图传播，`user_id` embedding 泛化能力会比较弱。

此时：

- `item_id` embedding 仍然有价值。
- `user_id` embedding 对冷用户帮助有限。
- 用户元特征和历史序列特征会更加重要。

可选做法：

- 保留 LightGCN 作为 ID 协同分支。
- 同时加入 `user_meta` 和 `history sequence` 分支弥补冷用户问题。
- 后续可以弱化 user_id embedding，增强 history encoder。

### 8.2 user_meta

用户元特征包括：

```text
u_gender_c
u_age_f
u_usage_f
```

建议：

- `u_gender_c` 可作为类别型 embedding。
- `u_usage_f` 可作为类别型 embedding 或连续特征。
- `u_age_f` 可以归一化后输入 Linear，也可以先作为连续值。
- 最终编码成一个 `user_meta_emb`。

### 8.3 item_meta

物品元特征较多，大多是视觉/音频连续特征：

```text
i_music_tempo_f
i_count_f
i_height_c
i_width_c
i_brightness_f
i_dif_brightness_f
i_E_2D_f
i_dif_E_2D_f
i_contrast_f
i_laplace_var_f
i_color_cast_f
i_hue_f
i_dif_hue_f
i_saturation_f
i_dif_saturation_f
i_value_f
i_dif_value_f
```

建议：

```text
i_* concat -> normalization -> MLP -> item_meta_emb
```

注意必须考虑数值尺度问题。例如：

- `i_laplace_var_f`
- `i_count_f`

这类字段数值范围较大，如果不归一化，可能主导训练。

### 8.4 c_video_type_c

`c_video_type_c` 建议作为类别型特征处理：

```text
video_type_emb = Embedding(num_types, dim)
```

也可以第一版继续使用类似当前模型的：

```text
Linear(1, 16)
```

但从语义上看，embedding 更合理。

### 8.5 history_eeg_310

历史 EEG 的输入形状是：

```text
[batch_size, history_len, 310]
```

有两种编码方式：

1. 使用 MLP：

```text
310 -> 64 -> eeg_dim
```

2. 复用当前 DGCNN：

```text
history_eeg_310 -> reshape [batch, history_len, 62, 5]
每个历史 step 跑 DGCNN -> eeg_dim
```

建议第一版使用 MLP。

原因：

- 速度更快。
- 实现更简单。
- 便于先验证新数据链路。
- 每个历史 step 都跑 DGCNN 会显著增加计算量。

后续可以再把 MLP 替换为 DGCNN，对比 EEG 图结构建模是否有收益。

### 8.6 历史四类情绪评分

历史情绪评分包括：

```text
history_interest
history_immersion
history_valence
history_arousal
```

每个历史 step 可以拼成：

```text
[interest, immersion, valence, arousal]
```

再经过：

```text
Linear(4, emotion_dim)
```

得到历史情绪 embedding。

### 8.7 历史 step embedding

每条历史记录可以编码为：

```text
history_step_emb =
    history_item_emb
    + history_eeg_emb
    + history_emotion_emb
```

或：

```text
history_step_emb = MLP(
    concat(history_item_emb, history_eeg_emb, history_emotion_emb)
)
```

第一版建议使用 concat + MLP，因为不同模态维度和语义差异较大，MLP 更灵活。

## 9. 融合方式建议

最终融合可以有两种路线。

### 9.1 concat + MLP

将所有分支表示拼接：

```text
user_id_emb
item_id_emb
user_meta_emb
item_meta_emb
video_type_emb
history_repr
```

然后：

```text
concat -> MLP -> pCTR
```

优点：

- 实现简单。
- 适合作为第一版 baseline。
- 方便排查维度和数据问题。

缺点：

- 特征间交互建模能力较弱。
- 不如注意力融合灵活。

### 9.2 token attention 融合

延续当前 `CTRLightGCN` 的设计思想，将不同来源特征都变成 token：

```text
user_id token
item_id token
user_meta token
item_meta token
video_type token
history token
```

然后使用：

```text
token gate
self-attention
global token cross-attention
predictor
```

优点：

- 复用当前模型已有设计思想。
- 各模态特征边界清楚。
- 后续扩展方便。
- 能建模不同模态之间的相互关系。

缺点：

- 比 concat + MLP 更复杂。
- 小数据集上更容易过拟合。
- 调参成本更高。

## 10. 最建议的第一版实现

第一版不要一步到位 Transformer，建议实现：

```text
CTRLightGCNStrictPreCTRMean
```

结构：

```text
当前 user/item 表示
+ user_meta MLP
+ item_meta MLP
+ video_type embedding
+ history step embedding mean pooling
-> fusion
-> pCTR
```

历史 step embedding：

```text
history_item_emb
+ history_eeg_mlp_emb
+ history_emotion_mlp_emb
-> history_step_emb
```

历史聚合：

```text
masked mean pooling(history_step_emb, history_length)
```

优点：

- 最稳。
- 最容易实现。
- 能最快验证新数据方案是否有效。
- 后续可以自然替换 history encoder 为 DIN、GRU 或 Transformer。

## 11. 后续模型版本建议

在第一版 mean pooling 跑通后，可以继续扩展：

```text
CTRLightGCNStrictPreCTRDIN
CTRLightGCNStrictPreCTRGRU
CTRLightGCNStrictPreCTRTransformer
```

推荐优先级：

```text
Mean baseline -> DIN attention -> GRU -> Transformer
```

其中 DIN attention 可以优先于 GRU，因为 CTR 推荐更关注“当前候选 item 与用户历史哪些交互相关”。

## 12. 注意事项

### 12.1 不要直接改旧 CTRLightGCNCTR

建议保留旧模型：

```text
CTRLightGCNCTR
```

作为初始版模型和对照组。

新模型应新增类，例如：

```text
CTRLightGCNStrictPreCTRMean
```

这样可以避免破坏旧实验。

### 12.2 第一版历史 EEG 建议用 MLP

虽然当前初始版模型中有 DGCNN，但第一版新模型不建议直接把历史 EEG 全部送入 DGCNN。

原因：

- 历史 EEG 形状是 `[batch, history_len, 310]`。
- 如果每个历史 step 都跑 DGCNN，计算成本较高。
- 当前派生数据集 CSV 已经较大，训练时再引入 DGCNN 会进一步增加压力。

建议：

```text
第一版：history_eeg_310 -> MLP
后续对照实验：history_eeg_310 -> DGCNN
```

### 12.3 必须处理 mask

历史序列是变长的，padding 后会出现无效位置。

所有历史聚合方式都必须使用 `history_length` 构造 mask。

否则 padding 的 0 会污染：

- mean pooling
- attention
- Transformer attention
- GRU 最后状态选择

### 12.4 注意数值归一化

`item_meta` 中存在尺度差异很大的连续特征。

如果直接输入 MLP，可能导致训练不稳定。

后续应考虑：

- 在数据处理阶段归一化。
- 或在模型内部使用 BatchNorm/LayerNorm。
- 或在 Reader 阶段增加标准化统计。

### 12.5 小数据集过拟合风险

当前数据总量只有几千条样本。复杂模型容易过拟合。

建议：

- 从简单 baseline 开始。
- 控制 embedding 维度。
- 使用 dropout。
- 使用 weight decay。
- 对 Transformer 层数和头数保持克制。

### 12.6 LightGCN 在按用户划分场景下要谨慎

如果 train/dev/test 按用户划分，那么 dev/test 用户没有参与训练图。

这会导致：

- dev/test 用户 ID embedding 学不到有效表示。
- LightGCN 用户侧泛化弱。

因此，新模型不应过度依赖 `user_id` embedding，而应更多依赖：

```text
user_meta
history sequence
item_meta
```

## 13. 总结建议

客观建议如下：

1. 不要直接改旧 `CTRLightGCNCTR`，保留它作为初始版模型。
2. 新增严格前置模型类，例如 `CTRLightGCNStrictPreCTRMean`。
3. 第一版用 MLP + masked mean pooling 跑通。
4. 第二版做候选 item 感知 attention，这是更适合推荐任务的增强。
5. 第三版再做 Transformer，用于建模更复杂的 EEG/情绪历史动态。
6. 历史 EEG 第一版先用 MLP，后续再对比 DGCNN，避免一开始计算量过大。

一句话总结：

> v1 模型应先从简单稳定的 `history step embedding + masked mean pooling + 当前 item/user/meta 特征融合` 开始，验证严格前置数据方案有效后，再逐步升级为 DIN attention、GRU 或 Transformer 历史序列建模。
