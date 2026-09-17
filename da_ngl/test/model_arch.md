# AssimilationNetv6 结构图（中文）

> 图形文件：`model_arch.svg`（矢量图，中文由你本地字体渲染；浏览器 / Inkscape / Word 都能打开）
> 本文件是同一张图的 Mermaid 版 + 文字说明。

## 1. 整体数据流

```mermaid
flowchart TD
  BG["背景 bg&nbsp;&nbsp;(B, 1, 69, 80, 120)<br/>FuXi 预报 lead 6h / 24h<br/>已用 ERA5 mean/std 标准化"]
  OBS["观测 obs&nbsp;&nbsp;(B, 73, 4, 80, 120)<br/>NGL GNSS ZTD，73 帧 × 5 min<br/>通道 = ZTD + 掩码 + lat + lon"]
  PAD["replicate pad：120 → 128<br/>（H,W 需被 16 整除，输出再裁回）"]
  FLAT["展平 73 帧 × 4 通道 → 292 通道<br/>站外格点已置 0"]
  SIDE["side_info = concat(bg, obs)&nbsp;&nbsp;(B, 361, 80, 128)<br/>361 = 69 + 292"]
  B0["① AssimilationBlock 0<br/>三路编码 → FussionStack ×2<br/>69/292/361 → 128 通道，下采样 ×2 → H/2"]
  B1["② AssimilationBlock 1<br/>128 → 256 通道，下采样 ×2 → H/4"]
  B2["③ AssimilationBlock 2（用 DecoderBlock）<br/>256 → 128 通道，上采样 ×2 → H/2"]
  SK["跳连拼接 concat(bg0, bg2) → (B, 256, H/2)"]
  DEC["输出解码器 DecoderBlock<br/>256 → 128 → 70 通道，PixelShuffle 上采样 → H×W"]
  RES["残差相加&nbsp;&nbsp;out = decoder(...) + 背景<br/>只加在前 bg_chans 个通道"]
  ENH["④ EnhanceStack 1 / 2<br/>各带残差：out = enhance(out) + out"]
  OUT["分析场 (B, 1, 70, 80, 120)<br/>0–68 = 分析后的 ERA5，69 = tp"]

  BG --> PAD --> SIDE
  OBS --> FLAT --> SIDE
  SIDE --> B0 --> B1 --> B2 --> SK --> DEC --> RES --> ENH --> OUT
  B0 -. "bg0 跳连（128ch @ H/2）" .-> SK
  BG -. "残差基准 bg_cp" .-> RES
```

## 2. AssimilationBlock 内部（① ② ③ 都是这个结构）

```mermaid
flowchart LR
  BG["bg"] --> EB["bg_encoder<br/>EncoderBlock / DecoderBlock"]
  OB["obs"] --> EO["obs_encoder<br/>EncoderBlock / DecoderBlock"]
  SI["side_info"] --> ES["side_encoder<br/>EncoderBlock / DecoderBlock"]
  EB --> F["FussionStackv2<br/>重复 depth 次 FussionNetv2"]
  EO --> F
  ES --> F
  F --> O1["bg'  = bg + out_bg"]
  F --> O2["obs' = obs + out_obs"]
  F --> O3["side' = out_side（唯一不带残差的一路）"]
```

## 3. FussionNetv2 内部（融合单元）

```mermaid
flowchart TD
  C["concat(bg, obs, side)"] --> D0["downsample_0：卷积下采样 ×2<br/>(bg+obs+side) → embed_dim"]
  D0 --> D1["downsample_1：下采样 ×2<br/>embed_dim → 2×embed_dim"]
  D1 --> U1["upsample_bg_1 → concat(h0) → upsample_bg_0"]
  D1 --> U2["upsample_obs_1 → concat(h0) → upsample_obs_0"]
  D1 --> U3["upsample_side_1 → concat(h0) → upsample_side_0"]
  U1 --> R1["bg = bg + out_bg"]
  U2 --> R2["obs = obs + out_obs"]
  U3 --> R3["side = out_side"]
```

## 4. 关键数字（当前配置）

| 项 | 值 |
| --- | --- |
| 背景通道 | 69（默认）／70（`include_fuxi_tp=true`，多一列 FuXi tp） |
| 观测张量 | 73 帧 × 4 通道（absolute）/ 5 通道（innovation）/ 6 通道（both），站外为 0 |
| 输出通道 | 70（0–68 ERA5 状态 + 69 tp） |
| 网格 | 80 × 120，训练时内部填充到 80 × 128（能被 16 整除），输出裁回 |
| embed_dim / depth | 128 / (2, 2, 2) |
| 参数量 | 约 73.7 M |
| 空间流程 | H×W → H/2 → H/4 → H/2 → H×W |
| 残差位置 | ① FussionNetv2 的 bg/obs 两路；② 输出端 `out = decoder + 背景`；③ EnhanceStack ×2 |
| 损失 | 纬度加权 MAE（70 通道；可选站点加权 `loss_station_weight` / `loss_nostation_weight`） |

## 5. 几个容易看错的地方

1. **背景既当输入、又当输出端的残差基准**：所以"分析 = 背景 + 网络修正"，
   网络的初始状态就是恒等映射（这也是为什么之前所有实验里 `分析 ≈ 背景`）。
2. **tp 通道默认没有背景**：残差相加只覆盖前 `bg_chans` 个通道，第 69 通道（tp）
   由解码器直接预测；开启 `include_fuxi_tp` 后 FuXi 的 tp 进入背景，残差才覆盖它
   （实测 tp MAE 0.31 → 0.237）。
3. **FussionNetv2 的 side 支路不加残差**，只有 bg / obs 两路是残差更新。
4. **观测不是逐帧分别编码的**：73 帧 × 4 通道先展平成 292 通道，再作为一整块送进
   `obs_encoder`（时间信息完全交给卷积在这 292 个通道里自己组合）。


---

# 附：AssimilationBlock 与 FussionNetv2 逐层展开

> 对应的矢量图：`model_arch_blocks.svg`（同样全中文）

## A. AssimilationBlockv2

```mermaid
flowchart LR
  BG["bg (B, Cb, H, W)"] --> EB["bg_encoder<br/>EncoderBlock 或 DecoderBlock"]
  OB["obs (B, Co, H, W)"] --> EO["obs_encoder<br/>EncoderBlock 或 DecoderBlock"]
  SI["side (B, Cs, H, W)"] --> ES["side_encoder<br/>EncoderBlock 或 DecoderBlock"]
  EB --> FS["FussionStackv2<br/>FussionNetv2 × depth"]
  EO --> FS
  ES --> FS
  FS --> O1["bg'  = bg + out_bg"]
  FS --> O2["obs' = obs + out_obs"]
  FS --> O3["side' = out_side（不加残差）"]
```

三个 stage 的编码器/融合配置（当前 `embed_dim=128, depth=(2,2,2)`）：

| stage | 编码器类型 | 编码器形状 | 融合输入→输出 | 融合内部宽度 | 空间 |
| --- | --- | --- | --- | --- | --- |
| ① block_0 | EncoderBlock | 69/292/361 → 128 → 128 | 128/128/128 → 同 | 128 / 256 | H → H/2 |
| ② block_1 | EncoderBlock | 128 → 256 → 256 | 256/256/256 → 同 | 128 / 256 | H/2 → H/4 |
| ③ block_2 | DecoderBlock | 256 → 256 → 128 | 128/128/128 → 同 | 128 / 256 | H/4 → H/2 |

注意 stage ② 的融合内部仍然是 128 宽（`FussionNetv2` 的 `embed_dim` 固定为 128），
只有进出两端的通道数是 256——所以 `downsample_0` 的输入是 3×256=768 通道，输出 128。

## B. FussionNetv2 内部（以 stage 0 为例，三路各 128 通道 @ H/2）

```mermaid
flowchart TD
  C["concat(bg, obs, side) → (B, 384, H/2, W/2)"] --> D0
  D0["downsample_0 = EncoderBlock(384 → 128 → 128, act=T)<br/>Conv 2×2 s2 → LN → SiLU → Conv 3×3<br/>→ h0 (B,128,H/4,W/4)"] --> D1
  D1["downsample_1 = EncoderBlock(128 → 256 → 256, act=T)<br/>Conv 2×2 s2 → LN → SiLU → Conv 3×3<br/>→ h1 (B,256,H/8,W/8)"] --> U1
  D1 --> U2
  D1 --> U3
  U1["up_1: DecoderBlock(256 → 256 → 128, act=T)<br/>Conv3×3 → LN → SiLU → Conv3×3 → PixelShuffle(2)<br/>→ (B,128,H/4,W/4)"] --> CAT
  D0 -. "h0 同尺度跳连（128 通道）" .-> CAT
  CAT["concat → (B, 256, H/4, W/4)"] --> U0
  U0["up_0: DecoderBlock(256 → 128 → 原通道数, act 取决支路)<br/>→ (B,128,H/2,W/2)"] --> RES
  RES["bg = bg + out_bg    obs = obs + out_obs    side = out_side"]
  U2["obs 支路：同上"] --> RES
  U3["side 支路：同上，但 up_0 的 act=True，且不加残差"] --> RES
```

三条支路的差别（代码里唯一的不对称）：

| 支路 | `up_0` 的 `act` | 输出处理 | 原因 |
| --- | --- | --- | --- |
| bg | `False`（线性输出） | `bg = bg + out_bg` | 输出是残差修正量 |
| obs | `False` | `obs = obs + out_obs` | 同上 |
| side | `True`（多一层 LN+SiLU） | `side = out_side` | 直接替换，不是残差 |

## C. 四个基本积木的逐层定义

| 积木 | 层（`nn.Sequential` 顺序） | 空间 |
| --- | --- | --- |
| `EncoderBlock(in, embed, out, k=3, act)` | `Conv2d(in→embed, k=2, s=2, p=0)` → `ln_norm(embed, 1e-6)` → `SiLU` → `Conv2d(embed→out, k=3, p=1)` → `act ? ln_norm(out,1e-6)+SiLU` | ÷2 |
| `DecoderBlock(in, embed, out, k=3, act)` | `Conv2d(in→embed, k=3, p=1)` → `ln_norm(embed,1e-6)` → `SiLU` → `Conv2d(embed→out×4, k=3, p=1)` → `PixelShuffle(2)` → `act ? ln_norm(out,1e-6)+SiLU` | ×2 |
| `ln_norm(C, eps=1e-5)` | `permute(NCHW→NHWC)` → `LayerNorm(C)` → `permute` 回 | 不变 |
| `EnhanceStack(in, e)` | `EncoderBlock(in→e→e)` → `EncoderBlock(e→2e→2e)` → `DecoderBlock(2e→2e→e)` → `concat(x0, ·)` → `DecoderBlock(2e→e→in, act=False)`；调用处再 `out = enhance(out) + out` | 先 ÷2 再 ×2 |

容易看漏的三点：

1. **下采样用的是 2×2、stride 2、padding 0 的"patch 卷积"**，不是常见的 3×3 stride 2；3×3 那层是保持分辨率做特征变换。
2. **升采样用 `PixelShuffle(2)`**（先卷积出 `out×4` 通道再重排），不是转置卷积或插值。
3. **`ln_norm` 只沿通道维做 LayerNorm**，没有 BN 的 running stats，所以训练/推理完全一致——这也是我之前排查"训练与推理不一致"时排除掉的一环。
