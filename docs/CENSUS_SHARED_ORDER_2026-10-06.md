# 普查：公开 EEG/fNIRS 数据集的呈现顺序（2026-10-06）

调研方法：由子代理读取事件文件（events.tsv、.evt、mrk.mat、notebook.mat）或数据说明，只下载小文件。
标"本会话核实"的条目由本会话亲自从数据中确认；标"未核实"的条目只有文档说明。

## A. 共用顺序（阳性）

| 数据集 | 模态 / 人数 | 顺序结构与证据 | 原始连续数据 + 事件 | 获取方式 |
|---|---|---|---|---|
| **Shin 2018 n-back** | EEG+fNIRS / 26 | 每个 run 9 个区组，所有人相同（**本会话核实**，原始 marker） | 有 | 本地 |
| **Tufts fNIRS2MW** | fNIRS / 68 | 16 个区组 `0123123023013012`，所有人相同（**本会话核实**：9 个完整被试 + 68 人的最后 4 个区组）；**休息时长随负荷变化** 20/20/30/40 s（本会话核实） | 只有 9 人有完整原始数据；68 人有最后 4 个区组 | 本地 |
| **PhysioNet mental-fnirs**（Mukli 2021） | fNIRS + TCD / 14 | 所有人固定为 0-back、1-back、0-back、2-back（每段约 134 s）；.evt 按**位置**编码 1–4（SUBJ001/002/005 已读）。2-back 永远在最后，**0 vs 2 与时间完全混淆** | 有（NIRx wl1/wl2 + .evt） | 97 MB，PhysioNet 开放获取 |
| **ds004514**（Rybář，Sci Data 2025） | EEG+fNIRS / 12 | 被试 02–12 的类别 / 任务 / 刺激序列哈希完全相同（数据层面核实）；论文自承"same seed for all participants except participant 1" | 有，BIDS | 25.9 GB |
| **Chen 2023 Stroop**（Sci Data，PMC10497617） | EEG+fNIRS / 21 | 论文写 "neutral, neutral, incongruent, incongruent"，**但数据显示实际为 中性-不一致-中性-不一致（交替），所有人相同**（本会话核实，见 `docs/CHEN_STROOP_2026-10-07.md`） | 有 | figshare，CC0 |
| **Tufts NIRS-X 听觉 n-back**（UIST 2024） | fNIRS / 53 | 实验室页面写明与视觉版相同的 16 区组顺序（仅文档，**未核实**数据） | 有 | Box，CC-BY-4.0 |
| 高原 Stroop（Sci Data 2024，PMC11001964） | EEG + 4 通道 fNIRS / 80 | 所有人同一个 120 试次序列（读取了压缩包内的说明文件） | 只有 2 s 分段，没有开始前窗口 | figshare，CC0 |
| PhysioNet EEGMAT | EEG / 36 | 静息永远在任务之前 | 只有 60 s 清洗后的片段 | PhysioNet |
| Shin 2017 A+B | EEG+fNIRS / 29 | 只在 session 层面共用（MI / MA 交替），session 内随机 | 有 | TU Berlin |
| ZuCo 1.0 / 2.0 | EEG + 眼动 / 12 / 18 | 原文写明区组和句子顺序 "identical for all subjects" | 有 | OSF |

## B. 每人顺序随机（阴性对照）

| 排名 | 数据集 | 证据 | 注意 |
|---|---|---|---|
| **1** | **Shin 2018 词语生成（WG vs 基线）**，本地已有，与 n-back 是同一批被试、同一台设备、同一个 session | **本会话核实**（EEG marker）：26 人共 24 种不同的 3-run 序列（VP017/018/023 相同）；每个 run 10/10；各位置 P(WG) = 0.37–0.63 | 随机规则对所有人相同：P(与上一个试次不同) = 0.62，存在"上一试次残留"这一共享通道 |
| 2 | Shin 2017 dataset B（心算 vs 基线），29 人 | 读取全部 29 人的 mrk.mat：87 个 run 中有 84 种不同序列 | 成对约束：每对试次 (2k, 2k+1) 中两种条件各一个（870/870），只有每对的第一个试次可以当干净的阴性对照 |
| 3 | COG-BCI（Hinss 2023），EEG，29 人 × 3 session | notebook.mat：87 个顺序中有 86 种不同 | 0/1/2-back 是分开的录音文件，没有跨条件的开始前窗口 |
| 4 | ds007554（Ajra 2026），EEG+fNIRS，30 人 | 论文称 "randomised order" | 数据中**无法核实**顺序（没有 scans.tsv，时间戳是转换时间） |
| 5 | ds006459 / ds006460 Stroop，fNIRS，17 人 | 17 人的序列全部不同 | 范式不是负荷 n-back |
| — | Kernel Flow n-back（43 人） | 论文称 "pseudo-randomized across … participants" | 没找到公开数据 |
| — | Herff 2014 | "pseudo-randomized" | 需要向作者申请，未核实 |

## C. 方法学注意

**随机化约束本身就是一个共享通道。** 每人随机的顺序如果遵循同一个约束（Shin 2017 的成对约束、Shin 2018 WG 的切换概率 0.62），仍然会形成跨被试一致的"上一试次 → 当前试次"关系。
阴性对照分析必须报告这个约束能带来的上限（"上一试次预言器"），并在固定上一试次标签的条件下检验当前标签。

## D. PhysioNet mental-fnirs 实测（2026-10-07，`scripts/27_mentalfnirs_position.py`，`results/mentalfnirs/`）

- 14 名被试中只有 **12 人**有 fNIRS 数据。顺序已从 .evt 核实：12 人都是 4 个区组（每个约 134 s，间隔约 10 s），事件码 1/2/4/8 按**位置**编码，条件为 0-back、1-back、0-back、2-back。采样率 3.9 Hz，48 个通道。
- MNE 自带的 NIRx 读取器会因为 .evt 末尾的空行报错，改用项目自己的 `eegfnirs.data.nirx` 读取。
- 测试：位置 1 和位置 3 都是 0-back。如果跨被试能区分它们，说明解码器读到的是时间，不是负荷。窗口为开始后 10–40 / 40–70 / 70–100 / 100–130 s，留一被试。
- **结果：所有对比都在机会水平，包括真正的负荷对比。** 0b vs 2：57%（p = 0.21）；同负荷的 0a vs 0b：49–55%（不显著）。
  使用全部 192 个特征时，0a vs 2 只有 31%（显著低于机会，是过拟合的迹象）；改用 4 个通道平均特征后，全部对比在 43–55% 之间。
- **结论**：12 人、每人每个条件只有一个区组，跨被试连负荷本身都解不出来，这个数据集**无法用于审计**（既不能支持也不能否定）。论文中只在普查表里列出它的顺序结构（"0 vs 2 与时间完全混淆"），并注明样本量不足以检验。
