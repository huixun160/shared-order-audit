# 查新报告：共用呈现顺序与跨被试泄漏（G0 关口）

日期：2026-10-06。对应 `docs/PLAN_SHARED_ORDER_AUDIT.md` 第 3 节。
方法：5 路并行检索（arXiv / OpenReview / bioRxiv / Europe PMC / Semantic Scholar / OpenAlex / 网页搜索，合计约 60 种说法），
关键条目由本会话亲自打开原文复核（标"**本会话核实**"）。

---

## 0. 结论先行

| 问题 | 结论 | 一句话 |
|---|---|---|
| **总体：共用顺序会让跨被试评估泄漏** | **部分已被做（定性）** | Barbu et al. 2015 在 fMRI 论文方法部分已用一段话指出该机制并给出对策；Li et al. 2021 写出了一般规则。Siskind 组两篇 ICLR 2026 投稿（已读全文）用了共用顺序数据，但只做被试内和"混合随机划分"，没做 LOSO。都没有量化跨被试泄漏、没有探针、没有普查 |
| Q1 有人指出 Tufts fNIRS2MW 固定顺序是混淆吗？ | **空白**（有未查到的缝隙，见 §6） | 所有描述顺序的论文都称其 "counterbalanced"，无人当作混淆 |
| Q2 有人指出 ZuCo 共用顺序导致跨被试泄漏吗？ | **空白**（顺序角度）；句子重叠泄漏**已被做** | ZuCo 原文自己写了顺序对所有被试相同，但没人把它当泄漏来源 |
| Q3 "开始前时间窗"作为跨被试泄漏诊断？ | **部分** | 被试内的开始前检查是老方法；跨被试 + 开始前窗口未见 |
| Q4 共用顺序数据集普查？ | **空白**（普查本身）；相邻工作存在 | 只有 Hu et al. 2021 在 4 个情绪数据集上做了"共用刺激"版本 |
| Q5 2026 年 6–10 月新论文？ | 未见直接撞车 | 泄漏审计类论文很活跃；ICLR 2027 投稿尚未全部公开，需持续盯 |

**对计划的含义**：不能声称"首次发现共用顺序会穿透 LOSO"。可以声称的新东西是：
(1) **定量**：在多个公开 EEG/fNIRS 数据集上测出泄漏幅度；
(2) **探针**：便宜、与模型无关的开始前窗口 LOSO 探针，并排除滤波伪迹；
(3) **普查**：共用顺序数据集清单 + 每个数据集的探针结果；
(4) **机制与边界**：解释为什么有的数据集上 LOSO 能消掉顺序效应（Li/Ahmed 的 OP1、Xu 的西瓜实验都回到机会水平），有的不能；
(5) 检验 Barbu 提出的对策（被试 × run 交叉验证）是否足够。

---

## 1. 最接近的先行工作（必须引用并区分）

### 1.1 Barbu et al. 2015 —— 最直接的先例（**本会话核实**）

- Barbu, Siddharth, Xiong, Corso, Fellbaum, Hanson, Hanson, Hélie, Malaia, Pearlmutter, **Siskind**, Talavage, Wilbur.
  "The Compositional Nature of Event Representations in the Human Brain". arXiv:[1505.06670](https://arxiv.org/abs/1505.06670)（2015）。fMRI，8 名被试，动词视频。
- 原文（PDF 全文提取）："All subjects were presented with the same stimuli and presentation order within and across runs."
  以及："Since the presentation order did not vary by subject, it is conceivable that classifier performance is due, in part, to the current stimulus in the context of previous stimuli in the same run"。
  对策："randomising presentation order across subject"，或 "Cross validation by run is an alternative control"。
- **区别**：只是一段方法说明，未量化、无探针、单数据集、fMRI。
- **对我们的影响**：核心机制不是新的。另外，它的对策"按被试 × run 交叉验证"必须在我们的数据上作为对照来测：在 Shin 2018 上三个 run 的顺序构成拉丁方，跨 run 训练/测试会打破 run 内的"前一区组→当前区组"映射，可能足以消除泄漏；在 Tufts 上只有一个 16 区组序列，没有 run 可以跨，这条对策用不了。

### 1.2 Siskind 组的其他工作

| 工作 | 核实 | 跨被试？ | 与本文关系 |
|---|---|---|---|
| Li, Johansen, Ahmed, …, Siskind. "The Perils and Pitfalls of Block Design for EEG Classification Experiments". *IEEE TPAMI* 43(1), 2021. doi:[10.1109/TPAMI.2020.2973153](https://doi.org/10.1109/TPAMI.2020.2973153)；accepted 版 [par.nsf.gov/10284053](https://par.nsf.gov/servlets/purl/10284053) | 全文（agent） | 只对 OP1 数据重做 LOSO，结果**机会水平** | 写出一般规则：训练集和测试集"collected with the same stimulus-class presentation order"就会污染。自有 6 名被试顺序相同但**明确不做跨被试分析** |
| Ahmed, Wilbur, Bharadwaj, Siskind. "Confounds in the Data—Comments on 'Decoding Brain Representations by Multimodal Learning…'". *IEEE TPAMI* 44(12):9217–9220, 2022. doi:10.1109/TPAMI.2021.3121268, [PMC9719395](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9719395/fullTextXML) | 全文（agent） | OP1 LOSO → **机会水平**，原文称 "analyzed across subjects to eliminate this confound" | 计划书里写的标题有误，已更正。它把 LOSO 当作**解决办法**——与我们的主张正好相反，必须正面回应 |
| Kilgallen, Pearlmutter, Siskind. "The Repeated-Stimulus Confound in Electroencephalography". arXiv:[2508.00531](https://arxiv.org/abs/2508.00531)（2025） | 全文（agent），ID 已核实 | 否 | 重复刺激混淆，16 篇论文虚高 4.46–7.42 点。不涉及共用顺序 |
| "Randomized Trials in EEG Classification Experiments Are Mandatory as Drift Is Pervasive"，ICLR 2026 投稿，匿名，OpenReview [81YN7QFPi4](https://openreview.net/forum?id=81YN7QFPi4)（`paper/4793_*.pdf`） | **本会话核实**（全文） | 否：多名被试数据**混合后随机 80/20 划分**；把多个被试拼成一个"虚拟被试" | 在 3 个公开数据集（两个静息态、一个听觉）上模拟区组设计，证明时间相关普遍存在，带通滤波去不掉。结论："the only known valid way … is through appropriate randomization" |
| "A Widely Used Protocol for EEG Classification Experiments Leads to a Confound"，ICLR 2026 投稿，匿名，OpenReview [UKPDpKGXAi](https://openreview.net/forum?id=UKPDpKGXAi)（`paper/4802_*.pdf`） | **本会话核实**（全文） | **否**：只做被试内分析，以及"6 名被试混合后随机划分"（pooled），**没有留一被试** | 新采数据：6 名被试，**A 组 4 个 session 所有被试顺序相同**，**B 组按被试随机顺序**。条件 I–IV 是 A/B 之间交叉训练/测试；训练与测试来自不同区组、不同顺序时 → 机会水平。原文明确反驳"pooling multiple subjects"能消除混淆——但用的是混合随机划分，不是 LOSO |

**风险提示（更新）**：
- Siskind 组是最可能做"跨被试版本"的团队。UKPDpKGXAi 的数据（A 组共用顺序 / B 组随机顺序，同一批被试）**只差一步**：在 A 组上做 LOSO 就是我们的核心实验。他们没做，但随时可能做。
- 反过来，这份数据如果公开，就是**理想的同被试内阳性/阴性对照**（同一批人，一半共用顺序，一半随机顺序）。论文称"we release our dataset"，但投稿匿名，链接未给出。第二阶段要找。
- 两篇都只针对视觉图像分类（Spampinato 数据集争议），不涉及 fNIRS、负荷任务，也没有开始前窗口。

### 1.3 反证：LOSO 在别处把顺序效应消掉了

- Li 2021 / Ahmed 2022：OP1 数据 LOSO → 机会水平（但 Li 原文承认不知道 OP1 各被试顺序是否相同）。
- Xu et al.（西瓜 "EEG"）附录 A.5：留被试外 → 机会水平（2.46% vs 2.50%），作者推荐留被试外作为对策。
  Xu, Wang, Xiao, Niu, Wang, Wu, Chen. "Beware of Overestimated Decoding Performance Arising from Temporal Autocorrelations in EEG Signals". arXiv:[2405.17024](https://arxiv.org/abs/2405.17024)；期刊版 "The impacts of temporal autocorrelations on EEG decoding", *BSPC* 113:108783 (2026)。

**我们需要的解释**：跨被试泄漏需要"与时间锁定、且跨被试一致的动态"——疲劳、警觉度、血流动力学残留、系统性漂移。西瓜没有这些；OP1 是快速图像呈现。区组设计的负荷任务（n-back）正好具备。这本身是一个可检验的论点（H4），应作为论文的一部分，而不是回避。
**也要如实面对**：Shin 上 EEG 只有 +1.7 点，fNIRS 不显著。如果 Tufts 也很弱，"LOSO 基本够用"可能才是正确结论（G1 关口）。

### 1.4 方法学威胁与概念先例

| 工作 | 核实 | 要点 |
|---|---|---|
| van Driel, Olivers, Fahrenfort. "High-pass filtering artifacts in multivariate classification of neural time series data". *J Neurosci Methods* 352:109080 (2021)；[bioRxiv 530220v2](https://www.biorxiv.org/content/10.1101/530220v2) | 预印本全文（agent） | 非因果高通（低至 0.1 Hz）会造成开始前伪解码；因果滤波"can never lead to displacement of information backward in time"。**Tufts 官方预处理是 3 阶零相位 Butterworth 0.001–0.2 Hz** → 必须从原始数据重新处理 |
| Grootswagers, Wardle, Carlson. *J Cogn Neurosci* 29(4):677–697 (2017)；arXiv:[1606.02840](https://arxiv.org/abs/1606.02840) | 全文（agent） | 开始前 100 ms 高于机会 = 预处理或交叉验证有错。被试内、随机顺序 |
| Schönmann, Szewczyk, de Lange, Heilbron. "Stimulus dependencies—rather than next-word prediction—can explain pre-onset brain encoding in naturalistic listening designs". *eLife* 14:RP106543 (**2026**)；[链接](https://elifesciences.org/articles/106543) | 全文（agent） | 刺激序列依赖产生开始前效应。概念同构，词级、MEG |
| Azizpour et al. arXiv:[2412.19622](https://arxiv.org/abs/2412.19622)；Abdoun et al. bioRxiv 10.1101/2025.09.16.676331 | 摘要（agent） | 同上，被试内 MEG/ECoG |
| Görgen, Hebart, Allefeld, Haynes. "The Same Analysis Approach". *NeuroImage* 2018；arXiv:[1703.06670](https://arxiv.org/abs/1703.06670) | 全文 grep（agent） | 把同一解码流程用在设计变量 / 零数据上；讨论了试次顺序混淆，指出 leave-one-run-out 下平衡设计也会失效。**探针的概念祖先**，被试内 fMRI |
| Madishetty. "CIPHER". arXiv:[2604.02362](https://arxiv.org/abs/2604.02362)（2026-03） | HTML（agent） | LOSO 下用"区组感知的标签置换"发现区组结构驱动近乎完美的准确率。单数据集，非开始前窗口，非共用顺序。**跨被试 + 设计结构的最近方法先例** |
| Hu, Zhao, Liu, Li, Lu. "A Novel Experiment Setting for Cross-subject Emotion Recognition". *EMBC* 2021, doi:10.1109/EMBC46164.2021.9630314 | 摘要（agent），全文未开 | 共用**刺激**（视频）使跨被试情绪识别偏高，4 个公开数据集。机制是刺激身份，不是位置/顺序 |
| Snoek, Miletić, Scholte. *NeuroImage* 2019；Kamrud et al. *Sensors* 21(9):3225 (2021)；Cai et al. *PLoS Comput Biol* 2019；Brookshire et al. *Front Neurosci* 2024 | 部分 | 混淆控制 / 被试泄漏 / 任务结构偏差。均不涉及共用顺序穿透 LOSO |

---

## 2. Q1：Tufts fNIRS2MW —— **空白**

- 原文（**本会话核实**，NeurIPS 2021 D&B [PDF](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/bd686fd640be98efaae0091fa301e613-Paper-round2.pdf)）：
  "the order of the values of n was predetermined, counter-balanced, and the same for every subject"；"a flattened version of a 4 × 4 array"（拉丁方）。正文写 "followed by 10-30 seconds of rest"。
- 拉丁方平衡的是**位置**，不平衡**一阶残留**（需要 Williams 设计）。序列 0-1-2-3-1-2-3-0-2-3-0-1-3-0-1-2 中，三个非首位的 0-back **全部**前接 3-back。可以据此反驳"已平衡"。
- **新发现、待核实**：agent 报告附录称休息时长随负荷变化（0/1-back 后 20 s、2-back 后 30 s、3-back 后 40 s），与正文 "10-30 s" 不一致。本会话在主文 PDF 中未找到附录这段（附录可能在补充材料）。**第二阶段从数据时间戳直接核实**——若属实，休息时长本身就是与标签绑定的第二个混淆，并直接影响"开始前窗口"的设计（前一区组休息越长，残留越少）。
- 引用论文：约 40 篇，22 篇读了全文，其中 10 篇使用数据。全部要么不提顺序，要么称 "counterbalanced"（Wang et al. 2024 arXiv:[2405.00213](https://arxiv.org/abs/2405.00213) / *Sensors* 25(12):3593 2025，[PMC12197191](https://pmc.ncbi.nlm.nih.gov/articles/PMC12197191/)）。
  Wang 发现区组之间差异很大，并把每个区组当作一个域来处理，但没有与顺序联系。
- Tufts 后续数据集 **NIRS-X**（UIST 2024，53 人，听觉 n-back）**使用同一固定顺序**（据数据集页面，agent 报告）→ 可以加入普查。
- **Huang 2021 的 OpenReview 审稿意见已读**（用户打印，`paper/111.pdf`，8 页，本会话逐页看过）：3 位审稿人和伦理审查分别讨论了数据格式、相关数据集、人口学代表性、generic 与 fine-tuning 的差距，**没有人提到区组顺序**。作者在回复中把 generic/fine-tune 的差距归因于 "Distribution shift over time"（每个被试的测试集固定为录音后 50%）。
  注意：在固定顺序下，"后 50%"对每个被试都是同一组区组位置（第 9–16 个），这一点也可以纳入分析。
- 读不到的：Long et al. 2026 *Cogn Neurodyn*（doi:[10.1007/s11571-026-10552-x](https://doi.org/10.1007/s11571-026-10552-x)，付费，PMC 禁运到 2027-12；在 Tufts 上做 0 vs 2-back，准确率 73.6%，结论是"延迟窗口重要"——**最可能涉及残留的一篇**）；NIRS-X 正文；IEEE/Springer 的 12 篇只看了摘要；Huang 2021 的 OpenReview 审稿意见（[QzNHE7QHhut](https://openreview.net/forum?id=QzNHE7QHhut)）。

## 3. Q2：ZuCo —— 顺序角度**空白**

- 事实本身不新：ZuCo 1.0（*Sci Data* 2018）/ 2.0（LREC 2020）原文写明区组和句子顺序 "identical for all subjects"。所以论文的贡献只能放在"作为混淆的量化与对照"上。
- **句子重叠泄漏（已被做）**：Yin et al. "Rethinking Cross-Subject Data Splitting for Brain-to-Text Decoding"，EMNLP 2025，arXiv:[2312.10987](https://arxiv.org/abs/2312.10987)（计划书里误记为 Jo et al.，已更正）；NEAR-Bench（ICLR 2027 投稿，[dDTERkcvdJ](https://openreview.net/forum?id=dDTERkcvdJ)）。
- 噪声基线：Jo et al. "Are EEG-to-Text Models Working?" arXiv:[2405.06459](https://arxiv.org/abs/2405.06459)。
- 相邻：COFETT（ACL 2026，arXiv:[2607.18749](https://arxiv.org/abs/2607.18749)）讲被试内漂移；**Bit Budget**（ICLR 2027 投稿，[AvTHamK9hj](https://openreview.net/forum?id=AvTHamK9hj)）发现 ZuCo EEG 几乎不携带文本信息，却能以 82.1% 解码任务/会话身份；**Whose Brain Is It?**（[QjEV2hgg3c](https://openreview.net/forum?id=QjEV2hgg3c)）发现检索余量的 94% 来自读者身份。
- "Removing Brainless Shortcuts"（[Gz6H1YZtjC](https://openreview.net/forum?id=Gz6H1YZtjC)）**与 ZuCo 无关**，讲的是重叠窗口泄漏单词时长（计划书描述有误）。
- **全文已核实**（用户手动下载，`paper/` 下）：Bit Budget（24 页）、Whose Brain Is It?（14 页）、NEAR-Bench（52 页）。全文搜索 presentation order / sentence order / serial position / drift / time-on-task / temporal confound：**没有一篇把顺序当作混淆**。
  Bit Budget 只在用 COFETT 时提到 "released presentation order"，作用是对齐试次；它的 82.1% 会话身份解码也没有归因到顺序。
  NEAR-Bench 的 "order" 只指注视词的顺序。

## 4. Q3–Q5 细节

- **Q3**：见 §1.4。"开始前窗口 + LOSO + 共用顺序"组合未见。最近的是 CIPHER（LOSO + 区组置换）和 SAA（设计变量探针）。
- **Q4**：没有多数据集的顺序普查。Hu 2021（共用刺激，4 个情绪数据集）是形式最接近的。
- **Q5**（2026-06 至 10 月）：Fan et al. arXiv:[2608.03176](https://arxiv.org/abs/2608.03176)（EEG-fNIRS，被试独立，**利用已知区组配额做匈牙利分配**——设计结构被主动利用的例子，只看了摘要）；Li & Tao arXiv:[2605.00865](https://arxiv.org/abs/2605.00865)（LOSO 泄漏审计，无顺序控制）；Zhang et al. arXiv:[2605.24524](https://arxiv.org/abs/2605.24524)（脑到语言来源归因）；"Best Practices in EEG Analysis" arXiv:[2609.36609](https://arxiv.org/abs/2609.36609)（零处提到顺序）。preprints.org 202609.1327（EEGMAT 泄漏，403，**未核实**——EEGMAT 在普查名单里，需要手动打开）。

## 5. 对计划书事实的更正

1. **Schroeder et al. 2025**（[PMC12259573](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12259573/fullTextXML)）："最多虚高 30 个点"不准确。30.4% 是 bootstrap 95% 置信区间的上界，并且是跨数据集平均；FBCSP 随机 k 折的平均增益约 19 点。原文称 Shin 顺序 "counterbalanced" 和 "pseudo-randomized"，全文只做被试内分析。
   **本会话已从原始 marker 核实**：26 名被试，每个 run 只有一种顺序（nback1 `320203032`、nback2 `203032320`、nback3 `032320203`）。"counterbalanced"只在被试内拉丁方意义上成立。
2. **Ahmed 2022** 标题见 §1.2。
3. **Khan et al. 2026**（arXiv:2606.16160）用的是**私有**驾驶模拟器数据（38 人，0/1/2-back，按被试 5 折，不是 LOSO）。`AUDIT_SCOPE.md` 里"两个独立评估收敛到约 55–56%"的说法仍成立，但要注明不同数据集、不同类别。
4. **BenchNIRS**：38.9%（LDA）是按被试分组 5 折，不是 LOSO。
5. **Schönmann** 的年份是 2026（eLife VOR）。
6. **arXiv 2312.10987** 是 Yin et al.，不是 Jo et al.。
7. **Wong et al.** "Preventing Data Leakage in Neural Decoding"，bioRxiv doi 10.64898/2026.01.26.701583：通用泄漏教程，不涉及顺序。作者列表各来源不一致，引用前需核对。

## 6. 尚未核实（需要你帮忙或接受为局限）

| 项目 | 原因 | 建议 |
|---|---|---|
| ~~OpenReview 6 项~~ | 已于 2026-10-06 由用户手动下载到 `paper/`，已全部核实 | — |
| Long et al. 2026 正文 | 付费，无访问权限 | **接受为局限**；论文中只引用其摘要结论 |
| UKPDpKGXAi 数据集是否公开 | 投稿匿名 | 2026-10-06 初搜未找到（只找到该组旧数据：IEEE DataPort "perils and pitfalls" 和 CVPR 2021 randomized trials, doi:10.21227/bc7e-6j47）；之后再查 |
| Tufts 休息时长随负荷变化 | 附录不在主 PDF | 第二阶段从数据核实 |
| Google Scholar "cited by" | 工具无法访问 | 接受为局限 |

## 7. 监控

- ICLR 2027 OpenReview 投稿（10 月中旬到 11 月陆续公开）：搜 "presentation order"、"block order"、"pre-onset"、"cross-subject leakage"。
- Kilgallen / Pearlmutter / Siskind 组的新 arXiv 论文。
- 投稿前重新搜一遍。
