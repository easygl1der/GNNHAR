# 报告结构草稿：基于 Zhang GNNHAR 框架的 Dow30、S&P 100 与 S&P 500 扩展实验

## 0. 写作定位

这份报告的目标不是完全复现 Zhang et al. (2024a) 的数值结果，而是在其 GNNHAR 框架下，用我们的数据和 implied volatility 扩展模型检验三个层次的问题。这里的基础模型来自 Corsi (2009) 的 HAR 思路，图结构扩展对应 Zhang et al. (2024b) 的 graph-based HAR / covariance forecasting line，非线性图模型对应 Zhang et al. (2024a) 的 GNNHAR。

1. GHAR、GNNHAR 及其多跳、多层、不同损失函数变体，从 Dow30 扩展到 S&P 100 再到 S&P 500 时，是否均能相对 HAR 提高预测准确性；
2. 在这些模型中加入 implied volatility 后，是否能带来额外预测贡献；
3. 当图节点数量和节点关系变得更丰富时，模型相对 HAR 的改进幅度是否进一步提高，也就是是否存在“提升的提升”。

主评估指标以 QLIKE 为核心，同时保留 MSE、MCS、DM、FVU、多层 / 多跳比较和 regime split 作为辅助证据。QLIKE 的使用参考 volatility forecast comparison 文献，尤其是 Patton (2011) 关于 imperfect volatility proxy 下 forecast comparison 的讨论；MCS 和 DM 分别对应 Hansen et al. (2011) 与 Diebold and Mariano (1995)。

### 0.1 从外行人和指导老师视角的当前阅读检查

这一小节只保留为写作定位提醒，具体内容已经吸收到后面的正文结构里。当前 draft 应该让一个不是 GNNHAR 或 volatility forecasting 领域的人也能读出四件事：我们为什么从 Zhang et al. 出发，为什么 implied volatility 是有经济含义的扩展，三张主表分别怎样回答 RQ1--RQ3，以及 S&P500 的异常结果为什么是需要诊断的研究现象，而不是简单删掉的坏数据。

后续写作要避免三个过强表述：第一，不能说本文完全复现了 Zhang 的数值结果，而应说我们复刻了 Zhang-style modeling and evaluation framework，并在不同数据源和更大 universe 上做扩展；第二，不能说 IV 的改善证明了因果效应，而应说 IV 版本提供了与 options-implied information 一致的预测增量证据；第三，不能说 S&P500 证明大图 GNN 无效，而应说当前实现和数据下没有支持“大图自动增强 GNNHAR 优势”的证据。

这份 draft 仍然是研究历程笔记。好的写法不是把所有内容改成正式论文口吻，而是保留我们如何提出问题、如何跑出结果、如何发现异常、如何一步步定位 `SATS` 和 S&P500 GNNHAR 问题的过程。

### 0.2 审稿意见后的叙述更新：守住本文重心

根据后续审稿式检查和作者讨论，正式 paper draft 的叙述重心需要进一步收紧。本文不应被写成“一个数据口径不够强但勉强做了 GNNHAR 的实验”，而应写成：

> 本文在 Zhang-style HAR / GHAR / GNNHAR 框架下，使用 option-linked volatility proxy 检验 implied volatility 与 graph neural depth 的相对预测价值。当前最稳定的结果不是“GNNHAR 在大图上全面胜出”，而是“IV 是稳定的信息来源，而 graph neural depth 在 QLIKE 和 large-universe setting 下并不稳健”。

这一定位有几个直接写作后果：

- 第一，30-day close-to-close historical volatility proxy 必须如实披露，不能把它写成 Zhang et al. 的高频 intraday RV。正式表述应使用 “daily variance-scale 30-day close-to-close historical-volatility proxy” 或 “proxy-volatility target”，避免让读者误以为我们在做 5-minute realized volatility 的直接复现。
- 第二，这个 proxy target 不应只被动写成缺陷。它更平滑、更规则，并且与 IV30 的 option horizon 自然对应。也就是说，如果 GNNHAR 在这个更平滑的目标上已经出现 large-universe instability，那么转向更高频、更非正则的 intraday RV 未必会让问题自动消失，反而可能放大训练和 QLIKE loss 的不稳定性。
- 第三，10-day HV/IV 数据已经是一个自然的 future robustness 方向，但当前 paper-ready 主结果仍是 30-day。因此 draft 里只能写 “10-day target is a sharper future robustness target”，不能写成已经完成的主证据。
- 第四，S&P500 calendar 不能再写成简单“数据源已经对齐”。实际情况是：Dow30 aligned full-model 和 S&P100 都是 234 dates，2025-07-07 到 2026-06-09；S&P500 是 223 dates，2025-07-14 到 2026-06-01。S&P500 的 223 个日期全部包含在 Dow30/S&P100 的 234-date calendar 内，缺少的是开头 5 个交易日和结尾 6 个交易日。因此正式写法应是 “nested near-calendar comparison” 或 “223-date subset of the 234-date Dow30/S&P100 calendar”，而不是完全不兼容的 calendar。
- 第五，MCS 小节已经应该同步为 Dow30、S&P100、S&P500 三组。Dow30 234-date MCS 已经补齐；仍需谨慎的是 aligned Dow30 regime diagnostics，而不是 MCS。
- 第六，Appendix 可以保留 repo paths、artifact paths、Colab / AutoDL / reproducibility notes，但正文必须把核心经济和统计证据前置：IV 的稳定预测价值、S&P500 GNNHAR low-volatility over-prediction、SATS stress case、以及 RQ3 不支持“larger graph automatically improves GNNHAR gain”。

因此后续写 paper 时，最推荐的中心论断是：

> We do not claim that GNNHAR universally dominates HAR. Instead, we show that when the Zhang-style graph-HAR/GNNHAR framework is extended with option-implied volatility and scaled across equity universes, the stable predictive information comes from IV, while nonlinear graph depth is fragile under QLIKE evaluation. This fragility appears even under a smooth 30-day volatility proxy, suggesting that graph neural volatility forecasting requires careful target choice, output scaling, and robustness diagnostics before being treated as a superior financial forecasting tool.

## 1. Introduction

### 1.0 Introduction 必须保留的作者论证链

这一节不能只写成“我们加入 IV 作为额外变量”这样一句话。Introduction 需要保留完整的研究动机链条，因为本文的贡献不是简单增加一个 predictor，而是从 Zhang et al. 的 future work 出发，解释为什么历史 RV 和图上的历史 spillover 信息本身存在经济含义上的边界。

建议按下面顺序写，不要跳步：

1. 先写 realized volatility forecasting 的传统背景。说明 HAR 为什么是强基准，因为它用日、周、月三个历史窗口刻画波动率的 persistence、clustering 和 heterogeneous time scales（Corsi, 2009；Andersen et al., 2001；Barndorff-Nielsen and Shephard, 2002）。
2. 再写 Zhang 的 GHAR 思路。HAR 只看单个资产自身历史，GHAR 把股票之间的 volatility spillover 加进来，用图邻接关系聚合邻居资产的历史 RV。这里要引用 Zhang 之前的 GHAR 文章或 Zhang et al. 对 GHAR 的描述，说明 graph-based HAR 是 HAR 的自然扩展（Zhang et al., 2024b）。
3. 再写 Zhang 的 GNNHAR。GHAR 本质上还是线性回归，GNNHAR 的关键改进是引入 neural network，在图上传播节点信息并加入非线性变换，用来捕捉 nonlinear volatility spillover。这是 Zhang 文章的核心贡献（Zhang et al., 2024a）。
4. 然后写 Zhang et al. 在 conclusion 中留下的 future work：扩展 predictor set，加入 limit order books、options 和 news。这里要明确说明，这不是随口一提，而是指出了“仅靠历史 RV 和历史价格信息”的局限（Zhang et al., 2024a；Li and Tang, 2021）。
5. 接着写我们的思考：Zhang 这个 future work 方向是有道理的，因为更高频的历史价格或历史 RV 并不必然充分预测未来波动。未来波动可能来自还没有完全进入历史价格的信息、尚未完全披露的消息、订单簿中的潜在流动性压力、以及期权市场中已经被交易出来的未来风险预期。
6. 最后引出本文：我们先选择 options / implied volatility 作为可结构化、可直接进入日度 panel 的第一步扩展，并在 Dow30、S&P100、S&P500 三个 universe 中检验 GHAR、GNNHAR 及其 IV 变体是否相对 HAR 改善预测。

这一段的核心作者表达需要保留：

- Zhang et al. 提到的 future work 有三类：limit order books、options 和 news；
- 我们不是把这三类都做完，而是把 options / IV 作为第一步，因为它最容易和日度 RV panel、HAR / GHAR / GNNHAR 框架对齐；
- 历史 RV 是 backward-looking 的，即使使用更高频数据构造 RV，也只是总结“已经实现的价格变化”；
- 金融市场中经常出现 good news is bad news 的情况，价格对新闻的方向和新闻表面含义可能不一致；
- 当前价格反映当前成交结果和当前供需关系，但不完整反映供需曲线的形状，也不完整反映订单簿厚度；
- 如果一边挂单薄、一边挂单厚，即使当前价格暂时稳定，后续较小的订单流也可能造成更大的价格跳动和 realized volatility；
- 如果市场中存在集中做市、撤单、订单簿某一侧突然变薄、或其他流动性结构变化，仅用日度历史 RV 很难提前判断；
- 新闻事件会导致波动，包括财报、利率决议、CPI、FOMC、就业和劳工市场数据等；
- 有些新闻尚未正式释放，但市场可能已经通过 options market 对未来不确定性进行定价；
- IV 可以理解为 options market 对未来波动的压缩统计量，反映市场参与者和资本对未来风险、尾部风险和不确定性的预期；
- 因此，本文把 IV 加入 HAR / GHAR / GNNHAR，是为了检验 forward-looking option-market information 是否能在历史 RV 和 graph spillover 之外提供增量预测力。

下面这些是作者原始思路，写正式正文时要尽量保留意思，不要被压缩成一句“IV contains forward-looking information”：

- “有时候更加高频的波动并不一定能通过历史信息来预测”。这里的意思是，哪怕 RV 是用高频价格构造的，它仍然是已经发生的价格变化；未来风险可能来自还没发生交易、还没完全公开、或者还没完全被历史价格吸收的信息。
- “good news is bad news”。这句话要保留，因为它说明新闻的表面语义和市场价格反应不一定同向。例如宏观数据表面利好，但市场可能因为担心利率路径、通胀粘性或政策收紧而提高风险定价。
- “价格始终反应的是当前的供需关系”。这句话要进一步写成：当前价格反映当前成交点，但不完整反映订单簿和供需曲线的形状；因此它不是未来波动的充分统计量。
- “一方挂单薄，一方挂单厚”。这句话对应 order-book depth 和 liquidity imbalance。它要用于说明为什么 limit order book 是 Zhang et al. future work 中自然的一类信息源：同一个当前价格背后可能对应完全不同的流动性结构。
- “有人想要集体做市，产生价格反常波动”。这里要谨慎学术化，可以写成 concentrated market making, strategic liquidity provision, order cancellation, or one-sided liquidity withdrawal may generate abnormal price movements. 重点不是断言操纵，而是说明交易层面的流动性结构可能导致历史日度 RV 无法提前反映的未来波动。
- “新闻确实会导致波动”。这一点要具体写财报、利率、CPI、FOMC、劳工就业率等事件，因为这些事件可以解释为什么 historical RV 之外的信息源有必要进入模型。
- “IV 蕴含着新闻，以及未被发出的新闻信息”。正式写法可以是：IV may embed public information, anticipated information releases, partially revealed private or institutional information, and compensation for tail risk. 这里不要写得像内幕交易断言，而要写成市场预期和风险补偿。
- “反映了资本的预期”。这一点可以写成：option prices aggregate the risk assessments and hedging demands of market participants; IV is therefore a market-implied summary of expected uncertainty.
- 本文的工作不是泛泛地说“更多变量更好”，而是沿着 Zhang et al. 的 future work，把 options market 的 forward-looking information 放进同一套 HAR / GHAR / GNNHAR 框架，然后用 Dow30、S&P100、S&P500 三个 universe 检验它是否真的改善预测。

可以在正式正文中写成如下逻辑段落：

> Zhang et al. leave an important direction for future work: expanding the predictor set to incorporate limit order books, options, and news (Zhang et al., 2024a; Li and Tang, 2021). This suggestion is economically meaningful rather than merely technical. Historical realized volatility, even when constructed from high-frequency prices, remains a backward-looking summary of realized price movements. Future volatility may instead be driven by latent order-book imbalance, liquidity fragility, news surprises, or forward-looking risk assessments that have not yet appeared in historical RV.

> In equity markets, the current price reflects the current transaction outcome and the prevailing supply-demand balance, but it does not fully reveal the shape of the supply-demand curve. A stock may appear stable at the current price while one side of the order book is thin and the other side is thick; in such a case, a relatively small future order flow can generate disproportionate price movement. Similarly, concentrated market making, order cancellation, or sudden thinning of one side of the book may generate abnormal volatility that is difficult to infer from daily historical RV alone.

> News provides another channel through which future volatility may deviate from what past volatility implies. Earnings announcements, CPI releases, FOMC meetings, labor market data, and interest-rate decisions often trigger sharp repricing. Moreover, the market may begin to price such uncertainty before the information is fully released. Implied volatility is therefore a natural first extension of the Zhang et al. framework because it summarizes option-market expectations about future volatility and may contain public news, not-yet-fully-public information, risk appetite, and tail-risk concerns (Bollerslev et al., 2018; Busch et al., 2010).

> This paper focuses on implied volatility as the first feasible extension among the three directions suggested by Zhang et al. Limit order book and news data are valuable, but they require high-frequency order-level synchronization or text-event alignment. IV can be organized as a daily panel and directly incorporated into HAR, GHAR, and GNNHAR. This allows us to ask whether forward-looking option-market information improves volatility forecasting beyond historical RV and graph-based spillover information.

### 1.1 Notations and empirical conventions

正式正文中 Introduction 下面需要先放一个 notations 小节，避免后面表格和公式突然出现大量符号。建议写成简洁但完整的符号说明：

- 用 $u\in\{\mathrm{Dow30},\mathrm{SP100},\mathrm{SP500}\}$ 表示股票 universe，用 $N_u$ 表示 universe $u$ 中进入模型的股票数量。当前正式结果中 $N_{\mathrm{Dow30}}=30$、$N_{\mathrm{SP100}}=91$、$N_{\mathrm{SP500}}=449$。
- 用 $i=1,\ldots,N_u$ 表示股票节点，用 $t$ 表示交易日。
- 用 $v_{i,t}$ 表示股票 $i$ 在日期 $t$ 的 daily realized variance target。当前数据中，它由 30-day close-to-close historical volatility proxy 转换到 daily variance scale；这和 Zhang 使用 5-minute intraday returns 构造 RV 不同，必须在数据节再次披露。
- 用 $IV30_{i,t}$ 表示 30-day mean implied volatility，用

$$
q_{i,t}
=
\left(\frac{IV30_{i,t}}{100}\right)^2/252
$$

表示转换到 daily variance scale 的 IV component。
- HAR 的历史 RV 特征写成

$$
x_{i,t-1}
=
\left(
v_{i,t-1},
\bar v_{i,t-5:t-2},
\bar v_{i,t-22:t-6}
\right),
$$

其中 $\bar v$ 表示对应窗口上的 average realized variance。
- IV extension 的节点特征写成

$$
z_{i,t-1}
=
\left(
v_{i,t-1},
\bar v_{i,t-5:t-2},
\bar v_{i,t-22:t-6},
q_{i,t-1}
\right).
$$

把所有节点堆叠后，$X_{t-1}\in\mathbb{R}^{N_u\times 3}$ 表示 non-IV feature matrix，$Z_{t-1}\in\mathbb{R}^{N_u\times 4}$ 表示 IV-augmented feature matrix。
- 用 $A_u$ 表示 GLASSO 估计得到的 adjacency matrix，用 $W_u=D_u^{-1/2}A_uD_u^{-1/2}$ 表示标准化后的 graph propagation matrix。GLASSO 的统计基础来自 Friedman et al. (2008) 的 sparse inverse covariance estimation。
- $K$-hop neighbors 指在图 $A_u$ 上距离不超过 $K$ 的节点集合。GHAR2H / GHAR3H 表示用 2-hop / 3-hop graph aggregation 的线性 graph-HAR 变体。
- $GNNHARkL$ 表示 $k$-layer GNNHAR，其中 $k\in\{1,2,3,4,5\}$。IV 版本写成 $GNNHARkL^{IV}$ 或在代码表格中写作 `GNNHARkL_*_IV`。
- 下标 $M$ 和 $Q$ 表示 estimation criterion。$M$ 表示模型用 MSE loss 训练，$Q$ 表示模型用 QLIKE loss 训练。例如 $GNNHAR2L_Q^{IV}$ 表示加入 IV、两层 GNN、用 QLIKE criterion 训练的模型。
- 对任一模型 $m$，预测写成 $\widehat v_{i,t}^{(m)}$。MSE 和 QLIKE loss 分别记为

$$
L^{MSE}_{i,t}(m)
=
\left(v_{i,t}-\widehat v_{i,t}^{(m)}\right)^2,
$$

$$
L^{QL}_{i,t}(m)
=
\frac{v_{i,t}}{\widehat v_{i,t}^{(m)}}
-\log\left(\frac{v_{i,t}}{\widehat v_{i,t}^{(m)}}\right)-1.
$$

- 主表中的 loss ratio 统一相对 $HAR_M$：

$$
R^{MSE}_{m,u}
=
\frac{\sum_{i,t}L^{MSE}_{i,t}(m)}
{\sum_{i,t}L^{MSE}_{i,t}(HAR_M)},
\qquad
R^{QL}_{m,u}
=
\frac{\sum_{i,t}L^{QL}_{i,t}(m)}
{\sum_{i,t}L^{QL}_{i,t}(HAR_M)}.
$$

对应 improvement 写成 $1-R^{MSE}_{m,u}$ 或 $1-R^{QL}_{m,u}$。RQ3 中的“提升的提升”就是比较这个 improvement 是否从 Dow30 到 S&P100 / S&P500 变大。
- MCS 表示 Model Confidence Set，用于识别在给定 confidence level 下不能从 best model set 中排除的模型（Hansen et al., 2011）；DM 表示 Diebold-Mariano pairwise forecast comparison，用于比较两个模型的 loss differential（Diebold and Mariano, 1995）。

这一节的写法要偏正式，避免像代码注释。它的作用是让读者后面看到 $HAR_M$、$GNNHAR2L_Q^{IV}$、$R^{QL}$、MCS star、DM p-value 时都知道含义。

### 1.2 研究背景：从 HAR 到 graph-based volatility forecasting

这一节需要按照 Zhang et al. 的行文方式展开，而不是一开始就直接进入我们的结果。

建议写成 4--5 个自然段：

第一段：realized volatility forecasting 的重要性。

- realized volatility forecasting 是金融计量、风险管理、资产配置、期权定价和压力测试中的核心问题（Andersen et al., 2001；Barndorff-Nielsen and Shephard, 2002）；
- 金融市场中的波动率具有 persistence、clustering 和 heterogeneous time-scale 的特征；
- HAR 模型之所以成为强基准，是因为它用日、周、月三个时间尺度刻画了市场参与者的异质记忆结构（Corsi, 2009）；
- 但 HAR 主要依赖单一资产自身的历史 realized volatility，对资产之间的 cross-sectional spillover 表达不足。

第二段：volatility spillover 和 GHAR。

- 实证金融中已经有大量文献说明，一个资产或市场的冲击会传导到其他资产或市场；
- Zhang et al. 之前的 GHAR 文章和后续 GNNHAR 文章都建立在这个思想上：如果股票之间存在 volatility spillover，那么预测单个股票的未来波动率时，不应只使用自身历史波动率，还应利用相邻股票的历史波动率（Zhang et al., 2024a, 2024b）；
- GHAR 的作用是把 HAR 与 graph information 结合起来，用图邻接矩阵聚合邻居节点的 realized volatility；
- 这一步的贡献在于把 HAR 从单资产时间序列模型扩展为 graph-based panel forecasting model。

第三段：GNNHAR 和非线性 spillover。

- GHAR 本质上仍然是线性模型，即邻居信息通过线性图聚合进入预测；
- Zhang et al. 的 GNNHAR 进一步引入 graph neural network，使模型能够捕捉 nonlinear volatility spillover（Zhang et al., 2024a）；
- GNNHAR 的优势在于：它不是预先规定 spillover 的线性形式，而是通过 neural network 在图结构上传播和变换节点信息；
- 这使得 GNNHAR 能够检验一个更强的问题：跨股票波动率关系是否不仅存在，而且具有非线性结构；
- 但 GNNHAR 也引入了新的问题，包括 hyperparameter sensitivity、深层 GNN 的 over-smoothing，以及在大图上的训练稳定性（Chen et al., 2020）。

第四段：Zhang et al. 留下的 future work 与 predictor set 扩展。

- Zhang et al. 在结论中明确提出，一个重要的 future work 是 expanding the predictor set，引入 limit order books、options 和 news 等额外信息来源（Zhang et al., 2024a；Li and Tang, 2021）；
- 这个方向是自然的，因为历史 realized volatility 本质上是 backward-looking information；
- 仅依赖历史价格或历史波动率，未必能捕捉市场对未来风险的预期，尤其是在信息冲击、流动性冲击和订单簿不平衡时。
- 这里要强调：Zhang et al. 的模型主要使用历史 RV 和 graph spillover。我们认为这个 future work 指向了本文最重要的扩展问题，即是否可以把市场对未来风险的定价也纳入同一套 Zhang-style forecasting framework。

这一段可以引用 Zhang 的原意，但不要长段照抄。可以写成：

> Zhang et al. point out that a natural direction for future research is to expand the predictor set by incorporating information sources such as limit order books, options, and news (Zhang et al., 2024a; Li and Tang, 2021). This observation is important because historical realized volatility summarizes past price movements, whereas financial markets often react to forward-looking expectations and latent information before such information is fully reflected in realized volatility.

### 1.3 为什么 IV 是自然扩展

这一节承接 Zhang 的 future work，把我们的 IV 扩展写出来。

建议写成 4--5 个自然段。这里要尽量保留我们的原始研究动机：Zhang et al. 的 future work 不只是形式上提到更多 predictor，而是确实指出了历史 RV 信息的边界。金融市场的未来波动并不总是能从历史价格或历史波动率中充分推断出来。

第一段：历史信息的局限。

- 更高频的历史价格和 realized volatility 并不一定充分预测未来波动；
- 金融市场中经常出现消息解释和价格反应方向不一致的情形，例如 good news is bad news；
- 即使价格已经反映当前成交结果，也未必反映订单簿深度、潜在流动性压力和未来风险预期；
- 一边挂单薄、一边挂单厚时，未来价格冲击和 realized volatility 可能并不由过去价格本身充分刻画。
- 价格始终反映的是当前成交和当前供需关系，但它未必完整揭示供需曲线的形状：如果买盘很薄、卖盘很厚，或者反过来，即使当前价格变化不大，后续一点点订单流也可能导致更大的波动；
- 因此，单纯用历史 RV 做预测，本质上是在用过去已经实现的价格变化推断未来风险，但未来风险有时来自尚未完全成交、尚未完全披露、或尚未完全被历史价格吸收的信息。

这一段可以写成接近如下的正式表述：

> Historical realized volatility is backward-looking by construction. Even if high-frequency prices are used to construct RV, the resulting measure summarizes realized price movements rather than the latent supply-demand imbalance, unexecuted order pressure, or forward-looking risk assessments embedded in other markets. In this sense, a richer predictor set is not merely a technical extension; it is economically motivated by the fact that future volatility may be driven by information that has not yet appeared in historical RV.

第二段：order book、news、options 三类信息的经济含义。

- limit order book 可以反映未成交订单、买卖盘厚度、流动性缺口和潜在冲击成本；
- order book 尤其能捕捉“当前价格看起来平稳，但订单簿结构已经脆弱”的情形；
- 如果市场中有人集中做市、撤单、或者某一侧挂单突然变薄，价格可能出现反常波动；这种信息通常不在历史日度 RV 中直接出现；
- news 会在财报、CPI、FOMC、就业数据、利率决议等事件期间引发剧烈波动；
- 新闻信息不仅包括已经公开的新闻，也包括市场对即将到来的新闻事件的预期，例如财报前、CPI 公布前、FOMC 前、就业数据前，市场通常会提前调整风险定价；
- options market 中的 implied volatility 则是市场参与者对未来波动的价格化预期；
- IV 可能包含公开新闻、未完全公开的信息、风险偏好、尾部风险担忧和机构资金对未来不确定性的定价。
- 从这个角度看，IV 是 options market 对未来风险的压缩统计量。它并不告诉我们具体是哪一条新闻、哪一类订单流或哪一个机构预期导致风险上升，但它把这些预期以 option prices 的形式反映出来。

第三段：为什么 options / IV 是我们当前最适合处理的信息源。

- Zhang et al. 提到 limit order books、options 和 news 三个方向；
- limit order book 数据非常有价值，但通常需要高频、盘口级别数据，数据获取和同步成本较高；
- news 数据也有价值，但需要文本识别、事件分类、发布时间对齐和可能的情绪 / surprise 度量；
- IV 数据相对更容易结构化为日度 panel，并且直接对应未来波动预期；
- 因此，IV 是一个自然的第一步：它把 options market 的 forward-looking information 加入 Zhang-style GNNHAR 框架，同时保持模型和评估流程相对清晰。

第四段：我们为什么选择 IV。

- 在 Zhang 提到的三个方向中，options 是最适合先纳入 GNNHAR 框架的 predictor source；
- IV 与 realized volatility 有清晰经济联系：RV 是过去实现的波动，IV 是市场对未来波动的风险中性预期；
- 因此，把 IV 加入 HAR / GHAR / GNNHAR，可以检验 forward-looking option-market information 是否能在历史 RV 和 graph spillover 之外提供增量预测力；
- 这也是本文相对 Zhang et al. 的主要扩展。

第五段：把这个思想和本文三个 RQ 接上。

- 如果 IV 只是重复历史 RV 中已有的信息，那么加入 IV 后的模型不应显著改善；
- 如果 IV 包含额外 forward-looking information，那么 `HAR_IV`、`GHAR_IV`、`GNNHAR_IV` 应该相对 non-IV 版本更好；
- 这里尤其要用 S&P500 讲清楚机制：不加入 IV 时，大 universe 的 historical-only 线性图模型几乎不能明显改善 `HAR_M`，而 non-IV GNNHAR 甚至明显变差；加入 IV 后，`HAR_M_IV` / `GHAR_M_IV` 成为最优或并列最优行，说明 IV 中确实包含历史 HAR lag 没有吸收的 forward-looking option-market information；
- 但也不能把 S&P500 写成 “GNNHAR-IV 已经成功”。更准确的结论是：IV 让 GNNHAR 的信息集更有意义，部分 QLIKE-trained GNNHAR-IV 行相对 non-IV GNNHAR 有改善，例如 S&P500 中 `GNNHAR2L_Q` 的 QLIKE ratio 从约 8.152 降到 7.802，`GNNHAR4L_Q` 从约 9.965 降到 7.808；但这些模型仍远差于 `HAR_M_IV` / `GHAR_M_IV`。因此 IV 改善了 nonlinear graph model 的信息基础，但没有解决当前 S&P500 GNNHAR 的尺度和低波动股票高估问题；
- 如果更大的图能够更好地利用 cross-sectional option-implied information，那么从 Dow30 到 S&P100 再到 S&P500，IV-augmented GHAR/GNNHAR 相对 HAR 的提升幅度应当进一步扩大；
- 这正是本文 RQ2 和 RQ3 的核心。

### 1.4 本文研究问题

把上面的动机压缩成本文的问题意识，可以这样理解：Zhang et al. 已经说明，历史 RV 的自身记忆和图上的 volatility spillover 都有预测价值；但他们的框架主要还是从 historical realized information 出发。我们的研究不是简单多加一个变量，而是问：如果把 options market 对未来风险的定价也放进同一套 HAR / GHAR / GNNHAR 框架，模型的预测表现会不会更好？进一步，如果 universe 从 Dow30 扩大到 S&P100、S&P500，图上节点关系更丰富，这种改进会不会被放大？

这里有一个需要提前讲清楚的边界。本文当前不是在复现 Zhang et al. 的数值结果，因为我们的 RV 和 IV 来自 30-day volatility proxy，而不是 Zhang 使用的 LOBSTER 高频 intraday RV。更准确地说，本文是在复刻 Zhang-style modeling and evaluation framework，然后在这个框架下做两个扩展：一个是加入 implied volatility，另一个是把 universe 扩到更大的 S&P500。这一点要在 Introduction 里讲出来，否则读者会误以为后面的表格应该和 Zhang Table 1 逐项数值可比。

但这个边界不能只写成“数据不如 Zhang 强”。30-day close-to-close HV proxy 的确更平滑，并且和 HAR monthly lag、IV30 之间有更强 horizon overlap；这会削弱把结果解释成纯 high-frequency realized-volatility forecasting contribution 的力度。不过从本文重心看，这个平滑目标也提供了一个有意义的 stability test：如果 large-universe GNNHAR 在相对平滑的 proxy-volatility target 上已经出现明显低波动股票高估和 loss instability，那么在更高频、更噪声、更非正则的 intraday RV 目标上，GNNHAR 未必会更稳定。因此正式写法应把本文定位为 Zhang-style proxy-volatility extension，而不是直接 intraday RV replication。

可以设置三个 research questions。这里的重点不是单独重复 Zhang 的原问题，而是把 Zhang 的模型族推广到更大的 universe，并加入 IV 后检验其预测增益：

- RQ1（model improvement across universes）：GHAR、GNNHAR 及其主要变体是否在 Dow30、S&P 100 和 S&P 500 三个 universe 中均能相对 HAR 提高预测准确性；这种提高在 MSE 和 QLIKE 两种损失函数下是否一致；
- RQ2（IV contribution）：在 HAR、GHAR、GNNHAR 及其变体中加入 implied volatility 后，是否能进一步降低 MSE / QLIKE forecast loss；IV 的贡献是否在不同 universe 和不同模型族中稳定存在；
- RQ3（improvement of improvement from richer graphs）：随着 universe 从 Dow30 扩展到 S&P 100 和 S&P 500，图节点数量和节点关系更丰富，GHAR / GNNHAR 相对 HAR 的改进幅度是否进一步增大。换句话说，我们关心的不只是模型是否优于 HAR，而是更大的图是否放大了这种相对 HAR 的提升。

这三个 RQ 要和 Zhang 的文章关系写清楚：

- RQ1 是 Zhang 主问题的扩展：Zhang 主要检验 GHAR / GNNHAR 是否优于 HAR，并在 DJIA 与 S&P100 上做 robustness；我们把同样的问题延伸到 Dow30、S&P100、S&P500，并且同时考察 MSE 与 QLIKE。
- RQ2 是本文最核心的新扩展：Zhang 在 future work 中提到 options，我们用 IV 具体实现这个方向，检验 option-market forward-looking information 是否在历史 RV 与图结构之外有增量贡献。
- RQ3 是本文关于大图的二阶问题：不是简单问“大图上的模型有没有提升”，而是问节点关系更丰富时，模型相对 HAR 的提升幅度是否进一步扩大，也就是“提升的提升”。如果这个假说成立，S&P500 中 graph / GNN models 的 gain 应该大于 Dow30；如果不成立，就说明更多节点也可能带来训练难度、噪声、过平滑或图构造误差。

### 1.5 本文工作与数据驱动回答

这里应当用我们已经跑出来的数据写简洁结论：

- Dow30：在标准窗口 aligned full-model rerun 中，`GNNHAR2L_Q_IV` 是 QLIKE 下最优模型，QLIKE ratio 约为 0.923，约比 $HAR_M$ 降低 7.7% QLIKE loss；MSE 下 `HAR_M_IV` 最优，MSE ratio 约为 0.959；
- S&P 100：IV 明显有用，但最优 QLIKE 模型是 `HAR_Q_IV`，多个 IV 模型在 MCS 中接近；
- S&P 500：正式 AutoDL 结果使用 449 个 ticker；当前 test dates 是 Dow30/S&P100 234-date calendar 的 223-date subset，覆盖 2025-07-14 到 2026-06-01；当前结果中最优 QLIKE 模型是 `GHAR_M_IV`，MSE 下 `HAR_M_IV` 与 `GHAR_M_IV` 显示为并列最优。这个结果应写成两层含义：第一，没有 IV 的 historical-only HAR/GHAR/GNNHAR 在大图里不能提供稳定优势；第二，引入 IV 后，最优模型变成 IV-augmented HAR/GHAR，说明 option-market forward-looking information 是当前最稳定的信息增量。GNNHAR-IV 相比 non-IV GNNHAR 有若干局部改善，但仍没有进入最优集合，因此不能说“带 IV 的 GNNHAR 已经解决 S&P500 问题”；
- 因此，我们现在的证据不支持“节点越多 GNNHAR 提升越强”的简单结论，更合理的说法是：大图下 GNNHAR 对训练、尺度、层数和正则化更敏感。

这一节需要明确告诉读者后文如何回答三个 RQ：

- 对 RQ1：用 full loss ratio tables、MCS 和 DM 检验，分别比较 HAR、GHAR、GNNHAR1L--5L 及其 IV 扩展；GHAR2H / GHAR3H 作为 Zhang Appendix E-style 的补充诊断放入附录，不作为正文主线；
- 对 RQ2：比较 IV 与 non-IV 版本，重点看 QLIKE ratio、MCS inclusion，以及不同 universe 中 IV 模型是否稳定进入前列；
- 对 RQ3：比较 Dow30、S&P100、S&P500 中模型相对 $HAR_M$ 的 improvement，即 $1 - R_m^{QL}$ 或 $1 - R_m^{MSE}$，判断节点关系丰富是否带来“提升的提升”。

### 1.6 文章结构

本文后续正文应严格仿照 Zhang et al. 的主结构，但要按照本文的三个 RQ 重新组织。完整正文包括七个主 section：

- Section 2 是 Preliminaries：交代图、邻接矩阵、$K$-hop neighbors、GNN layer、HAR、GHAR，以及 baseline GNNHAR。它回答“Zhang 的框架是什么”，为后面的 IV 扩展做铺垫。
- Section 3 是 Proposed Methodology：提出 HAR-IV、GHAR-IV 和 GNNHAR-IV，说明 IV 如何作为 forward-looking component 进入特征矩阵，说明当前模型是否包含 $IV\times RV$ 交互项（目前不包含），并给出 estimation criterion、forecast evaluation、MCS 和 DM test。
- Section 4 是 Empirical Analysis：介绍数据、样本、图构造、rolling out-of-sample design，并报告 Dow30、S&P100、S&P500 三个 universe 的完整主结果表。Section 4 的核心任务是给出数据事实，而不是把所有机制解释都放进去。
- Section 5 是 Statistical Analysis and Discussion：在 Section 4 主表基础上做进一步统计分析，包括 evaluation criterion 的影响、forecast ratios / errors、MCS 和 DM 的解释、FVU、GNN depth、MAD / smoothing。它负责解释“为什么这些结果是这样的”。线性 GHAR2H / GHAR3H 的 multi-hop 结果只在附录中作为补充诊断展开。
- Section 6 是 Robustness Tests：说明已有和待补的稳健性检验，包括 validation split、日期对齐、数据源、图构造、IV interaction、forecast horizon 和 exact hidden-state MAD。
- Section 7 是 Conclusion：直接回答三个 RQ，并把本文的 IV contribution、large-universe finding 和 future work 联系起来。

## 2. Preliminaries

这一节应该模仿 Zhang et al. 的 Section 2，但为了本文完整性，需要多交代 baseline GNNHAR。这里的目标是让读者在进入我们的 IV 模型前，已经清楚 HAR、GHAR、GNNHAR 的基本形式、图矩阵如何进入模型、以及 GNNHAR 的 $1L,2L,\ldots,5L$ 层数是什么意思。

写作顺序建议如下。

### 2.1 Graph definitions and adjacency matrices

先定义图：

$$
\mathcal{G}=(\mathcal{V},\mathcal{E}), \qquad
\mathcal{V}=\{1,\ldots,N\}.
$$

在本文中，每个 node 是一个股票，每条 edge 表示两个股票之间的 volatility spillover 或 return-based conditional dependence。令 $A\in\{0,1\}^{N\times N}$ 为二元邻接矩阵，若 $A_{ij}=1$，则股票 $i$ 与股票 $j$ 在估计图中相连。按照 Zhang 的设定，$A_{ii}=0$，即图聚合项不包含 self-loop，因为一个股票自身历史 RV 的影响由 HAR 部分单独建模。

需要解释 $K$-hop neighbors：

- $1$-hop neighbors 是和目标股票直接相连的股票；
- $2$-hop neighbors 是通过一个中间节点到达的股票；
- $K$-layer GNN 的 receptive field 包含 $K$-hop 以内的节点；
- 因此多层 GNNHAR 可以被理解为在更远的 volatility spillover neighborhood 中传播信息。

图归一化矩阵写为：

$$
W=D^{-1/2}AD^{-1/2}, \qquad
D_{ii}=\sum_{j=1}^N A_{ij}.
$$

这里要说明：$W$ 是实际进入 GHAR / GNNHAR 的矩阵。这个归一化和 Zhang 一致，用来稳定不同 degree 节点的邻居聚合。

### 2.2 Graph construction via GLASSO

这一小节可以仿照 Zhang 的 Section 3.1.1，但放在 preliminary 里交代清楚。我们的图不是外生给定，而是从 daily return panel 中估计出来。

设 $r_t=(r_{1,t},\ldots,r_{N,t})^\top$ 是 $N$ 个股票在日期 $t$ 的日收益率向量。GLASSO 估计稀疏 precision matrix：

$$
\widehat{\Theta}
=\arg\min_{\Theta\succ0}
\left\{
\operatorname{tr}(S\Theta)-\log\det(\Theta)
+\lambda \sum_{i\neq j}|\Theta_{ij}|
\right\},
$$

其中 $S$ 是收益率样本协方差矩阵。然后定义：

$$
A_{ij}=
\mathbf{1}\{\widehat{\Theta}_{ij}\neq 0\},\qquad i\neq j,
\quad A_{ii}=0.
$$

经济解释：如果 precision matrix 的 off-diagonal entry 不为零，则两个股票在控制其他股票后仍然存在条件相关关系；在 volatility forecasting 里，这被用作 graph-based spillover 的近似。

需要标注我们的实现：

- 每个 rolling origin 都使用训练窗口中的 daily returns 重新估计 GLASSO；
- notebook 中使用 `GraphicalLassoCV`，候选 alpha 为 $\{0.01,0.03,0.05,0.08,0.1,0.2\}$，失败时回退到固定 alpha；
- S&P500 formal AutoDL 结果保存了 22 个 graph matrices，对应 MSE / QLIKE 两组 rolling blocks；
- Dow30 和 S&P100 的当前 paper-ready 统计层里没有完整 graph matrices 文件，但模型运行时仍按 rolling window 构造了 $W$。

### 2.3 A brief review of graph neural network layers

这一节要模仿 Zhang 的 GNN review，不需要过度展开机器学习背景，但要给出 layer 矩阵。标准 GCN layer 可以写成：

$$
H^{(\ell+1)}
=
\sigma\!\left(
\widetilde{D}^{-1/2}\widetilde{A}\widetilde{D}^{-1/2}
H^{(\ell)}\Theta^{(\ell)}
\right),
$$

其中 $\widetilde{A}=A+I_N$。然后要说明 Zhang-style GNNHAR 和标准 GCN 的区别：

- 标准 GCN 通常加入 self-loop；
- Zhang 的 GNNHAR 不在 graph propagation 中加入 self-loop；
- 因为自身历史波动率由 HAR linear term 单独解释，graph propagation 只解释 spillover；
- 这一点让 GHAR 和 GNNHAR 的比较更清楚：GHAR 是线性邻居聚合，GNNHAR 是非线性邻居聚合。

本文采用的 Zhang-style GNN layer 写成：

$$
H^{(\ell+1)}
=
\operatorname{ReLU}\!\left(W H^{(\ell)}\Theta^{(\ell)}\right).
$$

这里 $H^{(0)}$ 是节点特征矩阵；在 baseline GNNHAR 中 $H^{(0)}=V_{:t-1}\in\mathbb{R}^{N\times 3}$，在我们的 IV extension 中 $H^{(0)}=Z_{:t-1}\in\mathbb{R}^{N\times 4}$。

### 2.4 HAR baseline

设 $v_{i,t}$ 是股票 $i$ 在日期 $t$ 的 realized variance target。HAR 使用日、周、月三个历史窗口：

$$
v_{i,t-1},\qquad
\bar v_{i,t-5:t-2}
=\frac{1}{4}\sum_{k=2}^{5}v_{i,t-k},
\qquad
\bar v_{i,t-22:t-6}
=\frac{1}{17}\sum_{k=6}^{22}v_{i,t-k}.
$$

定义

$$
V_{:t-1}
=
\left[
v_{t-1},\ \bar v_{t-5:t-2},\ \bar v_{t-22:t-6}
\right]\in\mathbb{R}^{N\times 3}.
$$

HAR baseline：

$$
\mathbb{E}(v_t\mid\mathcal{F}_{t-1})
=
\alpha+V_{:t-1}\beta.
$$

这里要说明：$HAR_M$ 表示用 MSE / OLS 估计的 HAR，$HAR_Q$ 表示用 QLIKE 作为 estimation criterion 训练的 HAR。

### 2.5 GHAR baseline

GHAR 在 HAR 的基础上加入 graph-aggregated lag features：

$$
\mathbb{E}(v_t\mid\mathcal{F}_{t-1})
=
\alpha+V_{:t-1}\beta
+WV_{:t-1}\gamma.
$$

这里 $WV_{:t-1}$ 表示邻居股票日、周、月 realized volatility 的图聚合。GHAR 仍然是线性模型；它检验的是一跳邻居信息是否相对 HAR 有预测增益。

多跳 GHAR 可以作为 diagnostic model：

$$
\mathbb{E}(v_t\mid\mathcal{F}_{t-1})
=
\alpha+V_{:t-1}\beta
+WV_{:t-1}\gamma_1
+W^2V_{:t-1}\gamma_2
+\cdots.
$$

本文用 `GHAR2H` / `GHAR3H` 来检验两跳和三跳邻居是否提供额外预测力。这个分析应与 Zhang Appendix E 的 multi-hop comparison 对齐。需要特别交代：Zhang 的主表并没有把 GHAR2H / GHAR3H 作为完整主模型族列入，而是在 Appendix E 中讨论 GHAR2Hop。因此本文也不把 `GHAR2H` / `GHAR3H` 放在正文主结果中展开，而是把它们放入附录，作为检验 linear multi-hop spillover 的补充诊断。

当前正式输出中，多跳 GHAR 的覆盖情况如下：

- Dow30：当前已经有两个标准窗口 aligned artifacts。第一，`20260619T071426Z_aligned_full_model` 是 full-model rerun，使用 2021-06-09 到 2026-06-09 的输入面板，test dates 为 2025-07-07 到 2026-06-09，共 234 个日期，包含 HAR/GHAR/GNNHAR1L--5L 及 IV variants，共 28 个预测文件；这是当前 Dow30 主结果口径。第二，`20260618T075711Z_wide_multihop_ghar` 是 multi-hop supplement，只用于 Appendix C 的 $GHAR2H/GHAR3H$ 诊断。旧的 2025-02-21 到 2026-01-23、232-date Dow30 full-model run 只应视为历史对照，不再作为当前跨 universe 日期对齐表格的主口径；
- S&P100：原 run 保存了 $GHAR2H_M,GHAR3H_M$ 及其 IV 版本；现在已经补充保存了 $GHAR2H_Q,GHAR3H_Q$ 及其 IV 版本，因此 S&P100 Appendix C 可以报告完整的 MSE-trained / QLIKE-trained multi-hop GHAR 诊断；
- S&P500：保存了 $GHAR2H_M,GHAR3H_M,GHAR2H_Q,GHAR3H_Q$ 及其 IV 版本。

正文表格中不需要把这些行全部展开。若正文需要一句交代，可以说 multi-hop GHAR results are reported in Appendix C and are not central to the main comparison. 附录表格必须按实际可用模型和日期口径报告：Dow30 的 full-model 主结果和 wide-window multi-hop supplement 是两个不同 artifact；S&P100 与 S&P500 的 multi-hop rows 可以进入同一 run-level table，因为它们和各自 truth/test-date arrays 对齐。

### 2.6 Baseline GNNHAR

GNNHAR 用非线性图传播替换 GHAR 中的线性 $WV_{:t-1}\gamma$。一层 GNNHAR 写为：

$$
H^{(1)}=\operatorname{ReLU}\!\left(WV_{:t-1}\Theta^{(0)}\right),
$$

$$
\mathbb{E}(v_t\mid\mathcal{F}_{t-1})
=
\alpha+V_{:t-1}\beta+H^{(1)}\gamma.
$$

两层和三层写为：

$$
H^{(2)}=\operatorname{ReLU}\!\left(WH^{(1)}\Theta^{(1)}\right),
\qquad
H^{(3)}=\operatorname{ReLU}\!\left(WH^{(2)}\Theta^{(2)}\right).
$$

一般地，$K$-layer GNNHAR 为：

$$
\mathbb{E}(v_t\mid\mathcal{F}_{t-1})
=
\alpha+V_{:t-1}\beta+H^{(K)}\gamma.
$$

需要写清楚：

- $K=1$ 时模型只聚合一跳 nonlinear spillover；
- $K=2,3,\ldots$ 时模型允许更远的 multi-hop spillover；
- 但层数越深不一定越好，因为 $K$ 太大可能导致 over-smoothing；
- Zhang 在 DJIA 主实验中主要报告 1L--3L，在 S&P100 robustness 中扩展到 4L--5L；
- 我们在 Dow30、S&P100、S&P500 中统一保留 1L--5L 的深度扩展，但最终表格应按照 Zhang 的顺序报告，而不是按照表现排序。

## 3. Proposed Methodology: GNNHAR-IV

这一节是本文相对 Zhang et al. 的核心扩展。Zhang 的 proposed methodology 是 GNNHAR；我们的 proposed methodology 是在同一套 HAR / GHAR / GNNHAR 框架中加入 implied volatility，形成 HAR-IV、GHAR-IV 和 GNNHAR-IV。

### 3.1 Implied volatility as a forward-looking component

令 $q_{i,t}$ 表示股票 $i$ 在日期 $t$ 的 implied volatility feature。当前 notebook 中使用的是 30-day mean implied volatility，并将年化百分比波动率转换到与 RV target 对齐的 daily variance scale。把 IV 放入 HAR-type realized-volatility forecasting 中，和 Busch et al. (2010) 以及 Li and Tang (2021) 中“options / implied variance 包含额外预测信息”的思路一致：

$$
q_{i,t}^{\mathrm{daily}}
=
\frac{(IV30_{i,t}/100)^2}{252}.
$$

当前实现使用 $q_{i,t-1}$ 作为滞后 IV 特征，避免未来信息泄露。定义 IV-augmented feature matrix：

$$
Z_{:t-1}
=
\left[
v_{t-1},\
\bar v_{t-5:t-2},\
\bar v_{t-22:t-6},\
q_{t-1}
\right]\in\mathbb{R}^{N\times 4}.
$$

这里要明确说明：当前已经跑出的模型不是 $IV\times RV$ interaction model，而是把 IV 作为第 4 个 component 直接加入 feature matrix。换句话说，当前结果回答的是“lagged IV 作为额外 forward-looking predictor 是否有增量贡献”，不是“IV 与 RV 的交互项是否显著”。

### 3.2 HAR-IV and GHAR-IV

HAR-IV 写为：

$$
\mathbb{E}(v_t\mid\mathcal{F}_{t-1})
=
\alpha+Z_{:t-1}\beta.
$$

这里 IV 的进入方式是 additive component。在线性回归意义下，它相当于在 HAR 的日、周、月 RV 特征之外，加入一个额外的 lagged IV regressor。

GHAR-IV 写为：

$$
\mathbb{E}(v_t\mid\mathcal{F}_{t-1})
=
\alpha+Z_{:t-1}\beta
+WZ_{:t-1}\gamma.
$$

需要注意：根据 notebook 当前实现，`ghar_predict_ols` 对整个 feature tensor 做 $WX$ 图聚合。因此在 GHAR-IV 里，不仅 RV 的日、周、月分量被图聚合，IV component 也会被 $W$ 聚合。也就是说，$WZ_{:t-1}$ 中包含 graph-aggregated lagged IV。这一点是我们相对 Zhang GHAR 的一个自然扩展。

### 3.3 GNNHAR-IV

GNNHAR-IV 把 $H^{(0)}$ 从 $V_{:t-1}\in\mathbb{R}^{N\times3}$ 改为 $Z_{:t-1}\in\mathbb{R}^{N\times4}$：

$$
H^{(0)}=Z_{:t-1},
$$

$$
H^{(\ell+1)}
=
\operatorname{ReLU}\!\left(W H^{(\ell)}\Theta^{(\ell)}\right),
\qquad
\ell=0,\ldots,K-1.
$$

最终预测为：

$$
\mathbb{E}(v_t\mid\mathcal{F}_{t-1})
=
\alpha+Z_{:t-1}\beta+H^{(K)}\gamma.
$$

解释要写清楚：

- $\alpha+Z_{:t-1}\beta$ 是自身历史 RV 与自身 lagged IV 的 direct effect；
- $H^{(K)}\gamma$ 是通过 GLASSO graph 传播后的 nonlinear spillover effect；
- 因为 $H^{(0)}$ 包含 IV，GNNHAR-IV 可以学习 graph-propagated IV information；
- 如果期权市场的 forward-looking information 在 cross-section 中也有 spillover，那么 GNNHAR-IV 理论上应比 GNNHAR 更有优势。

### 3.4 Optional interaction extension not estimated in the current run

你提出的 $IV\times RV$ 交互项是有意义的，但当前模型没有估计这一项。可以在 methodology 或 robustness/future work 中提出为可检验扩展：

$$
Z^{\mathrm{int}}_{:t-1}
=
\left[
V_{:t-1},\
q_{t-1},\
q_{t-1}\odot v_{t-1},\
q_{t-1}\odot \bar v_{t-5:t-2},\
q_{t-1}\odot \bar v_{t-22:t-6}
\right].
$$

这个扩展对应一个经济问题：IV 是否不仅提供独立的 forward-looking signal，而且会改变历史 RV 的边际预测含义。例如，当 IV 高时，同样的过去 RV 是否意味着更高的未来波动风险。当前报告不能把这个 interaction 写成已经跑出的结果，只能写为后续 robustness / model extension。正式写作时，methodology 中只需要定义这个候选扩展，Lin (2013) 式 regression-adjustment / covariate-interaction 类比和解释应放在 Section 6.4，并且必须说明这里不是 causal treatment-effect interpretation。

### 3.5 Estimation criterion

仿照 Zhang，把 estimation criterion 和 forecast loss 区分开：

- estimation criterion (EC)：训练模型时最小化的目标函数；
- forecast loss (FL)：out-of-sample 评价预测时使用的损失。

本文保留 Zhang 的两类 EC：

$$
\mathrm{MSE}
=
\frac{1}{N|\mathcal{T}_{train}|}
\sum_{i=1}^N
\sum_{t\in\mathcal{T}_{train}}
\left(v_{i,t}-\widehat v_{i,t}^{(F)}\right)^2,
$$

$$
QL
=
\frac{1}{N|\mathcal{T}_{train}|}
\sum_{i=1}^N
\sum_{t\in\mathcal{T}_{train}}
\left[
\frac{v_{i,t}}{\widehat v_{i,t}^{(F)}}
-\log\!\left(\frac{v_{i,t}}{\widehat v_{i,t}^{(F)}}\right)-1
\right].
$$

模型命名沿用 Zhang：$F_M$ 表示 MSE-trained model，$F_Q$ 表示 QLIKE-trained model。IV 版本在模型名后加 `_IV`。

这里要避免一个常见误解：Zhang 并不是建议“用 MSE 训练、用 QLIKE 评估”。他的设计是对同一模型同时估计 $F_M$ 和 $F_Q$，然后在 out-of-sample 中同时报告 MSE forecast loss 和 QLIKE forecast loss。也就是说，estimation criterion 和 forecast loss 是两个概念。Zhang et al. (2024a) 的实证叙述更重视 QLIKE forecast loss，原因是 QLIKE 在 volatility forecast comparison 中更常用，并且 Patton (2011) 的观点支持在 noisy / imperfect volatility proxy 下使用 robust forecast comparison loss。本文应沿用这个口径：主表同时报告 MSE 和 QLIKE，但主结论以 QLIKE 为核心，MSE 作为重要的稳定性和异常诊断指标。

当前实现细节必须按实际 run config 写清楚。这里不是笼统说“following Zhang”，而是说明哪些训练设定来自 Zhang Chao 论文中的 hyperparameter tuning，哪些来自公开代码默认值，哪些是本文实际运行时固定下来的参数。

Zhang Chao 论文和公开代码给出的信息并不完全是同一种证据。论文 Section 4.1 采用 rolling sample，每月重新估计模型，过去 1000 个交易日作为滚动窗口；主设定中约前三年用于 training，最近一年用于 validation，下一月作为 out-of-sample test。论文 Appendix D 明确做了 hidden dimension 的 validation grid search，候选集合为 $\{3,6,9,16,32\}$，并发现 $n_{\mathrm{hid}}=9$ 在 one-layer GNNHAR 的 validation MSE 和 QLIKE 上表现最好；多层 GNNHAR 随后沿用同一个 hidden dimension。Appendix D 同时说明 Adam learning rate 设为 $10^{-3}$，batch size 设为 32，并在 validation loss 出现 overfitting 信号时 early stop。因此，$n_{\mathrm{hid}}=9$ 不是任意预先给定的数，而是 Zhang 的 validation tuning 结果；learning rate 和 batch size 更像 Zhang 的实验设定，其中 learning rate 与代码默认一致，但 batch size 在论文和公开代码之间不同。

Zhang Chao 的公开 `GNNHAR.py` 默认值为 `valid_len=22`、`n_epochs=5000`、`n_hid=9`、`batch_size=128`、`lr=1e-3`、`numNN=1`，optimizer 为 Adam，并带有 validation loss based checkpointing。公开代码中的 `valid_len=22` 对应约一个月 validation window，更接近 Zhang Table 4 的 smaller-validation robustness design，而不是论文主设定中的约一年 validation window。本文正式 runs 为了在 Dow30、S&P100 和 S&P500 上保持可比，并控制计算成本，没有为每个 universe 重新做 Appendix-D-style hyperparameter search，而是固定使用 Zhang 选出的 $n_{\mathrm{hid}}=9$ 和公开代码默认的 long-training setting。

本文实际使用的训练参数如下。`lookback=1000` 表示每个 rolling origin 最多使用过去 1000 个交易日；其中最后 `valid_len=22` 个交易日作为 validation set，前面的 978 个交易日作为 training set。`window=22` 和 `block_stride=22` 表示每个 rolling block 预测接下来约一个月的 one-day-ahead targets，然后滚动到下一个 block。所有 neural / torch-estimated models 使用 Adam，learning rate 固定为 $10^{-3}$，batch size 固定为 128，maximum epochs 固定为 5000，single-network ensemble 即 `numNN=1` / `n_ensemble=1`。如果多个 ensemble members 存在，screening percentile 会按照 Zhang 代码逻辑取 validation loss 的下 50%; 但本文正式 runs 中实际是 single-network setting，所以这一筛选参数不会改变 ensemble composition。

| Universe | Rolling lookback | Train / validation per block | Test window / stride | Hidden dim | LR | Batch size | Max epochs | Ensemble | Model depths actually reported |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Dow30 | 1000 | 978 / 22 | 22 / 22 | 9 | $10^{-3}$ | 128 | 5000 | 1 | GNNHAR1L--3L; no main GHAR2H/GHAR3H rows |
| S&P100 | 1000 | 978 / 22 | 22 / 22 | 9 | $10^{-3}$ | 128 | 5000 | 1 | GNNHAR1L--5L; GHAR2H/GHAR3H available as supplement rows |
| S&P500 | 1000 | 978 / 22 | 22 / 22 | 9 | $10^{-3}$ | 128 | 5000 | 1 | GNNHAR1L--5L; GHAR2H/GHAR3H available for MSE/QLIKE and IV/non-IV |

Linear $HAR_M$、$GHAR_M$ 及其 IV 版本用 OLS 估计；QLIKE-trained linear rows, namely $HAR_Q$、$GHAR_Q$ and their IV versions, use the same torch/Adam optimization path as the QLIKE neural models, because QLIKE is not an ordinary least-squares criterion. GNNHAR rows, with or without IV, are always trained by Adam under the selected EC. This distinction is important: MSE-trained linear models are closed-form OLS, while QLIKE-trained linear models and all neural models are numerical optimization results.

因此，本文的参数设定应被表述为 Zhang-code aligned and Zhang-tuning-informed，而不是“完整复现 Zhang hyperparameter optimization”。与 Zhang 直接一致或继承的设定包括：rolling lookback 1000、monthly test window 22、learning rate $10^{-3}$、hidden dimension $9$、validation-based model selection、以及 MSE / QLIKE 两类 estimation criteria。与 Zhang 论文主设定不同或需要披露的地方包括：本文使用 one-month validation window 22 而不是主设定中约 one-year validation window；batch size 使用公开代码默认值 128 而不是 Appendix D 中写的 32；正式结果使用 single-network setting，而 Zhang 文中提到 ensemble averaging 可以缓解 stochastic optimizer 的不稳定性；并且本文没有在每个 universe 和 IV-augmented feature set 上重新做 hidden dimension、learning rate、batch size、epoch budget、ensemble size 的完整 grid search。

这个选择是可以解释的：本文的核心比较需要 Dow30、S&P100 和 S&P500 在同一套训练规则下可比，而且 S&P500 的完整 grid search 计算成本很高。固定 Zhang-selected $n_{\mathrm{hid}}=9$ 可以避免在每个 universe 上为 GNNHAR 额外调参后造成不公平比较。但这也留下一个需要在 robustness section 明确提出的问题：当 predictor set 从 RV 扩展到 RV+IV、universe 从 Dow30 扩大到 S&P500 时，Zhang 在原始 RV setting 中选出的 $n_{\mathrm{hid}}=9$ 是否仍然最优？更严格的后续版本应设计 Appendix-D-style robustness table，在每个 universe 上至少比较 hidden dimension $\{3,6,9,16,32\}$，并可进一步比较 batch size $\{32,128\}$、learning rate $\{10^{-3},3\times10^{-4}\}$、one-month vs one-year validation window、以及 single-network vs ensemble averaging。当前正文结论应因此避免把 GNNHAR 的弱表现完全解释为模型本身无效；一部分差异可能来自 fixed hyperparameter protocol。

### 3.6 Forecast evaluation, MCS, and DM tests

Forecast evaluation 继续使用 MSE 和 QLIKE。主表报告每个模型相对 $HAR_M$ 的 loss ratio：

$$
R_m^{QL}
=
\frac{\frac{1}{NT}\sum_{i,t}QL(v_{i,t},\widehat v_{i,t}^{(m)})}
{\frac{1}{NT}\sum_{i,t}QL(v_{i,t},\widehat v_{i,t}^{(HAR_M)})}.
$$

$R_m<1$ 表示模型优于 $HAR_M$，$R_m>1$ 表示模型差于 $HAR_M$。表格要仿照 Zhang：

- MSE 和 QLIKE 两列都报告；
- 最优 MSE 模型用 red 标注；
- 最优 QLIKE 模型用 blue 标注；
- MCS 5% confidence level 下不能被排除的模型加 `*`；
- 模型顺序按 HAR $\rightarrow$ GHAR $\rightarrow$ GNNHAR1L--5L，再重复 IV panel，不按结果好坏排序。

特别要注意表格口径。正文主表中的 ratio 必须统一相对 $HAR_M$，这与 Zhang Table 1 的设置一致。如果在附录或诊断表中按 QLIKE-trained panel 报告相对 $HAR_Q$ 的 ratio，需要明确标注分母。当前 S&P500 中 $HAR_Q$ 的 MSE 极大，因此某些 QLIKE-trained 模型的 `MSE ratio vs HAR_Q` 会看起来很低，但这并不代表它们相对 $HAR_M$ 的 MSE 表现好。正文不能混用这两种分母。

MCS 用于从多个候选模型中识别 statistically indistinguishable best model set（Hansen et al., 2011）。DM test 用于 pairwise forecast comparison（Diebold and Mariano, 1995）。本文 DM 不只做 vs HAR，还要仿照 Zhang et al. (2024a) 做两层比较：

- 正文主比较：每个模型 vs $HAR_M$，用于回答 graph / IV extension 是否相对 HAR baseline 改善预测；
- 正文或主附录比较：GHAR vs GHAR2H / GHAR3H，用于检验 linear multi-hop neighbors 是否提供额外信息；
- 正文或主附录比较：GNNHAR1L vs GNNHAR2L / 3L / 4L / 5L，用于检验增加 GNN depth 是否带来显著增益或过度平滑；
- 附录系统比较：GHAR vs GNNHAR1L / 2L / 3L / 4L / 5L，用于回答 nonlinear graph aggregation 是否显著超过 linear graph aggregation；
- IV 版本同样做一组，即 $GHAR^{IV}$ vs $GNNHAR1L^{IV}$--$GNNHAR5L^{IV}$，以及 $GNNHAR1L^{IV}$ vs deeper IV-GNNHAR versions；
- 如果篇幅有限，正文只报告核心结论，完整 DM statistics、$p$-values 和 loss differential 放入 Appendix C 或 Appendix D 的 pairwise DM table。

## 4. Empirical Analysis

这一节应仿照 Zhang 的 Section 4，但要服务于本文的三个 RQ。Section 4 只做 empirical setup 和 main result：先介绍数据来源、样本构造、图构造和 rolling out-of-sample design，然后给出 Dow30、S&P100、S&P500 三个完整主表，最后用 cross-universe summary 回答“大图节点关系是否带来提升的提升”。更细的统计解释、DM/FVU/MAD 和 forecast-ratio 分析放到 Section 5。

这里不要把 data 单独拆成前面的大章。Zhang 的写法是先在 empirical section 中说明 sample 和 evaluation，然后直接进入主表；我们也应该沿用这种行文。

### 4.1 Data and sample construction

本文使用三个 universe：

| Universe | Input raw ticker count | Model ticker count | Saved test dates | Current out-of-sample dates |
|---|---:|---:|---:|---|
| Dow30 | 30 | 30 | 234 | 2025-07-07 to 2026-06-09 |
| S&P100 | 101 | 91 | 234 | 2025-07-07 to 2026-06-09 |
| S&P500 | 503 | 449 | 223 | 2025-07-14 to 2026-06-01 |

补充说明：这张表现在采用当前 paper-ready 结果目录中的 sample 口径。`Input raw ticker count` 不是模型实际使用的股票数量，而是初始 universe 列表或指数成分股清单中的 ticker 数量。Dow30 原始清单包含 30 个成分股；S&P100 原始清单为 101 个 ticker，主要是因为指数成分股口径中可能同时包含 share-class tickers 或同一公司不同交易代码；S&P500 原始清单为 503 个 ticker，同样反映了指数清单中的多 share-class 代码，而不是 500 个唯一公司实体。`Model ticker count` 是经过清理后真正进入模型的节点数量。清理步骤包括：统一 ticker 格式；对齐 RV、IV 和 returns 三张面板；剔除 RV / IV / returns 覆盖率不足、缺失日期过多、无法通过 rolling-window 特征构造、或无法进入 GLASSO graph input 的股票。因此 S&P100 从 101 降到 91，S&P500 从 503 降到 449。Dow30 当前样本中 30 个 ticker 均通过覆盖率和面板交集检查，所以 model ticker count 仍为 30。

`Saved test dates` 和 `Current out-of-sample dates` 也需要解释清楚。当前三个 universe 并不是从同一个完全同步的数据快照、同一个 notebook run、同一个最终日期一次性生成的。Dow30 的 aligned rerun 使用 `GNNHAR_Research/data/dow30` 下 2021-06-09 到 2026-06-09 的 RV、IV、returns 面板，生成 2025-07-07 到 2026-06-09 的 234 个 out-of-sample test dates，truth shape 为 $234\times30$。S&P100 当前 paper-ready run 也是 234 个 test dates，日期为 2025-07-07 到 2026-06-09。S&P500 当前正式 run 来自 AutoDL / scale-experiment source，经过 0.99 coverage threshold 和 449 个 ticker 的 common-panel 筛选后，保存的 test period 为 2025-07-14 到 2026-06-01，共 223 个 test dates。也就是说，S&P500 的 current out-of-sample window 与 Dow30 / S&P100 不是完全同长，但它不是完全不同的日期口径：S&P500 的 223 个 dates 是 Dow30 / S&P100 234-date calendar 的子集，缺少的是最前面的 5 个 trading days 和最后面的 6 个 trading days。

这种不完全同长主要来自工程和数据口径，而不是模型定义本身：不同 universe 的原始数据快照更新日期不同，S&P500 的 coverage filtering 更严格，rolling origin 需要先满足 1000-day lookback 和 22-day validation window，且每 22 个 trading days 生成一个 test block；如果最后一个 block 的可用 truth / prediction 文件没有覆盖到 2026-06-09，就会出现 S&P500 结束日期早于 Dow30 / S&P100 的情况。也不能排除早期 Colab / AutoDL pipeline 中存在小的日期截断或 source synchronization issue。因此，正文中必须把这个日期不完全同长作为 sample limitation 披露，但语气不应过重：三个 universe 的结果可以作为 Zhang-style nested-calendar large-universe comparison，严格的 cross-universe equality of test calendar 则需要后续用同一份数据快照和同一套 rolling-origin generator 重新跑一次。需要单独说明的是，Dow30 aligned full-model rerun 当前已经覆盖 HAR/GHAR/GNNHAR1L--5L 及 IV variants，共 28 个模型；Dow30 aligned multi-hop supplement 另外覆盖 $GHAR2H/GHAR3H$ 的 MSE、QLIKE、IV 和 non-IV 预测。旧的 232-date Dow30 full-model run 可以作为历史对照，但不能再写成当前 out-of-sample 日期口径。

数据来源要写清楚：

- RV 使用 30-day close-to-close historical volatility panel；
- IV 使用 30-day mean implied volatility panel；
- 这些 volatility / IV 数据来自 VolVue / AlphaQuery-style option statistics 页面和本地清洗包，正式正文中应在 footnote 放数据页面链接，例如 `https://volvue.com/ticker/META/30-day/implied-volatility-mean` 和对应 historical-volatility 页面；
- GLASSO graph 使用 `daily_returns.csv` 中的 daily returns；S&P100 / S&P500 scale data metadata 显示这些 returns 来自 Yahoo adjusted-close daily returns；
- 所有 RV / IV 原始值是 annualized percentage volatility，not decimals，进入模型前按 notebook 转换为 daily variance scale。

必须标注的数据差异和局限：

- Zhang 使用 LOBSTER 高频 intraday data 构造 5-minute RV，样本为 2007-07-01 到 2021-06-30；
- 我们使用的是近五年 30-day historical volatility / 30-day implied volatility proxy，不是 LOBSTER 高频 RV；
- Dow30 当前 aligned rerun 的 raw input span 已更新为 2021-06-09 到 2026-06-09，和 S&P100 / S&P500 scale data 的 raw span 对齐；
- 旧 Dow30 232-date full-model run 的 input span 是 2021-01-27 到 2026-01-23；现在主样本应改用 `20260619T071426Z_aligned_full_model`，若引用旧表只能明确标注为历史对照；
- S&P500 的 397 vs 449 问题要写清楚：397 是 Google Drive upload 旧 complete-case panel，不是 AutoDL formal run source；正式 S&P500 分析使用 `data/scale_experiment/sp500`，覆盖筛选后得到 449 个 model tickers。

建议 footnote：

[^volvue-iv]: Example IV source page: `https://volvue.com/ticker/META/30-day/implied-volatility-mean`.
[^volvue-hv]: The historical-volatility source uses the corresponding 30-day close-to-close historical volatility pages.
[^yahoo]: The daily-return panel used for graph construction is stored as `daily_returns.csv`; for the scale data, metadata records Yahoo adjusted-close returns.

### 4.2 Out-of-sample forecasting design

这一节要仿照 Zhang 的 setup。写清楚：

- 每个 rolling origin 使用过去 1000 trading days；
- 当前 notebook / run config 中训练和验证切分要如实写。理论设定可以说按照 Zhang 使用 rolling train/validation split；实际 run config 中 Dow30 显示 `train_len=978`、`val_len=22`，部分 Colab 版本使用 `val_frac=0.25`，最终正文要以正式 run config 为准；
- 每 22 个 trading days 重新校准一次模型，近似 monthly recalibration；
- 这里的 22-day `window` 是 rolling test block 和 recalibration frequency，不是 one-month forecast horizon；
- 当前保存的预测目标是 one-day ahead $v_{i,t}$，不是 Zhang Table 1 中的 one-week / one-month horizon；
- 因此主表只能报告 one-day evaluation。如果要报告 weekly / monthly horizon，需要重新构造 $h=5$ 或 $h=22$ 的 target 并重跑。

这一点要写得非常明确，避免读者误会：

> Although models are recalibrated every 22 trading days, the current saved forecasts are one-day forecasts. The 22-day block is an evaluation and recalibration block, not a monthly forecasting horizon.

### 4.3 Main results: full model comparison

这里要放三个完整表格，每个 universe 一个表格，而且都用 LaTeX `[H]` 固定位置。每个表格都必须包含当前 universe 下全部模型，不只放局部对比。

这一节是整篇 empirical analysis 的核心。它不是为了简单报告哪个模型排第一，而是把三个 RQ 放在同一套表格里看。对每个 universe，我们都先看 non-IV panel 中 GHAR / GNNHAR 是否相对 $HAR_M$ 改善，这是 RQ1；再看同一模型加入 IV 后是否进一步降低 MSE 或 QLIKE，这是 RQ2；最后把 Dow30、S&P100、S&P500 的 improvement 放在一起比较，看 $1-R_{m,u}$ 是否随着节点规模扩大而变大，这是 RQ3。

在进入主表前，需要先交代 forecast loss 的解释边界。资产波动率和金融风险本身都是 latent concepts：风险并不是一个可以在现实世界中直接观测、并由某个绝对标准完全标定的物理量。Realized volatility 只是用价格路径事后构造出来的 proxy，implied volatility 也只是从 option prices 中反推出的 market-implied proxy。本文使用 MSE 和 QLIKE 评估预测，并不是说 $v_{i,t}$ 就是真实风险本身，也不是说最小化 MSE 或 QLIKE 就等于最大化风险管理或交易策略的最终收益。更准确地说，本文把 volatility forecasting 当作一个 Zhang-style statistical forecasting problem：给定 VolVue / AlphaQuery-style 数据中已经计算好的 realized-volatility proxy 和 implied-volatility proxy，检验不同模型的 one-day-ahead forecasts 是否更接近下一期 realized-volatility proxy。

因此，out-of-sample MSE / QLIKE comparison 只能回答一个有限但可操作的问题：在本文的数据定义下，哪个模型更好地预测了下一期 realized-volatility proxy。它不能直接回答“哪个模型最真实地度量了风险”，也不能直接回答“用哪个模型做交易或风控会获得最高收益”。从更高层次看，风险预测的最终价值应该由使用该风险度量后的经济结果来检验，例如 portfolio allocation、hedging、option trading、risk budgeting、VaR / expected shortfall backtesting 或 drawdown control。本文当前还没有把 forecasts 放入这些 decision-based tests 中，所以主表结果应理解为 proxy-based forecast accuracy evidence，而不是 final economic utility evidence。

这也是为什么本文同时保留 MSE 和 QLIKE。MSE 把 realized proxy 和 forecast 之间的平方误差当作评价对象，容易受到极端 realized-volatility observations 影响；QLIKE 在 volatility forecast comparison 文献中常用，并且对 noisy volatility proxy 有一定理论吸引力，但它仍然是在给定 proxy 后定义的 statistical loss。两者都是有用的 benchmark，却都不是“真实风险”的最终标尺。正式正文中要把这个边界写清楚，避免读者把 lower QLIKE / lower MSE 误读为模型已经完整捕捉了金融风险本身。

给第一次读这篇 draft 的人，需要先解释主表的读法。每一行是一个模型，每一列不是单独的统计量，而是在回答不同问题：

- `EC` 表示这个模型训练时用 MSE 还是 QLIKE 作为 estimation criterion；
- `IV` 表示是否加入 lagged implied volatility；
- `Hop/Layer` 表示线性图模型用了几跳邻居，或者 GNNHAR 用了几层 graph neural network；
- `MSE ratio` 和 `QLIKE ratio` 都是相对同一 universe 中 $HAR_M$ 的 out-of-sample loss ratio；
- ratio 小于 1 表示相对 $HAR_M$ 有改进，ratio 大于 1 表示比 $HAR_M$ 差；
- `gain` 是 $1-\text{ratio}$，所以正数是 improvement，负数是 deterioration。

这里的关键不是只找 winner，而是看三组关系。第一，看 non-IV 的 GHAR / GNNHAR 是否比 HAR 好，这是 Zhang-style graph model 的问题。第二，看同一模型加入 IV 后是否更好，这是本文的 IV contribution。第三，看这些 gain 是否从 Dow30 到 S&P100 / S&P500 变大，这是本文的“提升的提升”问题。如果读者只看每个表的最优模型，会错过我们真正想比较的结构。

表格顺序建议：

1. Strict Zhang-style non-IV panel:
   - $HAR_M$
   - $GHAR_M$
   - $GNNHAR1L_M,\ldots,GNNHAR5L_M$
   - $HAR_Q$
   - $GHAR_Q$
   - $GNNHAR1L_Q,\ldots,GNNHAR5L_Q$
2. IV extension panel:
   - $HAR_M^{IV}$
   - $GHAR_M^{IV}$
   - $GNNHAR1L_M^{IV},\ldots,GNNHAR5L_M^{IV}$
   - $HAR_Q^{IV}$
   - $GHAR_Q^{IV}$
   - $GNNHAR1L_Q^{IV},\ldots,GNNHAR5L_Q^{IV}$

`GHAR2H` / `GHAR3H` 不放入正文主表，避免主线过于复杂。它们放在 Appendix C 中作为 Zhang Appendix E-style 的 linear multi-hop diagnostic。当前保存预测的实际情况是：

- Dow30：当前 aligned full-model rerun 已保存 HAR/GHAR/GNNHAR1L--5L 及 IV variants，共 28 个模型，应作为正文 Dow30 主表；另一个 aligned multi-hop supplement 已保存 $GHAR2H/GHAR3H$ 的 MSE、QLIKE、IV 和 non-IV 版本，应作为 Appendix C 的 Dow30 aligned multi-hop table 报告；旧 232-date full-model rows 不应混入当前 aligned sample table；
- S&P100：现在可以完整报告 $GHAR2H_M,GHAR3H_M,GHAR2H_Q,GHAR3H_Q$ 及其 IV 版本；新增 QLIKE-trained multi-hop GHAR 预测已和 SP100 的 $234\times91$ truth array 对齐；
- S&P500：可以完整放 $GHAR2H/GHAR3H$ 的 MSE、QLIKE、IV 和 non-IV 版本。

这也要在 empirical setup 中说明：multi-hop GHAR 是本文为检验 linear multi-hop spillover 加入的 Zhang-style extension。Zhang 主文强调 GNNHAR depth，GHAR2Hop 主要在 Appendix E 中作为 supplementary diagnostic 出现；本文把它系统化为 2-hop 和 3-hop 线性图模型，但只在附录中报告，正文仅引用其核心结论。还要明确：Dow30 的 multi-hop supplement 和 Dow30 full-model run 的日期不同，因此不能把 supplement 的 rows 直接并入 Dow30 full loss table；S&P100 的新增 QLIKE multi-hop rows 则已经写回同一个 SP100 run 目录，可以和原 MSE multi-hop rows 一起报告。

每个表格列：

| Model | EC | IV | Hop/Layer | MSE ratio | QLIKE ratio | MSE gain | QLIKE gain | MCS-MSE | MCS-QL |
|---|---|---|---|---:|---:|---:|---:|---|---|

标注规则：

- red：当前 universe 下 MSE ratio 最低；
- blue：当前 universe 下 QLIKE ratio 最低；
- `*`：MCS 5% level 下属于 best model set；
- 不按 loss ratio 排序，按模型结构排序。

当前可写的数据驱动结论：

- Dow30：$GNNHAR2L_Q^{IV}$ 是标准窗口 aligned full-model rerun 的 QLIKE 最优，QLIKE ratio 约 0.923，相对 $HAR_M$ 约 7.7% improvement；MSE 下 $HAR_M^{IV}$ 最优，MSE ratio 约 0.959；
- S&P100：$HAR_Q^{IV}$ 是当前 QLIKE 最优，QLIKE ratio 约 0.933；MSE 下 $HAR_M^{IV}$ 最优，MSE ratio 约 0.965；IV 的贡献比 GNN depth 更稳定；
- S&P500：正式 AutoDL 结果使用 449 tickers；$GHAR_M^{IV}$ 是当前 QLIKE 最优，QLIKE ratio 约 0.973；MSE 下 $HAR_M^{IV}$ 最优，MSE ratio 约 0.966；深层 GNN 在大图上没有自动获得优势。

对 S&P500 的 QLIKE-trained panel 要加一个解释性脚注或正文注释。当前 `HAR_Q` 的 test MSE 约为 1923.30，而 `HAR_M` 的 test MSE 约为 10.63；`GHAR3H_Q^{IV}` 的 test MSE 约为 938.39。后者相对 `HAR_Q` 的 MSE ratio 约为 0.488，但相对 `HAR_M` 仍然非常差。因此，QLIKE-trained panel 中的 MSE ratio 如果以 $HAR_Q$ 为分母，会被 $HAR_Q$ 的极端 MSE 放大效应扭曲。正文应统一使用 $HAR_M$ 分母，并在 Section 5.1 解释这个异常。

主表写作顺序应该是：

1. 先解释每个 ratio 都是相对 $HAR_M$ 的 out-of-sample loss ratio；
2. 再逐个 universe 解释最优模型；
3. 然后比较 non-IV 和 IV panel；
4. 最后讨论 GHAR/GNNHAR 是否稳定优于 HAR。

这里不要只报告 winner。需要强调完整表格保留了所有失败或表现普通的模型，因为这些模型本身就是 RQ3 的证据：如果节点关系丰富真的带来系统性提升，那么 GNNHAR 的多层版本应该在 S&P100 和 S&P500 中更稳定地胜出；当前结果没有显示这一点。

下面三张表是可以直接进入 Section 4 的当前主结果表。表中所有 MSE ratio 和 QLIKE ratio 都以同一 universe 的 $HAR_M$ 为分母；gain 定义为 $1-\text{ratio}$。`GHAR2H` / `GHAR3H` 不放入正文主表，留到 Appendix C。当前 `loss_table.csv` 尚未合并 MCS membership，因此正文正式排版时若要加入 MCS-MSE / MCS-QL 星号，需要从 MCS 输出表再合并一次。

**Table 4.1. Dow30 out-of-sample loss ratios.** The current date-aligned Dow30 full-model sample uses 30 model tickers and 234 saved one-day-ahead test dates from 2025-07-07 to 2026-06-09. It is the `20260619T071426Z_aligned_full_model` run, with 28 prediction files covering HAR/GHAR/GNNHAR1L--5L and IV variants. ~~The archived 232-date Dow30 full-model run from 2025-02-21 to 2026-01-23 should be used as the current full-model evidence.~~ The archived 232-date run should now be treated only as historical evidence; the aligned rerun is the Dow30 main table source.

读这张 Dow30 表时，重点是小图环境下 graph / GNN / IV 是否能带来比较明显的 improvement。当前标准窗口结果中，QLIKE 最优模型是 $GNNHAR2L_Q^{IV}$，MSE 最优模型是 $HAR_M^{IV}$。这说明在 Dow30 这个较小 universe 中，图结构、非线性层和 IV 至少在部分 criterion 下确实可以相对 $HAR_M$ 提供较明显增益。

| Model | EC | IV | Hop/Layer | MSE ratio | QLIKE ratio | MSE gain | QLIKE gain |
|---|---|---|---|---:|---:|---:|---:|
| `HAR_M` | M | no | - | 1.000 | 1.000 | 0.0% | 0.0% |
| `GHAR_M` | M | no | 1H | 1.001 | 0.995 | -0.1% | 0.5% |
| `GNNHAR1L_M` | M | no | 1L | 1.004 | 1.009 | -0.4% | -0.9% |
| `GNNHAR2L_M` | M | no | 2L | 1.006 | 1.012 | -0.6% | -1.2% |
| `GNNHAR3L_M` | M | no | 3L | 1.004 | 1.014 | -0.4% | -1.4% |
| `GNNHAR4L_M` | M | no | 4L | 1.004 | 1.015 | -0.4% | -1.5% |
| `GNNHAR5L_M` | M | no | 5L | 1.008 | 1.039 | -0.8% | -3.9% |
| `HAR_Q` | Q | no | - | 1.007 | 0.983 | -0.7% | 1.7% |
| `GHAR_Q` | Q | no | 1H | 1.004 | 0.980 | -0.4% | 2.0% |
| `GNNHAR1L_Q` | Q | no | 1L | 1.005 | 0.980 | -0.5% | 2.0% |
| `GNNHAR2L_Q` | Q | no | 2L | 1.041 | 1.017 | -4.1% | -1.7% |
| `GNNHAR3L_Q` | Q | no | 3L | 1.043 | 1.013 | -4.3% | -1.3% |
| `GNNHAR4L_Q` | Q | no | 4L | 1.005 | 0.982 | -0.5% | 1.8% |
| `GNNHAR5L_Q` | Q | no | 5L | 1.004 | 0.978 | -0.4% | 2.2% |
| `HAR_M_IV` | M | yes | - | 0.959 | 0.966 | 4.1% | 3.4% |
| `GHAR_M_IV` | M | yes | 1H | 0.960 | 0.974 | 4.0% | 2.6% |
| `GNNHAR1L_M_IV` | M | yes | 1L | 0.968 | 0.986 | 3.2% | 1.4% |
| `GNNHAR2L_M_IV` | M | yes | 2L | 0.968 | 0.985 | 3.2% | 1.5% |
| `GNNHAR3L_M_IV` | M | yes | 3L | 0.968 | 0.996 | 3.2% | 0.4% |
| `GNNHAR4L_M_IV` | M | yes | 4L | 0.968 | 1.005 | 3.2% | -0.5% |
| `GNNHAR5L_M_IV` | M | yes | 5L | 0.969 | 0.972 | 3.1% | 2.8% |
| `HAR_Q_IV` | Q | yes | - | 0.981 | 0.927 | 1.9% | 7.3% |
| `GHAR_Q_IV` | Q | yes | 1H | 0.985 | 0.929 | 1.5% | 7.1% |
| `GNNHAR1L_Q_IV` | Q | yes | 1L | 0.983 | 0.929 | 1.7% | 7.1% |
| `GNNHAR2L_Q_IV` | Q | yes | 2L | 0.986 | 0.923 | 1.4% | 7.7% |
| `GNNHAR3L_Q_IV` | Q | yes | 3L | 0.985 | 0.931 | 1.5% | 6.9% |
| `GNNHAR4L_Q_IV` | Q | yes | 4L | 0.987 | 0.929 | 1.3% | 7.1% |
| `GNNHAR5L_Q_IV` | Q | yes | 5L | 0.984 | 0.931 | 1.6% | 6.9% |

**Table 4.2. S&P100 full-model out-of-sample loss ratios.** The table reports the current S&P100 A100 run with 91 model tickers and 234 saved one-day-ahead test dates from 2025-07-07 to 2026-06-09. Ratios are relative to $HAR_M$ in the same S&P100 run. Multi-hop GHAR rows are excluded from this main table and reported in Appendix C.

读这张 S&P100 表时，重点从“GNN 是否明显胜出”转向“IV 是否稳定有用”。当前结果里，最优 QLIKE 更偏向 $HAR_Q^{IV}$，而不是深层 GNNHAR。这对 RQ2 是支持证据，但对 RQ3 是一个提醒：节点更多并不自动意味着 GNN depth 的优势更大。

| Model | EC | IV | Hop/Layer | MSE ratio | QLIKE ratio | MSE gain | QLIKE gain |
|---|---|---|---|---:|---:|---:|---:|
| `HAR_M` | M | no | - | 1.000 | 1.000 | 0.0% | 0.0% |
| `GHAR_M` | M | no | 1H | 0.997 | 0.993 | 0.3% | 0.7% |
| `GNNHAR1L_M` | M | no | 1L | 1.000 | 0.997 | 0.0% | 0.3% |
| `GNNHAR2L_M` | M | no | 2L | 1.000 | 0.999 | -0.0% | 0.1% |
| `GNNHAR3L_M` | M | no | 3L | 1.001 | 1.000 | -0.1% | 0.0% |
| `GNNHAR4L_M` | M | no | 4L | 1.004 | 1.003 | -0.4% | -0.3% |
| `GNNHAR5L_M` | M | no | 5L | 1.003 | 1.010 | -0.3% | -1.0% |
| `HAR_Q` | Q | no | - | 1.006 | 0.980 | -0.6% | 2.0% |
| `GHAR_Q` | Q | no | 1H | 1.003 | 0.976 | -0.3% | 2.4% |
| `GNNHAR1L_Q` | Q | no | 1L | 1.004 | 0.980 | -0.4% | 2.0% |
| `GNNHAR2L_Q` | Q | no | 2L | 1.004 | 0.979 | -0.4% | 2.1% |
| `GNNHAR3L_Q` | Q | no | 3L | 1.003 | 0.977 | -0.3% | 2.3% |
| `GNNHAR4L_Q` | Q | no | 4L | 1.003 | 0.978 | -0.3% | 2.2% |
| `GNNHAR5L_Q` | Q | no | 5L | 1.003 | 0.977 | -0.3% | 2.3% |
| `HAR_M_IV` | M | yes | - | 0.965 | 0.979 | 3.5% | 2.1% |
| `GHAR_M_IV` | M | yes | 1H | 0.965 | 0.978 | 3.5% | 2.2% |
| `GNNHAR1L_M_IV` | M | yes | 1L | 0.972 | 0.987 | 2.8% | 1.3% |
| `GNNHAR2L_M_IV` | M | yes | 2L | 0.989 | 1.012 | 1.1% | -1.2% |
| `GNNHAR3L_M_IV` | M | yes | 3L | 0.968 | 0.990 | 3.2% | 1.0% |
| `GNNHAR4L_M_IV` | M | yes | 4L | 0.979 | 0.996 | 2.1% | 0.4% |
| `GNNHAR5L_M_IV` | M | yes | 5L | 0.980 | 0.994 | 2.0% | 0.6% |
| `HAR_Q_IV` | Q | yes | - | 0.981 | 0.933 | 1.9% | 6.7% |
| `GHAR_Q_IV` | Q | yes | 1H | 0.981 | 0.933 | 1.9% | 6.7% |
| `GNNHAR1L_Q_IV` | Q | yes | 1L | 0.981 | 0.935 | 1.9% | 6.5% |
| `GNNHAR2L_Q_IV` | Q | yes | 2L | 0.982 | 0.933 | 1.8% | 6.7% |
| `GNNHAR3L_Q_IV` | Q | yes | 3L | 0.981 | 0.940 | 1.9% | 6.0% |
| `GNNHAR4L_Q_IV` | Q | yes | 4L | 0.986 | 0.936 | 1.4% | 6.4% |
| `GNNHAR5L_Q_IV` | Q | yes | 5L | 0.981 | 0.934 | 1.9% | 6.6% |

**Table 4.3. S&P500 full-model out-of-sample loss ratios.** The table reports the formal S&P500 AutoDL result with 449 model tickers and 223 saved one-day-ahead test dates from 2025-07-14 to 2026-06-01. Ratios are relative to $HAR_M$ in the same S&P500 run. The large MSE ratios for QLIKE-trained models are retained because they are part of the current diagnostic evidence rather than rows to be filtered out.

读这张 S&P500 表时，不能只把 GNNHAR 的大 loss ratio 当成一个普通排序结果。它同时回答了两个问题：第一，IV-augmented HAR / GHAR 仍然相对稳定，说明 IV contribution 在大 universe 中没有消失；第二，当前 GNNHAR / GNNHAR-IV 在 S&P500 中出现数量级上的失效，需要在 Section 5 和 Section 6 里进一步诊断。换句话说，这张表对 RQ3 的回答是负面的：更大的图没有在当前实现中放大 GNNHAR 的优势，反而暴露了训练稳定性、尺度和低波动股票预测下界的问题。

还要在表下注明：S&P500 中 QLIKE-trained linear rows 的巨大 MSE ratio 主要由少数 extreme stock-date observations 支配，核心案例是 EchoStar (`SATS`) 在 2025-08-26 附近的真实价格跳跃进入 30-day RV rolling window 后，使 `HAR_Q` / `GHAR_Q` 等模型产生极端预测。这个问题在 Section 5.1 和 Appendix E 中用 `SATS` loss-contribution table 展开；它解释的是 QLIKE-trained panel 的 MSE 爆炸，而不是说整个 S&P500 表都是坏数据。

| Model | EC | IV | Hop/Layer | MSE ratio | QLIKE ratio | MSE gain | QLIKE gain |
|---|---|---|---|---:|---:|---:|---:|
| `HAR_M` | M | no | - | 1.000 | 1.000 | 0.0% | 0.0% |
| `GHAR_M` | M | no | 1H | 1.000 | 0.998 | -0.0% | 0.2% |
| `GNNHAR1L_M` | M | no | 1L | 6.282 | 10.922 | -528.2% | -992.2% |
| `GNNHAR2L_M` | M | no | 2L | 7.040 | 11.258 | -604.0% | -1025.8% |
| `GNNHAR3L_M` | M | no | 3L | 6.466 | 10.987 | -546.6% | -998.7% |
| `GNNHAR4L_M` | M | no | 4L | 6.743 | 11.098 | -574.3% | -1009.8% |
| `GNNHAR5L_M` | M | no | 5L | 6.272 | 10.919 | -527.2% | -991.9% |
| `HAR_Q` | Q | no | - | 180.880 | 2.910 | -17988.0% | -191.0% |
| `GHAR_Q` | Q | no | 1H | 161.079 | 2.851 | -16007.9% | -185.1% |
| `GNNHAR1L_Q` | Q | no | 1L | 17.051 | 7.857 | -1605.1% | -685.7% |
| `GNNHAR2L_Q` | Q | no | 2L | 16.106 | 8.152 | -1510.6% | -715.2% |
| `GNNHAR3L_Q` | Q | no | 3L | 17.289 | 7.854 | -1628.9% | -685.4% |
| `GNNHAR4L_Q` | Q | no | 4L | 19.880 | 9.965 | -1888.0% | -896.5% |
| `GNNHAR5L_Q` | Q | no | 5L | 17.652 | 7.856 | -1665.2% | -685.6% |
| `HAR_M_IV` | M | yes | - | 0.966 | 0.974 | 3.4% | 2.6% |
| `GHAR_M_IV` | M | yes | 1H | 0.966 | 0.973 | 3.4% | 2.7% |
| `GNNHAR1L_M_IV` | M | yes | 1L | 6.261 | 10.912 | -526.1% | -991.2% |
| `GNNHAR2L_M_IV` | M | yes | 2L | 6.380 | 10.956 | -538.0% | -995.6% |
| `GNNHAR3L_M_IV` | M | yes | 3L | 6.321 | 10.935 | -532.1% | -993.5% |
| `GNNHAR4L_M_IV` | M | yes | 4L | 6.578 | 11.048 | -557.8% | -1004.8% |
| `GNNHAR5L_M_IV` | M | yes | 5L | 6.361 | 10.951 | -536.1% | -995.1% |
| `HAR_Q_IV` | Q | yes | - | 124.370 | 2.677 | -12337.0% | -167.7% |
| `GHAR_Q_IV` | Q | yes | 1H | 83.373 | 2.698 | -8237.3% | -169.8% |
| `GNNHAR1L_Q_IV` | Q | yes | 1L | 13.818 | 7.823 | -1281.8% | -682.3% |
| `GNNHAR2L_Q_IV` | Q | yes | 2L | 12.851 | 7.802 | -1185.1% | -680.2% |
| `GNNHAR3L_Q_IV` | Q | yes | 3L | 12.930 | 7.824 | -1193.0% | -682.4% |
| `GNNHAR4L_Q_IV` | Q | yes | 4L | 13.632 | 7.808 | -1263.2% | -680.8% |
| `GNNHAR5L_Q_IV` | Q | yes | 5L | 13.241 | 7.826 | -1224.1% | -682.6% |

### 4.4 Cross-universe comparison and the "improvement of improvement"

这一小节放在 Section 4 里，不单独作为一个大章。它直接回答 RQ3：节点规模变大是否提高“相对 HAR 的提升幅度”。

实际表：

**Table 4.4. Best-model summary across universes.** For each universe, the table reports the best QLIKE model by QLIKE ratio and the best MSE model by MSE ratio, using $HAR_M$ as the common denominator within each universe. The gains are $1-R^{QL}$ and $1-R^{MSE}$, respectively.

| Universe | Best QLIKE model | QLIKE ratio vs $HAR_M$ | Gain $1-R^{QL}$ | Best MSE model | MSE ratio vs $HAR_M$ | Gain $1-R^{MSE}$ |
|---|---|---:|---:|---|---:|---:|
| Dow30 | $GNNHAR2L_Q^{IV}$ | 0.923 | 7.7% | $HAR_M^{IV}$ | 0.959 | 4.1% |
| S&P100 | $HAR_Q^{IV}$ | 0.933 | 6.7% | $HAR_M^{IV}$ | 0.965 | 3.5% |
| S&P500 | $GHAR_M^{IV}$ | 0.973 | 2.7% | $HAR_M^{IV}$ | 0.966 | 3.4% |

解释：

- 如果“节点更多、关系更丰富”会带来“提升的提升”，那么从 Dow30 到 S&P500，$1-R_m$ 应该扩大；
- 当前结果仍然没有支持该假说：使用标准窗口 Dow30 后，best QLIKE gain 约为 7.7%，S&P100 约为 6.7%，S&P500 约为 2.7%，没有随 universe size 扩大而增加；
- 因此当前证据不支持“更大的图天然增强 GNNHAR 相对 HAR 的优势”；
- 更合理的解释是：大图引入更多关系，也引入更多噪声、训练不稳定、尺度问题和 over-smoothing risk。
- 这些 gain 仍然是 relative to realized-volatility proxy 的 statistical loss gains，不应被写成真实风险度量能力或实际交易收益的最终排序。要回答经济价值问题，需要进一步把 forecasts 放入 portfolio / hedging / risk-control decision tests。
- S&P500 的 cross-universe 表还应该补一句机制解释：没有 IV 时，`GHAR_M` 只比 `HAR_M` 在 QLIKE 上小幅改善，non-IV GNNHAR 明显失败；加入 IV 后，`HAR_M_IV` / `GHAR_M_IV` 才成为最优。这个对比说明，S&P500 中稳定增益主要来自 options-implied forward-looking state，而不是单纯来自更大的 historical spillover graph。
- 对 GNNHAR-IV 的解释要克制：它相对 non-IV GNNHAR 有局部改善，说明 IV 中的复杂市场预期信息可能给 nonlinear graph aggregation 提供更好的输入；但当前 S&P500 GNNHAR-IV 的 loss ratio 仍远高于 1，因此它是“信息集改善但实现仍不稳定”的证据，而不是“GNNHAR-IV 已经是大图最优模型”的证据。

### 4.5 Interim empirical conclusion

Section 4 末尾需要有一个短结论，直接把三个 RQ 和主表对应起来：

- RQ1：从 Dow30 到 S&P100 再到 S&P500，GHAR/GNNHAR 及其变体并非都稳定优于 HAR。图结构和 GNN depth 在小图中更容易带来明显增益，但在大图中增益不稳定。
- RQ2：IV 是当前最稳定的新增信息来源。很多 universe 的 winner 或 near-winner 都是 IV version，这支持 Zhang future work 中“options information”方向。
- 尤其在 S&P500 上，要明确写出：historical-only 的线性 HAR/GHAR 和 non-IV GNNHAR 都没有给出令人满意的大图优势；加入 IV 后，最好的模型变成 `HAR_M_IV` / `GHAR_M_IV`。这说明 IV 可能压缩了市场对未来风险、新闻、尾部风险和风险偏好的复杂预期，并把这些 forward-looking 信息带入了日度 panel。
- 同时，GNNHAR-IV 相比 non-IV GNNHAR 的若干 QLIKE 改善可以作为一个中间发现：非线性图模型不是完全不能利用 IV，而是当前大图实现仍受输出尺度、低波动股票高估和训练稳定性制约。正式写作时应把这个点放在 Section 4.5 或 Section 5，而不是只在 appendix 里留作工程诊断。
- RQ3：当前没有看到“节点越多，model/HAR improvement 越大”。标准窗口 Dow30 的 best QLIKE gain 约 7.7%，S&P100 约 6.7%，S&P500 约 2.7%，因此大图可能同时带来更多 spillover information 和更多噪声、训练不稳定、over-smoothing risk。
- 同时要补一句边界：Section 4 的结论是基于 realized-volatility proxy 的 forecast accuracy 结论，不是对真实风险、资产配置收益或交易策略收益的最终评价。

这里还要明确一个当前结果层面的发现：S&P500 中 GNNHAR / GNNHAR-IV 的失败不是轻微排序变化，而是数量级上的异常。`GNNHAR` 预测在低波动股票上存在系统性高估，导致 MSE 和 QLIKE 均显著差于 HAR / GHAR。这个现象不能在正文中简单解释为“GNN 无效”，而应在 Section 5 和 Section 6 中作为模型诊断问题讨论。

然后自然过渡到 Section 5：主表说明了结果是什么，下一节解释这些结果为什么出现，以及这些结果在统计检验和诊断指标下是否稳健。

## 5. Statistical Analysis and Discussion

这一节仿照 Zhang 的 Section 5，但写法上要更像我们的研究笔记。Section 4 已经告诉我们“结果是什么”：Dow30 中 GNNHAR-IV 有明显优势，S&P100 中 IV 更稳定，S&P500 中 GNNHAR 出现异常失效。Section 5 要回答的是“为什么这些结果会这样，以及这些结果能不能被统计诊断支持”。因此这里不是重复主表，而是在主表之后做统计分析、异常定位和机制解释。

它要回答四件事：

1. 不同 estimation / evaluation criterion 是否改变模型排序；
2. MCS 和 DM test 是否支持某些模型显著优于其他模型；
3. FVU 是否显示 GNNHAR 和 IV 确实改变了预测函数；
4. multi-hop neighbors 和 deeper layers 是否带来额外信息，还是引入 smoothing 和不稳定。

这里可以放较小的局部表格和诊断图，但不能替代 Section 4 的三张完整主表。

### 5.1 Impact of evaluation criterion

仿照 Zhang 先讨论 MSE-trained 和 QLIKE-trained models 的差异。本小节的目的不是重新排序所有模型，而是解释为什么同一组 S&P500 结果在 MSE 和 QLIKE 下会给出非常不同的信号。读者应先记住两个事实：

1. MSE 对绝对误差极其敏感，因为误差会被平方；
2. QLIKE 是 volatility forecasting 中常用的 relative forecast loss，对低估 volatility 的惩罚较强，但对极端过度高估的增长不像 MSE 那样以平方速度放大。

Zhang 的发现是，QLIKE-trained models 往往在 turbulent periods 更有优势。本文也比较 $F_M$ 和 $F_Q$，尤其是 $HAR_M$ vs $HAR_Q$、$GHAR_M$ vs $GHAR_Q$、$GNNHARkL_M$ vs $GNNHARkL_Q$，以及对应 IV 版本。不过，S&P500 结果显示：当个别股票发生真实的 event-driven jump 时，QLIKE-trained linear models 可能产生很大的 MSE outliers。因此，这一节同时承担两个任务：一方面解释 Zhang-style QLIKE criterion 为什么重要，另一方面诊断它在我们 S&P500 数据中的极端个案。

从研究过程上说，这一节很重要，因为它记录了一个我们一开始也觉得“奇怪”的现象。S&P500 主表里，某些 QLIKE-trained 模型的 MSE 非常大，看起来像模型或数据出了问题；但如果只把它删掉，就会错过一个有价值的 stress case。我们后面通过 prediction array、stock-date loss contribution、Yahoo Finance 价格、local return panel、RV/IV panel 一层层核对，最后发现问题集中在 `SATS` 的真实股价跳跃上。也就是说，这不是单纯的坏数据，而是一个真实市场事件暴露了 QLIKE-trained linear models 在极端状态下的外推风险。

当前 S&P500 结果提供了一个必须谨慎解释的例子。QLIKE-trained linear models 的 MSE 可能被少数极端预测主导：在正式 S&P500 AutoDL 输出中，`HAR_Q` 的最大预测值约为 2887，而真实值最大约为 206；最严重的 squared-error 点集中在 `SATS` 的 2025 年 8 月底到 9 月期间。`HAR_Q` 的前 10 个 squared-error observation 贡献了约 32% 的总 MSE，前 100 个贡献约 93%。`GHAR3H_Q^{IV}` 把这些极端预测从约 2800 降到约 2000，因此相对 `HAR_Q` 的 MSE ratio 看起来改善明显，但它相对 $HAR_M$ 的 MSE 仍然很差。写作时应说明：QLIKE-trained 模型主要应按 QLIKE forecast loss 解释，MSE 列在这里更多是异常诊断而不是模型胜负的主依据。

这个异常不是数据错误，而是一个真实 market jump case。Yahoo Finance 日线和我们本地 `daily_returns.csv` 一致显示，EchoStar (`SATS`) 在 2025-08-26 单日 close-to-close return 约为 $70.25\%$，随后在 2025-09-08 又上涨约 $19.91\%$。由于本文使用的是 30-day realized-volatility proxy，这种单日暴涨会机械性进入 rolling window，并使 $v_{i,t-1}$、$\bar v_{i,t-5:t-2}$、$\bar v_{i,t-22:t-6}$ 在随后多个交易日持续处于极端高位。以 2025-09-23 为例，`SATS` 的 daily lag 约为 205.66，weekly lag 约为 201.52，monthly lag 约为 176.43，均接近或超过训练特征分布的 99.9% tail。

这个诊断过程本身也很重要，因为它记录了我们是怎么发现问题、再把问题定位到原始数据上的。最开始我们看到的是 S&P500 主表里 QLIKE-trained rows 的 MSE ratio 非常大，但 QLIKE loss 没有同样夸张；这说明问题不太像普通模型排序差异，而更像少数极端点在支配 MSE。于是我们检查 prediction arrays，发现最大的 squared-error observations 集中在 `SATS` 这一只股票。再进一步用 Yahoo Finance 日线、本地 `daily_returns.csv`、RV panel 和 IV panel 对齐后，我们确认 `SATS` 在 2025-08-26 发生了真实价格跳跃。这说明异常不是数据清洗错误，而是一个真实市场事件通过 30-day RV proxy 进入模型，并放大了 estimation criterion 和 forecast loss 的差异。

下面这张表的作用是把“模型为什么爆掉”连接回原始市场数据，而不是只报告一个抽象的 loss number。EchoStar (`SATS`) 在 2025-08-26 单日上涨约 $70.25\%$，并在 2025-09-08 又出现一次约 $19.91\%$ 的上涨。由于我们的因变量是 30-day realized-volatility proxy，这个单日跳跃会在 rolling window 中停留很多个交易日，所以后续的 daily、weekly、monthly HAR lag features 会同时变得非常极端。

**Table 5.x. `SATS` 的事件驱动型波动率跳升。** 这张表列出 EchoStar (`SATS`) 在 2025 年 8 月底价格跳跃前后的 close-to-close return、本文作为目标变量使用的 30-day realized-volatility proxy，以及 30-day implied-volatility mean。它要说明的是：这些大的 RV observations 来自真实股价变化，不是数据清洗产生的错误值。

| Date | Close | Return | 30d RV | IV mean | Comment |
|---|---:|---:|---:|---:|---|
| 2025-08-25 | 29.88 | 0.88% | 83.91 | 70.78 | before jump |
| 2025-08-26 | 50.87 | 70.25% | 197.72 | 59.45 | event jump |
| 2025-08-27 | 58.76 | 15.51% | 201.62 | 115.75 | RV remains elevated |
| 2025-09-08 | 80.63 | 19.91% | 198.76 | 77.12 | second jump |
| 2025-09-15 | 71.94 | -3.94% | 205.48 | 56.42 | lag features still extreme |
| 2025-09-23 | 73.64 | 0.68% | 200.89 | 50.48 | HAR lags near training tail |

下一张表解释为什么这个事件对 MSE 的影响特别大，但对 aggregate QLIKE 的影响没有同样大。这里一个 observation 指的是一个 stock-date pair。S&P500 测试集一共有 $449\times 223=100{,}127$ 个 stock-date observations，而 `SATS` 只贡献其中 223 个。因此：

- `Total MSE` 是所有 100,127 个 stock-date observations 的平均 squared error；
- `SATS MSE` 只是 `SATS` 这 223 个 observations 的平均 squared error；
- `SATS share of total SSE` 不是用 `SATS MSE / Total MSE` 计算，而是用 `SATS` 的 squared-error 总和除以全部 stock-date squared-error 总和。

这个分母区别非常关键。`SATS MSE` 可以非常大，而 `Total MSE` 没有同样大，是因为前者只是在一只股票上取平均，后者是在全部股票和日期上取平均。但另一方面，`SATS` 的误差又大到足以贡献 $HAR_Q$ 全部 squared-error sum 的 57.2%。所以这个表并不是在说 $493833.40/1923.30=57.2\%$，而是在说：虽然 `SATS` 只占 $1/449$ 的股票数量，但它贡献了超过一半的总平方误差。

**Table 5.y. `SATS` 对 S&P500 QLIKE-trained forecast losses 的贡献。** `Total MSE` 和 `Total QLIKE` 是在全部 $449\times223$ 个 stock-date observations 上取平均；`SATS MSE` 和 `SATS QLIKE` 只是在 223 个 `SATS` observations 上取平均。两个 share columns 表示 `SATS` 对 aggregate loss sum 的贡献比例，而不是两个平均损失列之间的比值。

| Model | Total MSE, all stock-dates | `SATS` MSE, `SATS` dates only | `SATS` share of total SSE | Total QLIKE, all stock-dates | `SATS` QLIKE, `SATS` dates only | `SATS` share of total QLIKE |
|---|---:|---:|---:|---:|---:|---:|
| $HAR_Q$ | 1923.30 | 493833.40 | 57.2% | 0.0132 | 0.1836 | 3.1% |
| $GHAR3H_Q^{IV}$ | 938.39 | 221631.07 | 52.6% | 0.0119 | 0.1442 | 2.7% |
| $GNNHAR2L_Q^{IV}$ | 136.64 | 20131.81 | 32.8% | 0.0355 | 0.0607 | 0.4% |

这张表给了我们更准确的解释：`SATS` 对 squared-error loss 是主导性的，但对 aggregate QLIKE 只贡献了几个百分点。直观地说，MSE 会把预测误差平方，所以当 `HAR_Q` 对 `SATS` 预测到 2500--2900、而真实 RV 大约只有 200 时，少数点就足以支配总 MSE；但 QLIKE 对极端过度高估的惩罚增长没有 MSE 那么快，所以这些点不会以同样比例支配 aggregate QLIKE。这就是为什么 S&P500 的 QLIKE-trained panel 中 MSE 列会爆炸，而 QLIKE 列没有同比例爆炸。

这个诊断给本文带来两个实质结论。第一，S&P500 的 QLIKE-trained MSE 异常不是无意义的坏数据，而是一个有解释价值的 stress case：真实价格跳跃使 30-day RV 和 HAR lag features 进入极端尾部，从而暴露出 QLIKE-trained linear extrapolation 在 event-driven jump 下的 outlier sensitivity。第二，IV 或更 flexible 的 graph / neural specification 在这些极端点上降低了部分爆炸预测，但这不应立即解释为 IV 信息本身完全负责；一个 alternative explanation 是，加入 IV 后模型参数和状态变量增加，模型有更大的自由度去吸收或缓冲 `SATS` 这种极端状态。正式论文中应把这写成待检验机制：IV improvement may reflect genuine option-market forward-looking information, additional model flexibility, or both. 这也为 Section 6 的 IV interaction、outlier-robust loss、以及 corrected GNN output robustness 提供动机。

可以放图：

- forecast errors boxplot；
- forecast ratios $\widehat v/v$ boxplot；
- calm / turbulent 分组 boxplot；
- 如果已有 PNG，就引用；如果还没整理，就写 `[Figure placeholder: grouped forecast error and forecast ratio boxplots, Zhang Fig. 5 style.]`。

建议补一张 S&P500 outlier diagnostic 小表或图，列出 `HAR_Q`、`GHAR3H_Q^{IV}`、最佳 MSE-trained 模型的 prediction quantiles、maximum prediction、top-$k$ squared-error contribution。这个表不用放主表，可以放 Section 5.1 或 Appendix E。它的作用是解释为什么同一个 QLIKE-trained panel 会出现 QLIKE ratio 看起来可接受、但 MSE 失真的现象。

### 5.2 MCS and pairwise DM interpretation

这一小节专门解释统计检验，不要把 MCS 和 DM 散落到很多地方。

MCS 的作用：

- 在同一个 candidate model set 中识别 statistically indistinguishable best model set；
- 它不是问某一个模型是否比 HAR 显著更好，而是问哪些模型不能从最优集合中被排除；
- 因此主表中的 `*` 应表示模型在 5% confidence level 下属于 MCS best set。

DM test 的作用：

- DM 是 pairwise forecast comparison；
- loss differential 可以写成

$$
d_t
=
L(e_{1,t})-L(e_{2,t}),
$$

其中 $L(\cdot)$ 可以是 MSE loss 或 QLIKE loss；
- 如果 $d_t>0$，模型 2 的 loss 更小；如果 $d_t<0$，模型 1 的 loss 更小；
- 报告时需要明确方向，例如 `Model A vs Model B` 的 positive statistic 是支持 B 还是支持 A。

本文 DM 的主要比较不应只强行 vs HAR。更贴近 Zhang 的比较包括：

- $GNNHAR1L$ vs $GNNHAR2L,\ldots,GNNHAR5L$：检验 deeper graph layers 是否提供额外预测信息；
- $GNNHARkL$ vs $GNNHARkL^{IV}$：检验 IV 是否在相同 layer depth 下提供额外信息；
- $HAR$ vs $HAR^{IV}$、$GHAR$ vs $GHAR^{IV}$：检验 options-implied forward-looking information 的边际贡献。

GHAR vs GHAR2H / GHAR3H 的 DM test 放入 Appendix C。正文只在解释 GNN depth 时引用其结论：线性多跳邻居没有显示出稳定的额外预测力，因此主文不把它作为核心模型族展开。

表格占位：

`[Table placeholder: MCS inclusion and directional DM-test summary, by universe and loss function.]`

### 5.3 FVU and the impact of nonlinearity

FVU 是 Zhang 用来分析 nonlinearity 的指标。这里要保留我们的研究逻辑：我们之所以做 FVU，不只是因为 Zhang 做了，而是因为我们预设两类信息可能带来 nonlinear structure：

1. GNNHAR 通过 ReLU graph layers 捕捉 nonlinear spillover；
2. IV 作为 option-market forward-looking information，可能包含新闻、风险偏好、尾部风险和机构预期，因此它与未来 RV 的关系未必是简单线性的。

可以写 FVU 定义：

$$
\mathrm{FVU}^{(m)}
=
\frac{\sum_{i,t}
\left(\widehat v_{i,t}^{(m)}-\widehat v_{i,t}^{(HAR_M)}\right)^2}
{\sum_{i,t}
\left(\widehat v_{i,t}^{(m)}-\bar{\widehat v}^{(m)}\right)^2}.
$$

具体公式要和最终脚本输出对齐。写作时要说明：

- FVU 是 effect size，不是显著性检验；
- 它衡量模型预测函数相对 baseline 移动了多少；
- 对比 $GHAR$ vs $GNNHAR1L$ 可以解释 nonlinear graph layer 是否真的改变预测；
- 对比 non-IV vs IV 可以解释 IV 是否改变预测结构；
- 对比 $GNNHAR^{IV}$ vs $GNNHAR$ 可以检验 IV 是否和 nonlinear graph aggregation 共同产生更大预测函数变化。

表格占位：

`[Table placeholder: FVU relative to $HAR_M$, by universe, model family, IV status, and regime.]`

### 5.4 Predictive information from multi-hop neighbors

仿照 Zhang 的 Section 5.3 和 Appendix E：

- 用 DM test 比较 $GNNHAR2L$ vs $GNNHAR1L$，$GNNHAR3L$ vs $GNNHAR1L$，以及 4L / 5L；
- IV 版本同样做一组；
- 重点不是“越深越好”，而是检验 deeper GNN layers 是否在 zero-hop 和 one-hop 信息之外提供显著预测力。

表格占位：

`[Table placeholder: QLIKE DM tests for GNNHAR depth comparisons, non-IV and IV versions.]`

写作结论应保持克制：Zhang 的结论是 multi-hop neighbors 不一定带来显著额外信息；我们的当前结果也应按 DM 的方向和 p-value 来判断，不要强行说深层有效。GHAR2H / GHAR3H 的线性多跳检验放入 Appendix C，正文只概括为：linear multi-hop GHAR does not provide robust incremental forecasting power.

### 5.5 Market regimes

仿照 Zhang 的 market regime subsection，但要说明我们的 regime split 当前不是严格 SPY RV regime。当前统计层使用 cross-sectional mean RV proxy from truth array；如果后续补 SPY RV，可以改成 Zhang 的 Bottom 90% / Top 10% design。

写法：

- calm period：market-level RV proxy below 90% quantile；
- turbulent period：market-level RV proxy above 90% quantile；
- 分别报告 MSE / QLIKE loss ratios；
- 说明这一节用于检查 IV 和 GNNHAR 的表现是否只来自某一类市场状态；
- 不把当前 proxy-regime 结果作为最核心结论。

这个小节和 RQ2 有联系：如果 IV 在 turbulent period 更有用，可以解释为 options market 对 tail risk、event risk 和 uncertainty repricing 更敏感。

### 5.6 Smoothing and MAD diagnostics

仿照 Zhang 的 MAD discussion：

$$
\mathrm{MAD}
=
\frac{\sum_{i=1}^N \bar d_i}
{\sum_{i=1}^N \mathbf{1}\{\bar d_i>0\}},
$$

其中 $\bar d_i$ 是节点 $i$ 与其 connected neighbors 的 average masked cosine distance。MAD 越小表示 node representations 越相似，over-smoothing risk 越强。

当前限制要写清楚：

- Zhang 的 exact MAD 使用 GNN final hidden representation $H^{(K)}$；
- 当前 saved forecasts 没有保存每个 rolling block 的 hidden states；
- 因此当前只能报告 prediction-level smoothing proxy 或在 future robustness 里重跑保存 hidden representations；
- 如果要严格复刻 Zhang Figure 7，需要修改 notebook，在 prediction 阶段保存 $H^{(K)}$。

### 5.7 What the current evidence says relative to Zhang

这一小节集中讨论的是：我们和 Zhang 到底在什么地方一致，什么地方不一样，以及这些不一样到底有什么研究意义。

- 与 Zhang 一致：多跳和更深层 GNN 不必然更好；QLIKE 是重要 evaluation / estimation criterion；graph structure 是合理的 spillover representation。
- 与 Zhang 不同：我们的数据是 30-day HV / IV proxy，不是 LOBSTER 高频 RV；我们的核心新增变量是 IV；S&P500 大图没有显示更强 GNN gain。
- 因此不能说我们复现了 Zhang 的数值结果，只能说我们在 Zhang-style framework 下做了 IV 和 large-universe extension。

这个“不一样”并不是缺陷本身。它反而说明了 Zhang future work 里那句“扩展 predictor set”是真的有经济含义：一旦把 options-implied information 放进来，真正稳定的改进可能来自 forward-looking signal，而不是单纯加深 graph layers。换句话说，本文不是在替 Zhang 做一遍同样的实验，而是在他们留下的框架上检验一个更具体的问题：当信息源从 historical RV 扩展到 IV 时，模型的 gain 会不会比单纯扩图更有解释力。

这也是为什么当前结果值得写进正文，而不是只作为“和 Zhang 不一致”的附注。它告诉我们，graph/GNN 结构本身并没有自动带来更大 universe 下的优势；相反，options market 的 forward-looking signal 在多个 universe 里更稳定。这一点会影响后面的 conclusion：我们应该强调 IV 才是本文最稳的新增信息，而不是把重心放在更深 GNN 上。

这里还要加一个对 30-day HV proxy 的主动解释。审稿人可能会问：本文到底是在 forecast realized volatility，还是在 forecast 一个 rolling 30-day historical volatility indicator？我们的回答应该是：本文明确是后者，即 daily variance-scale 30-day close-to-close HV proxy；这不是 intraday RV paper。但是这个目标并不使本文失去意义，因为它更平滑、更有持久性，也更接近 IV30 的期限结构。正因为它平滑，HAR / IV baseline 更强，GNNHAR 更难通过噪声拟合获胜；如果 GNNHAR 在这种更规则的目标上仍然不稳定，这反而支持本文关于 graph neural depth fragility 的主线。未来可以用已下载的 10-day HV / IV 数据做更短窗口 robustness，它会减少 monthly HAR overlap，但当前不能把 10-day 写成已完成主结果。

## 6. Robustness Tests

这一节仿照 Zhang Section 6，但要按照我们自己的项目情况写。Zhang 做 alternative validation set size 和 larger universe；我们的 larger universe 已经成为主研究问题，因此 robustness 不应只重复 S&P100 / S&P500。

### 6.1 Alternative validation split

Zhang 用 smaller validation dataset 做 robustness。我们可以设计：

- baseline：当前正式 run config；
- alternative：更接近 Zhang 36-month training + 12-month validation，或 47-month training + 1-month validation；
- 先在 Dow30 / S&P100 上做，不建议一开始全量 S&P500 重跑；
- 目标是检验 IV 和 GNN depth 的结论是否依赖 validation split。

当前状态：如果还没有重跑，就写 `[Robustness table to be added.]`，不能伪造结果。

### 6.2 Date alignment and source consistency

这是我们特别需要的 robustness，因为当前 Dow30 / S&P100 与 S&P500 的 test calendar 不是完全同长；但 S&P500 不是独立或不兼容 calendar，而是 Dow30 / S&P100 234-date calendar 的 223-date subset。

要写：

- 当前 Dow30 aligned rerun input span 是 2021-06-09 到 2026-06-09；
- SP100 / SP500 raw span 同样围绕 2021-06-09 到 2026-06-09，但 S&P500 final saved out-of-sample dates 是 2025-07-14 到 2026-06-01；
- 这会影响 cross-universe comparison 的严格程度，但影响比完全不同 calendar 小，因为 S&P500 dates 嵌套在 Dow30/S&P100 dates 中；
- ~~当前已经补了一个 Dow30 wide-window multi-hop GHAR aligned rerun：输入 span 为 2021-06-09 到 2026-06-09，test window 为 2025-07-07 到 2026-06-09，共 234 个 dates，与 S&P100 的当前 out-of-sample dates 对齐；~~
- ~~但这个 supplement 只包含 $GHAR2H/GHAR3H$ 的 linear multi-hop diagnostics，不包含完整 HAR/GHAR/GNNHAR full-model set，因此它只能支持 Appendix C 的日期对齐 multi-hop 讨论，不能替代 Dow30 主结果；~~
- 已补齐 Dow30 标准窗口 full-model rerun：`20260619T071426Z_aligned_full_model`，输入 span 为 2021-06-09 到 2026-06-09，test window 为 2025-07-07 到 2026-06-09，共 234 个 dates，与 S&P100 当前 out-of-sample dates 对齐；该 run 有 28 个模型预测、`loss_table.csv`、`dm_tests.csv` 和 `fvu.csv`，可以替代旧 Dow30 主结果；
- Dow30 的 `20260618T075711Z_wide_multihop_ghar` 仍作为 Appendix C 的 multi-hop supplement 使用；
- Dow30 234-date MCS 已经补齐，MCS 小节应同步写成 Dow30、S&P100、S&P500 三组；
- 后续 robustness 应优先把 S&P500 统一到 Dow30/S&P100 的完整 234-date calendar，并重算 aligned Dow30 的 regime diagnostics；
- 在正式投稿前，这一步很重要。

如果暂时不重跑 S&P500，正文中可以写成：current S&P500 evidence is a nested near-calendar scale comparison. This is acceptable for the current draft's research-note stage, but a final submission should either rerun S&P500 on the full 234-date calendar or report the current design explicitly as a nested-calendar robustness result.

### 6.3 Alternative graph construction

Zhang future work 也提到 alternative graphs。我们可以作为 robustness：

- GLASSO graph；
- correlation threshold graph；
- sector graph；
- supply-chain 或 analyst co-coverage graph 如果未来有数据；
- 比较 IV extension 在不同 graph 下是否仍然有效。

### 6.4 Large-universe GNN stability diagnostic

这是当前 S&P500 结果中最需要补充说明的 robustness / diagnostic。正式输出显示，GNNHAR 和 GNNHAR-IV 在 S&P500 上的 loss ratio 远大于 1，并且不是单纯的“略差于 HAR”。从预测分布看，GNNHAR 的预测值存在明显下界抬高现象：例如 `GNNHAR1L_M` 的预测最小值约为 29.69，而 truth 的 25% 分位数约为 20.27，中位数约为 26.60；`GNNHAR2L_Q^{IV}` 的预测最小值约为 26.70。结果是，GNNHAR 在低波动股票和低波动日期上系统性高估 volatility，低波动分位区间的 MSE 被显著放大。

这个现象可能来自当前实现的一个数值结构：GNNHAR 在 standardized target scale 上使用 ReLU 输出，然后再反标准化。若 standardized output 被 ReLU 限制为非负，则反标准化后的预测不能低于训练集均值附近的水平。对于 S&P500 这种横截面差异很大的 universe，这会严重压缩低波动股票的预测空间，把许多低波动 observation 推到接近全样本平均 volatility 的水平。这个机制可以解释为什么 GNNHAR 的 prediction quantile 下界明显高于 truth 的低分位数。

正式报告不应把这一点写成已经完全证明的唯一原因，而应写成 implementation diagnostic：

- 当前 GNNHAR 在 S&P500 上明显存在 low-volatility over-prediction；
- 这与 ReLU output on standardized target scale 的下界效应一致；
- 这也说明大图下的 GNNHAR 结果需要额外 robustness，而不能直接解释为“节点更多但非线性 spillover 不存在”；
- 后续应重跑一个 corrected-output robustness：保留 hidden layers 的 ReLU，但 final prediction layer 不使用 ReLU；或在 log-volatility scale 上训练并用 exp 反变换；同时加入 per-ticker normalization / fixed effects；
- 若 corrected-output GNNHAR 仍然明显差于 HAR/GHAR-IV，才能更有力地说 S&P500 中 nonlinear graph aggregation 没有带来额外预测力。

### 6.5 IV interaction and nonlinear IV effects

这里放你提出的 $IV\times RV$ interaction idea。这个设计可以写成本文一个重要的 robustness / model-extension check，而不是当前主实验已经完成的结果。

直观上，当前模型只把 $q_{i,t-1}$ 作为 additive component 加入：

$$
z_{i,t-1}
=
\left(
v_{i,t-1},
\bar v_{i,t-5:t-2},
\bar v_{i,t-22:t-6},
q_{i,t-1}
\right).
$$

这相当于假设 IV 提供一个独立的 forward-looking signal，但历史 RV 的边际预测含义不随 IV 水平变化。你提出的交互项则允许这种边际含义发生变化：

$$
z^{\mathrm{int}}_{i,t-1}
=
\left(
v_{i,t-1},
\bar v_{i,t-5:t-2},
\bar v_{i,t-22:t-6},
q_{i,t-1},
q_{i,t-1}v_{i,t-1},
q_{i,t-1}\bar v_{i,t-5:t-2},
q_{i,t-1}\bar v_{i,t-22:t-6}
\right).
$$

经济含义是：当 options market 已经通过 IV 反映更高的 future uncertainty、tail-risk concern 或 event risk 时，同样的历史 RV 可能对应不同的未来波动风险。也就是说，IV 不仅可能自己有预测力，还可能改变历史 RV components 的 marginal predictive effect。

这个想法可以和 Lin (2013) 的 regression-adjustment 思路类比。Lin 在 completely randomized experiment 中建议在 OLS adjustment 中加入 treatment-covariate interactions，使 treatment effect estimation 允许 covariate-specific heterogeneity，并改善大样本效率。本文不是随机实验，IV 也不是随机 treatment，因此不能把 $IV\times RV$ 写成 causal interaction 或 treatment-effect heterogeneity。更准确的表述是：我们借鉴这种“允许核心变量与协变量交互”的建模思想，在 forecasting regression 中检验 option-implied state 是否调节 historical RV 的预测斜率。

对线性 HAR/GHAR，这相当于标准 regression interaction term；对 GNNHAR-IV，这可以作为 richer node feature 输入 $H^{(0)}$，再通过 graph propagation 学习 nonlinear and cross-sectional interaction effects。正式 robustness 可以比较：

- additive IV model vs $IV\times RV$ interaction model；
- HAR-IV vs HAR-IV-Interaction；
- GHAR-IV vs GHAR-IV-Interaction；
- GNNHAR-IV vs GNNHAR-IV-Interaction；
- 在 Dow30 / S&P100 / S&P500 中分别比较 MSE、QLIKE、MCS 和 DM。

如果 interaction model 显著改善 QLIKE，说明 IV 的作用不只是提供额外变量，而是改变历史 RV 信息的使用方式；如果没有改善，则当前 additive IV specification 已经足够。

### 6.6 Forecast horizon robustness

Zhang 报告 one-day、one-week、one-month。当前我们只保存 one-day target。后续可以：

$$
v_{i,t:t+h}
=
\frac{1}{h}\sum_{k=0}^{h-1}v_{i,t+k},
\qquad h\in\{5,22\}.
$$

然后重跑主模型，检验 IV 和 graph spillover 在更长 horizon 下是否仍然有效。

### 6.7 Exact hidden-state MAD

如果要严格做 Zhang Figure 7，需要保存：

- rolling origin；
- model name；
- layer depth；
- ticker；
- final hidden representation $H^{(K)}$；
- adjacency matrix $A$ 或 $W$。

当前只能先保留为 future robustness / diagnostic。

## 7. Conclusion

结论需要直接回答三个 RQ。

结论的语气要像研究笔记，而不是把所有问题都说成已经解决。最好的写法是：先回答每个 RQ 当前数据支持什么，再写当前证据不支持什么，最后写下一步需要什么检验。这样读者会觉得我们对结果是诚实的，也能看出这个项目还有继续推进的空间。

可以用一句总括先定调：本文目前最稳定的发现不是“GNNHAR 在大图上全面胜出”，而是“IV 作为 option-market forward-looking information，在多个 universe 中比单纯增加 GNN 深度更稳定；即便在较平滑的 30-day HV proxy target 上，大图 GNNHAR 也显示出 scale-sensitive instability；大图是否能增强 graph / GNN 相对 HAR 的优势，还需要更严格的实现和 robustness 检验”。

### 7.1 Answer to RQ1

RQ1：GHAR、GNNHAR 及其变体从 Dow30 扩展到 S&P100 再到 S&P500，是否均能相对 HAR 提高预测准确性？

当前答案应克制。Dow30 中 $GNNHAR2L_Q^{IV}$ 在 QLIKE 下表现最好，说明在小图环境里，Zhang-style graph/GNN framework 加上 IV 可以产生明显增益。但到了 S&P100 和 S&P500，最稳定的 winner 更偏向 IV-augmented HAR / GHAR，而不是深层 GNN。因此我们不能笼统说 GHAR / GNNHAR 在所有 universe 都稳定优于 HAR。更重要的是，这个结论是在 30-day HV proxy 这种相对平滑的目标上得到的；它提示我们，GNNHAR 的弱表现不能简单归因于 intraday RV 噪声过大，而更可能与大图、输出尺度、低波动股票预测和 QLIKE criterion 下的训练稳定性有关。

更准确的说法是：graph framework 在部分 universe 中有预测价值，但它不是自动胜出的机制。模型是否优于 HAR，取决于数据源、图构造、训练稳定性、输出约束、IV 信息以及 evaluation criterion。这个结论比“GNNHAR 一定更好”更弱，但也更符合当前数据。

### 7.2 Answer to RQ2

RQ2：IV 是否有贡献？

当前答案相对明确：IV 是目前最稳定的新增信息来源。三个 universe 中，IV 版本模型普遍进入前列，尤其是 S&P100 和 S&P500 中，IV-augmented HAR / GHAR 比深层 GNN 更稳定。这支持 Zhang future work 中 options information 的方向。这里的贡献不应写成“IV 有用”这一句常识性发现，而应写成：在同一套 Zhang-style HAR / GHAR / GNNHAR backbone 中，真正稳定带来 out-of-sample improvement 的是 option-implied forward-looking state，而不是单纯增加 graph neural depth。

S&P500 的结果可以作为 RQ2 的核心例子。不加入 IV 时，模型基本只能依赖 historical volatility lag 和 graph-aggregated historical lag；这种 backward-looking 信息在大 universe 中没有稳定产生优胜模型。加入 IV 后，最优行转向 `HAR_M_IV` / `GHAR_M_IV`，说明 IV 可能把 options market 中关于 future volatility、event risk、tail risk 和 market expectation 的复杂信息压缩成可进入日度 panel 的 forward-looking state。对 GNNHAR 来说，IV 也确实让部分 QLIKE-trained GNNHAR-IV 相对 non-IV GNNHAR 改善；但这些改善不足以使它们超过 IV-augmented HAR/GHAR。因此结论应写成：GNNHAR 最好和 IV 一起使用，因为 IV 提供更有信息含量的节点状态；但当前 S&P500 证据仍显示，大图 GNNHAR 需要 corrected output、normalization 和 robustness 检验后才能被解释为有效的 nonlinear forecasting gain。

但这里不能把 IV 的改善直接写成因果结论。更稳妥的解释是：IV 反映 options market 对未来 volatility、tail risk、news uncertainty 和 risk appetite 的定价，因此它可能包含历史 RV 没有完全吸收的 forward-looking information。同时，加入 IV 也增加了模型可用的状态变量和灵活度，所以后续还需要 $IV\times RV$ interaction、outlier robustness 和更严格的 comparison 来区分“信息增量”和“模型灵活度增量”。

### 7.3 Answer to RQ3

RQ3：节点关系更丰富是否带来“提升的提升”？

当前证据不支持这个假说。标准窗口 Dow30 的 best QLIKE gain 约 7.7%，S&P100 约 6.7%，S&P500 约 2.7%。如果“节点关系更丰富”会自动放大模型相对 HAR 的提升，那么这个 gain 应该随着 universe 扩大而增加；但当前结果正好相反。这里要注明 S&P500 是 Dow30/S&P100 234-date calendar 的 223-date subset，因此这是 nested near-calendar scale evidence，不是最终 fully common-calendar evidence；但它已经足以说明“更大图自动更好”这一强说法在当前证据下站不住。

所以，本文目前更合理的判断是：更多节点既可能带来更多 spillover information，也会带来更多噪声、训练不稳定、over-smoothing risk、graph estimation error，以及实现层面的 scale sensitivity。S&P500 的 GNNHAR 异常表现尤其提醒我们，大图 GNN 的结果不能只从模型名称解释，还要检查 normalization、final output design、低波动股票预测偏差和极端 stock-date observations。

### 7.4 Future work

Future work 要回到 Introduction 的逻辑：

- limit order book：可以直接观察买卖盘厚度、未成交订单、流动性缺口和潜在价格冲击；
- news：财报、CPI、FOMC、就业、利率等事件会导致未来 volatility jump；
- options / IV：本文已经先把 options market information 放进模型，但 IV 可能只是新闻和风险预期的压缩统计量；
- 后续可以做 causal inference / mediation analysis，研究 news 如何影响 IV，IV 又如何影响 future RV；
- 也可以把 news sentiment / event surprise 作为协变量，与 IV 一起进入 GNNHAR；
- 还可以加入 $IV\times RV$ interaction，检验 IV 是否改变历史 RV 的边际预测效应；
- 最后需要做 S&P500 完整 234-date calendar rerun、aligned Dow30 regime diagnostics、10-day HV/IV robustness、weekly/monthly horizon 和 exact hidden-state MAD，提升与 Zhang article 的可比性。Dow30 234-date MCS 已补齐，不应再写成待补 MCS；待补的是 regime 和部分 robustness。

## Appendix A. Data Audit Tables

放：

- ticker alignment audit；
- raw source path；
- date range；
- missingness；
- SP500 397 vs 449 解释。

## Appendix B. Full Model Loss Tables

分别放：

- Dow30 aligned full loss table；
- S&P100 full loss table；
- S&P500 full loss table。

所有表格都应使用 `[H]` 固定位置。
表格行顺序应按模型结构排序，而不是按 loss ratio 排序：正文主表先 non-IV 的 HAR / GHAR / GNNHAR1L--5L，再 IV 版本的同一组模型；multi-hop GHAR 不放正文主表，放 Appendix C。最优 MSE 和最优 QLIKE 只用颜色或符号标注。

## Appendix C. DM Test Tables

放：

- GHAR vs GHAR2H / GHAR3H；
- GNNHAR1L vs 2L / 3L / 4L / 5L；
- IV 版本同样一组。

其中 GHAR2H / GHAR3H 是 Zhang Appendix E-style 的补充诊断，不是正文主结果的一部分。写作重点：

- Zhang 的 GHAR2Hop 结果显示 cross-sectional DM statistic 约为 $-1$，$p$-value 约为 $35\%$，因此不能认为 two-hop linear neighbors 提供了显著额外预测力；
- Dow30 的 wide-window multi-hop GHAR supplement 现在可作为日期对齐诊断使用。该 supplement 使用 2025-07-07 到 2026-06-09 的 234 个 test dates，truth shape 为 $234\times30$。绝对 loss 显示 IV 版本优于 non-IV 版本，MSE-trained 版本优于 QLIKE-trained 版本：$GHAR2H_M^{IV}$ 的 MSE / QLIKE 约为 0.9165 / 0.01535，$GHAR3H_M^{IV}$ 约为 0.9162 / 0.01537；non-IV $GHAR2H_M$ 与 $GHAR3H_M$ 的 MSE / QLIKE 约为 0.9558 / 0.01574 和 0.9555 / 0.01577。这个 supplement 现在有同日期 full-model baseline 可参照，但它仍然是独立 artifact；Appendix C 如果报告 absolute losses 应明确分母口径，若报告 ratio vs aligned `HAR_M` 则要从 `20260619T071426Z_aligned_full_model` 中取同日期 baseline；
- 本文的 S&P100 结果中，non-IV multi-hop GHAR 的 MSE 基本不变，但 QLIKE 没有改善：$GHAR2H_M$ 的 MSE / QLIKE ratio 约为 0.998 / 0.998，$GHAR3H_M$ 约为 0.998 / 1.001。IV 版本在 MSE 下接近当前 S&P100 最优 MSE 模型，但 QLIKE 仍弱于最优 IV HAR：$GHAR2H_M^{IV}$ 的 MSE / QLIKE ratio 约为 0.965 / 0.977，$GHAR3H_M^{IV}$ 约为 0.965 / 0.978。新增 QLIKE-trained multi-hop rows 明显变差，$GHAR2H_Q,GHAR3H_Q,GHAR2H_Q^{IV},GHAR3H_Q^{IV}$ 的 QLIKE ratios 分别约为 1.317、1.309、1.233、1.227；
- 本文的 S&P500 结果中，MSE-trained multi-hop GHAR 大多变差；QLIKE-trained IV multi-hop GHAR 在 QLIKE 上有改善，但 MSE 明显变差，因此证据是 mixed，不应在正文中解释为稳健提升；
- 因此附录结论应写为：consistent with Zhang Appendix E, linear multi-hop GHAR does not provide robust additional predictive power.

## Appendix D. FVU and Smoothing Diagnostics

放：

- FVU by regime；
- incremental FVU；
- prediction-level MAD proxy；
- 明确说明 exact hidden-state MAD 当前未保存。
- regime split 当前使用 cross-sectional RV proxy；如果未来补入 SPY RV，再改成 Zhang-style SPY 90% quantile split。

## Appendix E. S&P500 QLIKE and GNNHAR Diagnostics

这个附录服务于 Section 5.1 和 Section 6.4。它的目标不是重新选择 winner model，而是解释两个看起来反常的事实：第一，为什么 S&P500 的 QLIKE-trained linear models 在 MSE 下出现极大损失；第二，为什么 S&P500 的 GNNHAR predictions 在低波动股票上表现出系统性高估。附录表格应让读者能够从原始价格事件、RV/IV 变量、预测值、loss contribution 一路追踪到主表中的异常 loss ratios。

放两个诊断表组：

1. QLIKE-trained MSE outlier diagnostic：
   - `HAR_Q`、`GHAR3H_Q^{IV}`、`HAR_M`、`GHAR_M^{IV}`；
   - test MSE、test QLIKE、maximum prediction、truth maximum；
   - top 1 / 5 / 10 / 50 / 100 squared-error contribution；
   - worst observations table，特别标出 `SATS` 在 2025-08-28 到 2025-09-25 附近的 extreme predictions。
   - `SATS` loss-contribution table，至少包括 $HAR_Q$、$GHAR3H_Q^{IV}$、$GNNHAR2L_Q^{IV}$ 的 total MSE、`SATS` MSE、`SATS` share of total SSE、total QLIKE、`SATS` QLIKE、`SATS` share of total QLIKE。
   - `SATS` event table，列出 2025-08-25、2025-08-26、2025-08-27、2025-09-08、2025-09-15、2025-09-23 的 close、return、30d RV、IV mean，并说明 2025-08-26 的约 70.25% close-to-close return 是真实价格跳跃，不是数据错误。

2. GNNHAR low-volatility over-prediction diagnostic：
   - truth prediction quantiles vs `GNNHAR1L_M`、`GNNHAR1L_M^{IV}`、`GNNHAR2L_Q^{IV}`；
   - low-volatility buckets 中的 bias 和 MSE；
   - 解释 GNNHAR final ReLU on standardized target scale 可能导致 prediction floor 被抬高。

这一附录的写作目标不是否定 GNNHAR，而是解释为什么 S&P500 当前结果需要实现层面的 robustness。正文结论应保持克制：current S&P500 GNNHAR evidence is unfavorable, but part of the deterioration may reflect scale-sensitive implementation choices that should be re-tested.

## Appendix F. Reproducibility Notes

记录：

- 结果根目录：`outputs/paper_ready_20260617`；
- Zhang-style 统计层：`outputs/paper_ready_20260617/zhang_style_statistics`；
- 报告目录：`reports/zhang_style_statistics_20260618`；
- 关键脚本：
  - `scripts/analysis/build_zhang_style_statistics.py`
  - `scripts/analysis/build_zhang_style_report.py`
- SP500 AutoDL source：`data/scale_experiment/sp500`；
- Drive upload SP500 397-panel 不作为正式 SP500 AutoDL 分析源。

## Appendix G. Reference Map for the Draft

下面这份不是最终 bibliography，而是根据 academic MCP 检索结果整理出来的 reference map。它的作用是告诉我们：正文每一类说法应该优先引用哪些文献，哪些文献已经有比较稳定的 DOI / arXiv id，哪些只适合作为补充。

### G.1 Zhang / GHAR / GNNHAR 主线

- Zhang, Pu, Cucuringu, and Dong, *Forecasting realized volatility with spillover effects: Perspectives from graph neural networks*, International Journal of Forecasting, 2024. DOI: `10.1016/j.ijforecast.2024.09.002`.
  - 用在 Introduction、Preliminaries、Methodology、Empirical setup、Discussion。
  - 这是本文最核心的基准文章。我们要明确说：本文复刻的是 Zhang-style modeling and evaluation framework，不是复现其数值结果。
- Zhang, Pu, Cucuringu, and Dong, *Graph-based Methods for Forecasting Realized Covariances*, SSRN / working paper, 2024. DOI: `10.2139/ssrn.4274989`.
  - 用在 GHAR 背景和 graph-based HAR motivation。
  - 这篇解释了通过 graph neighborhood aggregation 扩展 HAR 的经济含义。
- Zhang et al. 的早期 GNNHAR working-paper versions:
  - *Graph Neural Networks for Forecasting Realized Volatility with Nonlinear Spillover Effects*, DOI: `10.2139/ssrn.4375165`.
  - *Graph Neural Networks for Forecasting Multivariate Realized Volatility with Spillover Effects*, arXiv: `2308.01419`.
  - 这些可以放在 footnote 或 related-work 里，不一定都进主引用，避免和 IJF 终稿重复。

### G.2 HAR / RV / volatility forecasting 基础

- Corsi (2009), *A simple approximate long-memory model of realized volatility*, Journal of Financial Econometrics.
  - 用在 HAR baseline 的第一处。
- Andersen, Bollerslev, Diebold, and Ebens (2001), *The distribution of realized stock return volatility*, Journal of Financial Economics.
  - 用在 realized volatility / realized variance 的基础背景。
- Barndorff-Nielsen and Shephard (2002), *Econometric analysis of realized volatility and its use in estimating stochastic volatility models*, JRSS-B.
  - 用在 RV 作为 integrated variance proxy 的理论背景。
- Liu, Patton, and Sheppard (2015), *Does anything beat 5-minute RV? A comparison of realized measures across multiple asset classes*, Journal of Econometrics.
  - 用在解释 Zhang 为什么使用高频 RV / 为什么我们要披露自己不是高频 RV。
- Bollerslev, Patton, and Quaedvlieg (2016), *Exploiting the errors: A simple approach for improved volatility forecasting*, Journal of Econometrics. DOI: `10.1016/j.jeconom.2015.10.007`.
  - 用在 rolling forecasting setup、volatility forecast evaluation、error exploitation。
- Bollerslev, Hood, Huss, and Pedersen (2018), *Risk Everywhere: Modeling and Managing Volatility*, Review of Financial Studies. DOI: `10.1093/rfs/hhy041`.
  - 用在 cross-sectional commonality / risk structure / volatility spillover 的背景。

### G.3 Forecast evaluation: QLIKE, DM, MCS

- Patton (2011), *Volatility Forecast Comparison Using Imperfect Volatility Proxies*, Journal of Econometrics. DOI: `10.1016/j.jeconom.2010.03.034`.
  - 用在 QLIKE 和 imperfect volatility proxy 的讨论，尤其是 Section 5.1。
- Patton and Sheppard (2009), *Evaluating volatility and correlation forecasts*, Handbook of Financial Time Series.
  - 用在说明 QLIKE 在 volatility forecast comparison 中的地位。
- Patton and Sheppard (2015), *Good volatility, bad volatility: Signed jumps and the persistence of volatility*, Review of Economics and Statistics.
  - 可用于 jump / volatility component 的背景，但不要过度拉到本文主线。
- Diebold and Mariano (1995), *Comparing Predictive Accuracy*, Journal of Business \& Economic Statistics. DOI: `10.1080/07350015.1995.10524599`.
  - 用在 DM test 定义。
- Harvey, Leybourne, and Newbold (1997), *Testing the equality of prediction mean squared errors*, International Journal of Forecasting.
  - 用在 small-sample / serial-dependence correction 的 DM test 说明。
- Hansen, Lunde, and Nason (2011), *The Model Confidence Set*, Econometrica. DOI: `10.3982/ECTA5771`.
  - 用在 MCS 定义和主表 star 标注。
- Hansen and Dumitrescu (2022), *How should parameter estimation be tailored to the objective?*, Journal of Econometrics.
  - 用在 estimation criterion vs forecast loss 的讨论。

### G.4 Graph construction / GNN / over-smoothing

- Friedman, Hastie, and Tibshirani (2008), *Sparse inverse covariance estimation with the graphical lasso*, Biostatistics. DOI: `10.1093/biostatistics/kxm045`.
  - 用在 GLASSO graph construction。
- Kipf and Welling (2017 / arXiv 2016), *Semi-Supervised Classification with Graph Convolutional Networks*. arXiv: `1609.02907`.
  - 用在 GCN layer 基础。
- Scarselli et al. (2008), *The graph neural network model*, IEEE Transactions on Neural Networks.
  - 可作为 GNN general background。
- Feng et al. (2022), *How powerful are K-hop message passing graph neural networks*.
  - 用在 $K$-hop / depth / multi-hop discussion。
- Chen et al. (2020), *Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View*. DOI: `10.1609/aaai.v34i04.5747`, arXiv: `1909.03211`.
  - 用在 MAD / over-smoothing diagnostics。

### G.5 IV / options / additional predictors

- Li and Tang (2021), *Automated volatility forecasting*. Academic MCP matched DOI / SSRN: `10.2139/ssrn.3776915`.
  - Zhang conclusion 里提到 options、limit order books、news 的 future-work 方向时可引用。
  - 本文 Introduction 可以用它来支撑 “additional information sources” 这条线。
- Busch, Christensen, and Nielsen (2010), *The role of implied volatility in forecasting future realized volatility and jumps in foreign exchange, stock, and bond markets*, Journal of Econometrics. DOI: `10.1016/j.jeconom.2010.03.014`.
  - 这是 IV / options information 与 future realized volatility 的核心引用之一，适合放在 Section 1.3 和 GNNHAR-IV motivation。
- *Forecasting Power of Implied Volatility: Evidence from Individual Equities*, 2005, SSRN DOI: `10.2139/ssrn.762644`.
  - 适合支撑 individual equities 层面的 IV forecasting motivation。
- Rahimikia and Poon (2020/2021 line in Zhang-related material) on limit order books and news for daily RV forecasting.
  - Academic MCP 对精确条目没有稳定返回，暂时作为待核实引用。可以先在正文写成 “prior studies using limit order books and news” 并保留 citation placeholder。

### G.6 Interaction / IV $\times$ RV extension

- Lin (2013), *Agnostic notes on regression adjustments to experimental data: Reexamining Freedman's critique*. DOI: `10.1214/12-AOAS583`, arXiv: `1208.2301`.
  - 用在 $IV\times RV$ interaction 的建模类比。
  - 必须说明：本文不是 randomized experiment，IV 不是 treatment，因此这里只是借鉴 “covariate interaction / heterogeneous slope” 的建模思想，不作 causal interpretation。

### G.7 当前最该优先放进正文的引用顺序

如果只先放一批最核心 references，建议按下面顺序：

1. Corsi (2009): HAR.
2. Zhang et al. (2024): GNNHAR 主文。
3. Zhang et al. (2024, SSRN `10.2139/ssrn.4274989`): GHAR / graph HAR。
4. Li and Tang (2021): predictor set expansion / options, LOB, news。
5. Patton (2011) and Patton and Sheppard (2009): QLIKE / volatility forecast comparison。
6. Hansen, Lunde, and Nason (2011): MCS。
7. Diebold and Mariano (1995): DM。
8. Friedman et al. (2008): GLASSO。
9. Kipf and Welling (2017) plus Chen et al. (2020): GNN layer and over-smoothing。
10. Lin (2013): $IV\times RV$ interaction as robustness / extension idea。

## References

Andersen, T. G., Bollerslev, T., Diebold, F. X., and Ebens, H. (2001). The distribution of realized stock return volatility. *Journal of Financial Economics*.

Barndorff-Nielsen, O. E., and Shephard, N. (2002). Econometric analysis of realized volatility and its use in estimating stochastic volatility models. *Journal of the Royal Statistical Society: Series B*.

Bollerslev, T., Hood, B., Huss, J., and Pedersen, L. H. (2018). Risk everywhere: Modeling and managing volatility. *Review of Financial Studies*. DOI: `10.1093/rfs/hhy041`.

Busch, T., Christensen, B. J., and Nielsen, M. O. (2010). The role of implied volatility in forecasting future realized volatility and jumps in foreign exchange, stock, and bond markets. *Journal of Econometrics*. DOI: `10.1016/j.jeconom.2010.03.014`.

Chen, D., Lin, Y., Li, W., Li, P., Zhou, J., and Sun, X. (2020). Measuring and relieving the over-smoothing problem for graph neural networks from the topological view. *Proceedings of the AAAI Conference on Artificial Intelligence*. DOI: `10.1609/aaai.v34i04.5747`.

Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*. DOI: `10.1093/jjfinec/nbp001`.

Diebold, F. X., and Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*. DOI: `10.1080/07350015.1995.10524599`.

Friedman, J., Hastie, T., and Tibshirani, R. (2008). Sparse inverse covariance estimation with the graphical lasso. *Biostatistics*. DOI: `10.1093/biostatistics/kxm045`.

Hansen, P. R., Lunde, A., and Nason, J. M. (2011). The model confidence set. *Econometrica*. DOI: `10.3982/ECTA5771`.

Kipf, T. N., and Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations*. arXiv: `1609.02907`.

Li, S. Z., and Tang, Y. (2021). Forecasting realized volatility: An automatic system using many features and many machine learning algorithms. *SSRN Electronic Journal*. DOI: `10.2139/ssrn.3776915`.

Lin, W. (2013). Agnostic notes on regression adjustments to experimental data: Reexamining Freedman's critique. *The Annals of Applied Statistics*. DOI: `10.1214/12-AOAS583`.

Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*. DOI: `10.1016/j.jeconom.2010.03.034`.

Zhang, C., Pu, X., Cucuringu, M., and Dong, X. (2024a). Forecasting realized volatility with spillover effects: Perspectives from graph neural networks. *International Journal of Forecasting*. DOI: `10.1016/j.ijforecast.2024.09.002`.

Zhang, C., Pu, X., Cucuringu, M., and Dong, X. (2024b). Graph-based methods for forecasting realized covariances. *SSRN Electronic Journal*. DOI: `10.2139/ssrn.4274989`.
