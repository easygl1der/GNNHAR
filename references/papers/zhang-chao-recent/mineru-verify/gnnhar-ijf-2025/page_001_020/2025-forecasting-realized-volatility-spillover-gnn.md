# Forecasting realized volatility with spillover effects: Perspectives from graph neural networks✩

<!-- image-->

Chao Zhang a,b,∗,1, Xingyue Pu b,d,1, Mihai Cucuringu b,c,e, Xiaowen Dong b,d

a FinTech Thrust, HKUST(GZ), Guangzhou, China

b Oxford-Man Institute of Quantitative Finance, University of Oxford, Oxford, UK

c Department of Statistics, University of Oxford, Oxford, UK

d Department of Engineering Science, University of Oxford, Oxford, UK

e The Alan Turing Institute, London, UK

## a r t i c l e i n f o

Article history:   
Dataset link: LOBSTER, github.com/chaozha   
ng-ox/GNNHAR   
Keywords:   
Graph neural network   
Realized volatility   
Spillover effect   
Quasi-likelihood   
Nonlinearity

## a b s t r a c t

We present a novel nonparametric methodology for modeling and forecasting multivariate realized volatilities using customized graph neural networks to incorporate spillover effects across stocks. The proposed model offers the benefits of incorporating spillover effects from multi-hop neighbors, capturing nonlinear relationships, and flexible training with different loss functions. The empirical findings suggest that incorporating spillover effects from multi-hop neighbors alone does not yield a clear advantage in terms of predictive accuracy. Furthermore, modeling nonlinear spillover effects enhances the forecasting accuracy of realized volatilities, particularly for short-term horizons of up to one week. More importantly, our results consistently indicate that training with the quasi-likelihood loss leads to substantial improvements in model performance compared to the commonly used mean squared error, primarily due to its superior handling of heteroskedasticity. A comprehensive series of empirical evaluations in alternative settings confirm the robustness of our results.

© 2024 International Institute of Forecasters. Published by Elsevier B.V. All rights are reserved, including those for text and data mining, AI training, and similar technologies.

## 1. Introduction

Modeling and forecasting stock return volatility plays a crucial role in the theory and practice of finance. Extensive attention has been devoted to this subject within the literature, encompassing numerous ARCH, GARCH, and stochastic volatility models. Due to the availability of high-frequency data, realized volatility (RV), calculated from the sum of squared intraday returns, has gained popularity in recent years. For example, Corsi (2009) put forward the heterogeneous autoregressive (HAR) model for predicting daily RVs using various lagged RV components over different time horizons. While these methods provided valuable insights into the dynamic dependence of volatilities, they neglected the volatility spillover effect among assets, as highlighted by Bollerslev, Hood, et al. (2018).

The volatility spillover effect refers to the phenomenon that certain big shocks of a specific asset (or market) may have an influence on the volatilities of other assets (or markets). Essentially, the discovery of volatility spillover effects is expected to benefit the understanding and forecasting of volatilities. For example, Buncic and Gisler (2016) documented that the VIX of the U.S. market plays an important role in forecasting the volatilities of other global assets markets. Degiannakis and Filis (2017) examined the cross-asset spillover effects from stocks, currencies, and commodities to improve RV predictions of crude oil. Bollerslev, Hood, et al. (2018), Li and Tang (2021) utilized the commonality in risk structures to improve the forecasting of future volatility. Basturk et al. (2022), Zhang et al. (2024) applied neural networks to predict volatility, finding that cross-sectional data pooling enhances forecasting accuracy.

<!-- image-->  
Fig. 1. Illustration of multi-hop and nonlinear volatility spillover. Note: The target node represents the volatility of IBM. The connections are only for illustration, and hence not necessarily consistent with our experiments.

There are a number of studies dedicated to incorporating the spillover effect into volatility modeling, e.g. BEKK-GARCH (Engle & Kroner, 1995) and VAR-GARCH (Ling & McAleer, 2003). In terms of modeling RV, Wilms et al. (2021) employed vector autoregression (VAR) to obtain the multivariate volatility forecasts for stock market indices. However, in high-dimensional scenarios, the aforementioned models may deliver poor out-of-sample forecasts due to the curse of dimensionality, as emphasized by Callot et al. (2017). Hecq et al. (2023) studied volatility spillovers using Granger causality analysis with VARs based on penalized least squares estimations. Most recently, Zhang et al. (2022) introduced graph-based methods to capture volatility spillover effects, and proposed a parsimonious model to augment HAR via neighborhood aggregation on a graph that represents a financial network, denoted graph HAR (or GHAR). In these graphs, each asset is modeled as a node, and an edge connecting two nodes encodes the existence of the spillover effect between their volatilities.

One natural question following GHAR is whether there exists any spillover effect between nodes that are beyond one step, also known as multi-hop neighbors (see the detailed definitions in Section 2.1). For example, as illustrated in Fig. 1, for the target node (i.e. IBM), in addition to the spillover effect of one-hop neighbors (i.e. JPM and GS), we are also interested in whether there is any spillover effect from two-hop neighbors (i.e. AXP, CVX, and BA). To the best of our knowledge, the exploration of spillover effects from multi-hop neighbors has not been extensively addressed in the literature on volatility modeling.

In addition to multi-hop effects, another interesting question is whether the volatility spillover is nonlinear. Choudhry et al. (2016) documented the existence of significant nonlinear spillover effects among four major markets—the U.S., Canada, Japan, and the U.K.—via a nonlinear causality test proposed by Bai et al. (2010). Wang et al. (2018) attempted to capture the nonlinear relationship between the volatilities of stocks and crude oil by incorporating the asymmetric effect of oil prices and regime shifts. While the existence of nonlinear volatility spillover effects has been documented and examined in previous studies, in this paper, we employ a deep learning approach (a graph neural network) to unveil nonparametric evidence about the existence of a nonlinear mechanism between cross-sectional volatilities, without explicitly assuming the presence of asymmetric effects or regime shifts in the pairwise interactions.

From a machine learning perspective, the incorporation of multi-hop neighbors expands the set of features, and the potential presence of nonlinear spillover effects introduces new functional forms to describe volatility dynamics. It is also worth emphasizing that the choice of estimation criterion (EC) plays a crucial role, as it represents the objective function for estimating model parameters. Traditional econometric models, such as GARCH, commonly employ conditional quasi-likelihood (QL) based on normal distributions for parameter estimation. Conversely, models focused on forecasting realized volatilities, such as HAR, utilize the mean squared error (MSE) as their EC. Therefore, an important question arises as to whether a preferred EC exists (Cipollini et al., 2020), especially when combined with the aforementioned aspects, namely the effect of multi-hop neighbors and non-linear relationships.

In the present work, we explore these three questions using graph neural networks (GNNs). GNNs are a class of deep learning models designed for performing inferences on graphs and graph-structured data. They are capable of learning node and graph-level representations that are useful for a wide range of tasks involving graph analysis, such as node classification, node regression, and graph clustering. GNNs have demonstrated successful applications in various financial domains, including stock movement prediction (Chen et al., 2018; Sawhney et al., 2020), credit risk prediction (Liang et al., 2021; Wang et al., 2019), and payment fraud detection (Liu et al., 2019, 2018). A recent study by Chen and Robert (2022) utilized a graph transformer network for intraday volatility forecasting. However, it is worth mentioning that their approach had some limitations, particularly in terms of interpretability and benchmarking. In addition, they did not thoroughly investigate factors such as multi-hop neighbors, nonlinearity, or the impact of estimation criteria, which are the focus of our current study.

In particular, we design a GNN-based framework to model volatility spillover effects and enhance volatility predictions. By replacing the linear neighborhood aggregation in the GHAR of Zhang et al. (2022) with a nonlinear operation, the proposed model is able to automatically learn the nonlinear spillover effects. Furthermore, the multi-layer setting of GNNs allows us to explore this nonlinearity in the multi-hop setting, i.e. spillover to neighbors that are more than one hop away in the financial network. Another notable advantage of our model lies in its flexibility to accommodate various ECs during the training phase.3 It should be emphasized that the goal here is not only to extend the original HAR model with neighborhood information but also to provide new perspectives from GNNs for the nonparametric modeling of volatility spillover effects, further improving the volatility forecasts.

The main contributions of our work are summarized as follows. First, we examine the spillover effect from multihop neighbors in the financial graph, and observe that the multi-hop spillover effect is not necessary, as long as zero-hop and one-hop neighbors are included. Second, we establish that the proposed GNN model with nonlinear operations significantly improves the forecasting performance of GHAR, indicating the existence of nonlinear spillover effects on one-hop neighbors. Third, compared to MSE-trained models, models employing QL as the EC generally achieve substantial improvements in predictive accuracy. With a further endeavor, we establish a natural link between QL-trained models and the multiplicative error model (Engle, 2002), highlighting their superior handling of error heteroskedasticity by assigning different degrees of importance to observations. Overall, our proposed GNN model trained with QL exhibits an average forecast error in MSE (resp. QL) approximately 13% (resp. 4%) lower than that of the standard HAR model. Additionally, we examine the robustness of our proposed models across various market conditions, an alternative data-splitting scheme, and an alternative universe, consistently observing enhanced prediction accuracy across all experimental settings.

The remainder of this paper is organized as follows. Section 2 contains preliminaries on the mathematical definitions of graphs, a brief review of GNN models, and two baseline models (HAR and GHAR). In Section 3, we introduce the proposed model (GNNHAR), evaluation criterion, and forecast evaluation approaches. Section 4 outlines the experimental setup and provides the key outof-sample results across various forecast horizons and market regimes. Furthermore, in Section 5, we conduct an extensive analysis concerning the impact of QL, nonlinearity, and multi-hop neighbors. In Section 6, we perform several robustness tests. We conclude our work and highlight future research directions in Section 7.

## 2. Preliminaries

In this section, we summarize the preliminary concepts and models. In particular, we provide the mathematical definitions of graphs and multi-hop neighbors in Section 2.1. In Section 2.2, we briefly review two popular graph neural networks that inspired our work. Section 2.3 revisits the baseline model HAR for forecasting realized volatilities, while Section 2.4 reviews another baseline model GHAR. Throughout this paper, capital bold letters indicate matrices, lowercase bold letters indicate vectors, and plain letters indicate scalars.

## 2.1. Graph definitions

Definition 2.1 (Graphs). A graph $\mathcal { G }$ is defined as $\mathcal { G } =$ $\{ \nu , \varepsilon \}$ , where $\mathcal { V } = \{ v _ { 1 } , . . . , v _ { N } \}$ is a set of N nodes and E is a set of edges, where $e _ { i j } = \left( v _ { i } , v _ { j } \right) \in \mathcal { E }$ denotes an edge connecting node $v _ { i }$ and node $v _ { j } .$

Definition 2.2 (Adjacency Matrix). An adjacency matrix A is a square matrix whose dimension is $N \times N$ , where A[i, j] represents the connection between $v _ { i }$ and $v _ { j }$ in the graph ${ \mathcal { G } } .$ If $\pmb { \cal A } [ i , j ] \in \{ 0 , 1 \}$ , ∀i, j, the graph is a binary graph.4 The diagonal elements of A are all zero, since edges from a node to itself are typically not considered in graphs. In this article, we mainly consider binary graphs without self-connections.

Definition 2.3 (K -hop Neighbors). Following Feng et al. (2022), we use the K-hop neighbors of node v to represent all the neighbors that have distance from node v less than or equal to K , based on the shortest path distance (SPD) kernel. In contrast, k-hop neighbors represent the neighbors with exact distance k from node v. Finally, we denote $Q _ { v , \mathcal { G } } ^ { K }$ as the set of K -hop neighbors of node v in graph G.

Example 1 (A Graph with 5 Nodes). In Fig. 2(a), we plot an example graph with five nodes and five undirected edges, where the node $v _ { 1 }$ is colored as a target node. Nodes $v _ { 2 }$ and $v _ { 4 }$ are the one-hop and two-hop neighbors of $v _ { 1 } ,$ respectively. Fig. 2(b) shows its adjacency matrix.

## 2.2. A brief review of GNNs

Graph neural networks (GNNs) are a class of deep learning models designed for performing inferences on graphs. The main idea is to learn a vector representation for every node defined on a graph while preserving both the graph topology structure and node content information (Wu et al., 2020). The node representations, for example, can be further applied to node classification or regression. To this end, many GNN variants utilize the idea of neighborhood aggregation to develop the layerwise forward propagation rules. In essence, neighborhood aggregation effectively generates a node v’s representation by aggregating its own feature vector $\pmb { h } _ { v } \in \mathbb { R } ^ { D }$ and the feature vectors of its connected nodes $\pmb { h } _ { u } \in \mathbb { R } ^ { D }$ where $u \in Q _ { v , \mathcal { G } } ^ { 1 }$ . Common examples of aggregation functions include sum, mean, and maximum. Early attempts at GNNs—regarding which, see Dai et al. (2018) and Scarselli et al. (2008)—update node representations by aggregating neighborhood information recursively until a stable equilibrium is reached. More efficiently, a novel notion of a convolution operator can be defined on irregular graphs to process neighborhood aggregation in parallel (so-called graph convolution).5 A considerable number of GNN variants and architectures are built from different graph convolution operators. We provide a brief introduction to a specific GNN architecture that is relevant to our volatility forecasting models.

<!-- image-->  
Fig. 2. Illustration of a graph and its corresponding adjacency matrix.

The graph convolutional network (GCN) was introduced by Kipf and Welling (2017). It approximates the graph convolution with the following layer-wise propagation rule6:

$$
\pmb { H } ^ { ( l + 1 ) } = \sigma \left( \tilde { \pmb { 0 } } ^ { - \frac 1 2 } \tilde { \pmb { A } } \tilde { \pmb { 0 } } ^ { - \frac 1 2 } \pmb { H } ^ { ( l ) } \pmb { \Theta } ^ { ( l ) } \right) ,\tag{1}
$$

where $\tilde { \pmb { A } } = \pmb { A } + \pmb { I } _ { N }$ is the adjacency matrix of the graph G with added self-connections, and O˜ is a diagonal matrix with $\begin{array} { r } { \tilde { \bf { O } } _ { i i } \ = \ \sum _ { j } \tilde { \bf { A } } _ { i j } } \end{array}$ . A is the regular adjacency matrix of one-hop neighbors. $\tilde { \pmb { 0 } } ^ { - \frac 1 2 } \tilde { \pmb { A } } \tilde { \pmb { 0 } } ^ { - \frac 1 2 }$ is the normalized adjacency matrix, introduced to stabilize the training of the GNN models. $\boldsymbol { \Theta } ^ { ( l ) } \in \mathbb { R } ^ { D ^ { ( l ) } \times D ^ { ( l + 1 ) } }$ is the layer-specific trainable weight matrix. $\pmb { H } ^ { ( l ) } \in \mathbb { R } ^ { N \times D ^ { ( l ) } }$ is the matrix of node representations at the lth layer. $\pmb { H } ^ { ( 0 ) }$ is the input node features. σ (·) denotes a nonlinear activation function, such as ReL $\mathrm { U ( \cdot ) } = \operatorname* { m a x } ( 0 , \cdot )$

When addressing various research problems, the above GNN layers can be combined with other deep learning layers in an end-to-end learning framework. Additionally, the exploration of multi-hop effects can be achieved by straightforwardly stacking multiple GNN layers within a model. A model that incorporates K-layer GNN layers is commonly referred to as a K-layer GNN model.

Definition 2.4 (Receptive Field). In a GNN model, the receptive field of a target node is the set of nodes of the graph that determine its representations; see Alon and Yahav (2020) and Feng et al. (2022).

Proposition 2.1. After K layers of graph convolution in a GNN model, every node representation is determined by the information from the nodes within K hops; see Feng et al. (2022).

The above proposition states that the size of the receptive field of every node is associated with the number of layers in a GNN model. Alon and Yahav (2020) found that when K is unnecessarily large, any two nodes could easily have highly overlapping receptive fields, and consequently attain highly similar node representations, which leads to the problem of over-smoothing (see Chen et al., 2020; Li et al., 2018). Therefore, a large K does not always help, and on the contrary, it may lead to indistinguishable node representations and thus weaken the forecasting or classification accuracy.

## 2.3. Forecasting RV with HAR

Assume the price process $P _ { i , s }$ of a financial asset i follows

$$
\mathrm { d } \log P _ { i , s } = \mu _ { i } \mathrm { d } s + \sigma _ { i , s } \mathrm { d } W _ { s } ^ { i } ,\tag{2}
$$

where $\mu _ { i }$ is the drift, $\sigma _ { i , s }$ is the instantaneous volatility, and $W _ { s } ^ { i }$ is the standard Brownian motion. The integrated variance (IV) of asset i at day t is defined as $\begin{array} { r l } { I V _ { i , t } } & { { } = } \end{array}$ $\begin{array} { r l } { \int _ { t - 1 } ^ { t } \sigma _ { i , s } ^ { 2 } \mathrm { d } s . } \end{array}$

Andersen et al. (2001) and Barndorff-Nielsen and Shephard (2002) showed that the sum of squared intraday returns is a consistent estimator of the unobserved $I V _ { i , t } .$ The daily RV for a particular asset i at day t is defined as $\begin{array} { r } { R V _ { i , t } \ = \ \sum _ { l = 1 } ^ { M } r _ { i , t ( l ) } ^ { 2 } , } \end{array}$ where $r _ { i , t ( l ) }$ is the lth ∆-min log returns during $\mathrm { d } \mathsf { a y } t , \mathsf { i . e . } r _ { i , t ( l ) } = \log p _ { i , t ( l \Delta ) } - \log p _ { i , t ( ( l - 1 ) \Delta ) } ,$ and $p _ { i , t ( l \varDelta ) }$ is the price at time l∆ at day t. We refer to ${ \pmb v } _ { t } =$ $( R V _ { 1 , t } , \cdot \cdot \cdot , R V _ { N , t } ) ^ { \prime }$ as the vector of cross-sectional realized volatilities. Here, we consider five-minute windows in a trading day, following Liu et al. (2015).7

Corsi (2009) proposed a heterogeneous autoregressive regression (HAR) model for modeling and forecasting RV where the lagged daily, weekly, and monthly volatility components are incorporated as features. Bollerslev, Hood, et al. (2018) recommended using pooled panel data instead of time-series data to improve the accuracy of RV forecasts. We adopt this approach to make the most of cross-sectional information. As a result, we model the cross-sectional RV for day t as follows:

$$
\begin{array} { r l } { \mathfrak { L } : } & { \mathbb { E } ( \pmb { v } _ { t } | \mathcal { F } _ { t - 1 } ) = \pmb { \alpha } + \beta _ { d } \pmb { v } _ { t - 1 } + \beta _ { w } \pmb { v } _ { t - 5 : t - 2 } } \\ & { \qquad + \beta _ { m } \pmb { v } _ { t - 2 : t - 6 } , } \\ & { \qquad = \pmb { \alpha } + \pmb { V } _ { : t - 1 } \pmb { \beta } , } \end{array}\tag{3}
$$

where $\mathcal { F } _ { t - 1 }$ is the information set consisting of all relevant information up to and including $t \_ - \ 1 . \ { \pmb v } _ { t - 5 : t - 2 } \ =$ $\begin{array} { r } { \frac { 1 } { 4 } \sum _ { k = 2 } ^ { 5 } v _ { t - k } } \end{array}$ and $\begin{array} { r c l } { \overline { { { \pmb { v } } } } _ { t - 2 2 : t - 6 } } & { = } & { \frac { 1 } { 1 7 } \sum _ { k = 6 } ^ { \bar { 2 } 2 } \pmb { v } _ { t - k } } \end{array}$ denote the weekly and monthly lagged RV, respectively,8 and $\pmb { V } _ { : t - 1 } =$ $\left[ \pmb { v } _ { t - 1 } , \pmb { v } _ { t - 5 : t - 2 } , \pmb { v } _ { t } . \right.$ −22:t−6] $\in ~ \mathbb { R } ^ { N \times 3 }$ . The choice of a daily, weekly, and monthly lag aims to capture the long-memory dynamic dependencies observed in most RV series.

## 2.4. Graph HAR (GHAR)

Zhang et al. (2022) augmented the HAR model to capture the volatility spillover effect via linear neighborhood aggregation on graphs.9 GHAR is defined as

$$
\begin{array} { r l } { \mathbf { G H A R } ( A ) : } & { \mathbb { E } ( v _ { t } | \mathcal { F } _ { t - 1 } ) = \alpha + \beta _ { d } v _ { t - 1 } + \beta _ { w } v _ { t - 5 : t - 2 } } \\ & { \mathrm { ~ \ ~ \ ~ } + \ \beta _ { m } v _ { t - 2 : t - 6 } } \\ & { \mathrm { ~ \ ~ \ ~ } + \ \gamma _ { d } W \cdot v _ { t - 1 } + \gamma _ { w } W } \\ & { \mathrm { ~ \ ~ \ } \cdot v _ { t - 5 : t - 2 } + \gamma _ { m } W \cdot v _ { t - 2 : t - 6 } , } \\ & { = \alpha + V _ { : t - 1 } \beta + W V _ { : t - 1 } \gamma , } \end{array}\tag{4}
$$

where $\pmb { \alpha } \in \mathbb { R } ^ { N } , \pmb { \beta } , \pmb { \gamma } \in \mathbb { R } ^ { 3 }$ are parameters to be estimated. $\pmb { W } \ = \ \pmb { 0 } ^ { - \frac 1 2 } \pmb { A } \pmb { 0 } ^ { - \frac 1 2 }$ is the normalized adjacency matrix without self-connections, where $\textbf { 0 } = ~ \mathrm { d i a g } \{ n _ { 1 } , . . . , n _ { N } \}$ and $\begin{array} { r } { n _ { i } = \sum _ { j } { \pmb { A } } [ i , j ] } \end{array}$ ], ∀i.10

$W \cdot v _ { t - 1 }$ represents the neighborhood aggregation over daily horizons, and similarly for weekly and monthly horizons. $\gamma _ { d } , \gamma _ { w }$ , and $\gamma _ { m }$ represent the effects of connected neighbors over different horizons. If we employ an empty graph, i.e. the elements of A are all zeros, (4) reduces to (3). When the off-diagonal elements of A are all ones, i.e. a complete graph, $W \cdot v _ { t - 1 }$ represents the global volatility, as studied by Bollerslev, Hood, et al. (2018).

## 3. Proposed methodology

To investigate the presence of multi-hop and nonlinear effects in modeling volatility spillovers, we propose a new class of forecasting models based on the GNNs in Section 3.1. Furthermore, Section 3.2 highlights the significance of using various criteria to estimate model coefficients. In Section 3.3, we introduce the forecast evaluation methods and emphasize the differences between estimation criteria and forecast evaluations.

## 3.1. GNN-enhanced HAR (GNNHAR)

As introduced in (4), GHAR in Zhang et al. (2022) assumes a linear relationship between the volatilities of two connected assets. However, if the spillover effect is nonlinear, linear models are misspecified and are likely to generate less accurate forecasts. Additionally, GHAR considers only the zero-hop and one-hop neighbors, and this lack of consideration for multi-hop neighbors may lead to incomplete information and less accurate predictions. In light of the abilities of GNNs discussed in Section 2, we propose the following GNN architecture for modeling the volatility spillover effect, allowing for nonlinearity and multi-hop neighbors to improve the prediction accuracy.

$$
\mathbf G \mathbf N \mathbf N ( \pmb { H } ^ { ( l ) } , \pmb { A } ) : \quad \pmb { H } ^ { ( l + 1 ) } = \mathrm { R e L U } \left( \pmb { O } ^ { - \frac 1 2 } \pmb { A } \pmb { O } ^ { - \frac 1 2 } \pmb { H } ^ { ( l ) } \pmb { Q } ^ { ( l ) } \right)\tag{5}
$$

where $\pmb { W } = \pmb { 0 } ^ { - \frac 1 2 } \pmb { A } \pmb { 0 } ^ { - \frac 1 2 }$ is the normalized adjacency matrix, used to avoid numerical instabilities and exploding/vanishing gradients during the training phrase. Note that $\pmb { H } ^ { ( 0 ) } = \pmb { \breve { V } } _ { : t - 1 } \in \mathbb { R } ^ { N \times 3 }$ , which is the matrix composed of the past daily, weekly, and monthly volatilities. $\mathbf { \dot { H } } ^ { ( l ) } \in$ $\mathbb { R } ^ { N \times D ^ { ( l ) } }$ is a matrix of node representations at the lth layer of the GNN, where $D ^ { ( l ) }$ is the dimension of node representations. $\boldsymbol { \dot { \Theta } } ^ { ( l ) } \in \mathbb { R } ^ { D ^ { ( l ) } \times D ^ { ( l + 1 ) } }$ is a matrix of trainable parameters (see Fig. 3).

In contrast to the GCN architecture shown in (1), our proposed GNN propagation rule does not include selfconnections; i.e. the diagonal elements in A are zeros. We conjecture that the mechanism of an individual stock’s past volatility on its future volatility differs from the spillover effect. As a result, we apply the above GNN propagation in (5) solely to model the spillover effect, while the impact of a stock’s own past volatility is modeled using the same linear model as in HAR.11 This allows for a clear and straightforward explanation of the performance gain of our proposed model compared to the baseline models, HAR and GHAR.

<!-- image-->  
Fig. 3. Illustration of the GNNHAR model.

We introduce a GNN-enhanced HAR model, referred to as GNNHAR1L in (6), by replacing the linear neighborhood aggregation in GHAR (i.e. the term $W { \boldsymbol { \mathbf { \mathit { \sigma } } } } _ { : t - 1 } \gamma$ in (4)) with the proposed GNN layer in (5). It is worth noting that the main difference between GNNHAR1L and GHAR is that GNNHAR1L uses a graph convolutional layer with a nonlinear activation function, in the form of

$$
\begin{array} { r l } { \pmb { H } ^ { ( 1 ) } = \mathrm { G N N } ( \pmb { V } _ { : t - 1 } , \pmb { A } ) , } & { { } } \\ { \mathbf { G N N H A R 1 L } ( \pmb { A } ) : } & { { } \mathbb { E } ( \pmb { v } _ { t } | \mathcal { F } _ { t - 1 } ) = \pmb { \alpha } + \pmb { V } _ { : t - 1 } \pmb { \beta } + \pmb { H } ^ { ( 1 ) } \pmb { \gamma } . } \end{array}\tag{6}
$$

As introduced in Section 2, the nonlinear multi-hop effects can be explored by stacking multiple layers of the GNN. We denote the two-layer and three-layer models as GNNHAR2L and GNNHAR3L, respectively.12 Specifically,

$$
\begin{array} { r l } { { \pmb { H } } ^ { ( 2 ) } = { \sf G N N } ( { \pmb { H } } ^ { ( 1 ) } , { \pmb { A } } ) , } & { } \\ { { \mathbb { G N } } { \cal N H } { \sf A R 2 } { \sf L } ( { \pmb { A } } ) : } & { { \mathbb { E } } ( { \pmb { v } } _ { t } | \mathcal { F } _ { t - 1 } ) = \pmb { \alpha } + { \pmb { V } } _ { : t - 1 } \beta + { \pmb { H } } ^ { ( 2 ) } \pmb { \gamma } . } \\ { { \pmb { H } } ^ { ( 3 ) } = { \sf G N N } ( { \pmb { H } } ^ { ( 2 ) } , { \pmb { A } } ) , } & { } \\ { { \mathbb { G N } } { \cal N H } { \sf A R 3 } { \sf L } ( { \pmb { A } } ) : } & { { \mathbb { E } } ( { \pmb { v } } _ { t } | \mathcal { F } _ { t - 1 } ) = \pmb { \alpha } + { \pmb { V } } _ { : t - 1 } \beta + { \pmb { H } } ^ { ( 3 ) } \pmb { \gamma } . } \end{array}\tag{7}
$$

(8)

Our empirical analysis (deferred to Appendix A) indicates that each node in the volatility spillover graphs for the components of the DJIA 30 index, chosen by GLASSO (see 3.1.1), is connected to other nodes within a maximum of three steps (i.e. the graph has a diameter of length three, which is the size of the longest shortest pairwise path distance in the graph).

Consequently, by employing a three-layer GNN, we can guarantee that the volatility representation of each asset encompasses information from all other assets. Hence, there is no requirement to investigate beyond a threelayer GNN. Nevertheless, it is worth noting that for different universes or graphs, the number of GNN layers may need to be re-evaluated according to the distribution of SPDs.

## 3.1.1. Graph construction

Before training the GNN models or GHAR, it is essential to predefine the adjacency matrix or graph structure. In much of the GNN literature, the graph structure, such as a citation network, is explicitly defined. Unlike these applications, financial graphs require estimation, typically from time series analyses of price-based economic variables. For example, Diebold and Yılmaz (2014) studied the connectedness built from variance decompositions, while Karpman et al. (2023) leveraged random forests alongside high-frequency trading data to infer edge relationships. Zhang et al. (2022) constructed different types of graphs for volatility modeling and concluded that adjacency matrices obtained through graphical LASSO (GLASSO) effectively capture the relationships between individual volatilities, thereby enhancing forecasting accuracy.

GLASSO was proposed by Friedman et al. (2008) as a sparsity-penalized maximum likelihood estimator for the precision matrix Θ (i.e. the inverse of the covariance matrix). It assumes that the input N-dim vector is drawn from a multivariate Gaussian distribution N (0, Σ ), where $\begin{array} { r } { \Sigma ~ = ~ \Theta ^ { - 1 } . 1 3 } \end{array}$ A principal advantage of GLASSO is its capacity to reveal the conditional independence between variables, here the assets, through the estimated precision matrix. If the ij-th entry of the precision matrix is zero, the ith asset and jth asset are conditionally independent. Therefore, the adjacency matrix A obtained by applying GLASSO to multivariate daily returns is defined as $A [ i , j ] =$ 1 if $\Theta [ i , j ] \neq 0 ;$ otherwise $\pmb { A } [ i , j ] = 0$ . Based on these compelling results of GLASSO, we adopt it to construct the adjacency matrix for graph-based models throughout this paper.14

## 3.2. Estimation criterion

The standard HAR model described in (3) is often estimated via ordinary least squares (OLS). In other words, the estimation criterion (EC) for its in-sample training is the MSE. When the errors in (3) are independent, homoscedastic, and normally (Gaussian) distributed, the OLS estimator is consistent under the asymptotic sense. Nonetheless, given the stylized facts of RV (such as heteroskedasticity and so on), the OLS estimator may not be an ideal choice and a better estimator may be available. For example, Hansen and Dumitrescu (2022) proved that the likelihood-based estimator is asymptotically efficient, although the likelihood-based estimator can also be vastly inferior if the underlying statistical model is misspecified. Clements and Preve (2021) empirically compared various estimation criteria on HAR and found that simple weighted least squares can yield substantial improvements to the predictive ability of the standard HAR.

Meanwhile, QL has served as a commonly employed metric for estimating traditional econometric models, including GARCH. Fan et al. (2014) and Hall and Yao (2003) demonstrated that the conditional Gaussian QL estimator is always consistent, even when the error term deviates from a normal distribution.

Utilizing the flexibility of neural networks and stochastic gradient descent algorithms, we are able to investigate whether different estimation criteria would result in disparate model predictions. Specifically, our primary focus revolves around the following estimation criteria: MSE and QL, defined as follows:

• MSE:

$$
\frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { \# \mathcal { T } _ { t r a i n } } \sum _ { t \in \mathcal { T } _ { t r a i n } } \left( R V _ { i , t } - \widehat { R V } _ { i , t } ^ { ( F ) } \right) ^ { 2 } ,\tag{9}
$$

• QL:

$$
\frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { \# \mathcal { T } _ { t r a i n } } \sum _ { t \in \mathcal { T } _ { t r a i n } } \left[ \frac { R V _ { i , t } } { \widehat { R V } _ { i , t } ^ { ( F ) } } - \log \left( \frac { R V _ { i , t } } { \widehat { R V } _ { i , t } ^ { ( F ) } } \right) - 1 \right] ,\tag{10}
$$

where $\widehat { R V } _ { i , t } ^ { ( F ) }$ represents the predicted value of $R V _ { i , t }$ by a specific model F , N is the number of stocks in our universe, $\mathcal { T } _ { t r a i n }$ is the training period, and $\# \mathcal { T } _ { t r a i n }$ is the length of the training period.

Lower values are preferred for both measures. For clarity, we use $F _ { M } \ ( F _ { Q } )$ to denote model F trained with MSE (QL). To the best of our knowledge, adopting QL as the estimation criterion to optimize volatility models, especially those grounded on neural networks, has not yet drawn considerable attention in the literature. An exception can be found in the work of Cipollini et al. (2020), who conducted an empirical assessment of the impact of various error criteria on linear HAR models. They observed that using QL led to slightly improved forecasts, though without offering further theoretical explanations.

In Appendix B, we show that the models trained with QL are linked to the multiplicative error model (MEM)

by Engle (2002). Hence, the comparison between models trained with MSE and QL essentially boils down to the comparison between additive models and multiplicative models (see below). According to Cipollini et al. (2021), those additive models have issues related to the heteroskedasticity of errors (ut ). However, when considering multiplicative models, the errors (z ) tend to be homoskedastic.

$$
R V _ { t } = \left\{ \begin{array} { l l } { \mathbb { E } \left( R V _ { t } | \mathcal { F } _ { t - 1 } \right) + u _ { t } , } & { u _ { t } \mathrm { ~ z e r o ~ m e a n } } \\ { \mathbb { E } \left( R V _ { t } | \mathcal { F } _ { t - 1 } \right) \times z _ { t } , } & { z _ { t } \mathrm { ~ u n i t ~ m e a n } . } \end{array} \right.
$$

From an empirical perspective, Clements and Preve (2021), Patton and Sheppard (2015) and Reisenhofer et al. (2022) estimated their models using different schemes of weighted least squares (WLS) to assign less importance during estimation to periods where volatility is less precisely estimated.

Next, we examine the weighting scheme implicitly employed in QL-trained HAR models. Fig. 4 first displays the aforementioned EC for different forecasts $\widehat { R V }$ when $R V ~ = ~ 1$ . Notably, the QL function exhibits asymmetry and imposes a higher penalty on under-predictions. This feature becomes particularly significant during turbulent periods, as the volatility forecasts tend to be smaller than the actual shocks. By placing emphasis on those underpredictions, models trained with QL have the potential to achieve improved prediction accuracy during such turbulent periods.

Proposition 3.1. The optimization of HAR models trained with QL can be achieved through iteratively reweighted least squares (IRLS), employing weights $w _ { k - 1 , t } \ = \ 1 / \widehat { R V } _ { k - 1 , t } ^ { 2 }$ at iteration k. Here, $\widehat { R V } _ { k - 1 , t }$ represents the fitted value from the preceding iteration (with the initial iteration performed using OLS).

This proposition further validates the observation that models trained with QL give greater emphasis to underpredictions. Its proof is provided in Appendix C. In line with Cipollini et al. (2021) and Clements and Preve (2021), we are not asserting the optimality of the weighting scheme in QL-trained models. Additionally, our analysis, limited to comparing two statistical loss functions for realized volatility, may not comprehensively justify the superiority of QL across various applications in finance.15 Nonetheless, they could serve as valuable benchmarks, due to their natural relations with the MEM and WLS, and their desirable theoretical properties.16

## 3.3. Forecast evaluation approaches

Regarding the performance of forecasts in out-ofsample tests, we continue to employ MSE and QL as our evaluation methods. However, it is important to distinguish between the concept of forecast loss (FL) and the estimation criterion (EC), as they serve distinct purposes. FL assesses the performance of RV forecasts during outof-sample testing, while the EC is utilized for model estimation within the in-sample period (Cipollini et al., 2020).

<!-- image-->  
Fig. 4. Comparison of the MSE and QL loss functions.

In order to determine the significance of the performance improvement compared to the baseline models, we employ two commonly used statistical tests found in the literature. As suggested by Patton and Sheppard (2009), QL demonstrates greater statistical power than MSE in the Diebold–Mariano (DM) test. Consequently, our focus in the analysis of the out-of-sample results is primarily on QL.

• The model confidence set (MCS) was proposed by Hansen et al. (2011) to identify a subset of models with significantly superior performance from model candidates at a given level of confidence. The MCS procedure renders it possible to make statements about the statistical significance from multiple pairwise comparisons. For additional details, we refer to the studies of Hansen et al. (2003, 2011).

• The Diebold–Mariano (DM) test was proposed by Diebold and Mariano (1995) to examine whether there are significant differences between two timeseries forecasts. The DM test was further modified by Harvey et al. (1997) to account for serial dependence in forecasts. In addition to comparing errors for each individual stock, we also follow Gu et al. (2020) to compare the cross-sectional average of prediction errors from two models. Further details on the DM test are available in Diebold and Mariano (1995).

## 4. Empirical analysis

In this section, we first introduce the data and provide details regarding the implementation. Subsequently, we present the main findings and conduct a stratified analysis to evaluate the performance across different market regimes.

## 4.1. Setup

The intraday data of Dow Jones Industrial Average (DJIA) components are obtained from the LOBSTER database.17 The time period under consideration is from July 1, 2007 to Jun 30, 2021.18 Following Bollerslev et al. (2016), we include only those stocks among the DJIA components that traded continuously throughout the entire period. As a result, 27 stocks are included in the final sample. Their ticker symbols are listed in Appendix A, where we also present summary statistics for the volatility estimates. Additionally, for robustness checks, we consider a larger universe of S&P 100 components. Further details regarding this analysis can be found in Section 6.2.

Our out-of-sample forecast comparisons are based on the RV forecasts for the set of models introduced in Sections 2 and 3. All models are recalibrated every month based on a rolling sample window of the past 1000 days, following Bollerslev et al. (2016), Bollerslev, Patton, and Quaedvlieg (2018), Symitsi et al. (2018) and Pascalau and Poirier (2021). Specifically, we use 36-month data for model training, and the recent 12-month data as the validation set to tune the hyperparameters and prevent overfitting.19 Finally, testing data are the samples in the following month; they are out-of-sample in order to provide objective assessments of the model performance. To this end, in aggregate, we obtain a 10-year out-of-sample period, that is, from July 1, 2011 to June 30, 2021.

The parameters in $\mathrm { H A R } _ { M }$ and ${ \mathrm { G H A R } } _ { M }$ are estimated by OLS using both the training and validation data, as there is no requirement for hyperparameter tuning. To estimate the parameters in the proposed GNNHARs, we adopt the Adam optimizer (Kingma & Ba, $2 0 1 4 ) . ^ { 2 0 }$ When QL is chosen as the EC, there are no available estimators in closed form. Therefore, we also employ Adam to optimize $\mathrm { H A R } _ { Q }$ and ${ \mathrm { G H A R } } _ { Q }$ using both the training and validation data. Given the stochastic nature of the $\mathsf { \bar { o p t i m i z e r } } , 2 1$ we employ an ensemble approach to enhance the robustness of GNNHAR models and QL-trained linear models (see Gu et al., 2020; Zhang et al., 2024). We train multiple models with random initialization and obtain final predictions by averaging the outputs of all networks. For further details on the hyperparameter choices in GNNHAR, refer to Appendix D.

One-day forecasting is not the only time horizon of interest to practitioners. Following the convention established in the literature (Symitsi et al., 2018; Zhang et al., 2022), we also examine whether the proposed methods can be applied to various forecasting horizons, e.g. one week or one month. The weekly and monthly target volatilities are defined as $\begin{array} { r } { \pmb { v } _ { t : t + h } = \sum _ { k = 0 } ^ { h } \pmb { v } _ { t + k } } \end{array}$ , where h = 4 and h = 21, respectively.

## 4.2. Main results

We begin our empirical analysis by comparing the outof-sample performance of the competing models under consideration. Table 1 presents the ratio of forecast losses for each model relative to the $\mathrm { H A R } _ { M }$ model (i.e. HAR estimated by OLS).

Table 1 first highlights the consistent improvement of the GHAR model over the standard HAR model in terms of forecast loss (FL), implying the importance of graph information. Furthermore, the first two columns of Table 1, which represent the results for the one-day horizon, demonstrate that our proposed GNNHAR model with a single hidden layer $\left( \mathrm { G N M H A R } 1 \mathrm { L } _ { M } \right)$ further improves the performance of the linear model ${ \mathrm { G H A R } } _ { M }$ . This finding underscores the significance of incorporating nonlinearity when modeling the spillover effect. However, it is worth noting that the performance starts to decline when additional GNN layers are added, particularly with three layers.

When considering models trained with QL, the results for the one-day horizon reveal that $\mathrm { H A R } _ { Q }$ achieves better forecasts than its counterpart $\mathrm { H A R } _ { M }$ $\mathsf { G N N H A R 1 L } _ { Q }$ further improves the predictive accuracy of $\mathrm { G N N H A R } 1 \mathrm { L } _ { M }$ and yields the best (resp. second-best) out-of-sample performance in terms of MSE (resp. QL). Specifically, at the daily forecast horizon, $\mathrm { G N N H A R 1 L } _ { Q }$ has about 13% (resp. 4%) lower average forecast error in MSE (resp. QL) compared to the standard $\mathrm { H A R } _ { M }$ model. In addition, the MCS test indicates that both $\mathrm { G N N H A R } 1 \mathrm { L } _ { Q }$ and $\mathrm { G N N H A R } 2 \mathrm { L } _ { Q }$ are included in the subset of best models, based on the QL forecast loss. Interestingly, $\mathsf { G N N H A R 3 L } _ { Q }$ delivers worse out-of-sample performance than GNNs with one or two layers, yet still outperforms its counterpart trained with

Table 1  
Out-of-sample forecast losses.
<table><tr><td rowspan="2"></td><td colspan="2">One day</td><td colspan="2">One week</td><td colspan="2">One month</td></tr><tr><td>MSE</td><td>QL</td><td>MSE</td><td>QL</td><td>MSE</td><td>QL</td></tr><tr><td>HARM</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td> ${ \mathrm { G H A R } } _ { M }$ </td><td>0.927</td><td>0.983</td><td>0.904</td><td>0.987</td><td>0.975*</td><td>1.036</td></tr><tr><td> $\mathtt { G N N H A R 1 L } _ { M }$ </td><td>0.907</td><td>0.979</td><td>0.940</td><td>0.943</td><td>1.021</td><td>0.968</td></tr><tr><td> $\mathtt { G N N H A R 2 L } _ { M }$ </td><td>0.967</td><td>0.977</td><td>1.034</td><td>0.953</td><td>1,134</td><td>1.032</td></tr><tr><td> $\mathsf { G N N H A R 3 L } _ { M }$ </td><td>1.210</td><td>0.982</td><td>1.014</td><td>0.961</td><td>1.046</td><td>0.958</td></tr><tr><td>HARO</td><td>0.927</td><td>0.981</td><td>0.939</td><td>0.945</td><td>1.069</td><td>0.986</td></tr><tr><td> $\mathrm { G H A R } _ { Q }$ </td><td>0.886</td><td>0.983</td><td>0.842*</td><td>0.936</td><td>1.151</td><td>0.954*</td></tr><tr><td> $\mathrm { G N M H A R 1 L } _ { Q }$ </td><td>0.867*</td><td>0.961*</td><td>0.855</td><td>0.913*</td><td>1.179</td><td>0.965</td></tr><tr><td> $G \mathsf { N N H A R 2 L } _ { Q }$ </td><td>0.879</td><td>0.959*</td><td>0.873</td><td>0.920</td><td>1.736</td><td>0.947*</td></tr><tr><td> $\mathsf { G N M H A R 3 L } _ { Q }$ </td><td>0.894</td><td>0.963</td><td>1.185</td><td>0.942</td><td>1.502</td><td>0.971</td></tr></table>

Note: The table reports the ratios of forecast losses of various models compared to the standard HARM model over the one-day, one-week, and one-month horizons. For each horizon, the model with the best out-of-sample performance in MSE (QL) is highlighted in red (blue). \* Asterisk indicates models that yield as accurate forecasts as the best model at the 5% significance level based on the MCS test.

MSE. These findings suggest that QL might serve as a more effective in-sample estimation criterion than MSE. In the subsequent sections, we provide further analysis to delve into these results.

The results for weekly and monthly horizons presented in Table 1 demonstrate that models incorporating graph information (including GHAR and various GNNHAR models) exhibit significantly superior forecast accuracy compared to the HAR model over longer horizons, up to one week. Specifically, when examining the QL loss for the one-week forecast horizon, we observe that $\mathrm { G N M H A R 1 L } _ { Q }$ achieves the best out-of-sample performance. However, as the prediction horizon extends, the ratios approach or even exceed one, particularly for MSE. This suggests that longer-term forecasting becomes less sensitive to graph information. Additionally, we notice that the discrepancy between the ratios based on MSE and QL becomes more pronounced over longer horizons. One possible explanation is that the QL loss is generally less affected by extreme observations in the testing samples (see Patton, 2011). This is particularly relevant considering that such extreme observations may occur more frequently over longer horizons.

## 4.3. Market regimes

To assess the stability of performance across different market regimes, we perform a stratified out-of-sample analysis on two sub-samples: relatively calm periods when the RV of the S&P 500 ETF index is below the 90% quantile of its entire sample distribution, and the turbulent periods when the RV is above its 90% quantile (see Pascalau & Poirier, 2021; Zhang et al., 2022).

The results presented in Table 2 demonstrate that the enhancements achieved through the introduction of nonlinearity and the selection of QL as the EC are generally consistent across different market regimes. Specifically, when considering calm days and the daily forecast horizon, the models $\mathrm { G N N H A R } 1 \mathrm { L } _ { M }$ and $\mathrm { G N N H A R } 2 \mathrm { L } _ { M }$ appear to be the most effective based on the MSE loss. On the other hand, when evaluating accuracy in terms of QL, the models GNNHAR1LQ and GNNHAR2LQ provide the most precise forecasts. This outcome is expected since the volatility process tends to be more stable during calm periods. Consequently, if the forecast user has a specific preference for a particular loss function, it would be advisable to optimize the model parameters accordingly. In other words, for stationary time series, the alignment of the training loss (i.e. EC) and the testing loss (i.e. FL) may produce improved forecasts.

Nevertheless, when examining turbulent days and the daily forecast horizon, models trained with QL exhibit greater percentage improvements compared to those trained with MSE across both losses. For instance, the average forecast MSE (QL) loss of $\mathrm { G N M H A R 1 L } _ { Q }$ is approximately 13% (2%) lower than $\mathsf { G N N H A R 1 L } _ { M } .$ . This suggests that models trained with QL may possess unique characteristics distinct from their MSE-trained counterparts during turbulent periods. This intriguing discovery is explored and analyzed in the subsequent section.

In addition, when considering longer forecast horizons and periods of calmness, $\mathrm { G N N H A R } 1 \mathrm { L } _ { M }$ produces significantly more accurate out-of-sample forecasts relative to other models in terms of MSE. Regarding the QL accuracy, $\mathrm { G N M H A R 1 L } _ { Q }$ outperforms other models for the weekly horizon, while $\mathtt { G N N H A R 2 L } _ { M }$ emerges as the topperforming model for the monthly horizon. When transitioning to the volatile periods, we continue to observe the superiority of QL-trained models (especially $\operatorname { G H A R } _ { Q } )$ over MSE-trained models, with the exception being the monthly forecast horizon and considering MSE as the FL.

## 5. Discussion

The objective of this section is to examine the reasons behind the superior performance of our proposed GNNHAR models trained with QL. Our analysis begins by investigating the impact of the choice of EC on the predictive accuracy of the models. We then delve into exploring the influence of model nonlinearity, followed by an examination of the predictive information obtained from multi-hop neighbors.

## 5.1. Impact of evaluation criterion

As mentioned above, QL deals with over- and underpredictions differently, which may account for the overall better performance of QL-trained models compared to MSE-trained models. In light of this observation, we examine the forecast errors $( \widehat { R V } _ { i , t } ^ { ( F ) } - R V _ { i , t } )$ and forecast ratios $( \widehat { R V } _ { i , t } ^ { ( F ) } / R V _ { i , t } )$ over the entire testing period and various sub-periods.22

Fig. 5 presents boxplots for forecast errors and ratios of various models. From subplots (a) and (b), we observe that in general, all models tend to exhibit a bias towards over-predictions (i.e. positive errors or ratios greater than one) rather than under-predictions, aligning with the findings of Clements and Preve (2021). Subplots (c) and (d) further unveil that this over-prediction tendency is primarily observed during calm periods. Conversely, subplots (e) and (f) indicate that these models are more inclined to under-predict volatilities during turbulent periods. This observation is not surprising, as the models do not explicitly incorporate any exogenous variables to aid in detecting changes in market conditions.

Table 2  
Stratified out-of-sample forecast losses.
<table><tr><td rowspan="2"></td><td colspan="2">One day</td><td colspan="2">One week</td><td colspan="2">One month</td></tr><tr><td>MSE</td><td>QL</td><td>MSE</td><td>QL</td><td>MSE</td><td>QL</td></tr><tr><td></td><td colspan="6">Panel A: Bottom 90%</td></tr><tr><td> $\mathrm { H A R } _ { M }$ </td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td> ${ \mathrm { G H A R } } _ { M }$ </td><td>0.961</td><td>0.998</td><td>0.949</td><td>1.001</td><td>0.967</td><td>1.027</td></tr><tr><td> $\mathtt { G N N H A R 1 L } _ { M }$ </td><td>0.943*</td><td>0.998</td><td>0.883*</td><td>0.960*</td><td>0.923*</td><td>0.924*</td></tr><tr><td> $\mathtt { G N N H A R 2 L } _ { M }$ </td><td>0.944*</td><td>0.990</td><td>0.901</td><td>0.954*</td><td>0.946*</td><td>0.921*</td></tr><tr><td> $\mathsf { G N N H A R 3 L } _ { M }$ </td><td>0.957</td><td>0.987</td><td>0.911</td><td>0.965</td><td>0.937*</td><td>0.930*</td></tr><tr><td> $\mathrm { H A R } _ { Q }$ </td><td>1.010</td><td>0.984</td><td>1.005</td><td>0.955*</td><td>1.159</td><td>0.942*</td></tr><tr><td> $\mathrm { G H A R } _ { Q }$ </td><td>0.989</td><td>1.007</td><td>1.076</td><td>1.001</td><td>1.257</td><td>1.084</td></tr><tr><td> $\mathrm { G N M H A R 1 L } _ { Q }$ </td><td>0.967</td><td>0.978*</td><td>0.944</td><td>0.943*</td><td>1.478</td><td>0.977</td></tr><tr><td> $G \mathsf { N N H A R 2 L } _ { Q }$ </td><td>0.976</td><td>0.979*</td><td>0.985</td><td>0.947*</td><td>1.433</td><td>0.973</td></tr><tr><td> $\mathsf { G N M H A R 3 L } _ { Q }$ </td><td>0.970</td><td>0.980*</td><td>1.062</td><td>0.957</td><td>1.662</td><td>0.969</td></tr><tr><td></td><td colspan="6">Panel B: Top 10%</td></tr><tr><td> $\mathrm { H A R } _ { M }$ </td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td> ${ \mathrm { G H A R } } _ { M }$ </td><td>0.916</td><td>0.910</td><td>0.897</td><td>0.959</td><td>0.976*</td><td>1.043</td></tr><tr><td> $\mathtt { G N N H A R 1 L } _ { M }$ </td><td>0.895</td><td>0.903</td><td>0.949</td><td>0.908</td><td>1.033</td><td>1.007</td></tr><tr><td> $\mathtt { G N N H A R 2 L } _ { M }$ </td><td>1.102</td><td>0.915</td><td>1.056</td><td>0.951</td><td>1,157</td><td>1,131</td></tr><tr><td> $\mathsf { G N N H A R 3 L } _ { M }$ </td><td>1.293</td><td>0.958</td><td>1.030</td><td>0.952</td><td>1.059</td><td>0.982</td></tr><tr><td> $\mathrm { H A R } _ { Q }$ </td><td>0.900</td><td>0.965</td><td>0.928</td><td>0.925</td><td>1.059</td><td>1.024</td></tr><tr><td> $\mathsf { G H A } \bar { \mathsf { R } } _ { 0 }$ </td><td>0.852</td><td>0.867*</td><td>0.804*</td><td>0.799*</td><td>1.149</td><td>0.841*</td></tr><tr><td> $\mathrm { G N M H A R 1 L } _ { Q }$ </td><td>0.834*</td><td>0.879</td><td>0.841</td><td>0.848</td><td>1.143</td><td>0.955</td></tr><tr><td> $\mathrm { G N M H A R } 2 \mathrm { L } _ { Q }$ </td><td>0.848</td><td>0.862*</td><td>0.924</td><td>0.861</td><td>1.773</td><td>0.886</td></tr><tr><td> $\mathsf { G N M H A R 3 L } _ { Q }$ </td><td>0.868</td><td>0.882</td><td>1.205</td><td>0.909</td><td>1.483</td><td>0.973</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Note: The table reports stratified losses during trading days with the bottom 90% (Panel A) and the top 10% (Panel B) RV of the S&P 500 ETF index over the one-day, one-week, and one-month horizons. For each horizon, the model with the best out-of-sample performance in MSE (QL) is highlighted in red (blue).  
\* Asterisk indicates models that yield as accurate forecasts as the best model at the 5% significance level based on the MCS test.

Furthermore, subplots (a) and (b) demonstrate that the bulk of the forecast errors (resp. ratios) of QL-trained models are generally closer to zero (resp. one) compared to MSE-trained models. Specifically, subplots (c) and (d) reveal that QL-trained models exhibit a reduced tendency to over-predict during calm periods, while subplots (e) and (f) suggest that they are less prone to excessive under-prediction during turbulent periods, when compared to the MSE-trained models.

## 5.2. Impact of nonlinearity

To examine the necessity of nonlinear relations, we provide the following analysis to shed light on the competitive performance of these models, particularly during volatile periods. Inspired by Chinco et al. (2019), we introduce, for each day t, the following metric to evaluate the fraction of variance of model F which is unexplained

<!-- image-->  
(a) Forecast errors.

<!-- image-->  
(b) Forecast ratios.

<!-- image-->  
(c) Forecast errors during calm days.

<!-- image-->  
(d) Forecast ratios during calm days.

<!-- image-->  
(e) Forecast errors during turbulent days.

<!-- image-->  
(f) Forecast ratios during turbulent days.  
Fig. 5. Grouped boxplots for models trained with MSE or QL.

Note: This figure presents boxplots illustrating three summary statistics: the median, and the Q1 and Q3 quantiles. Each group consists of two sets of boxplots, with the top (resp. bottom) set representing models utilizing QL (resp. MSE) as EC. (a)–(b) Forecast errors or ratios over the entire testing period. (c)–(d) Forecast errors or ratios over calm periods. (e)–(f) Forecast errors or ratios over turbulent periods.

Table 3 FVUs compared to HARM .

<table><tr><td rowspan="2"></td><td colspan="2">One day</td><td colspan="2">One week</td><td colspan="2">One month</td></tr><tr><td>Calm</td><td>Turb</td><td>Calm</td><td>Turb</td><td>Calm</td><td>Turb</td></tr><tr><td> $\mathrm { H A R } _ { M }$ </td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td> ${ \mathrm { G H A R } } _ { M }$ </td><td>0.044</td><td>0.061</td><td>0.054</td><td>0.099</td><td>0.066</td><td>0.092</td></tr><tr><td> $\mathbf { G N N H A R 1 L } _ { M }$ </td><td>0.077</td><td>0.165</td><td>0.117</td><td>0.244</td><td>0.178</td><td>0.300</td></tr><tr><td> $\mathsf { G N N H A R } 2 \mathrm { L } _ { M }$ </td><td>0.080</td><td>0.205</td><td>0.114</td><td>0.304</td><td>0.207</td><td>0.441</td></tr><tr><td> $\mathsf { G N N H A R 3 L } _ { M }$ </td><td>0.079</td><td>0.300</td><td>0.130</td><td>0.246</td><td>0.218</td><td>0.272</td></tr><tr><td> $\mathrm { H A R } _ { Q }$ </td><td>0.033</td><td>0.056</td><td>0.068</td><td>0.139</td><td>0.184</td><td>0.263</td></tr><tr><td> $\mathrm { G H A R } _ { Q }$ </td><td>0.077</td><td>0.128</td><td>0.108</td><td>0.216</td><td>0.228</td><td>0.779</td></tr><tr><td> ${ \mathrm { G N N H A R } } 1 { \mathrm { L } } _ { 0 }$ </td><td>0.060</td><td>0.134</td><td>0.102</td><td>0.244</td><td>0.216</td><td>0.886</td></tr><tr><td> $G \mathsf { N N H A R 2 L } _ { Q }$ </td><td>0.060</td><td>0.184</td><td>0.118</td><td>0.379</td><td>0.283</td><td>1.391</td></tr><tr><td> $\mathsf { G N N H A R 3 L } _ { \mathbb { Q } }$ </td><td>0.070</td><td>0.212</td><td>0.163</td><td>0.764</td><td>0.292</td><td>1.236</td></tr></table>

Note: The table reports the fractions of variance unexplained (FVUs) of multiple models compared by the baseline HAR, across different market regimes.

(FVU) by the standard $\mathrm { H A R } _ { M }$ mode $1 ^ { 2 3 }$ :

$$
\mathrm { F V U } _ { t } = \frac { \sum _ { i = 1 } ^ { N } \left( \widehat { R V } _ { i , t } ^ { ( F ) } - \widehat { R V } _ { i , t } ^ { ( \mathrm { H A R } _ { M } ) } \right) ^ { 2 } } { \sum _ { i = 1 } ^ { N } \left( \widehat { R V } _ { i , t } ^ { ( F ) } - \overline { { R V } } _ { t } ^ { ( F ) } \right) ^ { 2 } } ,\tag{11}
$$

where $\overline { { R V } } _ { t } ^ { ( F ) }$ is the average forecast RV of model F across stocks on day t. At one extreme, $\mathrm { F V U } _ { t } ~ = ~ 0$ means that the ${ \mathrm { H A R } } _ { M } { ' } s$ RV forecasts explain all of the variation in the predicted RVs provided by F , whereas, at the other extreme, $\mathrm { F V U } _ { t } = 1$ denotes that $\mathrm { H A R } _ { M }$ explains none of this variation.

Table 3 displays the FVUs of each model in comparison to $\mathrm { H A R } _ { M }$ . It is worth noting that nonlinear models, particularly those with multiple hidden layers, exhibit higher FVU values, as anticipated. In addition, the results for oneweek and one-month horizons in Table 3 suggest that the nonlinearity in volatility models seems to strengthen as the forecasting horizons increase. It is important to mention that the distinction between GHAR and GNNHAR1L lies in the presence of an additional hidden layer with a nonlinear activation function in GNNHAR1L. Consequently, the extra FVUs observed in GNNHAR1L can be considered as a measure of the degree of nonlinearity.

By comparing the first column and second column in Table 3, we observe higher FVU scores during turbulent days, regardless of the choice of EC. This suggests that nonlinear spillover effects are most likely to exist in turbulent periods, rather than in calm periods. In light of the results in Table 2, it can be inferred that a suitable level of model nonlinearity, such as that exhibited by GNNHAR1L, leads to improved predictive power during turbulent days. However, we find that overly complex models, such as GNNHAR3L, are unable to outperform the linear baseline. As a result, GNNHAR1L shows significant promise as a model for capturing nonlinearity while avoiding the overfitting problem.

## 5.3. Impact of multi-hop neighbors

We utilize the DM test to evaluate the statistical significance of two-hop neighbors by comparing the performance of GNNHAR2L and GNNHAR1L. Here, a positive (resp. negative) DM test value indicates the superiority of the GNNHAR1L (resp. GNNHAR2L) model. A p-value less than a given significance level a rejects the null hypothesis that GNNHAR2L and GNNHAR1L have the same forecasting power at the 1 − a confidence level.24

Fig. 6 illustrates the main results from the above hypothesis test. In terms of individual stocks, $\mathrm { G N N H A R } 2 \mathrm { L } _ { M }$ is only superior to $\mathrm { G N N H A R } 1 \mathrm { L } _ { M }$ in forecasting AXP’s volatilities, at the 5% confidence level. When considering the cross-sectional performance, the p-value is around 75%, from which we cannot reject the null hypothesis. This suggests that once the impact from itself and its onehop neighbors has been taken into account, two-hop neighbors are not deemed necessary. The comparison between GNNHAR2LQ and GNNHAR1LQ indeed supports these findings.

GNNs are known to suffer from the problem of oversmoothing, which is defined as the high similarity of node representations obtained at the output layer of GNNs; see Li et al. (2018). Such high similarity is often observed when stacking with multiple GNN layers that are more than necessary. With K layers, every node receives information from its K -hop neighbors.25 When K is large, node representations obtained from GNN information propagation become indistinguishable and weaken the forecasting accuracy.

Following the convention in the GNN literature (e.g. Chen et al., 2020), we use the mean average distance (MAD) to measure the similarity of node representations and identify whether there is any sign of over-smoothing in our GNNHAR models. The MAD takes as input the node representations $\pmb { H } \in \mathbb { R } ^ { N \times D }$ obtained at the final layer of the GNN, that is $\pmb { H } = \mathbf { G } \mathrm { N N } ( \pmb { V } _ { : t - 1 } , \pmb { A } )$ in (6). It is defined as follows 26

$$
\mathrm { M A D } = \frac { \sum _ { i = 1 } ^ { N } \bar { d } _ { i } } { \sum _ { i = 1 } ^ { N } \mathbb { 1 } _ { \bar { d } _ { i } > 0 } } , \mathrm { w h e r e ~ } \bar { d } _ { i } = \frac { \sum _ { j = 1 } ^ { N } \bar { D } _ { i j } } { \sum _ { j = 1 } ^ { N } \mathbb { 1 } _ { \bar { D } _ { i j > 0 } } } .\tag{12}
$$

Here, $\bar { \bf D }$ is the masked cosine distance matrix, i.e. $\bar { \textbf { D } } =$ $\mathbf { D } \circ \mathbf { A } ,$ where ◦ denotes the Hadamard product (elementwise multiplication), and $\begin{array} { r } { \pmb { D } _ { i j } = 1 - \frac { \mathbf { \tilde { \phi } } _ { H [ i , : ] \cdot H [ j , : ] } ^ { * } } { \| \pmb { H } [ i , : ] \| \| \pmb { H } [ j , : ] \| } } \end{array}$ . In the above definition, $\bar { d } _ { i }$ is the average distance between the representations of node i and its connected nodes. Overall, MAD represents an average level of how a node representation is similar to the representations of its connected neighbors in a graph.

In Fig. 7, three boxes represent GNNHAR models with one, two, and three GNN layers trained with MSE.27 Each

<!-- image-->  
(a) GNNHAR2LM vs GNNHAR1LM

<!-- image-->  
(b) GNNHAR2LQ vs GNNHAR1LQ  
Fig. 6. DM test between GNNHAR2L and GNNHAR1L.

Note: A positive (negative) number indicates superiority for the GNNHAR1L (GNNHAR2L) model. The y-axis represents the DM test values based on a QL between GNNHAR2L and GNNHAR1L, while the x-axis lists the stock symbols. Stars indicate the p-value, with orange, green, and blue representing significance at the 1%, 5%, and 10% levels, respectively. The horizon line represents the cross-sectional DM test value and its corresponding p-value. (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)

<!-- image-->  
Fig. 7. Smoothness of GNNHARs.  
Note: A small mean average distance (MAD) value indicates high similarity between node representations at the output layer of the GNN.

Table 4  
Out-of-sample forecast losses under a smaller validation dataset.
<table><tr><td rowspan="2"></td><td colspan="2">One day</td><td colspan="2">One week</td><td colspan="2">One month</td></tr><tr><td>MSE</td><td>QL</td><td>MSE</td><td>QL</td><td>MSE</td><td>QL</td></tr><tr><td> $\mathrm { H A R } _ { M }$ </td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td> ${ \mathrm { G H A R } } _ { M }$ </td><td>0.927</td><td>0.983</td><td>0.904</td><td>0.987</td><td>0.975*</td><td>1.031</td></tr><tr><td> $\mathbf { G N N H A R 1 L } _ { M }$ </td><td>0.942</td><td>0.978</td><td>0.931</td><td>0.945</td><td>1.008</td><td>0.975</td></tr><tr><td> $\mathsf { G N N H A R } 2 \mathrm { L } _ { M }$ </td><td>0.984</td><td>0.984</td><td>1.005</td><td>0.956</td><td>1.138</td><td>1.033</td></tr><tr><td> $\mathsf { G N N H A R 3 L } _ { M }$ </td><td>1.078</td><td>1.002</td><td>1.035</td><td>0.954</td><td>1.068</td><td>0.958</td></tr><tr><td> $\mathrm { H A R } _ { Q }$ </td><td>0.936</td><td>0.986</td><td>0.945</td><td>0.944</td><td>1.218</td><td>0.959</td></tr><tr><td> $\mathrm { G H A R } _ { Q }$ </td><td>0.942</td><td>0.982</td><td>0.993</td><td>0.945</td><td>1,174</td><td>0.954</td></tr><tr><td> $\mathrm { G N M H A R 1 L } _ { Q }$ </td><td>0.889*</td><td>0.967*</td><td>0.875*</td><td>0.912</td><td>1.226</td><td>0.961</td></tr><tr><td> $\mathrm { G N M H A R } 2 \mathrm { L } _ { Q }$ </td><td>0.896</td><td>0.968*</td><td>0.861*</td><td>0.907*</td><td>1.510</td><td>0.925*</td></tr><tr><td> $\mathsf { G N N H A R 3 L } _ { \mathbb { Q } }$ </td><td>1.152</td><td>0.981</td><td>1.060</td><td>0.929</td><td>1.572</td><td>0.972</td></tr></table>

Note: The table reports the out-of-sample losses of various models using 47 months as training data and the most recent one month as validation data. For each horizon, the model with the best out-of-sample performance in MSE (QL) is highlighted in red (blue). \* Asterisk indicates models that yield as accurate forecasts as the best model at the 5% significance level based on the MCS test.

box corresponds to the MAD values on a logarithmic scale, calculated across all out-of-sample samples. As the number of GNN layers increases, there is a decrease in log MAD that corresponds to an increase in smoothness. The three-layer GNNHAR has the lowest MAD score, suggesting potential over-smoothing of node representations. Specifically, the rows of $\mathsf { G N N } ( V _ { : t - 1 } , A )$ from GNNHAR3L in (6) become too similar to provide any node-specific predictive information. This partially explains the inferior performance of GNNHAR3L, as shown in Table 1.

## 6. Robustness tests

After presenting the main empirical results and analyzing the model performance across different market periods, we shift our focus to evaluating the robustness of the proposed models by considering two aspects: (i) an alternative validation set size, and (ii) a larger universe.

## 6.1. Alternative validation set size

Our main analysis is based on rolling samples of four years, using the first approximately three years as training data, and the recent year as validation data. Using a smaller validation dataset, such as one month, does not significantly alter our findings, as shown in Table 4.

## 6.2. Larger universe

To further assess the robustness of our findings and ascertain that they are not specific to the stocks under current consideration, we repeat the out-of-sample analysis using a larger dataset, including the components of the S&P 100 index.28 The experimental setups and the hyperparameter choices in GNNHAR remain the same as those described in Section 4.1. As illustrated in Table A.2, in the volatility spillover graphs for the S&P 100 index components, each node is connected to other nodes within a maximum of five steps. Consequently, we extend our analysis to include four- and five-layer versions of the GNNHAR model.

Table 5  
Out-of-sample forecast losses on S&P 100.
<table><tr><td rowspan="2"></td><td colspan="2">One day</td><td colspan="2">One week</td><td colspan="2">One month</td></tr><tr><td>MSE</td><td>QL</td><td>MSE</td><td>QL</td><td>MSE</td><td>QL</td></tr><tr><td>HARM</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td> $\mathrm { G H A R 1 L } _ { M }$ </td><td>0.948</td><td>0.988</td><td>0.909</td><td>0.994</td><td>0.972*</td><td>0.986</td></tr><tr><td> $\mathtt { G N N H A R 1 L } _ { M }$ </td><td>0.963</td><td>0.986</td><td>0.951</td><td>0.944</td><td>1.027</td><td>1.092</td></tr><tr><td> $\mathtt { G N N H A R 2 L } _ { M }$ </td><td>1.072</td><td>0.988</td><td>1.031</td><td>0.954</td><td>1.092</td><td>1.000</td></tr><tr><td> $\mathsf { G N N H A R 3 L } _ { M }$ </td><td>1.061</td><td>0.986</td><td>1.029</td><td>0.959</td><td>0.992</td><td>0.967</td></tr><tr><td> $\mathrm { G N N H A R } 4 \mathrm { L } _ { M }$ </td><td>1.047</td><td>0.992</td><td>1.042</td><td>0.975</td><td>1.079</td><td>0.978</td></tr><tr><td> $\mathsf { G N N H A R } 5 \mathrm { L } _ { M }$ </td><td>1.090</td><td>0.997</td><td>1.057</td><td>0.986</td><td>1.109</td><td>1.038</td></tr><tr><td> $\mathrm { H A R } _ { Q }$ </td><td>0.949</td><td>0.983</td><td>0.937</td><td>0.947</td><td>1.171</td><td>0.991</td></tr><tr><td> $\mathrm { G H A R } _ { Q }$ </td><td>0.919</td><td>0.984</td><td>0.850*</td><td>0.922</td><td>1.154</td><td>0.939*</td></tr><tr><td> $\mathrm { G N M H A R 1 L } _ { Q }$ </td><td>0.917*</td><td>0.969</td><td>0.858</td><td>0.916</td><td>1.231</td><td>1.017</td></tr><tr><td> $\mathrm { G N M H A R } 2 \mathrm { L } _ { Q }$ </td><td>0.915*</td><td>0.969</td><td>0.909</td><td>0.915*</td><td>1.206</td><td>0.941*</td></tr><tr><td> $\mathsf { G N M H A R 3 L } _ { Q }$ </td><td>0.938</td><td>0.966*</td><td>1.178</td><td>0.968</td><td>1.523</td><td>0.946</td></tr><tr><td> $G \mathsf { N N H A R 4 L } _ { Q }$ </td><td>0.985</td><td>0.970</td><td>1.165</td><td>0.972</td><td>1.563</td><td>0.971</td></tr><tr><td> $\mathsf { G N N H A R } 5 \mathrm { L } _ { Q }$ </td><td>0.951</td><td>0.968</td><td>1.193</td><td>0.975</td><td>1.741</td><td>0.989</td></tr></table>

Note: The table reports the ratios of forecast losses of various models compared to the standard $\mathrm { H A R } _ { M }$ model over one-day, one-week, and one-month horizons. For each horizon, the model with the best out-of-sample performance in MSE (QL) is highlighted in red (blue). \* Asterisk indicates models that yield as accurate forecasts as the best model at the 5% significance level based on the MCS test.

The out-of-sample forecasting performance on the volatilities of S&P 100 components is presented in Table 5. Firstly, we observe that GHAR consistently enhances forecasting accuracy compared to the traditional HAR model. Additionally, the nonlinear variant, GNNHAR1L, further improves upon the performance of GHAR over the oneday horizon. Generally, as we increase the number of layers in the GNNHAR models, their forecasting performance tends to decline. Nevertheless, we still observe the benefits of training models with the QL loss function. In summary, the findings presented in Table 5 align closely with those observed for the DJIA 30, providing consistent results across both datasets.

## 7. Conclusion

In this article, we proposed a novel methodology, GNNHAR, for modeling and forecasting RV while taking into account volatility spillover effects in the U.S. equity market. Our analysis suggests that the information from the multi-hop neighbors in the financial graph does not offer a clear advantage in predicting the volatility of any target stock. However, nonlinear spillover effects help improve the forecasting accuracy of the RV. Moreover, we found that utilizing QL as the training loss function leads to more accurate volatility forecasts than using the conventional MSE. Additionally, QL-trained nonlinear models demonstrated greater resilience during turbulent periods compared to calmer market conditions, unlike standard linear models which struggle in such regimes. Our comprehensive evaluation tests in alternative settings confirmed the robustness and effectiveness of our proposed methodology.

Table A.1  
Summary statistics of realized volatility.
<table><tr><td>Ticker</td><td>Mean</td><td>Std</td><td>Min</td><td>25%</td><td>50%</td><td>75%</td><td>Max</td><td>DJIA</td><td>S&amp;P 100</td></tr><tr><td>AAPL</td><td>2.30</td><td>3.39</td><td>0.07</td><td>0.70</td><td>1.25</td><td>2.46</td><td>38.30</td><td>✓</td><td>✓</td></tr><tr><td>ABT</td><td>1.41</td><td>1.95</td><td>0.12</td><td>0.57</td><td>0.89</td><td>1.50</td><td>34.32</td><td></td><td>✓</td></tr><tr><td>ACN</td><td>1.72</td><td>2.79</td><td>0.14</td><td>0.58</td><td>0.92</td><td>1.76</td><td>54.88</td><td></td><td>✓</td></tr><tr><td>ADBE</td><td>2.53</td><td>3.34</td><td>0.16</td><td>0.93</td><td>1.54</td><td>2.76</td><td>45.55</td><td></td><td>✓</td></tr><tr><td>ADP</td><td>1.41</td><td>2.51</td><td>0.10</td><td>0.49</td><td>0.78</td><td>1.39</td><td>44.36</td><td></td><td></td></tr><tr><td>AMGN</td><td>1.91</td><td>2.34</td><td>0.16</td><td>0.82</td><td>1.27</td><td>2.14</td><td>33.44</td><td></td><td></td></tr><tr><td>AMT</td><td>2.16</td><td>3.83</td><td>0.19</td><td>0.68</td><td>1.11</td><td>2.10</td><td>53.19</td><td></td><td></td></tr><tr><td>AMZN</td><td>3.22</td><td>4.48</td><td>0.11</td><td>1.02</td><td>1.84</td><td>3.59</td><td>62.14</td><td></td><td></td></tr><tr><td>AXP</td><td>3.19</td><td>6.32</td><td>0.12</td><td>0.64</td><td>1.15</td><td>2.67</td><td>91.45</td><td></td><td></td></tr><tr><td>BA</td><td>2.69</td><td>5.00</td><td>0.13</td><td>0.78</td><td>1.35</td><td>2.60</td><td>90.65</td><td></td><td></td></tr><tr><td>BAC</td><td>4.93</td><td>11.48</td><td>0.10</td><td>1.01</td><td>1.81</td><td>3.68</td><td>135.30</td><td></td><td></td></tr><tr><td>BDX</td><td>1.37</td><td>1.84</td><td>0.13</td><td>0.54</td><td>0.86</td><td>1.48</td><td>28.52</td><td></td><td></td></tr><tr><td>BMY</td><td>1.77</td><td>2.20</td><td>0.08</td><td>0.72</td><td>1.14</td><td>1.93</td><td>30.75</td><td></td><td></td></tr><tr><td>BSX</td><td>3.15</td><td>4.39</td><td>0.20</td><td>1.14</td><td>1.92</td><td>3.35</td><td>55.28</td><td></td><td></td></tr><tr><td>C</td><td>5.48</td><td>14.6</td><td>0.15</td><td>0.99</td><td>1.82</td><td>3.94</td><td>257.34</td><td></td><td></td></tr><tr><td>CAT</td><td>2.79</td><td>4.00</td><td>0.15</td><td>0.94</td><td>1.58</td><td>2.89</td><td>45.26</td><td></td><td></td></tr><tr><td>CB</td><td>1.82</td><td>3.66</td><td>0.07</td><td>0.44</td><td>0.75</td><td>1.62</td><td>61.54</td><td></td><td></td></tr><tr><td>CI</td><td>3.65</td><td>6.92</td><td>0.19</td><td>1.01</td><td>1.75</td><td>3.28</td><td>164.21</td><td></td><td></td></tr><tr><td>CMCSA</td><td>2.35</td><td>3.57</td><td>0.13</td><td>0.78</td><td>1.29</td><td>2.47</td><td>43.26</td><td></td><td></td></tr><tr><td>CME</td><td>3.07</td><td>5.49</td><td>0.18</td><td>0.84</td><td>1.38</td><td>2.72</td><td>68.79</td><td></td><td></td></tr><tr><td>COP</td><td>3.12</td><td>5.18</td><td>0.16</td><td>0.98</td><td>1.71</td><td>3.26</td><td>75.84</td><td></td><td></td></tr><tr><td>COST</td><td>1.44</td><td>2.11</td><td>0.0</td><td>0.51</td><td>0.79</td><td>1.44</td><td>26.30</td><td></td><td></td></tr><tr><td>CRM</td><td>4.00</td><td>4.93</td><td>0.22</td><td>1.44</td><td>2.41</td><td>4.64</td><td>61.67</td><td></td><td></td></tr><tr><td>CSCO</td><td>1.98</td><td>2.92</td><td>0.14</td><td>0.70</td><td>1.13</td><td>2.09</td><td>43.74</td><td></td><td></td></tr><tr><td>CVS</td><td>1.99</td><td>3.15</td><td>0.13</td><td>0.70</td><td>1.17</td><td>2.03</td><td>53.28</td><td></td><td></td></tr><tr><td>CVX</td><td>2.03</td><td>3.51</td><td>0.13</td><td>0.61</td><td>1.07</td><td>2.04</td><td>48.07</td><td></td><td></td></tr><tr><td>D</td><td>1.44</td><td>2.56</td><td>0.1</td><td>0.56</td><td>0.85</td><td>1.40</td><td>40.39</td><td></td><td></td></tr><tr><td>DHR</td><td>1.6</td><td>2.41</td><td>0.14</td><td>0.54</td><td>0.95</td><td>1.67</td><td>29.78</td><td></td><td></td></tr><tr><td>DIS</td><td>1.89</td><td>3.04</td><td>0.12</td><td>0.60</td><td>1.01</td><td>1.88</td><td>40.56</td><td></td><td></td></tr><tr><td>DUK</td><td>1.32</td><td>2.20</td><td>0.06</td><td>0.50</td><td>0.78</td><td>1.32</td><td>36.07</td><td></td><td></td></tr><tr><td>FIS</td><td>1.89</td><td>3.48</td><td>0.15</td><td>0.59</td><td>0.97</td><td>1.74</td><td>62.40</td><td></td><td></td></tr><tr><td>FISV</td><td>1.71</td><td>2.82</td><td>0.15</td><td>0.58</td><td>0.93</td><td>1.69</td><td>53.36</td><td></td><td></td></tr><tr><td>GE</td><td>3.08</td><td>5.54</td><td>0.09</td><td>0.68</td><td>1.43</td><td>3.05</td><td>77.33</td><td></td><td></td></tr><tr><td>GILD</td><td>2.36</td><td>2.67</td><td>0.23</td><td>1.03</td><td>1.55</td><td>2.64</td><td>33.62</td><td></td><td></td></tr><tr><td>GOOG</td><td>1.94</td><td>2.72</td><td>0.11</td><td>0.64</td><td>1.08</td><td>2.07</td><td>30.36</td><td></td><td></td></tr><tr><td>GS</td><td>3.24</td><td>6.27</td><td>0.19</td><td>0.92</td><td>1.49</td><td>2.81</td><td>112.41</td><td></td><td></td></tr><tr><td>HD</td><td>2.11</td><td>3.59</td><td>0.15</td><td>0.62</td><td>1.02</td><td>2.01</td><td>48.22</td><td>✓</td><td></td></tr><tr><td>HON</td><td>1.85</td><td>3.25</td><td>0.1</td><td>0.52</td><td>0.97</td><td>1.84</td><td>49.64</td><td>✓</td><td></td></tr><tr><td>IBM</td><td>1.38</td><td>2.33</td><td>0.11</td><td>0.47</td><td>0.75</td><td>1.34</td><td>30.22</td><td>✓</td><td>✓</td></tr><tr><td>INTC</td><td>2.29</td><td>3.12</td><td>0.14</td><td>0.86</td><td>1.39</td><td>2.44</td><td>42.90</td><td>✓</td><td>✓</td></tr><tr><td>INTU</td><td>2.00</td><td>2.81</td><td>0.15</td><td>0.75</td></tr><tr><td>SBUX</td><td>2.45</td><td>3.90</td><td>0.18</td><td>0.71</td><td>1.24</td><td>2.48</td><td>63.45</td><td></td><td>✓</td></tr><tr><td>SO</td><td>1.19</td><td>1.98</td><td>0.12</td><td>0.47</td><td>0.72</td><td>1.22</td><td>36.40</td><td></td><td>✓</td></tr><tr><td>SYK</td><td>1.67</td><td>2.61</td><td>0.08</td><td>0.62</td><td>0.98</td><td>1.76</td><td>49.51</td><td></td><td>✓</td></tr><tr><td>T</td><td>1.49</td><td>2.55</td><td>0.08</td><td>0.47</td><td>0.76</td><td>1.39</td><td>32.03</td><td></td><td>✓</td></tr><tr><td>TGT</td><td>2.46</td><td>4.02</td><td>0.11</td><td>0.76</td><td>1.24</td><td>2.34</td><td>53.02</td><td></td><td>✓</td></tr><tr><td>TJX</td><td>2.33</td><td>3.34</td><td>0.16</td><td>0.76</td><td>1.24</td><td>2.53</td><td>55.49</td><td></td><td>✓</td></tr><tr><td>TMO</td><td>1.89</td><td>2.74</td><td>0.16</td><td>0.71</td><td>1.14</td><td>1.99</td><td>40.82</td><td></td><td>✓</td></tr><tr><td>TRV</td><td>2.04</td><td>4.09</td><td>0.11</td><td>0.49</td><td>0.81</td><td>1.76</td><td>57.95</td><td>✓</td><td></td></tr><tr><td>TXN</td><td>2.33</td><td>3.02</td><td>0.16</td><td>0.84</td><td>1.41</td><td>2.57</td><td>48.68</td><td></td><td>✓</td></tr><tr><td>UNH</td><td>2.70</td><td>4.34</td><td>0.16</td><td>0.78</td><td>1.35</td><td>2.57</td><td>52.54</td><td>✓</td><td></td></tr><tr><td>UNP</td><td>2.53</td><td>3.94</td><td>0.14</td><td>0.83</td><td>1.39</td><td>2.52</td><td>45.94</td><td></td><td></td></tr><tr><td>UPS</td><td>1.58</td><td>2.35</td><td>0.10</td><td>0.51</td><td>0.88</td><td>1.72</td><td>31.67</td><td></td><td>✓</td></tr><tr><td>USB</td><td>3.20</td><td>6.88</td><td>0.13</td><td>0.62</td><td>1.16</td><td>2.64</td><td>95.38</td><td></td><td>✓</td></tr><tr><td>VZ</td><td>1.40</td><td>2.36</td><td>0.10</td><td>0.50</td><td>0.77</td><td>1.33</td><td>34.19</td><td>✓</td><td></td></tr><tr><td>WFC</td><td>4.05</td><td>8.89</td><td>0.11</td><td>0.73</td><td>1.39</td><td>3.24</td><td>106.81</td><td></td><td>✓</td></tr><tr><td>WMT</td><td>1.18</td><td>1.76</td><td>0.11</td><td>0.45</td><td>0.67</td><td>1.18</td><td>27.18</td><td>✓</td><td>✓</td></tr></table>

An intriguing avenue for future exploration involves expanding the predictor set to incorporate additional information sources, such as limit order books, options, and news (Li & Tang, 2021). Another interesting direction to explore is the robustness of the proposed methods when applied to different approaches to constructing financial graphs, such as those based on supply chains (Herskovic et al., 2020) and analyst co-coverage (Ali & Hirshleifer, 2020). It would be valuable to investigate whether these graphs provide unique information content and have the potential to enhance performance.

## CRediT authorship contribution statement

Chao Zhang: Writing – review & editing, Writing – original draft, Visualization, Validation, Software, Resources, Project administration, Methodology, Investigation, Formal analysis, Data curation, Conceptualization. Xingyue Pu: Writing – review & editing, Visualization, Validation, Software, Investigation, Formal analysis, Conceptualization. Mihai Cucuringu: Writing – review & editing, Supervision, Funding acquisition, Conceptualization. Xiaowen Dong: Writing – review & editing, Supervision, Funding acquisition, Conceptualization.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data and code availability

The data used in this paper is sourced from LOBSTER, which is subject to licensing restrictions and must be purchased by users, as redistribution is not permitted. The source code is available at: github.com/chaozhangox/GNNHAR.

Table A.2  
Frequency (in percentage) of the shortest path distance.
<table><tr><td>SPD</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>DJIA</td><td>57.7</td><td>41.8</td><td>0.5</td><td>0.0</td><td>0.0</td></tr><tr><td>S&amp;P 100</td><td>24.3</td><td>61.2</td><td>12.0</td><td>2.2</td><td>0.3</td></tr></table>

Note: For example, in the case of the S&P 100, 12% of pairs of nodes have their shortest path distance of size three.

## Acknowledgments

This work is supported by the Guangzhou-HKUST(GZ) Joint Funding Program (No. 2024A03J0630). The authors also thank the Oxford Suzhou Centre for Advanced Research for providing the computational facilities.

## Appendix A. Data statistics

See Tables A.1 and A.2.

## Appendix B. QL-trained models and MEM

Let $y _ { t }$ be a non-negative random variable, such as $R V _ { t }$ $y _ { t }$ follows an MEM model if it can be expressed as

$$
y _ { t } = \nu _ { t } z _ { t } , \quad \nu _ { t } = g \left( \theta ; \mathcal { F } _ { t - 1 } \right) , \quad z _ { t } \overset { \mathrm { i . i . d . } } \sim D ^ { + } \left( 1 , \sigma ^ { 2 } \right) .\tag{13}
$$

Here, the term $\nu _ { t }$ represents the non-negative expectation of $y _ { t }$ conditional on the information set $\mathcal { F } _ { t - 1 }$ available at time t − 1, and νt is determined by a function g with parameters $\theta . z _ { t }$ is a conditionally unpredictable homoskedastic component, with non-negative support and unit expected value. The standard MEM aligns with the autoregressive structure of the well-known GARCH(1,1) for $\nu _ { t }$ . In this paper, we utilize GNNs as g to model νt .

Supposing zt is gamma-distributed29 with scale 1 and shape 1, the density of $y _ { t }$ is

$$
f _ { y } ( y _ { t } ) \propto \frac { 1 } { \nu _ { t } } e ^ { - \frac { y _ { t } } { \nu _ { t } } } .
$$

Subsequently, the negative log likelihood, after omitting constants, can be expressed as follows30:

$$
L ( \theta ) = \sum _ { t = 1 } ^ { T } \left[ \log \left( \nu _ { t } \right) + \frac { y _ { t } } { \nu _ { t } } \right] .\tag{14}
$$

This is equivalent to (10) if we substitute $\nu _ { t }$ with $\widehat { R V } _ { t } ^ { ( F ) }$ and yt with $R V _ { t }$ , up to a constant factor.

## Appendix C. Training HAR via QL

Denote $\begin{array} { r l r l r l r l } { \pmb { \beta } } & { { } = } & { ( \alpha , \beta _ { d } , \beta _ { w } , \beta _ { m } ) ^ { \prime } } & { \in } & { { } \mathbb { R } ^ { 4 } , } & { \pmb { x } _ { t } } & { { } = } \end{array}$ $( 1 , R V _ { t - 1 } , R V _ { t - 5 : t - 2 } , R V _ { t - 2 2 : t - 6 } ) ^ { \prime } \in \mathbb { R } ^ { 4 }$ , and $\pmb { { \cal X } } = ( { \pmb { x } } _ { 2 3 } , \dots ,$ $\pmb { x } _ { T } ) ^ { \prime } \in \mathbb { R } ^ { ( T - 2 2 ) \times 4 }$ . The QL loss of the HAR model for a single time series is

$$
\mathcal { L } _ { Q } = \sum _ { t } \left[ \frac { R V _ { t } } { \beta ^ { \prime } \pmb { x } _ { t } } - \log \frac { R V _ { t } } { \beta ^ { \prime } \pmb { x } _ { t } } - 1 \right] .\tag{15}
$$

Then score function is given by

$$
\begin{array} { r l r } {  { \frac { \partial \mathcal { L } _ { 0 } } { \partial \beta } = \sum _ { t } \frac { - R V _ { t } } { ( \beta ^ { \prime } x _ { t } ) ^ { 2 } } x _ { t } + \frac { 1 } { \beta ^ { \prime } x _ { t } } x _ { t } } } \\ & { } & { = \sum _ { t } \frac { \beta ^ { \prime } x _ { t } - R V _ { t } } { ( \beta ^ { \prime } x _ { t } ) ^ { 2 } } x _ { t } } \\ & { } & { = \sum _ { t } w _ { \beta , t } ( \beta ^ { \prime } x _ { t } - R V _ { t } ) x _ { t } } \\ & { } & { = X ^ { \prime } W _ { \beta } ( X \beta - Y ) } \end{array}\tag{16}
$$

where $\begin{array} { r c l } { w _ { \beta , t } } & { = } & { \frac { 1 } { ( \beta ^ { \prime } \pmb { x } _ { t } ) ^ { 2 } } } \end{array}$ and ${ \pmb W } _ { \beta } ~ = ~ \mathrm { d i a g } \left\{ . . . w _ { \beta , t } . . . \right\}$ This leads to the first-order condition ${ \pmb X } ^ { \prime } { \pmb W } _ { \beta } ( { \pmb X } \beta - { \pmb Y } ) =$ $0 . { } ^ { 3 1 }$ The optimal solution $\beta$ appears in the weights $W _ { \beta } .$ Iteratively reweighted least squares (IRLS) is therefore recommended:

1. Select initial estimates $\beta _ { 0 } ,$ , such as the OLS.

2. At each iteration k, calculate the predictions $\widehat { R V } _ { k - 1 , t }$ $\mathbf { \xi } = \beta _ { k - 1 } ^ { \prime } \mathbf { x } _ { t }$ from the previous iteration, and the associated weights $w _ { k - 1 , t } = 1 / \widehat { R V } _ { k - 1 , t } ^ { 2 }$ and $W _ { k - 1 } =$ diag $\left\{ \dots \dots \ W { k - 1 , t } \dots \right\}$

3. Solve for new WLS estimates

$$
\pmb { \beta } _ { k } = \left[ \pmb { X } ^ { \prime } \pmb { W } _ { k - 1 } \pmb { X } \right] ^ { - 1 } \pmb { X } ^ { \prime } \pmb { W } _ { k - 1 } \pmb { Y } .\tag{17}
$$

4. Steps 2 and 3 are repeated until the estimated coefficients converge.

To gain further insights into the impact of the EC, we present the trajectories of $\beta _ { d }$ in the HAR models estimated using MSE or QL in Fig. C.1. As anticipated, there are substantial temporal variations in the rolling estimates of both models. In general, the estimates of $\beta _ { d }$ in HARQ exhibit greater variability compared to those in $\mathrm { H A R } _ { M }$ which can be attributed to the stochastic nature of the optimization algorithm employed in HARQ . However, the estimates of $\beta _ { d }$ in $\mathrm { H A R } _ { M }$ reveal two prominent changes occurring during December 2015 to February 2016 and March 2020 to April 2020, albeit in different directions.32 On the other hand, the $\beta _ { d }$ in $\mathrm { H A R } _ { Q }$ exhibits an increasing trend during turbulent periods. This suggests that QLtrained models have the ability to swiftly adapt to market changes and assign greater importance to observations associated with recent significant events. Future studies exploring the relationship between different estimators of HAR are therefore recommended.

## Appendix D. Hyperparameter tuning

Following the convention of stochastic optimization (Kingma & Ba, 2014), we set the batch size to $3 2 . ^ { 3 3 }$ The learning rate for Adam is set to $1 0 ^ { - 3 }$ . We stop the training procedure early if there is a sign of overfitting, that is, if the training loss keeps dropping but validation loss increases beyond a tolerance level.

To a large extent, the dimension of hidden representations or the number of hidden neurons in the lth layer, i.e. $D ^ { ( l ) }$ in (5), reflects the complexity of our models. Inadequate dimensions may lack the capability to effectively capture the underlying data structure, while excessively large dimensions could lead to overfitting and poor generalization performance. To mitigate this issue, we use a grid search over $D ^ { ( l ) } ~ \in ~ \{ 3 , 6 , \bar { 9 } , 1 6 , 3 2 \}$ on validation datasets. Fig. D.1 shows that a hidden dimension of nine in a one-layer GNNHAR model leads to the smallest MSE and QL on the validation data. The same conclusion holds true for the QL-trained models as well. When multiple GNN layers are utilized, we maintain the same $D ^ { ( l ) }$ value as determined in the one-layer model.

## Appendix E. GHAR with multi-hop (GHAR2Hop)

It is important to highlight that HAR can be interpreted as a model that only considers the zero-hop neighbors, i.e. the target node itself, while the GHAR takes into account both the zero-hop and one-hop neighbors. In order to explore the potential benefits of multi-hop neighbors for enhancing volatility forecasting, we delve into the investigation of whether they provide additional predictive power. To address this novel question, we consider the following model:

$$
\begin{array} { r } { \mathbf { G H A R 2 H o p } ( \pmb { A } ) : \quad R \pmb { V } _ { t } = \pmb { \alpha } + \pmb { V } _ { : t - 1 } \pmb { \beta } + \pmb { W } \pmb { V } _ { : t - 1 } \gamma } \\ { + \ \mathrm { H o p } 2 ( \pmb { A } ) \pmb { V } _ { : t - 1 } \pmb { \delta } + \pmb { u } _ { t } , } \end{array}\tag{18}
$$

where Hop2(A) maps the raw adjacent matrix (for onehop neighbors) to the adjacent matrix of two-hop neighbors. Specifically, $\mathrm { H o p } 2 ( \bar { A } ) = \mathrm { X O R } ( A ^ { 2 } \wedge ( \neg A ) , I _ { N } ) . \dot { A } ^ { 2 } [ \bar { i } , j ]$ has a non-zero if it is possible to go from node i to node $j$ in two or fewer steps, ¬A excludes the one-hop neighbors, and XOR confirms the diagonal of the two-hop adjacent matrix to be zero. For a visual representation and further details, we refer the reader to Example 1 and Fig. 2. In our experiments, we used the normalized adjacent matrix of two-hop neighbors and estimated (18) through OLS.

<!-- image-->  
Fig. C.1. Trajectories of $\beta _ { d }$ in HAR trained with different losses.

Note: The left y-axis represents the estimated values of $\beta _ { d }$ every month, while the right y-axis represents the daily RV of the S&P 500 ETF shown in bar charts.  
<!-- image-->

<!-- image-->  
Fig. D.1. Validation performance under different dimensions of hidden representations in GNNHAR1LM . Note: Each box is obtained from 10 replicated experiments with different random initial parameters.

The DM test results between GHAR2Hop and GHAR are presented in Fig. E.1. The cross-sectional DM test value is approximately −1, with a corresponding p-value of approximately 35%. These results reinforce the primary findings regarding the role of multi-hop neighbors, indicating that including two-hop neighbors may not provide substantial additional predictive power.

In Fig. E.2, we conduct a detailed examination of the coefficients associated with K -hop neighbors across different forecasting horizons. Based on the given definitions, the zero-hop coefficients for the daily (resp. weekly and monthly) horizon represent $\beta _ { d }$ (resp. $\beta _ { w }$ and $\beta _ { m } ) ,$ the one-hop coefficients correspond to $\gamma _ { d }$ (resp. $\gamma _ { w }$ and $\gamma _ { m } )$ , and the two-hop coefficients denote $\delta _ { d }$ (resp. $\delta _ { w }$ and $\delta _ { m } )$ . Fig. E.2 reveals that the coefficients at zero hops are positive over three horizons (i.e. $\beta _ { d } , \beta _ { w } , \beta _ { m } > 0 )$ , consistent with previous literature (Bollerslev, Patton, & Quaedvlieg, 2018). We also observe that the daily coefficients are positive on average but rapidly decay with distance (i.e. $\beta _ { d } ~ > ~ \gamma _ { d } ~ > ~ \delta _ { d } )$ . Specifically, the daily coefficient associated with two-hop neighbors is approximately oneeighth (one-sixteenth) relative to the coefficient of their one-hop (zero-hop) counterparts. Another interesting observation is that the weekly and monthly coefficients are negative, potentially due to high collinearity, as highlighted by Zhang et al. (2022). Nonetheless, the magnitude of these coefficients diminishes as the distance increases, suggesting that the influence of the two-hop neighbors may be negligible.

<!-- image-->  
Fig. E.1. DM test between GHAR2Hop and GHAR.

Note: A positive (negative) number indicates superiority for the GHAR (GHAR2Hop) model. The y-axis represents the DM test values based on QLs between GHAR2Hop and GHAR, while the x-axis lists the stock symbols. Stars indicate the p-values, with orange, green, and blue representing significance at the 1%, 5%, and 10% levels, respectively. The horizon line represents the cross-sectional DM test value and its corresponding p-value. (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)

<!-- image-->  
Fig. E.2. Coefficients in GHAR2Hop. Note: This figure describes the average coefficients of different hop neighborhoods over multiple horizons.

## References

Acemoglu, Daron, Ozdaglar, Asuman, & Tahbaz-Salehi, Alireza (2010). Cascades in networks and aggregate volatility: Technical report, National Bureau of Economic Research.

Ali, Usman, & Hirshleifer, David (2020). Shared analyst coverage: Unifying momentum spillover effects. Journal of Financial Economics, 136(3), 649–675.

Alon, Uri, & Yahav, Eran (2020). On the bottleneck of graph neural networks and its practical implications. In International conference on learning representations.

Andersen, Torben G., Bollerslev, Tim, Diebold, Francis X., & Ebens, Heiko (2001). The distribution of realized stock return volatility. Journal of Financial Economics, 61(1), 43–76.

Andersen, Torben G., Bollerslev, Tim, & Meddahi, Nour (2011). Realized volatility forecasting and market microstructure noise. Journal of Econometrics, 160(1), 220–234.

Anselin, Luc (2022). Spatial econometrics. In Handbook of spatial analysis in the social sciences (pp. 101–122).

Bai, Zhidong, Wong, Wing-Keung, & Zhang, Bingzhi (2010). Multivariate linear and nonlinear causality tests. Mathematics and Computers in Simulation, 81(1), 5–17.

Barndorff-Nielsen, Ole E., & Shephard, Neil (2002). Econometric analysis of realized volatility and its use in estimating stochastic volatility

models. Journal of the Royal Statistical Society. Series B. Statistical Methodology, 64(2), 253–280.

Basturk, Nalan, Schotman, Peter C., & Schyns, Hugo (2022). A neural network with shared dynamics for multi-step prediction of value-at-risk and volatility. Available at SSRN 3871096.

Bauwens, Luc, Hafner, Christian M., & Laurent, Sébastien (2012). Handbook of volatility models and their applications: vol. 3, John Wiley & Sons.

Bollerslev, Tim, Hood, Benjamin, Huss, John, & Pedersen, Lasse Heje (2018). Risk everywhere: Modeling and managing volatility. The Review of Financial Studies, 31(7), 2729–2773.

Bollerslev, Tim, Patton, Andrew J., & Quaedvlieg, Rogier (2016). Exploiting the errors: A simple approach for improved volatility forecasting. Journal of Econometrics, 192(1), 1–18.

Bollerslev, Tim, Patton, Andrew J., & Quaedvlieg, Rogier (2018). Modeling and forecasting (un) reliable realized covariances for more reliable financial decisions. Journal of Econometrics, 207(1), 71–91.

Bucci, Andrea (2020). Realized volatility forecasting with neural networks. Journal of Financial Econometrics, 18(3), 502–531.

Buncic, Daniel, & Gisler, Katja I. M. (2016). Global equity market volatility spillovers: A broader role for the United States. International Journal of Forecasting, 32(4), 1317–1339.

Callot, Laurent A. F., Kock, Anders B., & Medeiros, Marcelo C. (2017). Modeling and forecasting large realized covariance matrices and portfolio choice. Journal of Applied Econometrics, 32(1), 140–158.

Caporin, Massimiliano, Rossi, Eduardo, & De Magistris, Paolo Santucci (2017). Chasing volatility: A persistent multiplicative error model with jumps. Journal of Econometrics, 198(1), 122–145.

Chen, Deli, Lin, Yankai, Li, Wei, Li, Peng, Zhou, Jie, & Sun, Xu (2020). Measuring and relieving the over-smoothing problem for graph neural networks from the topological view. In AAAI conference on artificial intelligence, vol. 34 (pp. 3438–3445).

Chen, Qinkai, & Robert, Christian-Yann (2022). Multivariate realized volatility forecasting with graph neural network. In Proceedings of the third ACM international conference on AI in finance (pp. 156–164).

Chen, Yingmei, Wei, Zhongyu, & Huang, Xuanjing (2018). Incorporating corporation relationship via graph convolutional neural networks for stock price prediction. In Proceedings of the 27th ACM international conference on information and knowledge management (pp. 1655–1658).

Chinco, Alex, Clark-Joseph, Adam D., & Ye, Mao (2019). Sparse signals in the cross-section of returns. The Journal of Finance, 74(1), 449–492.

Choudhry, Taufiq, Papadimitriou, Fotios I., & Shabi, Sarosh (2016). Stock market volatility and business cycle: Evidence from linear and nonlinear causality tests. Journal of Banking & Finance, 66, 89–101.

Cipollini, Fabrizio, Gallo, Giampiero M., & Otranto, Edoardo (2021). Realized volatility forecasting: Robustness to measurement errors. International Journal of Forecasting, 37(1), 44–57.

Cipollini, Fabrizio, Gallo, Giampiero M., & Palandri, Alessandro (2020). Realized variance modeling: Decoupling forecasting from estimation. Journal of Financial Econometrics, 18(3), 532–555.

Clements, Adam, & Preve, Daniel P. A. (2021). A practical guide to harnessing the HAR volatility model. Journal of Banking & Finance, 133, Article 106285.

Corsi, Fulvio (2009). A simple approximate long-memory model of realized volatility. Journal of Financial Econometrics, 7(2), 174–196.

Dai, Hanjun, Kozareva, Zornitsa, Dai, Bo, Smola, Alex, & Song, Le (2018). Learning steady-states of iterative algorithms over graphs. In International conference on machine learning (pp. 1106–1114). PMLR.

Defferrard, Michaël, Bresson, Xavier, & Vandergheynst, Pierre (2016). Convolutional neural networks on graphs with fast localized spectral filtering. In Advances in neural information processing systems.

Degiannakis, Stavros, & Filis, George (2017). Forecasting oil price realized volatility using information channels from other asset classes. Journal of International Money and Finance, 76, 28–49.

Diebold, Francis X., & Mariano, Roberto S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253–263.

Diebold, Francis X., & Yılmaz, Kamil (2014). On the network topology of variance decompositions: Measuring the connectedness of financial firms. Journal of Econometrics, 182(1), 119–134.

Engle, Robert (2002). New frontiers for ARCH models. Journal of Applied Econometrics, 17(5), 425–446.

Engle, Robert F., & Kroner, Kenneth F. (1995). Multivariate simultaneous generalized ARCH. Econometric Theory, 11(1), 122–150.

Fan, Jianqing, Qi, Lei, & Xiu, Dacheng (2014). Quasi-maximum likelihood estimation of GARCH models with heavy-tailed likelihoods. Journal of Business & Economic Statistics, 32(2), 178–191.

Feng, Jiarui, Chen, Yixin, Li, Fuhai, Sarkar, Anindya, & Zhang, Muhan (2022). How powerful are K-hop message passing graph neural networks. In Advances in neural information processing systems.

Friedman, Jerome, Hastie, Trevor, & Tibshirani, Robert (2008). Sparse inverse covariance estimation with the graphical LASSO. Biostatistics, 9(3), 432–441.

Goyenko, Ruslan, Kelly, Bryan T., Moskowitz, Tobias J., Su, Yinan, & Zhang, Chao (2024). Trading volume alpha. Available at SSRN.

Gu, Shihao, Kelly, Bryan, & Xiu, Dacheng (2020). Empirical asset pricing via machine learning. The Review of Financial Studies, 33(5), 2223–2273.

Hall, Peter, & Yao, Qiwei (2003). Inference in ARCH and GARCH models with heavy-tailed errors. Econometrica, 71(1), 285–317.

Hansen, Peter Reinhard, & Dumitrescu, Elena-Ivona (2022). How should parameter estimation be tailored to the objective? Journal of Econometrics, 230(2), 535–558.

Hansen, Peter Reinhard, Lunde, Asger, & Nason, James M. (2003). Choosing the best volatility models: The model confidence set approach. Oxford Bulletin of Economics and Statistics, 65, 839–861.

Hansen, Peter Reinhard, Lunde, Asger, & Nason, James M. (2011). The model confidence set. Econometrica, 79(2), 453–497.

Harvey, David, Leybourne, Stephen, & Newbold, Paul (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting, 13(2), 281–291.

Hecq, Alain, Margaritella, Luca, & Smeekes, Stephan (2023). Granger causality testing in high-dimensional VARs: A post-double-selection procedure. Journal of Financial Econometrics, 21(3), 915–958.

Herskovic, Bernard, Kelly, Bryan, Lustig, Hanno, & Van Nieuwerburgh, Stijn (2020). Firm volatility in granular networks. Journal of Political Economy, 128(11), 4097–4162.

Karpman, Kara, Basu, Sumanta, Easley, David, & Kim, Sanghee (2023). Learning financial networks with high-frequency trade data. Data Science in Science, 2(1), Article 2166624.

Kingma, Diederik P., & Ba, Jimmy (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

Kipf, Thomas N., & Welling, Max (2017). Semi-supervised classification with graph convolutional networks. In International conference on learning representations.

LeSage, James. P. (1999). The theory and practice of spatial econometrics: vol. 28, (no. 11), (pp. 1–39). Toledo, Ohio: University of Toledo.

Li, Qimai, Han, Zhichao, & Wu, Xiao-Ming (2018). Deeper insights into graph convolutional networks for semi-supervised learning. In AAAI conference on artificial intelligence.

Li, Sophia Zhengzi, & Tang, Yushan (2021). Automated volatility forecasting. Available at SSRN 3776915.

Liang, Ting, Zeng, Guanxiong, Zhong, Qiwei, Chi, Jianfeng, Feng, Jinghua, Ao, Xiang, & Tang, Jiayu (2021). Credit risk and limits forecasting in e-commerce consumer lending service via multi-view-aware mixture-of-experts nets. In Proceedings of the 14th ACM international conference on web search and data mining (pp. 229–237).

Ling, Shiqing, & McAleer, Michael (2003). Asymptotic theory for a vector ARMA-GARCH model. Econometric Theory, 19(2), 280–310.

Liu, Ziqi, Chen, Chaochao, Li, Longfei, Zhou, Jun, Li, Xiaolong, Song, Le, & Qi, Yuan (2019). Geniepath: Graph neural networks with adaptive receptive paths. In Proceedings of the AAAI conference on artificial intelligence (pp. 4424–4431).

Liu, Ziqi, Chen, Chaochao, Yang, Xinxing, Zhou, Jun, Li, Xiaolong, & Song, Le (2018). Heterogeneous graph neural networks for malicious account detection. In Proceedings of the 27th ACM international conference on information and knowledge management (pp. 2077–2085).

Liu, Han, Lafferty, John, & Wasserman, Larry (2009). The nonparanormal: Semiparametric estimation of high dimensional undirected graphs. Journal of Machine Learning Research, 10(10), 2295–2328.

Liu, Lily Y., Patton, Andrew J., & Sheppard, Kevin (2015). Does anything beat 5-minute RV? A comparison of realized measures across multiple asset classes. Journal of Econometrics, 187(1), 293–311.

Masters, Dominic, & Luschi, Carlo (2018). Revisiting small batch training for deep neural networks. arXiv preprint arXiv:1804.07612.

Pascalau, Razvan, & Poirier, Ryan (2021). Increasing the information content of realized volatility forecasts. Journal of Financial Econometrics, 21(4), 1064–1098.

Patton, Andrew J. (2011). Volatility forecast comparison using imperfect volatility proxies. Journal of Econometrics, 160(1), 246–256.

Patton, Andrew J., & Sheppard, Kevin (2009). Evaluating volatility and correlation forecasts. In Handbook of financial time series (pp. 801–838). Springer.

Patton, Andrew J., & Sheppard, Kevin (2015). Good volatility, bad volatility: Signed jumps and the persistence of volatility. The Review of Economics and Statistics, 97(3), 683–697.

Reisenhofer, Rafael, Bayer, Xandro, & Hautsch, Nikolaus (2022). HARNet: A convolutional neural network for realized volatility forecasting. arXiv preprint arXiv:2205.07719.

Sawhney, Ramit, Agarwal, Shivam, Wadhwa, Arnav, & Shah, Rajiv (2020). Deep attentive learning for stock movement prediction from social media text and company correlations. In Proceedings of the 2020 conference on empirical methods in natural language processing (pp. 8415–8426).

Scarselli, Franco, Gori, Marco, Tsoi, Ah Chung, Hagenbuchner, Markus, & Monfardini, Gabriele (2008). The graph neural network model. IEEE Transactions on Neural Networks, 20(1), 61–80.

Sheppard, Kevin (2010). Financial econometrics notes (pp. 333–426). University of Oxford.