# DOW 30 股票波动率预测研究方案

## 研究目标

基于 DOW 30 股票数据，构建并比较多种波动率预测模型，重点研究：

1. **图结构对模型性能的影响**：比较 GLASSO 构建的邻接矩阵与随机邻接矩阵
2. **损失函数的选择**：比较 MSE 与 QLIKE 损失函数的预测效果
3. **外生变量的作用**：评估隐含波动率(IV)对预测性能的提升效果

## 核心研究内容

### 1. 邻接矩阵构建
- **GLASSO 邻接矩阵**：基于股票收益率数据，使用图 LASSO 方法构建稀疏精度矩阵
- **随机邻接矩阵**：作为基准对照，评估图结构的真实贡献

### 2. 模型架构对比

#### 基准模型（仅使用 RV）
- **HAR**：异质自回归模型
- **GHAR**：图异质自回归模型  
- **GNNHAR**：图神经网络异质自回归模型

#### 扩展模型（RV + IV）
- **HAR+IV**：加入隐含波动率的线性扩展
- **GHAR+IV**：图模型的隐含波动率扩展
- **GNNHAR+IV**：图神经网络的隐含波动率扩展

### 3. 损失函数比较
- **MSE**：均方误差损失
- **QLIKE**：拟似然损失（专门针对波动率预测设计）

## 数学框架

### 符号定义
- $v_t \in \mathbb{R}^N$：第 $t$ 日 N 只股票的已实现波动率向量
- $x_t \in \mathbb{R}^N$：第 $t$ 日 N 只股票的隐含波动率向量
- $A \in \mathbb{R}^{N\times N}$：邻接矩阵（由收益率构造）
- $W = D^{-1/2} A D^{-1/2}$：归一化邻接矩阵

### HAR 特征聚合
**已实现波动率 (RV)**：
- 日频：$v_{t-1}$
- 周频：$v_{t-5:t-2} = \frac{1}{4}\sum_{k=2}^{5}v_{t-k}$
- 月频：$v_{t-22:t-6} = \frac{1}{17}\sum_{k=6}^{22}v_{t-k}$

**隐含波动率 (IV)**：
- 日频：$x_{t-1}$
- 周频：$x_{t-5:t-2} = \frac{1}{4}\sum_{k=2}^{5}x_{t-k}$
- 月频：$x_{t-22:t-6} = \frac{1}{17}\sum_{k=6}^{22}x_{t-k}$

## 模型详细说明

### 1. HAR+IV 模型
$$E(v_t|\mathcal{F}_{t-1}) = \alpha + V_{:t-1}\beta + X_{:t-1}\delta$$

其中：
- $\beta \in \mathbb{R}^3$：RV HAR 系数（日、周、月）
- $\delta \in \mathbb{R}^3$：IV HAR 系数（日、周、月）

### 2. GHAR+IV 模型
$$E(v_t|\mathcal{F}_{t-1}) = \alpha + V_{:t-1}\beta + WV_{:t-1}\gamma + X_{:t-1}\delta + WX_{:t-1}\eta$$

参数：
- $\beta, \gamma \in \mathbb{R}^3$：自身与邻居 RV HAR 系数
- $\delta, \eta \in \mathbb{R}^3$：自身与邻居 IV HAR 系数

### 3. GNNHAR+IV 模型

**节点特征**：
$$Z_{:t-1} = [V_{:t-1}, X_{:t-1}] \in \mathbb{R}^{N\times 6}$$

**图卷积层**：
$$H^{(l+1)} = \sigma(WH^{(l)}\Theta^{(l)}), \quad l = 0,\ldots,L-1$$

**输出预测**：
$$\hat{v}_t = H^{(L)}\theta^{(\text{out})}$$

### 4. 损失函数

**MSE 损失**：
$$\mathcal{L}_{\text{MSE}} = \frac{1}{NT}\sum_{t}\sum_{i} (v_{i,t} - \hat{v}_{i,t})^2$$

**QLIKE 损失**：
$$\mathcal{L}_{\text{QLIKE}} = \frac{1}{NT}\sum_{t}\sum_{i}\left(\frac{v_{i,t}}{\hat{v}_{i,t}+\epsilon} - \log\frac{v_{i,t}}{\hat{v}_{i,t}+\epsilon} - 1\right)$$

## 实验设计

### 数据准备
1. 加载 DOW 30 股票的价格和隐含波动率数据
2. 计算对数收益率和已实现波动率
3. 构建 HAR 特征（日、周、月聚合）
4. 数据集划分：70% 训练，30% 测试

### 邻接矩阵构建
1. **GLASSO 方法**：基于收益率协方差矩阵的稀疏估计
2. **随机基准**：生成随机邻接矩阵作为对照

### 模型训练与评估
1. 对每个模型分别使用 MSE 和 QLIKE 损失函数训练
2. 在测试集上评估预测性能
3. 比较不同图结构和损失函数的效果
4. 分析隐含波动率的贡献

### 评估指标
- **MSE**：均方误差
- **QLIKE**：拟似然损失
- **MAE**：平均绝对误差
- **R²**：决定系数

## 预期成果

1. **图结构效应**：量化 GLASSO 构建的图结构相对于随机图的性能提升
2. **损失函数比较**：确定最适合波动率预测的损失函数
3. **外生变量价值**：评估隐含波动率对预测精度的贡献
4. **模型排序**：建立不同模型的性能排序和适用场景

## 代码实现框架

```python
import os
from os.path import join

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data_dir = "/content"

# 1) 读入 DOW30 return / RV / IV
ret_df = pd.read_csv(join(data_dir, "dow30_daily_returns_2021_2026.csv"),
                     index_col=0, parse_dates=True)
rv_df  = pd.read_csv(join(data_dir, "merged_rv_data_filled.csv"),
                     index_col=0, parse_dates=True)
iv_df  = pd.read_csv(join(data_dir, "merged_iv_data_filled.csv"),
                     index_col=0, parse_dates=True)

# 2) 对齐日期和股票
for df in [ret_df, rv_df, iv_df]:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

common_dates  = ret_df.index.intersection(rv_df.index).intersection(iv_df.index)
common_stocks = ret_df.columns.intersection(rv_df.columns).intersection(iv_df.columns)

ret_df = ret_df.loc[common_dates, common_stocks].sort_index()
rv_df  = rv_df.loc[common_dates, common_stocks].sort_index()
iv_df  = iv_df.loc[common_dates, common_stocks].sort_index()

print("Aligned shapes:")
print("  ret_df:", ret_df.shape)
print("  rv_df :", rv_df.shape)
print("  iv_df :", iv_df.shape)

# 3) 用 return 做 GLASSO adjacency → W_glasso
def build_W_glasso(ret_df, min_samples_factor=2.0):
    subret = ret_df.copy()
    subret = subret.fillna(method='ffill').fillna(method='bfill').fillna(0.0)

    n = subret.shape[1]
    if len(subret) < min_samples_factor * (n**2):
        print(f"Warning: samples={len(subret)} may be small vs n^2.")

    glasso = GraphicalLassoCV(cv=3, max_iter=100).fit(subret.values)
    print(f"GraphicalLasso alpha: {glasso.alpha_:.4f}")
    prec = glasso.precision_
    A = (prec != 0).astype(float)
    np.fill_diagonal(A, 0.0)
    print("Adjacency sparsity:", A.mean())

    d = A.sum(axis=1)
    d_safe = d + 1e-8
    D_m12 = np.diag(1.0 / np.sqrt(d_safe))
    W_arr = D_m12 @ A @ D_m12

    return pd.DataFrame(W_arr, index=subret.columns, columns=subret.columns)

W_glasso = build_W_glasso(ret_df)
print("W_glasso shape:", W_glasso.shape)


# ========== HAR 聚合 ==========
def har_agg_three_scales(df):
    """
    返回 {'d','w','m'} 三个 DataFrame
    d: t-1; w: (t-2..t-5)/4; m: (t-6..t-22)/17
    """
    df_d = df.shift(1)
    df_w = sum(df.shift(k) for k in range(2, 6)) / 4.0
    df_m = sum(df.shift(k) for k in range(6, 23)) / 17.0
    return {'d': df_d, 'w': df_w, 'm': df_m}

rv_har = har_agg_three_scales(rv_df)
iv_har = har_agg_three_scales(iv_df)

# ========== HAR / HAR+IV 面板 ==========
def build_panel(rv_df, rv_har, iv_har=None):
    dates = rv_df.index
    tickers = rv_df.columns
    rows = []
    for date in dates:
        v_t  = rv_df.loc[date]
        rv_d = rv_har['d'].loc[date]
        rv_w = rv_har['w'].loc[date]
        rv_m = rv_har['m'].loc[date]
        if iv_har is not None:
            iv_d = iv_har['d'].loc[date]
            iv_w = iv_har['w'].loc[date]
            iv_m = iv_har['m'].loc[date]
        for tic in tickers:
            row = {
                'Date': date,
                'Ticker': tic,
                'Target': v_t[tic],
                'RV_d': rv_d[tic],
                'RV_w': rv_w[tic],
                'RV_m': rv_m[tic],
            }
            if iv_har is not None:
                row.update({
                    'IV_d': iv_d[tic],
                    'IV_w': iv_w[tic],
                    'IV_m': iv_m[tic],
                })
            rows.append(row)
    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.sort_values(['Date','Ticker']).reset_index(drop=True)
    return df

panel_HAR    = build_panel(rv_df, rv_har, iv_har=None)
panel_HAR_IV = build_panel(rv_df, rv_har, iv_har=iv_har)

print("panel_HAR:", panel_HAR.shape)
print("panel_HAR_IV:", panel_HAR_IV.shape)


def panel_to_date_dict(panel):
    dct = {}
    for d, sub in panel.groupby('Date'):
        dct[d] = sub.copy()
    return dct, sorted(dct.keys())

panel_HAR_dic, dates_HAR       = panel_to_date_dict(panel_HAR)
panel_HAR_IV_dic, dates_HAR_IV = panel_to_date_dict(panel_HAR_IV)

def add_graph_features(dates, panel_dic, W, use_iv=False):
    all_rows = []
    for d in dates:
        df = panel_dic[d].copy()
        tickers = df['Ticker'].values
        W_sub = W.loc[tickers, tickers].values

        rv_mat = df[['RV_d','RV_w','RV_m']].values
        grv = W_sub @ rv_mat
        df['GRV_d'] = grv[:,0]
        df['GRV_w'] = grv[:,1]
        df['GRV_m'] = grv[:,2]

        if use_iv:
            iv_mat = df[['IV_d','IV_w','IV_m']].values
            giv = W_sub @ iv_mat
            df['GIV_d'] = giv[:,0]
            df['GIV_w'] = giv[:,1]
            df['GIV_m'] = giv[:,2]

        all_rows.append(df)
    return pd.concat(all_rows).reset_index(drop=True)

panel_GHAR_glasso   = add_graph_features(dates_HAR,    panel_HAR_dic,    W_glasso, use_iv=False)
panel_GHARIV_glasso = add_graph_features(dates_HAR_IV, panel_HAR_IV_dic, W_glasso, use_iv=True)

print("panel_GHAR_glasso:", panel_GHAR_glasso.shape)
print("panel_GHARIV_glasso:", panel_GHARIV_glasso.shape)


def qlike_loss_np(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y = y_true[mask]
    f = np.maximum(y_pred[mask], eps)
    ratio = y / f
    return np.mean(ratio - np.log(ratio) - 1)

# 简单时间切分：前70%日期训练，后30%测试
all_dates = sorted(rv_df.index)
T = len(all_dates)
split_idx = int(T * 0.7)
train_dates = set(all_dates[:split_idx])
test_dates  = set(all_dates[split_idx:])

def split_panel(panel):
    train = panel[panel['Date'].isin(train_dates)].copy()
    test  = panel[panel['Date'].isin(test_dates)].copy()
    return train, test


def fit_evaluate_linear(panel, feature_cols, loss_in='MSE'):
    """
    panel: 含 Date/Ticker/Target + 特征
    feature_cols: 用作回归的特征名列表
    loss_in: 'MSE' 或 'QL'，表示训练阶段的目标
    """
    train_df, test_df = split_panel(panel)
    X_train = train_df[feature_cols].values
    y_train = train_df['Target'].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df['Target'].values

    model = LinearRegression()
    model.fit(X_train, y_train)
    yhat_train = model.predict(X_train)
    yhat_test  = model.predict(X_test)

    if loss_in.upper() == 'MSE':
        train_loss = mean_squared_error(y_train, yhat_train)
    else:
        train_loss = qlike_loss_np(y_train, yhat_train)

    test_mse = mean_squared_error(y_test, yhat_test)
    test_ql  = qlike_loss_np(y_test, yhat_test)

    return {
        'train_loss': train_loss,
        'test_MSE': test_mse,
        'test_QLIKE': test_ql,
        'model': model,
        'test_df': test_df,
        'y_test': y_test,
        'yhat_test': yhat_test,
    }


results = []

# 1) HAR
har_feats = ['RV_d','RV_w','RV_m']
for loss_in in ['MSE','QL']:
    res = fit_evaluate_linear(panel_HAR, har_feats, loss_in=loss_in)
    results.append({
        'Model': 'HAR',
        'Train_Loss_Type': loss_in,
        'Train_Loss': res['train_loss'],
        'Test_MSE': res['test_MSE'],
        'Test_QLIKE': res['test_QLIKE'],
    })

# 2) HAR+IV
hariv_feats = ['RV_d','RV_w','RV_m','IV_d','IV_w','IV_m']
for loss_in in ['MSE','QL']:
    res = fit_evaluate_linear(panel_HAR_IV, hariv_feats, loss_in=loss_in)
    results.append({
        'Model': 'HAR+IV',
        'Train_Loss_Type': loss_in,
        'Train_Loss': res['train_loss'],
        'Test_MSE': res['test_MSE'],
        'Test_QLIKE': res['test_QLIKE'],
    })

# 3) GHAR (GLASSO W)
ghar_feats = ['RV_d','RV_w','RV_m','GRV_d','GRV_w','GRV_m']
for loss_in in ['MSE','QL']:
    res = fit_evaluate_linear(panel_GHAR_glasso, ghar_feats, loss_in=loss_in)
    results.append({
        'Model': 'GHAR (GLASSO)',
        'Train_Loss_Type': loss_in,
        'Train_Loss': res['train_loss'],
        'Test_MSE': res['test_MSE'],
        'Test_QLIKE': res['test_QLIKE'],
    })

# 4) GHAR+IV (GLASSO W)
ghariv_feats = ['RV_d','RV_w','RV_m','IV_d','IV_w','IV_m',
                'GRV_d','GRV_w','GRV_m','GIV_d','GIV_w','GIV_m']
for loss_in in ['MSE','QL']:
    res = fit_evaluate_linear(panel_GHARIV_glasso, ghariv_feats, loss_in=loss_in)
    results.append({
        'Model': 'GHAR+IV (GLASSO)',
        'Train_Loss_Type': loss_in,
        'Train_Loss': res['train_loss'],
        'Test_MSE': res['test_MSE'],
        'Test_QLIKE': res['test_QLIKE'],
    })

summary = pd.DataFrame(results).sort_values(['Model','Train_Loss_Type'])
print("\n=== GHAR vs GHAR+IV: Test MSE / QLIKE (DOW30, GLASSO W) ===")
print(summary.to_string(index=False))
```

输出结果为
```
=== GHAR vs GHAR+IV: Test MSE / QLIKE (DOW30, GLASSO W) ===
           Model Train_Loss_Type  Train_Loss  Test_MSE  Test_QLIKE
   GHAR (GLASSO)             MSE    0.804931  1.064870    0.000849
   GHAR (GLASSO)              QL    0.000773  1.064870    0.000849
GHAR+IV (GLASSO)             MSE    0.774244  1.004503    0.000817
GHAR+IV (GLASSO)              QL    0.000747  1.004503    0.000817
             HAR             MSE    0.813729  1.118893    0.000868
             HAR              QL    0.000779  1.118893    0.000868
          HAR+IV             MSE    0.780215  1.050391    0.000839
          HAR+IV              QL    0.000755  1.050391    0.000839
```


我是否还需要画什么图片，或者补充什么可以让我的论文更容易被接受，更有价值的信息？？




