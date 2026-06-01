import pandas as pd

# 读取DJIA_ret_FH1.csv文件
df = pd.read_csv('./data/mock/DJIA_ret_FH1.csv', header=None)

"""
GHAR工具函数模块
提供数据加载、预处理、邻接矩阵计算等功能
"""

import numpy as np
import pandas as pd
from sklearn.covariance import GraphicalLassoCV
from sklearn.linear_model import LinearRegression
from os.path import join


# ==================== 数据加载函数 ====================

def load_feature_data(data_path, universe='DJIA'):
    """
    加载波动率特征数据
    
    Args:
        data_path: 数据文件夹路径
        universe: 数据集名称
    Returns:
        feature_df: 波动率特征DataFrame
    """
    feature_df = pd.read_csv(join(data_path, f'{universe}_var_FH1.csv'), index_col=0)
    feature_df.fillna(method="ffill", inplace=True)
    feature_df = feature_df.sort_index(axis=1)
    return feature_df


def load_target_data(data_path, universe='DJIA', horizon=1):
    """
    加载目标波动率数据
    
    Args:
        data_path: 数据文件夹路径
        universe: 数据集名称
        horizon: 预测期数
    Returns:
        vech_df: 目标波动率DataFrame
    """
    var_df = pd.read_csv(join(data_path, f'{universe}_var_FH{horizon}.csv'), index_col=0)
    var_df.fillna(method="ffill", inplace=True)
    var_df = var_df.sort_index(axis=1)
    return var_df


def load_return_data(data_path, universe='DJIA'):
    """
    加载收益率数据（用于计算邻接矩阵）
    
    Args:
        data_path: 数据文件夹路径
        universe: 数据集名称
    Returns:
        ret_df: 收益率DataFrame
    """
    ret_df = pd.read_csv(join(data_path, f'{universe}_ret_FH1.csv'), index_col=0)
    ret_df.fillna(method="ffill", inplace=True)
    ret_df = ret_df.sort_index(axis=1)
    return ret_df


# ==================== HAR特征工程 ====================

def create_har_features(feature_df, target_df, har_lags=[1, 5, 22]):
    """
    创建HAR特征：计算不同滞后期的平均波动率
    
    Args:
        feature_df: 波动率特征DataFrame
        target_df: 目标波动率DataFrame
        har_lags: 滞后期列表 [日频, 周频, 月频]
    
    Returns:
        subdf_dic: 按日期组织的特征字典
        date_l: 日期列表
    """
    subdf_l = []
    
    for target_var in target_df.columns:
        subdf = pd.DataFrame()
        subdf['Target'] = target_df[target_var].copy()
        subdf['Date'] = target_df.index
        subdf['Ticker'] = target_var
        
        # 为每个滞后期创建特征
        indpt_df_l = []
        for lag in har_lags:
            tmp_indpdt_df = 0
            for il in range(1, 1 + lag):
                tmp_indpdt_df += feature_df[target_var].shift(il)
            indpt_df_l.append(tmp_indpdt_df / lag)
        
        # 合并特征
        explain_df = pd.concat(indpt_df_l, axis=1)
        explain_df.columns = ['var+lag%d' % i for i in har_lags]
        
        subdf = pd.merge(subdf, explain_df, left_index=True, right_index=True)
        subdf.replace([np.inf, -np.inf], np.nan, inplace=True)
        subdf.dropna(inplace=True)
        subdf_l.append(subdf)
    
    df = pd.concat(subdf_l)
    df.reset_index(drop=True, inplace=True)
    
    # 按日期组织数据
    date_l = sorted(list(set(df['Date'].tolist())))
    subdf_dic = {date: df[df['Date'] == date] for date in date_l}
    
    print(f'✓ HAR特征创建完成！共 {len(date_l)} 个日期')
    return subdf_dic, date_l


# ==================== 邻接矩阵计算 ====================

def compute_glasso_adjacency(ret_df):
    """
    使用Graphical LASSO计算稀疏邻接矩阵
    
    Args:
        ret_df: 收益率DataFrame
    Returns:
        adj_df: 归一化的邻接矩阵DataFrame
        alpha: 正则化参数
        sparsity: 邻接矩阵稀疏度
    """
    n = ret_df.shape[1]
    tickers = ret_df.columns
    
    # Graphical LASSO估计
    cov = GraphicalLassoCV().fit(ret_df)
    alpha = cov.alpha_
    print(f'  Alpha (正则化参数): {alpha:.4f}')
    
    # 转换为邻接矩阵
    corr = cov.precision_ != 0
    sparsity = corr.mean()
    print(f'  邻接矩阵稀疏度: {sparsity:.3f}')
    
    # 移除自环
    corr_adj = corr - np.identity(n)
    
    # 对称归一化: D^{-1/2} A D^{-1/2}
    d_sqrt_inv = np.diag(np.sqrt(1 / (corr_adj.sum(1) + 1e-8)))
    adj_normalized = np.dot(np.dot(d_sqrt_inv, corr_adj), d_sqrt_inv)
    
    adj_df = pd.DataFrame(adj_normalized, columns=tickers, index=tickers)
    return adj_df, alpha, sparsity


def create_identity_adjacency(tickers):
    """
    创建单位邻接矩阵（对应HAR模型）
    
    Args:
        tickers: 股票代码列表
    Returns:
        adj_df: 单位矩阵DataFrame
    """
    n = len(tickers)
    adj_df = pd.DataFrame(np.identity(n), index=tickers, columns=tickers)
    return adj_df


def apply_graph_aggregation(subdf_dic, date_l, adj_df_l):
    """
    应用图聚合：将邻接矩阵与特征相乘
    
    Args:
        subdf_dic: 特征字典
        date_l: 日期列表
        adj_df_l: 邻接矩阵列表
    Returns:
        df: 聚合后的特征DataFrame
    """
    new_subdf_l = []
    
    for date in date_l:
        subdf = subdf_dic[date]
        tmp_subdf_l = []
        clms = [i for i in subdf.columns if 'lag' in i]
        
        # 对每个邻接矩阵进行聚合
        for k, adj_df in enumerate(adj_df_l):
            # 矩阵乘法: A @ Features
            aggregated = np.dot(adj_df.values, subdf[clms].values)
            tmp_subdf = pd.DataFrame(
                aggregated, 
                columns=['sec' + str(k) + i for i in clms], 
                index=subdf.index
            )
            tmp_subdf_l.append(tmp_subdf)
        
        # 合并原始列和聚合列
        new_subdf = pd.concat([subdf[['Target', 'Date', 'Ticker']]] + tmp_subdf_l, axis=1)
        new_subdf_l.append(new_subdf)
    
    df = pd.concat(new_subdf_l)
    df.reset_index(drop=True, inplace=True)
    print('✓ 图聚合完成！')
    return df


# ==================== 数据分割 ====================

def split_train_test(df, train_start_date, train_end_date, test_start_date, test_end_date):
    """
    分割训练集和测试集
    
    Args:
        df: 完整数据DataFrame
        train_start_date: 训练开始日期
        train_end_date: 训练结束日期
        test_start_date: 测试开始日期
        test_end_date: 测试结束日期
    
    Returns:
        train_df: 训练集
        test_df: 测试集
    """
    train_df = df[(df['Date'] >= train_start_date) & (df['Date'] < train_end_date)]
    test_df = df[(df['Date'] >= test_start_date) & (df['Date'] < test_end_date)]
    
    print(f'训练集大小: {len(train_df)} 样本')
    print(f'测试集大小: {len(test_df)} 样本')
    
    return train_df, test_df


def df_to_arrays(df, feature_cols):
    """
    将DataFrame转换为numpy数组
    
    Args:
        df: DataFrame
        feature_cols: 特征列名列表
    Returns:
        X: 特征矩阵
        y: 目标向量
    """
    X = df[feature_cols].values
    y = df['Target'].values
    return X, y


# ==================== 模型训练与预测 ====================

def train_har_model(train_X, train_y):
    """
    训练HAR/GHAR线性回归模型
    
    Args:
        train_X: 训练特征
        train_y: 训练目标
    Returns:
        model: 训练好的模型
    """
    model = LinearRegression()
    model.fit(train_X, train_y)
    
    print(f'✓ 模型训练完成')
    print(f'  系数: {model.coef_[:5]}...')  # 只显示前5个
    print(f'  截距: {model.intercept_:.6f}')
    
    return model


def predict_and_adjust(model, test_X, test_df, train_df):
    """
    预测并调整负值预测
    
    Args:
        model: 训练好的模型
        test_X: 测试特征
        test_df: 测试DataFrame
        train_df: 训练DataFrame（用于获取最小值）
    Returns:
        pred_df: 预测结果DataFrame
    """
    # 预测
    predictions = model.predict(test_X)
    
    # 创建预测DataFrame
    pred_df = test_df[['Ticker', 'Date']].copy()
    pred_df['Prediction'] = predictions
    pred_df = pred_df.pivot(index='Date', columns='Ticker', values='Prediction')
    
    print(f'调整前最小预测值: {pred_df.min().min():.6f}')
    
    # 调整负值预测
    for ticker in pred_df.columns:
        ticker_train = train_df[train_df['Ticker'] == ticker]['Target']
        min_val = ticker_train.min()
        pred_df[ticker] = pred_df[ticker].clip(lower=min_val)
    
    print(f'调整后最小预测值: {pred_df.min().min():.6f}')
    
    return pred_df


# ==================== 评估函数 ====================

def evaluate_predictions(pred_df, true_df):
    """
    评估预测结果
    
    Args:
        pred_df: 预测DataFrame
        true_df: 真实值DataFrame
    Returns:
        metrics: 评估指标字典
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    
    # 对齐数据
    common_dates = pred_df.index.intersection(true_df.index)
    common_tickers = pred_df.columns.intersection(true_df.columns)
    
    pred_values = pred_df.loc[common_dates, common_tickers].values.flatten()
    true_values = true_df.loc[common_dates, common_tickers].values.flatten()
    
    # 计算指标
    mse = mean_squared_error(true_values, pred_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, pred_values)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'N_samples': len(pred_values)
    }
    
    print('\n📊 评估结果:')
    for key, val in metrics.items():
        print(f'  {key}: {val:.6f}')
    
    return metrics



import sys
sys.path.append('./GNNHAR')  # 添加路径
import pandas as pd
import numpy as np

# 设置数据路径
DATA_PATH = './data/mock'
feature_df = load_feature_data(DATA_PATH)
target_df = load_target_data(DATA_PATH, horizon=1)
ret_df = load_return_data(DATA_PATH)
subdf_dic, date_l = create_har_features(feature_df, target_df, har_lags=[1, 5, 22])
def create_predefined_adjacency(stock_order=['IBM', 'JPM', 'GS', 'CVX', 'AXP', 'BA']):
    """
    创建预定义的邻接矩阵（基于模拟网络结构）
    
    网络结构（根据图片）:
    - IBM (0-hop): 连接到 JPM, GS, BA
    - JPM (1-hop): 连接到 IBM, GS, CVX
    - GS (1-hop): 连接到 IBM, JPM, BA
    - CVX (2-hop): 连接到 JPM, AXP
    - AXP (2-hop): 连接到 CVX, BA
    - BA (2-hop): 连接到 IBM, GS, AXP
    
    Args:
        stock_order: 股票顺序列表
    Returns:
        W: 归一化的邻接矩阵 DataFrame
        A: 原始邻接矩阵 DataFrame
    """
    # 邻接矩阵 A (根据图片的连接关系)
    A = np.array([
        [0, 1, 1, 0, 0, 1],  # IBM -> JPM, GS, BA
        [1, 0, 1, 1, 0, 0],  # JPM -> IBM, GS, CVX
        [1, 1, 0, 0, 0, 1],  # GS -> IBM, JPM, BA
        [0, 1, 0, 0, 1, 0],  # CVX -> JPM, AXP
        [0, 0, 0, 1, 0, 1],  # AXP -> CVX, BA
        [1, 0, 1, 0, 1, 0]   # BA -> IBM, GS, AXP
    ])
    
    # 度矩阵 O = diag(3, 3, 3, 2, 2, 3)
    degrees = A.sum(axis=1)
    O = np.diag(degrees)
    
    # 对称归一化: W = O^{-1/2} A O^{-1/2}
    O_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
    W = O_inv_sqrt @ A @ O_inv_sqrt
    
    # 转换为 DataFrame
    A_df = pd.DataFrame(A, index=stock_order, columns=stock_order)
    W_df = pd.DataFrame(W, index=stock_order, columns=stock_order)
    
    print("原始邻接矩阵 A:")
    print(A_df)
    print(f"\n度矩阵 O: {degrees}")
    print("\n归一化邻接矩阵 W = O^{-1/2} A O^{-1/2}:")
    print(W_df.round(4))
    
    return W_df, A_df


def compute_qlike_loss(y_true, y_pred):
    """
    计算 QLIKE 损失函数
    
    QL = (RV / RV_pred) - log(RV / RV_pred) - 1
    
    Args:
        y_true: 真实值
        y_pred: 预测值
    Returns:
        qlike: QLIKE损失
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)


import sys
sys.path.append('./GNNHAR')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 设置数据路径
DATA_PATH = './data/mock'

# 确保股票顺序与邻接矩阵一致
STOCK_ORDER = ['IBM', 'JPM', 'GS', 'CVX', 'AXP', 'BA']

feature_df = load_feature_data(DATA_PATH)
target_df = load_target_data(DATA_PATH, horizon=1)
ret_df = load_return_data(DATA_PATH)

# 确保列顺序一致
feature_df = feature_df[STOCK_ORDER]
target_df = target_df[STOCK_ORDER]
ret_df = ret_df[STOCK_ORDER]
W_true, A_true = create_predefined_adjacency(STOCK_ORDER)
subdf_dic, date_l = create_har_features(feature_df, target_df, har_lags=[1, 5, 22])
train_ret = ret_df.iloc[:600]
W_glasso, alpha, sparsity = compute_glasso_adjacency(train_ret)
train_end_idx = 600
test_end_idx = 700

split_date = date_l[min(train_end_idx, len(date_l)-100)]
test_end_date = date_l[min(test_end_idx, len(date_l)-1)]

# Cell 6: 训练和评估函数
def train_and_evaluate_model(subdf_dic, date_l, adj_list, model_name,
                              split_date, test_end_date):
    """
    训练并评估模型
    
    Args:
        subdf_dic: HAR特征字典
        date_l: 日期列表
        adj_list: 邻接矩阵列表
        model_name: 模型名称
        split_date: 训练/测试分割日期
        test_end_date: 测试结束日期
    
    Returns:
        results: 包含预测、真实值和评估指标的字典
    """
    print(f"\n{'='*80}")
    print(f"训练 {model_name} 模型")
    print(f"{'='*80}")
    
    # 应用图聚合
    df_agg = apply_graph_aggregation(subdf_dic, date_l, adj_list)
    
    # 分割数据
    train_df, test_df = split_train_test(
        df_agg, date_l[0], split_date, split_date, test_end_date
    )
    
    # 提取特征
    feature_cols = [col for col in train_df.columns if 'lag' in col]
    print(f"使用 {len(feature_cols)} 个特征")
    
    train_X, train_y = df_to_arrays(train_df, feature_cols)
    test_X, test_y = df_to_arrays(test_df, feature_cols)
    
    # 训练模型
    model = train_har_model(train_X, train_y)
    
    # 预测
    pred_df = predict_and_adjust(model, test_X, test_df, train_df)
    true_df = test_df.pivot(index='Date', columns='Ticker', values='Target')
    
    # 计算评估指标
    pred_values = pred_df.values.flatten()
    true_values = true_df.loc[pred_df.index, pred_df.columns].values.flatten()
    
    mse = mean_squared_error(true_values, pred_values)
    qlike = compute_qlike_loss(true_values, pred_values)
    
    print(f"\n📊 {model_name} 评估结果:")
    print(f"  MSE:   {mse:.6f}")
    print(f"  RMSE:  {np.sqrt(mse):.6f}")
    print(f"  QLIKE: {qlike:.6f}")
    
    # 返回结果
    results = {
        'model_name': model_name,
        'model': model,
        'pred_df': pred_df,
        'true_df': true_df,
        'mse': mse,
        'qlike': qlike,
        'coefficients': model.coef_,
        'intercept': model.intercept_
    }
    
    return results

# Cell 7: 训练 HAR 模型（baseline）
print("\n" + "="*80)
print("模型1: HAR (无邻接矩阵)")
print("="*80)

# HAR使用单位矩阵
adj_identity = create_identity_adjacency(STOCK_ORDER)
results_har = train_and_evaluate_model(
    subdf_dic, date_l, [adj_identity], 'HAR',
    split_date, test_end_date
)

# Cell 8: 训练 GHAR 模型（使用真实邻接矩阵）
print("\n" + "="*80)
print("模型2: GHAR (真实邻接矩阵)")
print("="*80)

# GHAR使用单位矩阵 + 真实邻接矩阵
results_ghar_true = train_and_evaluate_model(
    subdf_dic, date_l, [adj_identity, W_true], 'GHAR (True Adjacency)',
    split_date, test_end_date
)

# Cell 9: 训练 GHAR 模型（使用错误的GLASSO邻接矩阵）
print("\n" + "="*80)
print("模型3: GHAR (GLASSO邻接矩阵 - 错误)")
print("="*80)

results_ghar_glasso = train_and_evaluate_model(
    subdf_dic, date_l, [adj_identity, W_glasso], 'GHAR (GLASSO Adjacency)',
    split_date, test_end_date
)

# 最简单版本
W_random = pd.DataFrame(
    np.random.rand(6, 6),
    index=['IBM', 'JPM', 'GS', 'CVX', 'AXP', 'BA'],
    columns=['IBM', 'JPM', 'GS', 'CVX', 'AXP', 'BA']
)
W_random = (W_random + W_random.T) / 2  # 对称化
np.fill_diagonal(W_random.values, 0)    # 对角线为0

print(W_random)

print("\n" + "="*80)
print("模型3: GHAR (随机邻接矩阵)")
print("="*80)

# GHAR使用单位矩阵 + 真实邻接矩阵
results_ghar_random = train_and_evaluate_model(
    subdf_dic, date_l, [adj_identity, W_random], 'GHAR (Random Adjacency)',
    split_date, test_end_date
)






# Cell 10: 对比结果
print("\n" + "="*80)
print("模型性能对比")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': ['HAR', 'GHAR (True)', 'GHAR (GLASSO)', 'GHAR (Random)'],
    'MSE': [
        results_har['mse'],
        results_ghar_true['mse'],
        results_ghar_glasso['mse'],
        results_ghar_random['mse']
    ],
    'QLIKE': [
        results_har['qlike'],
        results_ghar_true['qlike'],
        results_ghar_glasso['qlike'],
        results_ghar_random['qlike']
    ]
})

comparison_df['MSE_Improvement_vs_HAR(%)'] = (
    (results_har['mse'] - comparison_df['MSE']) / results_har['mse'] * 100
)
comparison_df['QLIKE_Improvement_vs_HAR(%)'] = (
    (results_har['qlike'] - comparison_df['QLIKE']) / results_har['qlike'] * 100
)

print(comparison_df.to_string(index=False))

print("\n关键发现:")
print(f"1. GHAR(真实邻接矩阵) vs HAR:")
print(f"   - MSE降低: {comparison_df.iloc[1]['MSE_Improvement_vs_HAR(%)']:.2f}%")
print(f"   - QLIKE降低: {comparison_df.iloc[1]['QLIKE_Improvement_vs_HAR(%)']:.2f}%")

print(f"\n2. GHAR(GLASSO) vs HAR:")
print(f"   - MSE降低: {comparison_df.iloc[2]['MSE_Improvement_vs_HAR(%)']:.2f}%")
print(f"   - QLIKE降低: {comparison_df.iloc[2]['QLIKE_Improvement_vs_HAR(%)']:.2f}%")

print(f"\n3. GHAR(随机邻接矩阵) vs HAR:")
print(f"   - MSE降低: {comparison_df.iloc[3]['MSE_Improvement_vs_HAR(%)']:.2f}%")
print(f"   - QLIKE降低: {comparison_df.iloc[3]['QLIKE_Improvement_vs_HAR(%)']:.2f}%")

if comparison_df.iloc[2]['MSE_Improvement_vs_HAR(%)'] < 0:
    print("   ⚠️ 使用GLASSO邻接矩阵导致性能下降！")

if comparison_df.iloc[3]['MSE_Improvement_vs_HAR(%)'] < 0:
    print("   ⚠️ 使用随机邻接矩阵导致性能下降！")








# ==================== Train GNNHAR Models with Random Adjacency ====================

print("\n" + "="*80)
print("Training GNNHAR Models with Random Adjacency Matrix")
print("="*80)

# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Use CPU for this demo
device = torch.device('cpu')

# ==================== GNN Model Definitions ====================

class GraphConvLayer(nn.Module):
    """Graph Convolution Layer: H_out = A @ (H_in @ W) + b"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
            nn.init.ones_(self.bias)
        else:
            self.bias = None
    
    def forward(self, node_feature, adj):
        # node_feature: (batch_size, N, in_features)
        # adj: (N, N)
        h = torch.matmul(node_feature, self.weight)  # (batch, N, out_features)
        output = torch.matmul(adj, h)  # (N, N) @ (batch, N, out_features)
        if self.bias is not None:
            return output + self.bias
        return output


class GNNHAR1L(nn.Module):
    """1-layer GNNHAR model"""
    def __init__(self, n_hid=9):
        super(GNNHAR1L, self).__init__()
        
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)  # Linear HAR component
        
        H2 = self.gcn1(node_feat, adj)  # Graph convolution
        H2 = self.relu(H2)
        H2 = self.mlp1(H2)
        
        res = H1 + H2
        res = self.relu(res)
        
        return res.squeeze(-1)


class GNNHAR2L(nn.Module):
    """2-layer GNNHAR model"""
    def __init__(self, n_hid=9):
        super(GNNHAR2L, self).__init__()
        
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.gcn2 = GraphConvLayer(n_hid, n_hid, bias=False)
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)
        
        # 2 layers of GCN
        H2 = self.relu(self.gcn1(node_feat, adj))
        H2 = self.relu(self.gcn2(H2, adj))
        H2 = self.mlp1(H2)
        
        res = H1 + H2
        res = self.relu(res)
        
        return res.squeeze(-1)


class GNNHAR3L(nn.Module):
    """3-layer GNNHAR model"""
    def __init__(self, n_hid=9):
        super(GNNHAR3L, self).__init__()
        
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.gcn2 = GraphConvLayer(n_hid, n_hid, bias=False)
        self.gcn3 = GraphConvLayer(n_hid, n_hid, bias=False)
        self.mlp1 = nn.Linear(n_hid, 1, bias=False)
        self.relu = nn.ReLU()
    
    def forward(self, node_feat, adj):
        H1 = self.linear1(node_feat)
        
        # 3 layers of GCN
        H2 = self.relu(self.gcn1(node_feat, adj))
        H2 = self.relu(self.gcn2(H2, adj))
        H2 = self.relu(self.gcn3(H2, adj))
        H2 = self.mlp1(H2)
        
        res = H1 + H2
        res = self.relu(res)
        
        return res.squeeze(-1)


# ==================== Training Function with QLIKE Loss ====================

def train_gnnhar_model_qlike(X_train, y_train, X_test, y_test, adj_matrix, 
                       model_class, model_name, n_hid=9, n_epochs=1000, 
                       lr=0.001, batch_size=32):
    """
    Train a GNNHAR model with QLIKE loss
    
    Args:
        X_train: (T_train, N, 3) training features
        y_train: (T_train, N) training targets
        X_test: (T_test, N, 3) test features
        y_test: (T_test, N) test targets
        adj_matrix: (N, N) adjacency matrix
        model_class: Model class (GNNHAR1L, GNNHAR2L, or GNNHAR3L)
        model_name: Name for display
        n_hid: Hidden dimension
        n_epochs: Training epochs
        lr: Learning rate
        batch_size: Batch size
    
    Returns:
        results: Dictionary with predictions and metrics
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name} with QLIKE Loss")
    print(f"{'='*80}")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    adj_t = torch.FloatTensor(adj_matrix).to(device)
    
    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = model_class(n_hid=n_hid).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # QLIKE loss function
    def QLIKE_Loss(y_true, y_pred):
        """
        QLIKE loss: mean(y_true/y_pred - log(y_true/y_pred) - 1)
        """
        # Add small epsilon for numerical stability
        y_pred_stable = y_pred + 1e-4
        y_true_stable = y_true + 1e-4
        
        ratio = y_true_stable / y_pred_stable
        loss = torch.mean(ratio - torch.log(ratio) - 1)
        return loss
    
    # Training loop
    model.train()
    train_losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = []
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch_X, adj_t)
            loss = QLIKE_Loss(batch_y, pred)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss.append(loss.item())
        
        avg_loss = np.mean(epoch_loss)
        train_losses.append(avg_loss)
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch}/{n_epochs}, QLIKE Loss: {avg_loss:.6f}")
    
    print(f"  Final training loss: {train_losses[-1]:.6f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test_t, adj_t).cpu().numpy()
        y_test_np = y_test_t.cpu().numpy()
    
    # Calculate both MSE and QLIKE metrics
    def QLIKE_eval(y_true, y_pred):
        """QLIKE evaluation metric"""
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)
    
    mse = mean_squared_error(y_test_np.flatten(), pred_test.flatten())
    qlike = QLIKE_eval(y_test_np.flatten(), pred_test.flatten())
    
    print(f"\n📊 {model_name} Results:")
    print(f"  MSE:   {mse:.6f}")
    print(f"  RMSE:  {np.sqrt(mse):.6f}")
    print(f"  QLIKE: {qlike:.6f}")
    
    # Extract parameters
    with torch.no_grad():
        linear_weight = model.linear1.weight.cpu().numpy().flatten()
        linear_bias = model.linear1.bias.cpu().numpy()[0]
    
    print(f"\n  Linear HAR coefficients (α, β, γ):")
    print(f"    α (daily):   {linear_weight[0]:.6f}")
    print(f"    β (weekly):  {linear_weight[1]:.6f}")
    print(f"    γ (monthly): {linear_weight[2]:.6f}")
    print(f"    Intercept:   {linear_bias:.6f}")
    
    results = {
        'model_name': model_name,
        'model': model,
        'pred': pred_test,
        'true': y_test_np,
        'mse': mse,
        'qlike': qlike,
        'train_losses': train_losses,
        'linear_coef': linear_weight,
        'linear_bias': linear_bias,
        'loss_type': 'QLIKE'
    }
    
    return results


# ==================== Prepare Data for GNNHAR ====================

print("\n" + "="*80)
print("Preparing data for GNNHAR models")
print("="*80)

# Extract features for each time step
def extract_features_for_gnn(dates, subdf_dic, stock_order):
    """Extract features in shape (T, N, 3) for GNNHAR"""
    T = len(dates)
    N = len(stock_order)
    X = np.zeros((T, N, 3))
    y = np.zeros((T, N))
    
    for t, date in enumerate(dates):
        subdf = subdf_dic[date]
        for i, ticker in enumerate(stock_order):
            ticker_data = subdf[subdf['Ticker'] == ticker]
            if len(ticker_data) > 0:
                X[t, i, 0] = ticker_data['var+lag1'].values[0]
                X[t, i, 1] = ticker_data['var+lag5'].values[0]
                X[t, i, 2] = ticker_data['var+lag22'].values[0]
                y[t, i] = ticker_data['Target'].values[0]
    
    return X, y

train_dates = [d for d in date_l if date_l[0] <= d < split_date]
test_dates = [d for d in date_l if split_date <= d < test_end_date]

X_train_gnn, y_train_gnn = extract_features_for_gnn(train_dates, subdf_dic, STOCK_ORDER)
X_test_gnn, y_test_gnn = extract_features_for_gnn(test_dates, subdf_dic, STOCK_ORDER)

print(f"Training data shape: X={X_train_gnn.shape}, y={y_train_gnn.shape}")
print(f"Test data shape: X={X_test_gnn.shape}, y={y_test_gnn.shape}")


# ==================== Train GNNHAR Models with Random Adjacency ====================

print("\n" + "="*80)
print("Training GNNHAR1L with Random Adjacency")
print("="*80)

results_gnnhar1l_random = train_gnnhar_model_qlike(
    X_train_gnn, y_train_gnn, X_test_gnn, y_test_gnn,
    W_random.values, GNNHAR1L, 'GNNHAR1L (Random, QLIKE)',
    n_hid=9, n_epochs=1000, lr=0.001, batch_size=32
)

print("\n" + "="*80)
print("Training GNNHAR2L with Random Adjacency")
print("="*80)

results_gnnhar2l_random = train_gnnhar_model_qlike(
    X_train_gnn, y_train_gnn, X_test_gnn, y_test_gnn,
    W_random.values, GNNHAR2L, 'GNNHAR2L (Random, QLIKE)',
    n_hid=9, n_epochs=1000, lr=0.001, batch_size=32
)

print("\n" + "="*80)
print("Training GNNHAR3L with Random Adjacency")
print("="*80)

results_gnnhar3l_random = train_gnnhar_model_qlike(
    X_train_gnn, y_train_gnn, X_test_gnn, y_test_gnn,
    W_random.values, GNNHAR3L, 'GNNHAR3L (Random, QLIKE)',
    n_hid=9, n_epochs=1000, lr=0.001, batch_size=32
)


# ==================== Comprehensive Comparison ====================

print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON")
print("="*80)

# Compile all results
all_results_final = pd.DataFrame({
    'Model': [
        'HAR',
        'GHAR (True)',
        'GHAR (GLASSO)',
        'GHAR (Random)',
        'GNNHAR1L (Random)',
        'GNNHAR2L (Random)',
        'GNNHAR3L (Random)'
    ],
    'Type': [
        'Linear', 'Linear', 'Linear', 'Linear',
        'GNN-QLIKE', 'GNN-QLIKE', 'GNN-QLIKE'
    ],
    'Adjacency': [
        'Identity', 'True', 'GLASSO', 'Random',
        'Random', 'Random', 'Random'
    ],
    'MSE': [
        results_har['mse'],
        results_ghar_true['mse'],
        results_ghar_glasso['mse'],
        results_ghar_random['mse'],
        results_gnnhar1l_random['mse'],
        results_gnnhar2l_random['mse'],
        results_gnnhar3l_random['mse']
    ],
    'QLIKE': [
        results_har['qlike'],
        results_ghar_true['qlike'],
        results_ghar_glasso['qlike'],
        results_ghar_random['qlike'],
        results_gnnhar1l_random['qlike'],
        results_gnnhar2l_random['qlike'],
        results_gnnhar3l_random['qlike']
    ]
})

# Calculate improvements
baseline_mse = results_har['mse']
baseline_qlike = results_har['qlike']

all_results_final['MSE_Improvement(%)'] = (baseline_mse - all_results_final['MSE']) / baseline_mse * 100
all_results_final['QLIKE_Improvement(%)'] = (baseline_qlike - all_results_final['QLIKE']) / baseline_qlike * 100

print("\n" + all_results_final.to_string(index=False))

# Find best models
best_mse_idx = all_results_final['MSE'].idxmin()
best_qlike_idx = all_results_final['QLIKE'].idxmin()

print(f"\n{'='*80}")
print("BEST MODELS")
print("="*80)
print(f"\n🏆 Best Model (MSE): {all_results_final.loc[best_mse_idx, 'Model']}")
print(f"   MSE = {all_results_final.loc[best_mse_idx, 'MSE']:.6f}")
print(f"   QLIKE = {all_results_final.loc[best_mse_idx, 'QLIKE']:.6f}")
print(f"   Improvement over HAR: {all_results_final.loc[best_mse_idx, 'MSE_Improvement(%)']:.2f}%")

print(f"\n🏆 Best Model (QLIKE): {all_results_final.loc[best_qlike_idx, 'Model']}")
print(f"   MSE = {all_results_final.loc[best_qlike_idx, 'MSE']:.6f}")
print(f"   QLIKE = {all_results_final.loc[best_qlike_idx, 'QLIKE']:.6f}")
print(f"   Improvement over HAR: {all_results_final.loc[best_qlike_idx, 'QLIKE_Improvement(%)']:.2f}%")

# Key insights
print(f"\n{'='*80}")
print("KEY INSIGHTS")
print("="*80)

print(f"\n1. Linear GHAR Models:")
print(f"   GHAR (True):    MSE={results_ghar_true['mse']:.6f}, QLIKE={results_ghar_true['qlike']:.6f}")
print(f"   GHAR (GLASSO):  MSE={results_ghar_glasso['mse']:.6f}, QLIKE={results_ghar_glasso['qlike']:.6f}")
print(f"   GHAR (Random):  MSE={results_ghar_random['mse']:.6f}, QLIKE={results_ghar_random['qlike']:.6f}")

print(f"\n2. Non-linear GNNHAR Models (Random Adjacency):")
print(f"   GNNHAR1L: MSE={results_gnnhar1l_random['mse']:.6f}, QLIKE={results_gnnhar1l_random['qlike']:.6f}")
print(f"   GNNHAR2L: MSE={results_gnnhar2l_random['mse']:.6f}, QLIKE={results_gnnhar2l_random['qlike']:.6f}")
print(f"   GNNHAR3L: MSE={results_gnnhar3l_random['mse']:.6f}, QLIKE={results_gnnhar3l_random['qlike']:.6f}")

print(f"\n3. Random Adjacency Performance:")
ghar_random_rank = (all_results_final['MSE'] < results_ghar_random['mse']).sum() + 1
gnnhar_random_best = min(results_gnnhar1l_random['mse'], results_gnnhar2l_random['mse'], results_gnnhar3l_random['mse'])
print(f"   GHAR (Random) rank: {ghar_random_rank}/{len(all_results_final)}")
print(f"   Best GNNHAR (Random): MSE={gnnhar_random_best:.6f}")
if gnnhar_random_best < results_ghar_random['mse']:
    print(f"   → Non-linearity helps even with random adjacency")
else:
    print(f"   → Linear GHAR sufficient for random adjacency")

print(f"\n4. Comparison with True Adjacency:")
improvement = (results_ghar_random['mse'] - results_ghar_true['mse']) / results_ghar_true['mse'] * 100
print(f"   True vs Random GHAR: {improvement:+.2f}% (negative = random worse)")
if improvement > 0:
    print(f"   ⚠ Random adjacency outperforms true adjacency!")
else:
    print(f"   ✓ True adjacency better than random as expected")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)


# ==================== Additional Experiment: MSE-trained GNNHAR with Random Adjacency ====================

def train_gnnhar_model_mse(X_train, y_train, X_test, y_test, adj_matrix, 
                       model_class, model_name, n_hid=9, n_epochs=1000, 
                       lr=0.001, batch_size=32):
    """
    Train a GNNHAR model with MSE loss
    
    Args:
        X_train: (T_train, N, 3) training features
        y_train: (T_train, N) training targets
        X_test: (T_test, N, 3) test features
        y_test: (T_test, N) test targets
        adj_matrix: (N, N) adjacency matrix
        model_class: GNNHAR1L, GNNHAR2L, or GNNHAR3L
        model_name: Model name for display
        n_hid: Hidden dimension
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        
    Returns:
        results: Dictionary with MSE, QLIKE, and model parameters
    """
    print("\n" + "="*80)
    print(f"Training {model_name} with MSE Loss")
    print("="*80)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    adj_tensor = torch.FloatTensor(adj_matrix.values)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = model_class(n_hid=n_hid)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # MSE Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch_X, adj_tensor)
            
            # Compute MSE loss
            loss = criterion(pred, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Print progress
        if epoch % 200 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"  Epoch {epoch}/{n_epochs}, MSE Loss: {avg_loss:.6f}")
    
    print(f"  Final training loss: {epoch_loss / len(train_loader):.6f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Test predictions
        test_pred = model(X_test_tensor, adj_tensor).numpy()
        
        # Ensure positive predictions
        test_pred = np.maximum(test_pred, 1e-6)
        
        # Compute MSE
        mse = np.mean((y_test - test_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Compute QLIKE
        qlike_values = y_test / test_pred - np.log(y_test / test_pred) - 1
        qlike = np.mean(qlike_values)
    
    print(f"\n📊 {model_name} Results:")
    print(f"  MSE:   {mse:.6f}")
    print(f"  RMSE:  {rmse:.6f}")
    print(f"  QLIKE: {qlike:.6f}")
    
    # Extract linear HAR coefficients
    with torch.no_grad():
        if hasattr(model, 'har_weight'):
            har_coef = model.har_weight.numpy()
            har_intercept = model.har_intercept.numpy()
            print(f"\n  Linear HAR coefficients (α, β, γ):")
            print(f"    α (daily):   {har_coef[0]:.6f}")
            print(f"    β (weekly):  {har_coef[1]:.6f}")
            print(f"    γ (monthly): {har_coef[2]:.6f}")
            print(f"    Intercept:   {har_intercept:.6f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'qlike': qlike,
        'predictions': test_pred
    }


print("\n\n" + "="*80)
print("ADDITIONAL EXPERIMENT: MSE-Trained GNNHAR with Random Adjacency")
print("="*80)

# Train GNNHAR models with Random adjacency using MSE loss
print("\n" + "="*80)
print("Training GNNHAR Models with Random Adjacency (MSE Loss)")
print("="*80)

# Use the same data preparation as before
X_train_gnn, y_train_gnn = extract_features_for_gnn(
    train_dates, subdf_dic, STOCK_ORDER
)
X_test_gnn, y_test_gnn = extract_features_for_gnn(
    test_dates, subdf_dic, STOCK_ORDER
)

print(f"Training data shape: X={X_train_gnn.shape}, y={y_train_gnn.shape}")
print(f"Test data shape: X={X_test_gnn.shape}, y={y_test_gnn.shape}")

# Train GNNHAR1L with Random adjacency (MSE)
print("\n" + "="*80)
print("Training GNNHAR1L with Random Adjacency (MSE)")
print("="*80)
results_gnnhar1l_random_mse = train_gnnhar_model_mse(
    X_train_gnn, y_train_gnn, X_test_gnn, y_test_gnn,
    W_random, GNNHAR1L, "GNNHAR1L (Random, MSE)",
    n_hid=9, n_epochs=1000, lr=0.001, batch_size=32
)

# Train GNNHAR2L with Random adjacency (MSE)
print("\n" + "="*80)
print("Training GNNHAR2L with Random Adjacency (MSE)")
print("="*80)
results_gnnhar2l_random_mse = train_gnnhar_model_mse(
    X_train_gnn, y_train_gnn, X_test_gnn, y_test_gnn,
    W_random, GNNHAR2L, "GNNHAR2L (Random, MSE)",
    n_hid=9, n_epochs=1000, lr=0.001, batch_size=32
)

# Train GNNHAR3L with Random adjacency (MSE)
print("\n" + "="*80)
print("Training GNNHAR3L with Random Adjacency (MSE)")
print("="*80)
results_gnnhar3l_random_mse = train_gnnhar_model_mse(
    X_train_gnn, y_train_gnn, X_test_gnn, y_test_gnn,
    W_random, GNNHAR3L, "GNNHAR3L (Random, MSE)",
    n_hid=9, n_epochs=1000, lr=0.001, batch_size=32
)

# ==================== Comprehensive Comparison Including MSE-trained ====================

print("\n\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON (INCLUDING MSE-TRAINED RANDOM GNNHAR)")
print("="*80)

# Compile all results
all_results_extended = {
    'Model': [],
    'Type': [],
    'Adjacency': [],
    'Loss_Function': [],
    'MSE': [],
    'QLIKE': [],
    'MSE_Improvement(%)': [],
    'QLIKE_Improvement(%)': []
}

# Add baseline and previous results
models_data = [
    ('HAR (Baseline)', 'Linear', 'Identity', 'MSE', results_har['mse'], results_har['qlike']),
    ('GHAR (True)', 'Linear', 'True', 'MSE', results_ghar_true['mse'], results_ghar_true['qlike']),
    ('GHAR (GLASSO)', 'Linear', 'GLASSO', 'MSE', results_ghar_glasso['mse'], results_ghar_glasso['qlike']),
    ('GHAR (Random)', 'Linear', 'Random', 'MSE', results_ghar_random['mse'], results_ghar_random['qlike']),
    ('GNNHAR1L (Random, QLIKE)', 'GNN', 'Random', 'QLIKE', results_gnnhar1l_random['mse'], results_gnnhar1l_random['qlike']),
    ('GNNHAR2L (Random, QLIKE)', 'GNN', 'Random', 'QLIKE', results_gnnhar2l_random['mse'], results_gnnhar2l_random['qlike']),
    ('GNNHAR3L (Random, QLIKE)', 'GNN', 'Random', 'QLIKE', results_gnnhar3l_random['mse'], results_gnnhar3l_random['qlike']),
    ('GNNHAR1L (Random, MSE)', 'GNN', 'Random', 'MSE', results_gnnhar1l_random_mse['mse'], results_gnnhar1l_random_mse['qlike']),
    ('GNNHAR2L (Random, MSE)', 'GNN', 'Random', 'MSE', results_gnnhar2l_random_mse['mse'], results_gnnhar2l_random_mse['qlike']),
    ('GNNHAR3L (Random, MSE)', 'GNN', 'Random', 'MSE', results_gnnhar3l_random_mse['mse'], results_gnnhar3l_random_mse['qlike']),
]

baseline_mse = results_har['mse']
baseline_qlike = results_har['qlike']

for model_name, model_type, adjacency, loss_fn, mse, qlike in models_data:
    all_results_extended['Model'].append(model_name)
    all_results_extended['Type'].append(model_type)
    all_results_extended['Adjacency'].append(adjacency)
    all_results_extended['Loss_Function'].append(loss_fn)
    all_results_extended['MSE'].append(mse)
    all_results_extended['QLIKE'].append(qlike)
    
    mse_improvement = (baseline_mse - mse) / baseline_mse * 100
    qlike_improvement = (baseline_qlike - qlike) / baseline_qlike * 100
    
    all_results_extended['MSE_Improvement(%)'].append(mse_improvement)
    all_results_extended['QLIKE_Improvement(%)'].append(qlike_improvement)

# Create DataFrame
df_extended = pd.DataFrame(all_results_extended)

# Print table
print("\n" + "="*80)
print("COMPREHENSIVE MODEL COMPARISON (MSE vs QLIKE Loss)")
print("="*80)
print(df_extended.to_string(index=False))

# Identify best models
print("\n" + "="*80)
print("BEST MODELS")
print("="*80)

best_mse_idx = df_extended['MSE'].idxmin()
best_qlike_idx = df_extended['QLIKE'].idxmin()

print(f"\n🏆 Best Model (MSE): {df_extended.loc[best_mse_idx, 'Model']}")
print(f"   MSE = {df_extended.loc[best_mse_idx, 'MSE']:.6f}")
print(f"   QLIKE = {df_extended.loc[best_mse_idx, 'QLIKE']:.6f}")
print(f"   Improvement over HAR: {df_extended.loc[best_mse_idx, 'MSE_Improvement(%)']:.2f}%")

print(f"\n🏆 Best Model (QLIKE): {df_extended.loc[best_qlike_idx, 'Model']}")
print(f"   MSE = {df_extended.loc[best_qlike_idx, 'MSE']:.6f}")
print(f"   QLIKE = {df_extended.loc[best_qlike_idx, 'QLIKE']:.6f}")
print(f"   Improvement over HAR: {df_extended.loc[best_qlike_idx, 'QLIKE_Improvement(%)']:.2f}%")

# Compare MSE vs QLIKE training
print("\n" + "="*80)
print("KEY INSIGHTS: MSE vs QLIKE Training with Random Adjacency")
print("="*80)

print("\n1. GNNHAR Models with Random Adjacency - QLIKE Loss:")
print(f"   GNNHAR1L: MSE={results_gnnhar1l_random['mse']:.6f}, QLIKE={results_gnnhar1l_random['qlike']:.6f}")
print(f"   GNNHAR2L: MSE={results_gnnhar2l_random['mse']:.6f}, QLIKE={results_gnnhar2l_random['qlike']:.6f}")
print(f"   GNNHAR3L: MSE={results_gnnhar3l_random['mse']:.6f}, QLIKE={results_gnnhar3l_random['qlike']:.6f}")

print("\n2. GNNHAR Models with Random Adjacency - MSE Loss:")
print(f"   GNNHAR1L: MSE={results_gnnhar1l_random_mse['mse']:.6f}, QLIKE={results_gnnhar1l_random_mse['qlike']:.6f}")
print(f"   GNNHAR2L: MSE={results_gnnhar2l_random_mse['mse']:.6f}, QLIKE={results_gnnhar2l_random_mse['qlike']:.6f}")
print(f"   GNNHAR3L: MSE={results_gnnhar3l_random_mse['mse']:.6f}, QLIKE={results_gnnhar3l_random_mse['qlike']:.6f}")

# Find best among random adjacency GNNHARs
random_gnnhar_models = df_extended[df_extended['Model'].str.contains('GNNHAR') & 
                                     df_extended['Adjacency'].str.contains('Random')]
best_random_gnnhar_idx = random_gnnhar_models['MSE'].idxmin()

print(f"\n3. Best GNNHAR (Random Adjacency): {df_extended.loc[best_random_gnnhar_idx, 'Model']}")
print(f"   MSE: {df_extended.loc[best_random_gnnhar_idx, 'MSE']:.6f}")
print(f"   QLIKE: {df_extended.loc[best_random_gnnhar_idx, 'QLIKE']:.6f}")
print(f"   Loss Function Used: {df_extended.loc[best_random_gnnhar_idx, 'Loss_Function']}")

# Compare with linear models
print(f"\n4. Comparison with Linear GHAR (Random):")
ghar_random_mse = results_ghar_random['mse']
best_gnnhar_random_mse = df_extended.loc[best_random_gnnhar_idx, 'MSE']
improvement = (ghar_random_mse - best_gnnhar_random_mse) / ghar_random_mse * 100
print(f"   GHAR (Random): MSE={ghar_random_mse:.6f}")
print(f"   Best GNNHAR (Random): MSE={best_gnnhar_random_mse:.6f}")
print(f"   → Improvement: {improvement:.2f}%")
if improvement > 0:
    print(f"   ✓ Nonlinearity (GNN) helps with random adjacency!")
else:
    print(f"   ⚠ Linear model performs better")

# Save results to CSV
print("\n" + "="*80)
print("Saving extended results to CSV")
print("="*80)
df_extended.to_csv('./data/mock/comprehensive_comparison_mse_qlike.csv', index=False)
print("✓ Saved: ./data/mock/comprehensive_comparison_mse_qlike.csv")

print("\n" + "="*80)
print("EXPERIMENT COMPLETE")
print("="*80)
print("\n📊 Summary:")
print(f"   Total models tested: {len(df_extended)}")
print(f"   Best MSE: {df_extended['MSE'].min():.6f} ({df_extended.loc[best_mse_idx, 'Model']})")
print(f"   Best QLIKE: {df_extended['QLIKE'].min():.6f} ({df_extended.loc[best_qlike_idx, 'Model']})")
print(f"   Baseline (HAR): MSE={baseline_mse:.6f}, QLIKE={baseline_qlike:.6f}")


# ==================== Save Comprehensive Results Summary ====================

print("\n" + "="*80)
print("Generating Comprehensive Results Summary")
print("="*80)

from datetime import datetime

# Create comprehensive summary file
summary_file = './data/mock/experiment_summary.txt'

with open(summary_file, 'w', encoding='utf-8') as f:
    # Header
    f.write("="*80 + "\n")
    f.write("GNNHAR SIMULATION STUDY - COMPREHENSIVE RESULTS SUMMARY\n")
    f.write("="*80 + "\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Script: random_gnn.py\n")
    f.write("="*80 + "\n\n")
    
    # Section 1: Experimental Setup
    f.write("="*80 + "\n")
    f.write("SECTION 1: EXPERIMENTAL SETUP\n")
    f.write("="*80 + "\n\n")
    
    f.write("Data Configuration:\n")
    f.write(f"  - Number of stocks: {len(STOCK_ORDER)}\n")
    f.write(f"  - Stock universe: {', '.join(STOCK_ORDER)}\n")
    f.write(f"  - Total time periods: {len(date_l)}\n")
    f.write(f"  - Training periods: {len(train_dates)}\n")
    f.write(f"  - Test periods: {len(test_dates)}\n")
    f.write(f"  - Training date range: {train_dates[0]} to {train_dates[-1]}\n")
    f.write(f"  - Test date range: {test_dates[0]} to {test_dates[-1]}\n\n")
    
    f.write("Model Hyperparameters:\n")
    f.write(f"  - Hidden dimension: 9\n")
    f.write(f"  - Learning rate: 0.001\n")
    f.write(f"  - Training epochs: 1000\n")
    f.write(f"  - Batch size: 32\n")
    f.write(f"  - Optimizer: Adam\n\n")
    
    # Section 2: Adjacency Matrices
    f.write("="*80 + "\n")
    f.write("SECTION 2: ADJACENCY MATRICES\n")
    f.write("="*80 + "\n\n")
    
    f.write("2.1 True Adjacency Matrix (A_true):\n")
    f.write("-" * 80 + "\n")
    f.write(A_true.to_string() + "\n\n")
    
    f.write("2.2 Normalized True Adjacency Matrix (W_true):\n")
    f.write("-" * 80 + "\n")
    f.write(W_true.to_string() + "\n\n")
    
    f.write("2.3 GLASSO Estimated Adjacency Matrix (W_glasso):\n")
    f.write("-" * 80 + "\n")
    f.write(f"Alpha (regularization): {alpha:.4f}\n")
    f.write(f"Sparsity: {sparsity:.3f}\n")
    f.write(W_glasso.to_string() + "\n\n")
    
    f.write("2.4 Random Adjacency Matrix (W_random):\n")
    f.write("-" * 80 + "\n")
    f.write(W_random.to_string() + "\n\n")
    
    # Section 3: Model Performance Results
    f.write("="*80 + "\n")
    f.write("SECTION 3: MODEL PERFORMANCE RESULTS\n")
    f.write("="*80 + "\n\n")
    
    f.write("3.1 Baseline and Linear Models:\n")
    f.write("-" * 80 + "\n")
    baseline_linear = df_extended[df_extended['Type'] == 'Linear'].copy()
    f.write(baseline_linear.to_string(index=False) + "\n\n")
    
    f.write("3.2 GNNHAR Models (QLIKE Loss):\n")
    f.write("-" * 80 + "\n")
    gnnhar_qlike = df_extended[(df_extended['Type'] == 'GNN') & 
                                (df_extended['Loss_Function'] == 'QLIKE')].copy()
    f.write(gnnhar_qlike.to_string(index=False) + "\n\n")
    
    f.write("3.3 GNNHAR Models (MSE Loss):\n")
    f.write("-" * 80 + "\n")
    gnnhar_mse = df_extended[(df_extended['Type'] == 'GNN') & 
                              (df_extended['Loss_Function'] == 'MSE')].copy()
    f.write(gnnhar_mse.to_string(index=False) + "\n\n")
    
    f.write("3.4 Complete Model Comparison Table:\n")
    f.write("-" * 80 + "\n")
    f.write(df_extended.to_string(index=False) + "\n\n")
    
    # Section 4: Best Models
    f.write("="*80 + "\n")
    f.write("SECTION 4: BEST MODELS\n")
    f.write("="*80 + "\n\n")
    
    f.write("4.1 Best Model by MSE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Model: {df_extended.loc[best_mse_idx, 'Model']}\n")
    f.write(f"Type: {df_extended.loc[best_mse_idx, 'Type']}\n")
    f.write(f"Adjacency: {df_extended.loc[best_mse_idx, 'Adjacency']}\n")
    f.write(f"Loss Function: {df_extended.loc[best_mse_idx, 'Loss_Function']}\n")
    f.write(f"MSE: {df_extended.loc[best_mse_idx, 'MSE']:.6f}\n")
    f.write(f"QLIKE: {df_extended.loc[best_mse_idx, 'QLIKE']:.6f}\n")
    f.write(f"MSE Improvement: {df_extended.loc[best_mse_idx, 'MSE_Improvement(%)']:.2f}%\n")
    f.write(f"QLIKE Improvement: {df_extended.loc[best_mse_idx, 'QLIKE_Improvement(%)']:.2f}%\n\n")
    
    f.write("4.2 Best Model by QLIKE:\n")
    f.write("-" * 80 + "\n")
    f.write(f"Model: {df_extended.loc[best_qlike_idx, 'Model']}\n")
    f.write(f"Type: {df_extended.loc[best_qlike_idx, 'Type']}\n")
    f.write(f"Adjacency: {df_extended.loc[best_qlike_idx, 'Adjacency']}\n")
    f.write(f"Loss Function: {df_extended.loc[best_qlike_idx, 'Loss_Function']}\n")
    f.write(f"MSE: {df_extended.loc[best_qlike_idx, 'MSE']:.6f}\n")
    f.write(f"QLIKE: {df_extended.loc[best_qlike_idx, 'QLIKE']:.6f}\n")
    f.write(f"MSE Improvement: {df_extended.loc[best_qlike_idx, 'MSE_Improvement(%)']:.2f}%\n")
    f.write(f"QLIKE Improvement: {df_extended.loc[best_qlike_idx, 'QLIKE_Improvement(%)']:.2f}%\n\n")
    
    # Section 5: Key Findings
    f.write("="*80 + "\n")
    f.write("SECTION 5: KEY FINDINGS\n")
    f.write("="*80 + "\n\n")
    
    f.write("Finding 1: Adjacency Matrix Comparison (Linear GHAR)\n")
    f.write("-" * 80 + "\n")
    f.write(f"GHAR (True):    MSE={results_ghar_true['mse']:.6f}, QLIKE={results_ghar_true['qlike']:.6f}\n")
    f.write(f"GHAR (GLASSO):  MSE={results_ghar_glasso['mse']:.6f}, QLIKE={results_ghar_glasso['qlike']:.6f}\n")
    f.write(f"GHAR (Random):  MSE={results_ghar_random['mse']:.6f}, QLIKE={results_ghar_random['qlike']:.6f}\n")
    f.write(f"HAR (Baseline): MSE={results_har['mse']:.6f}, QLIKE={results_har['qlike']:.6f}\n\n")
    
    ghar_true_imp = (baseline_mse - results_ghar_true['mse']) / baseline_mse * 100
    ghar_glasso_imp = (baseline_mse - results_ghar_glasso['mse']) / baseline_mse * 100
    ghar_random_imp = (baseline_mse - results_ghar_random['mse']) / baseline_mse * 100
    
    f.write("Improvements over HAR:\n")
    f.write(f"  - True adjacency:   {ghar_true_imp:.2f}%\n")
    f.write(f"  - GLASSO adjacency: {ghar_glasso_imp:.2f}%\n")
    f.write(f"  - Random adjacency: {ghar_random_imp:.2f}%\n")
    
    if ghar_random_imp > ghar_true_imp:
        f.write("\n⚠ SURPRISING: Random adjacency outperforms true adjacency!\n")
    else:
        f.write("\n✓ As expected: True adjacency performs best.\n")
    f.write("\n")
    
    f.write("Finding 2: MSE vs QLIKE Training (Random Adjacency)\n")
    f.write("-" * 80 + "\n")
    f.write("QLIKE-trained models:\n")
    f.write(f"  GNNHAR1L: MSE={results_gnnhar1l_random['mse']:.6f}, QLIKE={results_gnnhar1l_random['qlike']:.6f}\n")
    f.write(f"  GNNHAR2L: MSE={results_gnnhar2l_random['mse']:.6f}, QLIKE={results_gnnhar2l_random['qlike']:.6f}\n")
    f.write(f"  GNNHAR3L: MSE={results_gnnhar3l_random['mse']:.6f}, QLIKE={results_gnnhar3l_random['qlike']:.6f}\n\n")
    
    f.write("MSE-trained models:\n")
    f.write(f"  GNNHAR1L: MSE={results_gnnhar1l_random_mse['mse']:.6f}, QLIKE={results_gnnhar1l_random_mse['qlike']:.6f}\n")
    f.write(f"  GNNHAR2L: MSE={results_gnnhar2l_random_mse['mse']:.6f}, QLIKE={results_gnnhar2l_random_mse['qlike']:.6f}\n")
    f.write(f"  GNNHAR3L: MSE={results_gnnhar3l_random_mse['mse']:.6f}, QLIKE={results_gnnhar3l_random_mse['qlike']:.6f}\n\n")
    
    # Find best among each category
    best_qlike_trained_mse = min(results_gnnhar1l_random['mse'], 
                                   results_gnnhar2l_random['mse'], 
                                   results_gnnhar3l_random['mse'])
    best_mse_trained_mse = min(results_gnnhar1l_random_mse['mse'], 
                                 results_gnnhar2l_random_mse['mse'], 
                                 results_gnnhar3l_random_mse['mse'])
    
    f.write(f"Best QLIKE-trained: MSE={best_qlike_trained_mse:.6f}\n")
    f.write(f"Best MSE-trained:   MSE={best_mse_trained_mse:.6f}\n")
    
    if best_mse_trained_mse < best_qlike_trained_mse:
        improvement = (best_qlike_trained_mse - best_mse_trained_mse) / best_qlike_trained_mse * 100
        f.write(f"\n✓ MSE training superior by {improvement:.2f}%\n")
    else:
        f.write("\n⚠ QLIKE training performs better\n")
    f.write("\n")
    
    f.write("Finding 3: Linear vs Nonlinear (Random Adjacency)\n")
    f.write("-" * 80 + "\n")
    best_random_gnnhar = df_extended[df_extended['Model'].str.contains('GNNHAR') & 
                                      (df_extended['Adjacency'] == 'Random')]['MSE'].min()
    ghar_random_mse = results_ghar_random['mse']
    
    f.write(f"GHAR (Linear, Random):       MSE={ghar_random_mse:.6f}\n")
    f.write(f"Best GNNHAR (Nonlinear, Random): MSE={best_random_gnnhar:.6f}\n")
    
    if best_random_gnnhar < ghar_random_mse:
        improvement = (ghar_random_mse - best_random_gnnhar) / ghar_random_mse * 100
        f.write(f"\n✓ Nonlinearity (GNN) improves by {improvement:.2f}%\n")
    else:
        f.write("\n⚠ Linear model performs better\n")
    f.write("\n")
    
    # Section 6: Statistical Summary
    f.write("="*80 + "\n")
    f.write("SECTION 6: STATISTICAL SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write("MSE Statistics:\n")
    f.write(f"  Mean:   {df_extended['MSE'].mean():.6f}\n")
    f.write(f"  Median: {df_extended['MSE'].median():.6f}\n")
    f.write(f"  Min:    {df_extended['MSE'].min():.6f}\n")
    f.write(f"  Max:    {df_extended['MSE'].max():.6f}\n")
    f.write(f"  Std:    {df_extended['MSE'].std():.6f}\n\n")
    
    f.write("QLIKE Statistics:\n")
    f.write(f"  Mean:   {df_extended['QLIKE'].mean():.6f}\n")
    f.write(f"  Median: {df_extended['QLIKE'].median():.6f}\n")
    f.write(f"  Min:    {df_extended['QLIKE'].min():.6f}\n")
    f.write(f"  Max:    {df_extended['QLIKE'].max():.6f}\n")
    f.write(f"  Std:    {df_extended['QLIKE'].std():.6f}\n\n")
    
    # Section 7: Model Rankings
    f.write("="*80 + "\n")
    f.write("SECTION 7: MODEL RANKINGS\n")
    f.write("="*80 + "\n\n")
    
    f.write("7.1 Top 5 Models by MSE:\n")
    f.write("-" * 80 + "\n")
    top5_mse = df_extended.nsmallest(5, 'MSE')[['Model', 'MSE', 'QLIKE', 'MSE_Improvement(%)']]
    for i, (idx, row) in enumerate(top5_mse.iterrows(), 1):
        f.write(f"{i}. {row['Model']}\n")
        f.write(f"   MSE={row['MSE']:.6f}, QLIKE={row['QLIKE']:.6f}, Improvement={row['MSE_Improvement(%)']:.2f}%\n")
    f.write("\n")
    
    f.write("7.2 Top 5 Models by QLIKE:\n")
    f.write("-" * 80 + "\n")
    top5_qlike = df_extended.nsmallest(5, 'QLIKE')[['Model', 'MSE', 'QLIKE', 'QLIKE_Improvement(%)']]
    for i, (idx, row) in enumerate(top5_qlike.iterrows(), 1):
        f.write(f"{i}. {row['Model']}\n")
        f.write(f"   MSE={row['MSE']:.6f}, QLIKE={row['QLIKE']:.6f}, Improvement={row['QLIKE_Improvement(%)']:.2f}%\n")
    f.write("\n")
    
    # Section 8: Conclusions
    f.write("="*80 + "\n")
    f.write("SECTION 8: CONCLUSIONS\n")
    f.write("="*80 + "\n\n")
    
    f.write("Key Takeaways:\n\n")
    
    f.write("1. Adjacency Matrix Impact:\n")
    if ghar_random_imp > ghar_true_imp:
        f.write("   Random adjacency matrices unexpectedly outperformed true adjacency,\n")
        f.write("   suggesting that optimal W for prediction may differ from true causal structure.\n\n")
    else:
        f.write("   True adjacency performed best, as theoretically expected.\n\n")
    
    f.write("2. Loss Function Comparison:\n")
    if best_mse_trained_mse < best_qlike_trained_mse:
        f.write("   MSE training proved superior to QLIKE training, even on QLIKE metric.\n")
        f.write("   This contradicts some literature and warrants further investigation.\n\n")
    else:
        f.write("   QLIKE training performed better, consistent with theoretical expectations.\n\n")
    
    f.write("3. Nonlinearity Benefits:\n")
    if best_random_gnnhar < ghar_random_mse:
        f.write("   GNN's nonlinear layers improved upon linear GHAR, even with random adjacency.\n")
        f.write("   This suggests value in nonlinear modeling beyond network structure.\n\n")
    else:
        f.write("   Linear models proved sufficient; nonlinearity did not add value.\n\n")
    
    f.write("4. Overall Performance:\n")
    best_improvement_mse = df_extended['MSE_Improvement(%)'].max()
    best_improvement_qlike = df_extended['QLIKE_Improvement(%)'].max()
    f.write(f"   Best MSE improvement over HAR: {best_improvement_mse:.2f}%\n")
    f.write(f"   Best QLIKE improvement over HAR: {best_improvement_qlike:.2f}%\n\n")
    
    # Section 9: Files Generated
    f.write("="*80 + "\n")
    f.write("SECTION 9: OUTPUT FILES\n")
    f.write("="*80 + "\n\n")
    
    f.write("Generated Files:\n")
    f.write("  1. comprehensive_comparison_mse_qlike.csv - Detailed model comparison table\n")
    f.write("  2. experiment_summary.txt - This comprehensive summary file\n\n")
    
    # Footer
    f.write("="*80 + "\n")
    f.write("END OF REPORT\n")
    f.write("="*80 + "\n")

print(f"✓ Saved comprehensive summary: {summary_file}")

# Also save detailed results for each model type
model_details_file = './data/mock/model_details.txt'

with open(model_details_file, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("DETAILED MODEL RESULTS\n")
    f.write("="*80 + "\n\n")
    
    # HAR Model
    f.write("HAR (Baseline Model)\n")
    f.write("-" * 80 + "\n")
    f.write(f"MSE:   {results_har['mse']:.6f}\n")
    f.write(f"QLIKE: {results_har['qlike']:.6f}\n\n")
    
    # GHAR Models
    f.write("GHAR Models\n")
    f.write("-" * 80 + "\n")
    f.write(f"True Adjacency:   MSE={results_ghar_true['mse']:.6f}, QLIKE={results_ghar_true['qlike']:.6f}\n")
    f.write(f"GLASSO Adjacency: MSE={results_ghar_glasso['mse']:.6f}, QLIKE={results_ghar_glasso['qlike']:.6f}\n")
    f.write(f"Random Adjacency: MSE={results_ghar_random['mse']:.6f}, QLIKE={results_ghar_random['qlike']:.6f}\n\n")
    
    # GNNHAR with Random Adjacency (QLIKE Loss)
    f.write("GNNHAR Models (Random Adjacency, QLIKE Loss)\n")
    f.write("-" * 80 + "\n")
    f.write(f"GNNHAR1L: MSE={results_gnnhar1l_random['mse']:.6f}, QLIKE={results_gnnhar1l_random['qlike']:.6f}\n")
    f.write(f"GNNHAR2L: MSE={results_gnnhar2l_random['mse']:.6f}, QLIKE={results_gnnhar2l_random['qlike']:.6f}\n")
    f.write(f"GNNHAR3L: MSE={results_gnnhar3l_random['mse']:.6f}, QLIKE={results_gnnhar3l_random['qlike']:.6f}\n\n")
    
    # GNNHAR with Random Adjacency (MSE Loss)
    f.write("GNNHAR Models (Random Adjacency, MSE Loss)\n")
    f.write("-" * 80 + "\n")
    f.write(f"GNNHAR1L: MSE={results_gnnhar1l_random_mse['mse']:.6f}, QLIKE={results_gnnhar1l_random_mse['qlike']:.6f}\n")
    f.write(f"GNNHAR2L: MSE={results_gnnhar2l_random_mse['mse']:.6f}, QLIKE={results_gnnhar2l_random_mse['qlike']:.6f}\n")
    f.write(f"GNNHAR3L: MSE={results_gnnhar3l_random_mse['mse']:.6f}, QLIKE={results_gnnhar3l_random_mse['qlike']:.6f}\n\n")

print(f"✓ Saved model details: {model_details_file}")

print("\n" + "="*80)
print("ALL RESULTS SAVED SUCCESSFULLY")
print("="*80)
print("\n📁 Output Files:")
print(f"   1. {summary_file}")
print(f"   2. {model_details_file}")
print(f"   3. ./data/mock/comprehensive_comparison_mse_qlike.csv")
print("\n✓ Experiment pipeline complete!")



