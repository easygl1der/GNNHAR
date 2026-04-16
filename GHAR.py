"""
Linear models to forecast the realized volatility, including HAR and GHAR. HAR is a special case of GHAR, assuming the adjacency matrix is identity.
"""

import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=22, help="forward-looking period")  # 前瞻期窗口大小
parser.add_argument("--horizon", type=int, default=1, help="forecasting horizon")  # 预测时间跨度
parser.add_argument("--model_name", type=str, default='GHAR', help="model name")  # 模型名称
parser.add_argument("--adj_name", type=str, default='iden+glasso', help="adj choices")  # 邻接矩阵选择
parser.add_argument("--universe", type=str, default='DJIA', help="data name")  # 数据集名称
parser.add_argument("--version", type=str, default='Forecast_Var', help="version name")  # 版本名称

opt = parser.parse_args()
print(opt)

# 构建特定版本的标识符，用于文件命名和路径管理
this_version = '_'.join(
    [opt.version,
     opt.model_name,
     opt.adj_name,
     opt.universe,
     'W' + str(opt.window),
     'F' + str(opt.horizon)])

# 设置数据路径和模型保存路径
path = 'your_local_path'
model_save_path = join('your_model_storage_path', this_version)
os.makedirs(model_save_path, exist_ok=True)


def load_feature_data(universe):
    """
    加载特征数据（用于构建HAR模型的滞后项）
    
    Args:
        universe: 数据集名称（如DJIA）
    
    Returns:
        feature_df: 特征数据框，包含各股票的已实现波动率数据
    """
    feature_df = pd.read_csv(join(path, 'Data', f'{universe}_var_FH1.csv'), index_col=0)
    feature_df.fillna(method="ffill", inplace=True)  # 前向填充缺失值
    feature_df = feature_df[feature_df.index <= '2021-07-01']  # 截取到指定日期
    feature_df = feature_df.sort_index(axis=1)  # 按列名排序
    return feature_df


def load_data(universe, horizon):
    """
    加载目标变量数据（待预测的已实现波动率）
    
    Args:
        universe: 数据集名称
        horizon: 预测时间跨度
    
    Returns:
        vech_df: 目标变量数据框
    """
    var_df = pd.read_csv(join(path, 'Data', f'{universe}_var_FH{horizon}.csv'), index_col=0)
    var_df.fillna(method="ffill", inplace=True)  # 前向填充缺失值
    vech_df = var_df[var_df.index <= '2021-07-01']  # 截取到指定日期
    vech_df = vech_df.sort_index(axis=1)  # 按列名排序
    return vech_df


def load_ret(universe):
    """
    加载股票收益率数据（用于计算邻接矩阵）
    
    Args:
        universe: 数据集名称
    
    Returns:
        ret_df: 收益率数据框
    """
    ret_df = pd.read_csv(join(path, 'Data', f'{universe}_ret_FH1.csv'), index_col=0)
    ret_df.fillna(method="ffill", inplace=True)  # 前向填充缺失值
    ret_df = ret_df[ret_df.index <= '2021-07-01']  # 截取到指定日期
    ret_df = ret_df.sort_index(axis=1)  # 按列名排序
    return ret_df


def preprocess_HAR(feature_df, vech_df):
    """
    预处理HAR模型数据，构建滞后项特征
    
    Args:
        feature_df: 特征数据框
        vech_df: 目标变量数据框
    
    Returns:
        subdf_dic: 按日期分组的数据字典
        date_l: 日期列表
    """
    subdf_l = []
    # 获取所有资产列表并排序
    all_assets_l = [i for i in vech_df.columns if i not in ['Date', 'Time']]
    all_assets_l.sort()

    # HAR模型的滞后期：日（1天）、周（5天）、月（22天）
    har_lags = [1, 5, 22]
    
    # 为每个目标变量构建HAR特征
    for target_var in vech_df:
        subdf = pd.DataFrame()
        subdf['Target'] = vech_df[target_var].copy()  # 目标变量
        subdf['Date'] = vech_df.index  # 日期
        subdf['Ticker'] = target_var  # 股票代码
        
        indpt_df_l = []
        # 构建不同滞后期的平均值作为特征
        for lag in har_lags:
            tmp_indpdt_df = 0
            # 计算过去lag天的平均值
            for il in range(1, 1+lag):
                tmp_indpdt_df += feature_df[target_var].shift(il)

            indpt_df_l.append(tmp_indpdt_df / lag)

        # 合并所有滞后特征
        explain_df = pd.concat(indpt_df_l, axis=1)
        explain_df.columns = ['var+lag%d' % i for i in har_lags]

        # 合并目标变量和特征变量
        subdf = pd.merge(subdf, explain_df, left_index=True, right_index=True)
        subdf.replace([np.inf, -np.inf], np.nan, inplace=True)  # 替换无穷值
        subdf.dropna(inplace=True)  # 删除缺失值
        subdf_l.append(subdf)

    # 合并所有股票的数据
    df = pd.concat(subdf_l)
    df.reset_index(drop=True, inplace=True)

    # 获取所有日期并排序
    date_l = list(set(df['Date'].tolist()))
    date_l.sort()

    # 按日期分组数据
    subdf_dic = {}
    for date in date_l:
        subdf = df[df['Date'] == date]
        subdf_dic[date] = subdf

    print('Finish preparation!')
    return subdf_dic, date_l


def preprocess_adj_l(date_l, subdf_dic, adj_df_l):
    """
    使用邻接矩阵对特征进行图卷积变换
    
    Args:
        date_l: 日期列表
        subdf_dic: 按日期分组的数据字典
        adj_df_l: 邻接矩阵列表
    
    Returns:
        df: 经过图卷积变换后的数据框
    """
    new_subdf_l = []
    for date in date_l:
        subdf = subdf_dic[date]
        tmp_subdf_l = []
        # 获取所有滞后特征列
        clms = [i for i in subdf.columns if 'lag' in i]
        
        # 对每个邻接矩阵进行图卷积操作
        for k, adj_df in enumerate(adj_df_l):
            # 矩阵乘法：邻接矩阵 × 特征矩阵，实现图卷积
            tmp_subdf = pd.DataFrame(np.dot(adj_df, subdf[clms]), 
                                   columns=['sec'+str(k)+i for i in clms], 
                                   index=subdf.index)
            tmp_subdf_l.append(tmp_subdf)
        
        # 合并原始信息和变换后的特征
        new_subdf = pd.concat([subdf[['Target', 'Date', 'Ticker']]]+tmp_subdf_l, axis=1)
        new_subdf_l.append(new_subdf)

    # 合并所有日期的数据
    df = pd.concat(new_subdf_l)
    df.reset_index(drop=True, inplace=True)
    print('Finish transformation!')
    return df


def df2arr(df, vars_l):
    """
    将数据框转换为numpy数组格式，用于机器学习模型
    
    Args:
        df: 输入数据框
        vars_l: 特征变量列表
    
    Returns:
        all_inputs: 特征矩阵
        all_targets: 目标变量向量
    """
    all_inputs = df[vars_l].values
    all_targets = df[['Target']].values
    return all_inputs, all_targets


def GLASSO_Precision(subret):
    """
    使用Graphical Lasso方法计算精度矩阵并构建邻接矩阵
    
    Args:
        subret: 股票收益率数据框，每列代表一只股票的收益率时间序列
    
    Returns:
        adj_df: 标准化后的邻接矩阵，用于表示股票间的关系网络
    """
    from sklearn.covariance import GraphicalLassoCV
    
    # 获取股票数量和股票代码
    n = subret.shape[1]
    tickers = subret.columns
    
    # 使用交叉验证的Graphical Lasso拟合数据，自动选择最优的正则化参数alpha
    cov = GraphicalLassoCV().fit(subret)
    print('Alpha in GLASSO: %.3f' % cov.alpha_)
    
    # 从精度矩阵中提取非零元素，构建二值化的相关性矩阵
    corr = cov.precision_ != 0
    print('Sparsity of Adj: %.3f' % corr.mean())
    
    # 去除对角线元素（自相关），得到邻接矩阵
    corr_adj = corr - np.identity(n)
    
    # 计算度矩阵的逆平方根，用于标准化邻接矩阵
    # 加上小常数1e-8防止除零错误
    d_sqrt_inv = np.diag(np.sqrt(1/(corr_adj.sum(1)+1e-8)))
    
    # 应用对称标准化：D^(-1/2) * A * D^(-1/2)
    # 这样可以确保邻接矩阵的特征值在合理范围内
    adj_df = pd.DataFrame(np.dot(np.dot(d_sqrt_inv, corr_adj), d_sqrt_inv), 
                         columns=tickers, index=tickers)
    
    return adj_df


def Train(ret_df, vech_df, subdf_dic, date, date_l):
    """
    训练GHAR模型并进行预测
    
    Args:
        ret_df: 收益率数据框
        vech_df: 目标变量数据框
        subdf_dic: 按日期分组的数据字典
        date: 当前预测日期
        date_l: 日期列表
    """
    timestamp = date_l.index(date)
    # 确定训练和测试的时间范围
    s_p = max(timestamp-1000, 0)  # 训练开始位置（最多往前1000个时间点）
    f_p = min(timestamp + opt.window, len(date_l)-1)  # 测试结束位置

    s_date = date_l[s_p]  # 训练开始日期
    f_date = date_l[f_p]  # 测试结束日期

    # 提取训练期间的收益率数据（用于构建邻接矩阵）
    subret = ret_df[ret_df.index < date]
    subret = subret[subret.index >= s_date]

    # 提取训练期间的目标变量数据
    subdata = vech_df[vech_df.index < date]
    subdata = subdata[subdata.index >= s_date]
    tickers = subret.columns

    n = vech_df.shape[1]  # 股票数量
    adj_name_l = opt.adj_name.split('+')  # 解析邻接矩阵类型
    adj_df_l = []
    
    # 构建不同类型的邻接矩阵
    for adj_name in adj_name_l:
        if adj_name == 'iden':
            # 单位矩阵（对应传统HAR模型）
            adj_df = pd.DataFrame(np.identity(n), index=tickers, columns=tickers)
        elif adj_name == 'glasso':
            # 基于Graphical Lasso的邻接矩阵
            adj_df = GLASSO_Precision(subret)
        else:
            # 零矩阵（默认情况）
            adj_df = pd.DataFrame(np.zeros((n, n)), index=tickers, columns=tickers)

        adj_df_l.append(adj_df)

    # 使用邻接矩阵对数据进行图卷积变换
    df = preprocess_adj_l(date_l[s_p:f_p+1], subdf_dic, adj_df_l)
    
    # 获取所有特征变量列名
    vars_l = [i for i in df.columns if 'lag' in i]
    
    # 划分训练集和测试集
    train_df = df[df['Date'] >= s_date]
    train_df = train_df[train_df['Date'] < date]
    print(train_df)
    
    test_df = df[df['Date'] >= date]
    test_df = test_df[test_df['Date'] < f_date]
    print(test_df)
    
    # 转换为numpy数组格式
    train_x, train_y = df2arr(train_df, vars_l)
    test_x, test_y = df2arr(test_df, vars_l)
    
    # 训练线性回归模型
    best_model = LinearRegression()
    best_model.fit(train_x, train_y)
    print(best_model.coef_)
    
    # 进行预测
    test_pred_df = test_df[['Ticker', 'Date']]
    test_pred_df['Pred_VHAR'] = best_model.predict(test_x)
    # 将预测结果重塑为日期×股票的矩阵格式
    test_pred_df = test_pred_df.pivot(index='Date', columns='Ticker', values='Pred_VHAR')
    
    test_pred_df.columns = list(test_pred_df.columns)
    test_pred_df.index = list(test_pred_df.index)
    
    print('Before: %.3f' % test_pred_df.min().min())
    
    # 调整负预测值：将负值替换为训练数据中的最小值
    # 这是因为已实现波动率不能为负
    for clm in test_pred_df.columns:
        clm_pred_df = test_pred_df[clm]
        clm_train_df = train_df[train_df['Ticker'] == clm]['Target']
        clm_pred_df[clm_pred_df <= 0] = clm_train_df.min()
        test_pred_df[clm] = clm_pred_df
    
    print('After: %.3f' % test_pred_df.min().min())
    
    # 保存预测结果
    save_path = join(path, 'Var_Pred_Results', this_version)
    os.makedirs(save_path, exist_ok=True)
    
    test_pred_df.to_csv(join(save_path, 'Pred_%s.csv' % date))


def connect_pred():
    """
    合并所有预测结果文件，生成最终的预测结果
    """
    save_path = join(path, 'Var_Pred_Results', this_version)
    files_l = os.listdir(save_path)
    pred_files = [i for i in files_l if 'Pred_' in i]  # 筛选预测文件
    pred_files.sort()  # 按文件名排序
    
    test_pred_df_l = []
    # 读取所有预测文件
    for i in pred_files:
        test_pred_df = pd.read_csv(join(save_path, i), index_col=0)
        test_pred_df_l.append(test_pred_df)

    # 合并所有预测结果
    test_pred_df = pd.concat(test_pred_df_l)
    print(test_pred_df)

    # 保存最终的合并结果
    sum_path = join(path, 'Var_Results_Sum')
    os.makedirs(sum_path, exist_ok=True)
    test_pred_df.to_csv(join(sum_path, this_version + '_pred.csv'))


if __name__ == '__main__':
    # 加载数据
    feature_df = load_feature_data(opt.universe)  # 特征数据
    vech_df = load_data(opt.universe, opt.horizon)  # 目标变量数据
    ret_df = load_ret(opt.universe)  # 收益率数据

    n = vech_df.shape[1]  # 股票数量

    # 预处理HAR特征
    subdf_dic, date_l = preprocess_HAR(feature_df, vech_df)

    print('Training Starts Now ...')
    idx = date_l.index('2011-07-01')  # 训练开始的日期索引

    # 滚动窗口训练和预测
    # 每隔opt.window个时间点进行一次训练和预测
    for date in date_l[idx::opt.window]:
        print(' * ' * 20 + date + ' * ' * 20)
        Train(ret_df, vech_df, subdf_dic, date, date_l)

    # 合并所有预测结果
    connect_pred()
