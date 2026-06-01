"""
Compute the daily variance from 5-min return data
Compute the variance data for multi-horizon and various universes
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats.mstats import winsorize
from sklearn.linear_model import HuberRegressor, LinearRegression, LassoCV, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from multiprocessing import cpu_count, Pool
from joblib import dump
from os.path import join
import os
from datetime import datetime
from sklearn import preprocessing
from numpy import linalg as LA
import scipy

# 道琼斯工业平均指数成分股列表
DJIA_stocks_l = ['MMM', 'AXP', 'AMGN', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'HD', 'HON', 'IBM', 'GS', 'NKE',
                 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'PG', 'CRM', 'TRV', 'UNH', 'VZ', 'WMT']
DJIA_stocks_l.sort()

# 标普100指数成分股列表
SP100_stocks_l = ['AAPL', 'ABT', 'ACN', 'ADBE', 'ADP', 'AMGN', 'AMT', 'AMZN', 'AXP', 'BA', 'BAC', 'BDX', 'BMY',
                 'BSX', 'C', 'CAT', 'CB', 'CI', 'CMCSA', 'CME', 'COP', 'COST', 'CRM', 'CSCO', 'CVS', 'CVX', 'D',
                 'DHR', 'DIS', 'DUK', 'FIS', 'FISV', 'GE', 'GILD', 'GOOG', 'GS', 'HD', 'HON', 'IBM', 'INTC', 'INTU',
                 'ISRG', 'JNJ', 'JPM', 'KO', 'LLY', 'LMT', 'LOW', 'MA', 'MCD', 'MDT', 'MMM', 'MO', 'MRK', 'MS',
                 'MSFT', 'NFLX', 'NKE', 'NVDA', 'ORCL', 'PEP', 'PFE', 'PG', 'PNC', 'QCOM', 'SBUX', 'SO', 'SYK',
                 'T', 'TGT', 'TJX', 'TMO', 'TXN', 'UNH', 'UNP', 'UPS', 'USB', 'VZ', 'WFC', 'WMT']
SP100_stocks_l.sort()

# 数据名称字典，映射指数名称到对应的股票列表
data_name_dic = {'DJIA': DJIA_stocks_l, 'SP100': SP100_stocks_l}

def load_data(path):
    """
    加载5分钟收益率数据并进行预处理
    
    Args:
        path: 数据文件路径
        
    Returns:
        ret_data: 预处理后的收益率数据
    """
    # 从CSV文件加载5分钟收益率数据
    ret_data = pd.read_csv(join(path, 'Data', 'data_5min.csv'))

    # 提取股票列名（排除Date和Time列）
    stocks_l = [i for i in ret_data.columns if i not in ['Date', 'Time']]
    # 将收益率从小数转换为百分比（乘以100）
    ret_data[stocks_l] *= 100

    # 对数据进行winsorize处理，避免LOBSTER数据中的测量误差
    up = 99.5  # 上分位数
    low = 0.5  # 下分位数
    for clm in ret_data.columns:
        if clm not in ['Date', 'Time']:
            # 计算上下分位数
            max_p = np.nanpercentile(ret_data[clm], up)
            min_p = np.nanpercentile(ret_data[clm], low)

            # 将超出分位数范围的值替换为分位数值
            ret_data.loc[ret_data[clm] > max_p, clm] = max_p
            ret_data.loc[ret_data[clm] < min_p, clm] = min_p

    return ret_data


def compute_variance(sub_data):
    """
    计算数据的方差
    
    Args:
        sub_data: 子数据集
        
    Returns:
        var_sum: 方差数据
    """
    # 获取股票列名
    stocks_l = [i for i in sub_data.columns if i not in ['Date', 'Time']]
    # 计算收益率的平方
    sq_data = sub_data[stocks_l] ** 2
    # 对平方收益率求和（至少需要1个非空值）
    var_sum = sq_data.sum(min_count=1)
    # 转换为DataFrame格式
    var_sum = pd.DataFrame(var_sum).T
    return var_sum


def Compute_Horizon(path, univese, ret_vol, horizon):
    """
    计算不同时间跨度和股票池的方差/收益率数据
    
    Args:
        path: 数据路径
        univese: 股票池名称（如'DJIA', 'SP100'）
        ret_vol: 数据类型（'ret'表示收益率，'var'表示方差）
        horizon: 时间跨度（天数）
    """
    # 根据数据类型加载相应的日度数据
    if ret_vol == 'ret':
        daily_var_data = pd.read_csv(join(path, 'Data', 'daily_return.csv'), index_col=0)
    elif ret_vol == 'var':
        daily_var_data = pd.read_csv(join(path, 'Data', 'daily_variance.csv'), index_col=0)
    else:
        print('Please choose ret or var')
        return
        
    # 计算多期累积数据
    var_data = 0
    for i in range(horizon):
        var_data += daily_var_data.shift(-i)  # 向前移动i期并累加
    
    # 删除包含NaN的行
    var_data.dropna(inplace=True)
    # 选择指定股票池的数据
    var_univ = var_data[data_name_dic[univese]]
    # 保存结果到CSV文件
    var_univ.to_csv(join(path, 'Data', f'{univese}_{ret_vol}_FH{horizon}.csv'))


if __name__ == '__main__':
    # 设置数据路径
    path = 'your_local_path'

    # 加载收益率数据
    ret_data = load_data(path)
    # 获取股票列表
    stocks_l = [i for i in ret_data.columns if i not in ['Date', 'Time']]
    # 获取日期列表并排序
    date_l = list(set(ret_data['Date'].tolist()))
    date_l.sort()

    ### 计算日度收益率
    # 按日期分组并求和（至少需要1个非空值）
    daily_return_data = ret_data.groupby(by='Date').sum(min_count=1)
    daily_return_data.index = list(daily_return_data.index)
    # 保存日度收益率数据
    daily_return_data.to_csv(join(path, 'Data', 'daily_return.csv'))
    
    ### 计算日度方差
    # 按日期分组并应用方差计算函数
    var_df = ret_data.groupby(by='Date').apply(compute_variance)
    var_df.index = date_l
    
    # 设置列名为股票名称
    var_df.columns = stocks_l
    # 保存日度方差数据
    var_df.to_csv(join(path, 'Data', 'daily_variance.csv'))
    
    ### 计算不同时间跨度的方差数据
    horizon = 5  # 设置时间跨度为5天
    # 对不同股票池计算多期收益率和方差数据
    for name in ['DJIA30', 'SP100']:
        Compute_Horizon(path, name, 'ret', horizon)  # 计算多期收益率
        Compute_Horizon(path, name, 'var', horizon)  # 计算多期方差