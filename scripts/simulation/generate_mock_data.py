"""
生成模拟数据，模拟Figure 1中的线性和非线性波动率溢出效应
包含6只股票：IBM, JPM, GS, CVX, AXP, BA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 设置随机种子以保证可重复性
np.random.seed(42)

# 股票列表（按照网络结构顺序）
STOCKS = ['IBM', 'JPM', 'GS', 'CVX', 'AXP', 'BA']

class VolatilityNetworkSimulator:
    """
    模拟具有线性和非线性溢出效应的波动率网络（异方差模型）
    
    按照图片中的异方差模型构建：
    ┌──────────────────────────────────────────────────────────────┐
    │ 模型1 (IBM - 目标节点):                                       │
    │   RV²_IBM,t = β₀ + β₁·RV²_IBM,t-1 + β₂·RV²_GS,t-1           │
    │              + β₃·RV⁴_IBM,t-1 + ε₁                           │
    │   (包含自回归、GS线性溢出、自身非线性项)                        │
    ├──────────────────────────────────────────────────────────────┤
    │ 模型2 (JPM - 1-hop邻居):                                      │
    │   RV²_JPM,t = β'₀ + β'₁·RV²_JPM,t-1 + β'₂·RV²_AXP,t-1       │
    │              + β'₃·RV⁴_CVX,t-1 + ε₂                          │
    │   (包含自回归、AXP线性溢出、CVX非线性溢出)                   │
    ├──────────────────────────────────────────────────────────────┤
    │ 模型3 (GS - 1-hop邻居):                                       │
    │   RV²_GS,t = β''₀ + β''₁·RV²_GS,t-1 + β''₂·RV²_BA,t-1 + ε₃  │
    │   (包含自回归、BA线性溢出)                                    │
    ├──────────────────────────────────────────────────────────────┤
    │ 叶子节点 (CVX, AXP, BA):                                      │
    │   RV²_t = β₀ + ε  (纯随机游走)                               │
    └──────────────────────────────────────────────────────────────┘
    
    其中 εᵢ ~ N(0, σ²ᵢ) 是异方差误差项
    所有β参数初始值设为 0.1
    """
    
    def __init__(self, T=1000, base_vol=0.5):
        """
        Args:
            T: 时间步数
            base_vol: 基础波动率水平
        """
        self.T = T
        self.base_vol = base_vol
        self.stocks = STOCKS
        self.n_stocks = len(STOCKS)
        
        # 股票索引映射
        self.stock_idx = {stock: i for i, stock in enumerate(STOCKS)}
    
    def generate_base_volatility(self):
        """
        生成基础波动率序列（独立成分）
        使用AR(1)过程模拟波动率的持续性
        """
        volatilities = {}
        
        for stock in self.stocks:
            # AR(1)参数
            phi = 0.7  # 自回归系数
            sigma = 0.1  # 创新标准差
            
            # 初始化
            rv = np.zeros(self.T)
            rv[0] = self.base_vol
            
            # 生成AR(1)过程
            for t in range(1, self.T):
                innovation = np.random.normal(0, sigma)
                rv[t] = (1 - phi) * self.base_vol + phi * rv[t-1] + innovation
                # 确保波动率为正
                rv[t] = max(rv[t], 0.01)
            
            volatilities[stock] = rv
        
        return volatilities
    
    def add_linear_spillover(self, volatilities, source, target, strength=0.3, lag=1):
        """
        添加线性溢出效应: target_t = target_t + strength * source_{t-lag}
        
        Args:
            volatilities: 波动率字典
            source: 源股票
            target: 目标股票
            strength: 溢出强度
            lag: 滞后期数
        """
        for t in range(lag, self.T):
            volatilities[target][t] += strength * volatilities[source][t - lag]
    
    def add_nonlinear_spillover(self, volatilities, source, target, strength=0.2, lag=1):
        """
        添加非线性（二次）溢出效应: target_t = target_t + strength * source_{t-lag}^2
        
        Args:
            volatilities: 波动率字典
            source: 源股票
            target: 目标股票  
            strength: 溢出强度
            lag: 滞后期数
        """
        for t in range(lag, self.T):
            # 二次项捕捉非线性关系
            nonlinear_effect = strength * (volatilities[source][t - lag] ** 2)
            volatilities[target][t] += nonlinear_effect
    
    def add_interaction_effect(self, volatilities, source1, source2, target, strength=0.1):
        """
        添加交互效应: target_t = target_t + strength * source1_t * source2_t
        
        Args:
            volatilities: 波动率字典
            source1, source2: 源股票
            target: 目标股票
            strength: 交互强度
        """
        for t in range(1, self.T):
            interaction = strength * volatilities[source1][t-1] * volatilities[source2][t-1]
            volatilities[target][t] += interaction
    
    def generate_network_data(self):
        """
        按照图片中的异方差模型生成数据
        
        模型设定（包含自回归项和溢出效应）：
        RV²_IBM,t = β₀ + β₁·RV²_IBM,t-1 + β₂·RV²_GS,t-1 + β₃·RV⁴_IBM,t-1 + ε₁
        RV²_JPM,t = β'₀ + β'₁·RV²_JPM,t-1 + β'₂·RV²_AXP,t-1 + β'₃·RV⁴_CVX,t-1 + ε₂
        RV²_GS,t = β''₀ + β''₁·RV²_GS,t-1 + β''₂·RV²_BA,t-1 + ε₃
        
        其中 εᵢ ~ N(0, σ²ᵢ) 是异方差误差项
        """
        print("按照异方差模型生成数据...")
        
        # 参数设置（按照图片说明，初始值为0.1）
        params = {
            'IBM': {'beta0': 0.1, 'beta1': 0.1, 'beta2': 0.1, 'beta3': 0.1, 'sigma': 0.05},
            'JPM': {'beta0': 0.1, 'beta1': 0.1, 'beta2': 0.1, 'beta3': 0.1, 'sigma': 0.05},
            'GS': {'beta0': 0.1, 'beta1': 0.1, 'beta2': 0.1, 'sigma': 0.05},
            'CVX': {'beta0': 0.1, 'sigma': 0.05},
            'AXP': {'beta0': 0.1, 'sigma': 0.05},
            'BA': {'beta0': 0.1, 'sigma': 0.05}
        }
        
        # 初始化波动率（使用平方形式 RV²）
        rv_squared = {stock: np.zeros(self.T) for stock in self.stocks}
        
        # 设置初始值
        for stock in self.stocks:
            rv_squared[stock][0] = params[stock]['beta0']
        
        # 逐时间步生成（按照拓扑顺序）
        for t in range(1, self.T):
            # === 第一层：叶子节点（纯随机游走）===
            # CVX: 只有截距和误差
            rv_squared['CVX'][t] = (
                params['CVX']['beta0'] + 
                np.random.normal(0, params['CVX']['sigma'])
            )
            
            # BA: 只有截距和误差
            rv_squared['BA'][t] = (
                params['BA']['beta0'] + 
                np.random.normal(0, params['BA']['sigma'])
            )
            
            # AXP: 只有截距和误差（作为JPM的输入）
            rv_squared['AXP'][t] = (
                params['AXP']['beta0'] + 
                np.random.normal(0, params['AXP']['sigma'])
            )
            
            # === 第二层：GS (受BA影响) ===
            # RV²_GS,t = β''₀ + β''₁·RV²_GS,t-1 + β''₂·RV²_BA,t-1 + ε₃
            rv_squared['GS'][t] = (
                params['GS']['beta0'] +
                params['GS']['beta1'] * rv_squared['GS'][t-1] +
                params['GS']['beta2'] * rv_squared['BA'][t-1] +
                np.random.normal(0, params['GS']['sigma'])
            )
            
            # === 第三层：JPM (受AXP线性、CVX非线性影响) ===
            # RV²_JPM,t = β'₀ + β'₁·RV²_JPM,t-1 + β'₂·RV²_AXP,t-1 + β'₃·RV⁴_CVX,t-1 + ε₂
            rv_squared['JPM'][t] = (
                params['JPM']['beta0'] +
                params['JPM']['beta1'] * rv_squared['JPM'][t-1] +
                params['JPM']['beta2'] * rv_squared['AXP'][t-1] +
                params['JPM']['beta3'] * (rv_squared['CVX'][t-1] ** 2) +  # RV⁴项
                np.random.normal(0, params['JPM']['sigma'])
            )
            
            # === 第四层：IBM (受GS线性、自身非线性影响) ===
            # RV²_IBM,t = β₀ + β₁·RV²_IBM,t-1 + β₂·RV²_GS,t-1 + β₃·RV⁴_IBM,t-1 + ε₁
            rv_squared['IBM'][t] = (
                params['IBM']['beta0'] +
                params['IBM']['beta1'] * rv_squared['IBM'][t-1] +
                params['IBM']['beta2'] * rv_squared['GS'][t-1] +
                params['IBM']['beta3'] * (rv_squared['IBM'][t-1] ** 2) +  # RV⁴项
                np.random.normal(0, params['IBM']['sigma'])
            )
        
        # 转换回 RV (取平方根，确保为正)
        volatilities = {}
        for stock in self.stocks:
            rv_squared[stock] = np.maximum(rv_squared[stock], 0.001)  # 避免负值
            volatilities[stock] = np.sqrt(rv_squared[stock])
        
        print("\n数据生成关系式（异方差模型）:")
        print("RV²_CVX,t = β₀ + ε (纯随机)")
        print("RV²_BA,t = β₀ + ε (纯随机)")
        print("RV²_AXP,t = β₀ + ε (纯随机)")
        print("RV²_GS,t = β₀ + β₁·RV²_GS,t-1 + β₂·RV²_BA,t-1 + ε")
        print("RV²_JPM,t = β₀ + β₁·RV²_JPM,t-1 + β₂·RV²_AXP,t-1 + β₃·RV⁴_CVX,t-1 + ε")
        print("RV²_IBM,t = β₀ + β₁·RV²_IBM,t-1 + β₂·RV²_GS,t-1 + β₃·RV⁴_IBM,t-1 + ε")
        print(f"\n所有β初始值 = 0.1, σ = 0.05")
        
        return volatilities
    
    def create_har_features(self, volatilities):
        """
        创建HAR特征：1天、5天、22天的滞后平均波动率
        
        Returns:
            lag1, lag5, lag22: (T, n_stocks) 数组
        """
        rv_matrix = np.array([volatilities[stock] for stock in self.stocks]).T  # (T, n_stocks)
        
        # 计算不同时间跨度的平均
        lag1 = np.zeros_like(rv_matrix)
        lag5 = np.zeros_like(rv_matrix)
        lag22 = np.zeros_like(rv_matrix)
        
        for t in range(self.T):
            # 1天滞后
            if t >= 1:
                lag1[t] = rv_matrix[t-1]
            
            # 5天平均
            if t >= 5:
                lag5[t] = np.mean(rv_matrix[t-5:t], axis=0)
            
            # 22天平均
            if t >= 22:
                lag22[t] = np.mean(rv_matrix[t-22:t], axis=0)
        
        return lag1, lag5, lag22
    
    def save_to_csv(self, volatilities, output_dir='data/mock'):
        """
        保存数据为CSV格式，与原GNNHAR代码兼容
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建日期索引
        start_date = datetime(2020, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(self.T)]
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]
        
        # 转换为DataFrame
        rv_df = pd.DataFrame(volatilities, index=date_strings)
        rv_df.index.name = 'Date'
        
        # 保存为与原代码兼容的格式
        # 1. 保存已实现波动率 (用于特征)
        rv_df.to_csv(f'{output_dir}/DJIA_var_FH1.csv')
        print(f"已保存: {output_dir}/DJIA_var_FH1.csv")
        
        # 2. 保存目标变量 (下一期的RV)
        target_df = rv_df.shift(-1).dropna()
        target_df.to_csv(f'{output_dir}/DJIA_var_FH1_target.csv')
        print(f"已保存: {output_dir}/DJIA_var_FH1_target.csv")
        
        # 3. 保存收益率数据 (用于计算邻接矩阵，这里用波动率的平方根模拟)
        ret_df = rv_df.apply(lambda x: np.sqrt(x) * np.random.randn(len(x)))
        ret_df.to_csv(f'{output_dir}/DJIA_ret_FH1.csv')
        print(f"已保存: {output_dir}/DJIA_ret_FH1.csv")
        
        return rv_df
    
    def visualize_network_effects(self, volatilities):
        """
        可视化网络效应
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Volatility Network Simulation (Figure 1)', fontsize=16, fontweight='bold')
        
        time_range = range(min(200, self.T))  # 只显示前200个时间点
        
        for idx, stock in enumerate(self.stocks):
            ax = axes[idx // 2, idx % 2]
            ax.plot(time_range, volatilities[stock][:len(time_range)], 
                   linewidth=1.5, alpha=0.8)
            ax.set_title(f'{stock} Realized Volatility', fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Volatility')
            ax.grid(True, alpha=0.3)
            
            # 标注节点类型
            if stock == 'IBM':
                ax.text(0.02, 0.98, '0-hop (Target)', transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            elif stock in ['JPM', 'GS']:
                ax.text(0.02, 0.98, '1-hop neighbor', transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.8))
            else:
                ax.text(0.02, 0.98, '2-hop neighbor', transform=ax.transAxes,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('data/mock/network_volatilities.png', dpi=150, bbox_inches='tight')
        print("已保存可视化图: data/mock/network_volatilities.png")
        plt.close()
    
    def analyze_correlations(self, volatilities):
        """
        分析股票间的相关性
        """
        rv_matrix = np.array([volatilities[stock] for stock in self.stocks]).T
        corr_matrix = np.corrcoef(rv_matrix.T)
        
        # 创建相关性热图
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # 设置刻度
        ax.set_xticks(range(self.n_stocks))
        ax.set_yticks(range(self.n_stocks))
        ax.set_xticklabels(self.stocks)
        ax.set_yticklabels(self.stocks)
        
        # 添加数值标签
        for i in range(self.n_stocks):
            for j in range(self.n_stocks):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=10)
        
        ax.set_title('Correlation Matrix of Simulated Volatilities', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig('data/mock/correlation_matrix.png', dpi=150, bbox_inches='tight')
        print("已保存相关性矩阵: data/mock/correlation_matrix.png")
        plt.close()
        
        return corr_matrix
    
    def verify_causal_relationships(self, volatilities):
        """
        验证生成的数据是否符合预设的异方差模型
        通过回归分析检验（使用RV²作为因变量）
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        
        print("\n" + "=" * 60)
        print("异方差模型验证（回归分析）")
        print("=" * 60)
        
        results = []
        
        # 准备数据：RV² (去掉前22个时间点)
        rv_squared = {stock: (volatilities[stock][22:] ** 2) for stock in self.stocks}
        
        # 1. 验证 IBM 模型: RV²_IBM,t = f(RV²_IBM,t-1, RV²_GS,t-1, RV⁴_IBM,t-1)
        X_ibm = np.column_stack([
            rv_squared['IBM'][:-1],              # RV²_IBM,t-1
            rv_squared['GS'][:-1],               # RV²_GS,t-1
            rv_squared['IBM'][:-1] ** 2          # RV⁴_IBM,t-1
        ])
        y_ibm = rv_squared['IBM'][1:]
        model_ibm = LinearRegression().fit(X_ibm, y_ibm)
        r2_ibm = r2_score(y_ibm, model_ibm.predict(X_ibm))
        print(f"\nIBM模型: RV²_IBM ~ RV²_IBM,t-1 + RV²_GS,t-1 + RV⁴_IBM,t-1")
        print(f"  β = [{model_ibm.coef_[0]:.4f}, {model_ibm.coef_[1]:.4f}, {model_ibm.coef_[2]:.4f}]")
        print(f"  R² = {r2_ibm:.4f}")
        results.append(('IBM', model_ibm.coef_, r2_ibm))
        
        # 2. 验证 JPM 模型: RV²_JPM,t = f(RV²_JPM,t-1, RV²_AXP,t-1, RV⁴_CVX,t-1)
        X_jpm = np.column_stack([
            rv_squared['JPM'][:-1],              # RV²_JPM,t-1
            rv_squared['AXP'][:-1],              # RV²_AXP,t-1
            rv_squared['CVX'][:-1] ** 2          # RV⁴_CVX,t-1
        ])
        y_jpm = rv_squared['JPM'][1:]
        model_jpm = LinearRegression().fit(X_jpm, y_jpm)
        r2_jpm = r2_score(y_jpm, model_jpm.predict(X_jpm))
        print(f"\nJPM模型: RV²_JPM ~ RV²_JPM,t-1 + RV²_AXP,t-1 + RV⁴_CVX,t-1")
        print(f"  β = [{model_jpm.coef_[0]:.4f}, {model_jpm.coef_[1]:.4f}, {model_jpm.coef_[2]:.4f}]")
        print(f"  R² = {r2_jpm:.4f}")
        results.append(('JPM', model_jpm.coef_, r2_jpm))
        
        # 3. 验证 GS 模型: RV²_GS,t = f(RV²_GS,t-1, RV²_BA,t-1)
        X_gs = np.column_stack([
            rv_squared['GS'][:-1],               # RV²_GS,t-1
            rv_squared['BA'][:-1]                # RV²_BA,t-1
        ])
        y_gs = rv_squared['GS'][1:]
        model_gs = LinearRegression().fit(X_gs, y_gs)
        r2_gs = r2_score(y_gs, model_gs.predict(X_gs))
        print(f"\nGS模型: RV²_GS ~ RV²_GS,t-1 + RV²_BA,t-1")
        print(f"  β = [{model_gs.coef_[0]:.4f}, {model_gs.coef_[1]:.4f}]")
        print(f"  R² = {r2_gs:.4f}")
        results.append(('GS', model_gs.coef_, r2_gs))
        
        print("\n" + "=" * 60)
        print("✓ 如果R²接近真实参数设置，说明异方差模型被正确实现")
        print("✓ 预期：所有β系数应接近0.1（初始设定值）")
        print("=" * 60)
        
        return results


def main():
    """主函数：生成模拟数据"""
    print("=" * 60)
    print("Figure 1 Network Data Simulation")
    print("=" * 60)
    
    # 初始化模拟器
    simulator = VolatilityNetworkSimulator(T=1000, base_vol=0.5)
    
    # 生成网络数据
    volatilities = simulator.generate_network_data()
    
    # 保存数据
    print("\n保存数据文件...")
    rv_df = simulator.save_to_csv(volatilities)
    
    # 可视化
    print("\n生成可视化...")
    simulator.visualize_network_effects(volatilities)
    
    # 相关性分析
    print("\n分析相关性...")
    corr_matrix = simulator.analyze_correlations(volatilities)
    
    # 验证因果关系
    print("\n验证因果关系...")
    causal_results = simulator.verify_causal_relationships(volatilities)
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print("数据统计摘要")
    print("=" * 60)
    print(rv_df.describe())
    
    print("\n相关性矩阵:")
    print(pd.DataFrame(corr_matrix, index=STOCKS, columns=STOCKS).round(3))
    
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("数据路径: data/mock/")
    print("可以在 simplified.py 中设置 DATA_PATH='data/mock' 使用这些数据")
    print("=" * 60)


if __name__ == '__main__':
    main()

