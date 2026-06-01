import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.covariance import GraphicalLassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ==========================================
# 1. 工具函数与数据预处理
# ==========================================

def calculate_har_features(df, lags=[1, 5, 22]):
    """
    根据定义计算 HAR 特征:
    Day: t-1
    Week: average(t-5 ... t-2)
    Month: average(t-22 ... t-6)
    """
    df_values = df.values
    N, T = df_values.shape[1], df_values.shape[0]
    
    # 特征容器: T x N x 3 (Day, Week, Month)
    features = np.zeros((T, N, 3))
    
    for t in range(22, T):
        # Day: v_{t-1}
        features[t, :, 0] = df_values[t-1, :]
        
        # Week: v_{t-5:t-2} -> mean of indices t-5 to t-2 (inclusive)
        # Python slice t-5 : t-1 (excl t-1) is indices t-5, t-4, t-3, t-2
        features[t, :, 1] = np.mean(df_values[t-5:t-1, :], axis=0)
        
        # Month: v_{t-22:t-6} -> mean of indices t-22 to t-6
        features[t, :, 2] = np.mean(df_values[t-22:t-5, :], axis=0)
        
    # 对齐数据，去掉前22天无法计算的部分
    return features[22:], df_values[22:]

def get_glasso_adj(returns, alpha=0.01):
    """
    使用 GLASSO 计算邻接矩阵 W
    returns: DataFrame or array (T x N)
    """
    # 标准化
    scaler = StandardScaler()
    ret_std = scaler.fit_transform(returns)
    
    # Graphical Lasso
    # 注意：实际应用中可能需要调节 alpha 参数
    try:
        glasso = GraphicalLassoCV(cv=3)
        glasso.fit(ret_std)
        cov = glasso.covariance_
        prec = glasso.precision_
    except:
        # 如果 CV 失败，回退到固定 alpha
        from sklearn.covariance import GraphicalLasso
        glasso = GraphicalLasso(alpha=alpha)
        glasso.fit(ret_std)
        prec = glasso.precision_
        
    # 取绝对值作为权重，忽略对角线
    adj = np.abs(prec)
    np.fill_diagonal(adj, 0)
    
    # 归一化: D^{-1/2} A D^{-1/2}
    degrees = np.sum(adj, axis=1)
    # 避免除以0
    degrees[degrees==0] = 1e-5
    d_inv_sqrt = np.diag(np.power(degrees, -0.5))
    
    W = d_inv_sqrt @ adj @ d_inv_sqrt
    return torch.FloatTensor(W)

def get_random_adj(N):
    """生成随机邻接矩阵用于对比"""
    A = np.random.rand(N, N)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    
    degrees = np.sum(A, axis=1)
    d_inv_sqrt = np.diag(np.power(degrees, -0.5))
    W = d_inv_sqrt @ A @ d_inv_sqrt
    return torch.FloatTensor(W)

# ==========================================
# 2. 模型定义 (PyTorch)
# ==========================================

class QLIKELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, y_true, y_pred):
        # QLIKE: v/v_hat - log(v/v_hat) - 1
        # Equivalent to: v/v_hat + log(v_hat) - log(v) - 1
        # Ensure positivity
        y_pred = torch.clamp(y_pred, min=self.eps)
        loss = (y_true / y_pred) - torch.log(y_true / y_pred) - 1
        return torch.mean(loss)

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def predict(self, x_rv, x_iv=None, adj=None):
        raise NotImplementedError

class HAR_Model(BaseModel):
    def __init__(self, num_nodes, use_iv=False):
        super().__init__()
        self.use_iv = use_iv
        # RV HAR coefficients: beta (3 dims: d, w, m)
        # Shared across nodes or node-specific? 
        # 文档公式 implying node-specific alpha but often shared beta in panel HAR.
        # 为了更灵活，这里使用 Linear 层模拟，input_dim=3 (or 6), output_dim=1 per node.
        # 我们对每个节点使用共享权重的参数（Panel HAR）或者每个节点独立参数。
        # 你的公式 alpha_i 是独立的，beta 是共享的 (vector form V * beta)。
        # 这里实现 Panel 形式 (共享 beta)，但在 Output 层加个 Bias 作为 alpha
        
        input_dim = 3
        if use_iv:
            input_dim = 6
            
        self.linear = nn.Linear(input_dim, 1, bias=True) # Bias is alpha (averaged)
        # 如果需要每个节点独立的 alpha，可以使用 nn.Parameter(N) 手动加
        self.node_bias = nn.Parameter(torch.zeros(num_nodes))

    def forward(self, x_rv, x_iv=None, adj=None):
        # x_rv: (Batch, N, 3)
        if self.use_iv:
            x = torch.cat([x_rv, x_iv], dim=2) # (Batch, N, 6)
        else:
            x = x_rv # (Batch, N, 3)
            
        out = self.linear(x) # (Batch, N, 1)
        out = out.squeeze(-1) + self.node_bias # (Batch, N)
        return nn.functional.softplus(out) # Ensure positive volatility

class GHAR_Model(BaseModel):
    def __init__(self, num_nodes, use_iv=False):
        super().__init__()
        self.use_iv = use_iv
        # Own params
        self.linear_own = nn.Linear(3 if not use_iv else 6, 1, bias=False)
        # Neighbor params
        self.linear_neighbor = nn.Linear(3 if not use_iv else 6, 1, bias=False)
        
        self.node_bias = nn.Parameter(torch.zeros(num_nodes))

    def forward(self, x_rv, x_iv=None, adj=None):
        # x_rv: (Batch, N, 3)
        if self.use_iv:
            x = torch.cat([x_rv, x_iv], dim=2)
        else:
            x = x_rv
            
        # 自身项: X * beta
        own_term = self.linear_own(x).squeeze(-1) # (Batch, N)
        
        # 邻居项: W * X * gamma
        # 需要先聚合邻居特征: (W @ X_transpose).transpose
        # PyTorch batch matmul:
        # x: (B, N, F) -> permute -> (B, F, N)
        # adj: (N, N)
        # WX: adj @ x (if x is N x F)
        # For batch: we want (W @ x[b])
        
        # x shape: (B, N, F)
        # adj shape: (N, N)
        # We need sum_j W_ij x_j
        neighbor_feat = torch.einsum('nm,bmf->bnf', adj, x)
        neighbor_term = self.linear_neighbor(neighbor_feat).squeeze(-1) # (Batch, N)
        
        out = own_term + neighbor_term + self.node_bias
        return nn.functional.softplus(out)

class GNNHAR_Model(BaseModel):
    def __init__(self, num_nodes, use_iv=False, hidden_dim=8, layers=2):
        super().__init__()
        self.use_iv = use_iv
        input_dim = 6 if use_iv else 3
        
        self.layers = nn.ModuleList()
        # Layer 0
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        # Layer 1...L-1
        for _ in range(layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.out_layer = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()
        
    def forward(self, x_rv, x_iv=None, adj=None):
        if self.use_iv:
            h = torch.cat([x_rv, x_iv], dim=2)
        else:
            h = x_rv
            
        for layer in self.layers:
            # GCN aggregation: W * H * Theta
            # 1. W * H
            h_agg = torch.einsum('nm,bmf->bnf', adj, h)
            # 2. * Theta (Linear layer)
            h = layer(h_agg)
            h = self.act(h)
            
        # Readout
        out = self.out_layer(h).squeeze(-1)
        return nn.functional.softplus(out)

# ==========================================
# 3. 训练与评估流程
# ==========================================

def train_model(model, train_data, test_data, adj, loss_type='MSE', epochs=200, lr=0.01):
    """
    train_data: tuple (x_rv, x_iv, y)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if loss_type == 'MSE':
        criterion = nn.MSELoss()
    else:
        criterion = QLIKELoss()
        
    train_x_rv, train_x_iv, train_y = train_data
    test_x_rv, test_x_iv, test_y = test_data
    
    # 转 Tensor
    adj = adj.to(torch.float32)
    train_x_rv = torch.FloatTensor(train_x_rv)
    train_x_iv = torch.FloatTensor(train_x_iv)
    train_y = torch.FloatTensor(train_y)
    test_x_rv = torch.FloatTensor(test_x_rv)
    test_x_iv = torch.FloatTensor(test_x_iv)
    test_y = torch.FloatTensor(test_y)
    
    loss_history = []
    
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        pred = model(train_x_rv, train_x_iv, adj)
        loss = criterion(train_y, pred)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
    # Evaluation
    model.eval()
    with torch.no_grad():
        pred_test = model(test_x_rv, test_x_iv, adj)
        # Calculate both MSE and QLIKE for reporting
        mse_val = mean_squared_error(test_y.numpy().flatten(), pred_test.numpy().flatten())
        
        # Manual QLIKE calc
        y_true = test_y
        y_pred = torch.clamp(pred_test, min=1e-6)
        qlike_val = torch.mean((y_true/y_pred) - torch.log(y_true/y_pred) - 1).item()
        
    return mse_val, qlike_val

# ==========================================
# 4. 主程序：复现你的测试目标
# ==========================================

def main_experiment():
    # --- A. 数据加载 (这里使用模拟数据，请替换为读取你的 'test.txt' 中的文件) ---
    print("正在准备数据...")
    N_assets = 30 # Dow 30
    T_days = 500
    
    # 模拟 RV 和 IV (对数正态分布)
    np.random.seed(42)
    df_RV = pd.DataFrame(np.exp(np.random.randn(T_days, N_assets)*0.5 - 2), columns=[f'Asset_{i}' for i in range(N_assets)])
    df_IV = pd.DataFrame(np.exp(np.random.randn(T_days, N_assets)*0.4 - 1.8), columns=[f'Asset_{i}' for i in range(N_assets)])
    # 模拟 Returns (用于 GLASSO)
    df_Ret = pd.DataFrame(np.random.randn(T_days, N_assets)*0.01, columns=[f'Asset_{i}' for i in range(N_assets)])
    
    # 1. 构造 HAR 特征
    rv_feats, rv_target = calculate_har_features(df_RV)
    iv_feats, _ = calculate_har_features(df_IV)
    
    # 对齐 target (预测下一天，所以 target 对应 input 的下一行)
    # input: 0 ... T-2, target: 1 ... T-1
    X_RV = rv_feats[:-1]
    X_IV = iv_feats[:-1]
    Y = rv_target[1:]
    
    # 划分训练集测试集
    split = int(len(Y) * 0.8)
    train_data = (X_RV[:split], X_IV[:split], Y[:split])
    test_data = (X_RV[split:], X_IV[split:], Y[split:])
    
    # 2. 构造邻接矩阵
    print("构造邻接矩阵...")
    W_glasso = get_glasso_adj(df_Ret.iloc[:split+22]) # 仅使用训练集信息构造图
    W_random = get_random_adj(N_assets)
    
    results = []

    # --- 实验 1: 检验 W (Glasso) vs Random W ---
    print("\n--- 实验 1: W(Glasso) vs Random W ---")
    for w_name, w_mat in [("GLASSO", W_glasso), ("Random", W_random)]:
        # 测试 GHAR 和 GNNHAR
        for m_name, M_Class in [("GHAR", GHAR_Model), ("GNNHAR", GNNHAR_Model)]:
            model = M_Class(num_nodes=N_assets, use_iv=False) # 基础模型不带IV
            mse, ql = train_model(model, train_data, test_data, w_mat, loss_type='MSE')
            results.append({"Exp": "W_Check", "Model": m_name, "W": w_name, "LossFn": "MSE", "Test_MSE": mse, "Test_QL": ql})
            print(f"Model: {m_name}, W: {w_name} -> MSE: {mse:.6f}, QLIKE: {ql:.6f}")

    # --- 实验 2: 检验 Loss 函数 (MSE vs QLIKE) ---
    print("\n--- 实验 2: Loss Function Comparison (MSE vs QL) ---")
    # 使用最优的模型配置（假设 GLASSO 更好，使用 GNNHAR 为例）
    W_best = W_glasso
    for loss_name in ['MSE', 'QLIKE']:
        model = GNNHAR_Model(num_nodes=N_assets, use_iv=False)
        mse, ql = train_model(model, train_data, test_data, W_best, loss_type=loss_name)
        results.append({"Exp": "Loss_Check", "Model": "GNNHAR", "W": "GLASSO", "LossFn": loss_name, "Test_MSE": mse, "Test_QL": ql})
        print(f"Train Loss: {loss_name} -> Test MSE: {mse:.6f}, Test QLIKE: {ql:.6f}")

    # --- 实验 3: 检验外生变量 IV 的影响 (Base vs +IV) ---
    print("\n--- 实验 3: Exogenous Variable IV Impact ---")
    # 使用 MSE Loss 和 GLASSO W 作为基准
    models_to_test = [
        ("HAR", HAR_Model),
        ("GHAR", GHAR_Model),
        ("GNNHAR", GNNHAR_Model)
    ]
    
    for m_name, M_Class in models_to_test:
        # Without IV
        model_base = M_Class(num_nodes=N_assets, use_iv=False)
        mse_b, ql_b = train_model(model_base, train_data, test_data, W_glasso, loss_type='MSE')
        
        # With IV
        model_iv = M_Class(num_nodes=N_assets, use_iv=True)
        mse_iv, ql_iv = train_model(model_iv, train_data, test_data, W_glasso, loss_type='MSE')
        
        results.append({"Exp": "IV_Impact", "Model": m_name, "IV": "No", "Test_MSE": mse_b})
        results.append({"Exp": "IV_Impact", "Model": m_name + "+IV", "IV": "Yes", "Test_MSE": mse_iv})
        
        improv = (mse_b - mse_iv) / mse_b * 100
        print(f"{m_name} MSE: {mse_b:.6f} vs {m_name}+IV MSE: {mse_iv:.6f} (Improv: {improv:.2f}%)")

    # 汇总展示
    res_df = pd.DataFrame(results)
    print("\n所有实验结果汇总:")
    print(res_df)

if __name__ == "__main__":
    main_experiment()