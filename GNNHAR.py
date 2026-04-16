"""
Proposed GNNHAR models to forecast the realized volatility. 
Include HAR, GHAR, GNNHAR1L, GNNHAR2L, and GNNHAR3L, with different loss functions, implemented in PyTorch.
For linear regressions with MSE loss, we also provide another implementation in GHAR.py, through the LinearRegression class in sklearn.

This module implements Graph Neural Network extensions of the Heterogeneous Autoregressive (HAR) model
for volatility forecasting. The models capture both temporal dependencies (through HAR features) and 
cross-sectional dependencies (through graph neural networks) in financial time series data.

Key Models:
- HAR: Baseline heterogeneous autoregressive model
- GHAR: Graph HAR with single graph convolution
- GNNHAR1L/2L/3L: Multi-layer graph neural network extensions with 1, 2, or 3 GCN layers

The implementation supports:
- Multiple loss functions (MSE, QLIKE)
- Ensemble training with multiple random seeds
- Rolling window forecasting
- Adjacency matrix computation using Graphical Lasso
"""

import argparse
import os
from os.path import join

import numpy as np
import pandas as pd
from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, Subset
import torch.optim as optim

# Check if CUDA is available and set tensor type accordingly
# This allows the code to run on both CPU and GPU environments
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Command line argument parser for model configuration
# These parameters control all aspects of model training and architecture
parser = argparse.ArgumentParser()
parser.add_argument("--window", type=int, default=22, help="moving window size for rolling forecasts")
parser.add_argument("--horizon", type=int, default=1, help="forecasting horizon (days ahead)")
parser.add_argument("--valid_len", type=int, default=22, help="validation period length")
parser.add_argument("--model_name", type=str, default='GNNHAR1L', help="model architecture: HAR/GHAR/GNNHAR1L/GNNHAR2L/GNNHAR3L")
parser.add_argument("--adj_name", type=str, default='glasso', help="adjacency matrix method: glasso or none")
parser.add_argument("--universe", type=str, default='DJIA', help="asset universe (e.g., DJIA, SP500)")
parser.add_argument("--loss", type=str, default='MSE', help="loss function: MSE or QLike")
parser.add_argument("--n_epochs", type=int, default=5000, help="number of training epochs")
parser.add_argument("--n_hid", type=int, default=9, help="number of hidden neurons in GCN layers")
parser.add_argument("--batch_size", type=int, default=128, help="batch size for training")
parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
parser.add_argument("--ens", type=int, default=0, help="ensemble index (for parallel training)")
parser.add_argument("--numNN", type=int, default=1, help="number of neural networks in ensemble")
parser.add_argument("--version", type=str, default='Forecast_Var', help="experiment version name")

opt = parser.parse_args()
print(opt)

# Create a unique version string based on hyperparameters for model identification
# This ensures different configurations don't overwrite each other's results
this_version = '_'.join(
    [opt.version,
     opt.loss,
     opt.model_name,
     opt.adj_name,
     opt.universe,
     'E' + str(opt.n_epochs),
     'H' + str(opt.n_hid),
     'BS' + str(opt.batch_size),
     'LR' + str(opt.lr),
     'W' + str(opt.window),
     'F' + str(opt.horizon),
     'Val' + str(opt.valid_len)])

# Set up paths for data and model storage
# TODO: Update these paths to your actual data and model directories
path = 'your_local_path'  # Root path for data files
model_save_path = join('your_model_storage_path', this_version)  # Path to save trained models
os.makedirs(model_save_path, exist_ok=True)


def load_feature_data(universe):
    """
    Load feature data for the specified universe (e.g., DJIA).
    
    Args:
        universe (str): Asset universe identifier (e.g., 'DJIA', 'SP500')
        
    Returns:
        pd.DataFrame: Feature data with dates as index and assets as columns
        
    Note:
        - Uses forward fill to handle missing values
        - Filters data up to 2021-07-01 (training cutoff)
        - Sorts columns for consistent ordering
    """
    feature_df = pd.read_csv(join(path, 'Data', f'{universe}_var_FH1.csv'), index_col=0)
    feature_df.fillna(method="ffill", inplace=True)  # Forward fill missing values
    feature_df = feature_df[feature_df.index <= '2021-07-01']  # Filter by date
    feature_df = feature_df.sort_index(axis=1)  # Sort columns alphabetically
    return feature_df


def load_data(universe, horizon):
    """
    Load variance data for the specified universe and forecasting horizon.
    
    Args:
        universe (str): Asset universe identifier
        horizon (int): Forecasting horizon in days
        
    Returns:
        pd.DataFrame: Variance data (targets) with dates as index and assets as columns
        
    Note:
        This loads the target variables (realized variances) that we want to forecast
    """
    var_df = pd.read_csv(join(path, 'Data', f'{universe}_var_FH{horizon}.csv'), index_col=0)
    var_df.fillna(method="ffill", inplace=True)  # Forward fill missing values
    vech_df = var_df[var_df.index <= '2021-07-01']  # Filter by date
    vech_df = vech_df.sort_index(axis=1)  # Sort columns alphabetically
    return vech_df


def load_ret(universe):
    """
    Load return data for the specified universe.
    
    Args:
        universe (str): Asset universe identifier
        
    Returns:
        pd.DataFrame: Return data used for computing adjacency matrices
        
    Note:
        Return data is used to estimate the covariance/precision matrix for graph construction
    """
    ret_df = pd.read_csv(join(path, 'Data', f'{universe}_ret_FH1.csv'), index_col=0)
    ret_df.fillna(method="ffill", inplace=True)  # Forward fill missing values
    ret_df = ret_df[ret_df.index <= '2021-07-01']  # Filter by date
    ret_df = ret_df.sort_index(axis=1)  # Sort columns alphabetically
    return ret_df


def get_lag_avg(df, lag):
    """
    Calculate the average of lagged values for HAR model features.
    
    Args:
        df (pd.DataFrame): Input time series data
        lag (int): Number of lags to average over
        
    Returns:
        pd.DataFrame: Averaged lagged values
        
    Note:
        This creates the HAR components:
        - lag=1: Daily component (1-day average)
        - lag=5: Weekly component (5-day average) 
        - lag=22: Monthly component (22-day average)
    """
    res = pd.DataFrame(columns=df.columns, index=df.index).fillna(0)
    for l in range(1, lag + 1):
        res += (1 / lag) * df.shift(l)  # Average of past 'lag' periods
    return res


def preprocess_adj_l(date_l, subdf_dic, adj_df_l):
    """
    Preprocess adjacency matrices for graph neural network layers.
    
    Args:
        date_l (list): List of dates to process
        subdf_dic (dict): Dictionary mapping dates to dataframes
        adj_df_l (list): List of adjacency matrices
        
    Returns:
        pd.DataFrame: Transformed features with graph convolution applied
        
    Note:
        This function applies adjacency matrix transformations to lag features,
        effectively performing graph convolution preprocessing
    """
    new_subdf_l = []
    for date in date_l:
        subdf = subdf_dic[date]
        # print(subdf)
        tmp_subdf_l = []
        clms = [i for i in subdf.columns if 'lag' in i]  # Get lag columns
        # print(clms)
        for k, adj_df in enumerate(adj_df_l):
            # print(adj_df)
            # Apply adjacency matrix transformation to lag features
            # This is equivalent to one step of graph convolution: A * X
            tmp_subdf = pd.DataFrame(np.dot(adj_df, subdf[clms]), columns=['sec'+str(k)+i for i in clms], index=subdf.index)
            tmp_subdf_l.append(tmp_subdf)
        # Concatenate original target/date/ticker columns with transformed features
        new_subdf = pd.concat([subdf[['Target', 'Date', 'Ticker']]]+tmp_subdf_l, axis=1)
        new_subdf_l.append(new_subdf)

    df = pd.concat(new_subdf_l)
    df.reset_index(drop=True, inplace=True)
    print('Finish transformation!')
    return df


def GLASSO_Precision(subret):
    """
    Compute adjacency matrix using Graphical Lasso (GLASSO) precision matrix.
    
    Args:
        subret (pd.DataFrame): Return data for estimating precision matrix
        
    Returns:
        pd.DataFrame: Normalized adjacency matrix based on precision matrix sparsity
        
    Note:
        - Uses cross-validation to select optimal regularization parameter
        - Creates binary adjacency based on non-zero precision matrix entries
        - Applies symmetric normalization: D^(-1/2) * A * D^(-1/2)
        - Removes self-loops (diagonal entries)
    """
    from sklearn.covariance import GraphicalLassoCV
    n = subret.shape[1]
    tickers = subret.columns
    cov = GraphicalLassoCV().fit(subret)  # Fit GLASSO with cross-validation
    print('Alpha in GLASSO: %.3f' % cov.alpha_)  # Regularization parameter
    corr = cov.precision_ != 0  # Binary adjacency based on non-zero precision entries
    print('Sparsity of Adj: %.3f' % corr.mean())  # Fraction of non-zero entries
    corr_adj = corr - np.identity(n)  # Remove self-loops (diagonal entries)
    
    # Apply symmetric normalization: D^(-1/2) * A * D^(-1/2)
    # This ensures the adjacency matrix has proper scaling for GCN layers
    d_sqrt_inv = np.diag(np.sqrt(1/(corr_adj.sum(1)+1e-8)))  # Degree normalization
    adj_df = pd.DataFrame(np.dot(np.dot(d_sqrt_inv, corr_adj), d_sqrt_inv), columns=tickers, index=tickers)
    return adj_df


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer for processing node features with adjacency matrix.
    
    Implements the standard GCN operation: H' = A * H * W + b
    where A is the adjacency matrix, H is the input features, W is learnable weights
    
    Args:
        in_features (int): Number of input features per node
        out_features (int): Number of output features per node
        bias (bool): Whether to include bias term
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()

        # Learnable weight matrix W with Xavier initialization
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # Optional bias term initialized to ones
        if bias is True:
            self.bias = nn.Parameter(torch.FloatTensor(1, out_features))
            nn.init.ones_(self.bias)
        else:
            self.bias = None

    def forward(self, node_feature, adj):
        """
        Forward pass: H' = A * H * W + b
        
        Args:
            node_feature (torch.Tensor): Input node features [batch_size, N, in_features]
            adj (torch.Tensor): Adjacency matrix [N, N]
            
        Returns:
            torch.Tensor: Output node features [batch_size, N, out_features]
        """
        h = torch.matmul(node_feature, self.weight)  # Linear transformation: H * W
        output = torch.matmul(adj, h)  # Graph convolution: A * (H * W)
        if self.bias is not None:
            return output + self.bias
        return output

# HAR model - Heterogeneous Autoregressive model for volatility forecasting
class HAR(nn.Module):
    """
    HAR (Heterogeneous Autoregressive) model - baseline linear model.
    
    The HAR model captures volatility clustering by using three components:
    - Daily: 1-day lagged volatility
    - Weekly: 5-day average lagged volatility  
    - Monthly: 22-day average lagged volatility
    
    This is the baseline model without any graph structure.
    """
    def __init__(self):
        super(HAR, self).__init__()

        # Linear layer mapping 3 HAR features (daily, weekly, monthly) to 1 output
        self.linear1 = nn.Linear(3, 1, bias=True)
        self.relu = nn.ReLU()  # ReLU activation to ensure positive volatility

    def forward(self, node_feat, adj):
        """
        Forward pass for HAR model.
        
        Args:
            node_feat (torch.Tensor): HAR features [batch_size, N, 3] 
                                     (daily, weekly, monthly components)
            adj (torch.Tensor): Adjacency matrix [N, N] (not used in HAR)
            
        Returns:
            torch.Tensor: Predicted volatilities [batch_size, N]
        """
        H1 = self.linear1(node_feat)  # Linear transformation of HAR features
        res = self.relu(H1)  # Apply ReLU activation for positive outputs

        return res.squeeze(-1)  # Remove last dimension: [batch_size, N, 1] -> [batch_size, N]
    

# GHAR model - Graph HAR with single graph convolution
class GHAR(nn.Module):
    """
    GHAR (Graph HAR) model - HAR with single graph convolution layer.
    
    Extends HAR by adding a graph convolution component that captures
    cross-sectional dependencies between assets. The final output combines
    both the linear HAR component and the graph component via residual connection.
    """
    def __init__(self, n_hid):
        super(GHAR, self).__init__()

        # Linear HAR component (same as baseline HAR)
        self.linear1 = nn.Linear(3, 1, bias=True)

        # Graph convolution component
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        """
        Forward pass combining linear HAR and graph convolution.
        
        Args:
            node_feat (torch.Tensor): HAR features [batch_size, N, 3]
            adj (torch.Tensor): Adjacency matrix [N, N]
            
        Returns:
            torch.Tensor: Predicted volatilities [batch_size, N]
        """
        H1 = self.linear1(node_feat)  # Linear HAR component

        H2 = self.gcn1(node_feat, adj)  # Graph convolution component
        res = H1 + H2  # Residual connection combines both components
        res = self.relu(res)  # Apply activation

        return res.squeeze(-1)

# 1-layer GNNHAR - Graph Neural Network HAR with one GCN layer
class GNNHAR1L(nn.Module):
    """
    1-layer GNNHAR model with MLP after graph convolution.
    
    Architecture:
    1. Linear HAR component: 3 -> 1
    2. Graph component: 3 -> n_hid (GCN) -> 1 (MLP)
    3. Residual connection + activation
    
    The MLP after GCN allows for more complex transformations compared to GHAR.
    """
    def __init__(self, n_hid):
        super(GNNHAR1L, self).__init__()

        # Linear HAR component
        self.linear1 = nn.Linear(3, 1, bias=True)

        # Graph neural network component
        self.gcn1 = GraphConvLayer(3, n_hid, bias=False)  # Graph convolution
        self.mlp1 = nn.Linear(n_hid, 1, bias = False)  # MLP to reduce dimension
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        """
        Forward pass with graph convolution followed by MLP.
        
        Args:
            node_feat (torch.Tensor): HAR features [batch_size, N, 3]
            adj (torch.Tensor): Adjacency matrix [N, N]
            
        Returns:
            torch.Tensor: Predicted volatilities [batch_size, N]
        """
        H1 = self.linear1(node_feat)  # Linear HAR component

        H2 = self.gcn1(node_feat, adj)  # Graph convolution
        H2 = self.relu(H2)  # Activation after GCN
        H2 = self.mlp1(H2)  # MLP projection to output dimension

        res = H1 + H2  # Residual connection
        res = self.relu(res)  # Final activation

        return res.squeeze(-1)


class GNNHAR2L(nn.Module):
    """
    2-layer GNNHAR model with two graph convolution layers.
    
    Architecture:
    1. Linear HAR component: 3 -> 1
    2. Graph component: 3 -> n_hid (GCN1) -> n_hid (GCN2) -> 1 (MLP)
    3. Residual connection + activation
    
    Deeper graph component can capture more complex graph patterns and
    multi-hop relationships between assets.
    """
    def __init__(self, nhid):
        super(GNNHAR2L, self).__init__()

        # Linear HAR component
        self.linear1 = nn.Linear(3, 1, bias=True)

        # Two-layer graph neural network
        self.gcn1 = GraphConvLayer(3, nhid, bias=False)      # First GCN layer
        self.gcn2 = GraphConvLayer(nhid, nhid, bias = False) # Second GCN layer

        self.mlp1 = nn.Linear(nhid, 1, bias = False)  # Final projection layer
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        """
        Forward pass with two graph convolution layers.
        
        Args:
            node_feat (torch.Tensor): HAR features [batch_size, N, 3]
            adj (torch.Tensor): Adjacency matrix [N, N]
            
        Returns:
            torch.Tensor: Predicted volatilities [batch_size, N]
        """
        H1 = self.linear1(node_feat)  # Linear HAR component

        # 2-layer GCN with activations between layers
        H2 = self.relu(self.gcn1(node_feat, adj))  # First GCN + activation
        H2 = self.relu(self.gcn2(H2, adj))         # Second GCN + activation

        # Project to output dimension
        H2 = self.mlp1(H2)

        res = H1 + H2  # Residual connection
        res = self.relu(res)  # Final activation

        return res.squeeze(-1)


class GNNHAR3L(nn.Module):
    """
    3-layer GNNHAR model with three graph convolution layers.
    
    Architecture:
    1. Linear HAR component: 3 -> 1
    2. Graph component: 3 -> n_hid (GCN1) -> n_hid (GCN2) -> n_hid (GCN3) -> 1 (MLP)
    3. Residual connection + activation
    
    Deepest model that can capture the most complex graph patterns,
    but may be prone to overfitting with limited data.
    """
    def __init__(self, nhid):
        super(GNNHAR3L, self).__init__()

        # Linear HAR component
        self.linear1 = nn.Linear(3, 1, bias=True)

        # Three-layer graph neural network
        self.gcn1 = GraphConvLayer(3, nhid, bias=False)      # First GCN layer
        self.gcn2 = GraphConvLayer(nhid, nhid, bias = False) # Second GCN layer
        self.gcn3 = GraphConvLayer(nhid, nhid, bias = False) # Third GCN layer

        self.mlp1 = nn.Linear(nhid, 1, bias = False)  # Final projection layer
        self.relu = nn.ReLU()

    def forward(self, node_feat, adj):
        """
        Forward pass with three graph convolution layers.
        
        Args:
            node_feat (torch.Tensor): HAR features [batch_size, N, 3]
            adj (torch.Tensor): Adjacency matrix [N, N]
            
        Returns:
            torch.Tensor: Predicted volatilities [batch_size, N]
        """
        H1 = self.linear1(node_feat)  # Linear HAR component

        # 3-layer GCN with activations between layers
        H2 = self.relu(self.gcn1(node_feat, adj))  # First GCN + activation
        H2 = self.relu(self.gcn2(H2, adj))         # Second GCN + activation
        H2 = self.relu(self.gcn3(H2, adj))         # Third GCN + activation

        # Project to output dimension
        H2 = self.mlp1(H2)

        res = H1 + H2  # Residual connection
        res = self.relu(res)  # Final activation

        return res.squeeze(-1)
    

def Compute_Adj(ret_df, vech_df, date, date_l):
    """
    Compute adjacency matrix and time indices for training/validation/testing.
    
    Args:
        ret_df (pd.DataFrame): Return data for adjacency computation
        vech_df (pd.DataFrame): Variance data (not used for adjacency)
        date (str): Current forecasting date
        date_l (list): List of all available dates
        
    Returns:
        tuple: (adj_df, s_p, v_p, timestamp, f_p)
            - adj_df: Adjacency matrix as tensor
            - s_p: Start index for training period
            - v_p: Start index for validation period  
            - timestamp: Current date index
            - f_p: End index for forecasting period
            
    Note:
        Time periods are defined as:
        - Training: [s_p, v_p) - up to 1000 days before current date
        - Validation: [v_p, timestamp) - last opt.valid_len days before current date
        - Testing: [timestamp, f_p) - next opt.window days after current date
    """
    timestamp = date_l.index(date)
    # Define time periods for training, validation, and forecasting
    s_p = max(timestamp-1000, 0)  # Start of training period (up to 1000 days back)
    v_p = timestamp - opt.valid_len  # Start of validation period
    f_p = min(timestamp + opt.window, len(date_l)-1)  # End of forecasting period

    s_date = date_l[s_p]
    v_date = date_l[v_p]
    f_date = date_l[f_p]

    # Get return data for adjacency matrix computation (only past data)
    subret = ret_df[ret_df.index < date]
    subret = subret[subret.index >= s_date]

    # Get variance data for the same period (for consistency, though not used for adjacency)
    subdata = vech_df[vech_df.index < date]
    subdata = subdata[subdata.index >= s_date]

    n = vech_df.shape[1]
    adj_name = opt.adj_name
    tickers = subret.columns

    # Compute adjacency matrix based on specified method
    if adj_name == 'glasso':
        adj_df = GLASSO_Precision(subret)  # Use Graphical Lasso
    else:
        # Default to zero adjacency (equivalent to no graph structure)
        adj_df = pd.DataFrame(np.zeros((n, n)), columns=tickers, index=tickers)

    print((s_date, v_date, f_date))  # Print time period for debugging
    adj_df = Tensor(adj_df.values)  # Convert to tensor for PyTorch
    return adj_df, s_p, v_p, timestamp, f_p


def df2arr(df, vars_l):
    """
    Convert DataFrame to PyTorch tensors for model input.
    
    Args:
        df (pd.DataFrame): Input dataframe
        vars_l (list): List of variable column names
        
    Returns:
        tuple: (all_inputs, all_targets) as PyTorch tensors
        
    Note:
        This is a utility function for data preprocessing, though not used in main pipeline
    """
    all_inputs = Tensor(df[vars_l].values)
    all_targets = Tensor(df[['Target']].values)
    return all_inputs, all_targets


class Loss(nn.Module):
    """
    Custom loss function supporting both MSE and QLIKE losses.
    
    QLIKE (Quasi-Likelihood) loss is specifically designed for volatility forecasting
    and is more robust to outliers compared to MSE. It's defined as:
    QLIKE = mean(y_pred/y_true - log(y_pred/y_true))
    
    MSE is the standard mean squared error loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, outputs, forecast_y):
        """
        Compute loss based on specified loss function.
        
        Args:
            outputs (torch.Tensor): Model predictions
            forecast_y (torch.Tensor): True target values
            
        Returns:
            torch.Tensor: Computed loss value
        """
        if opt.loss == 'QLike':
            # QLIKE loss for volatility forecasting
            # Add small epsilon to prevent division by zero
            true_fore = outputs / (forecast_y + 1e-4)  # Stabilize training
            l_v = torch.mean(true_fore - torch.log(true_fore))
        else:
            # Standard MSE loss
            mseloss = nn.MSELoss()
            l_v = mseloss(outputs, forecast_y)
        return l_v


def Train_Single(train_loader, valid_loader, model_index, seed, date):
    """
    Train a single model instance with specified random seed.
    
    Args:
        train_loader (DataLoader): Training data loader
        valid_loader (DataLoader): Validation data loader
        model_index (int): Index of current model in ensemble
        seed (int): Random seed for reproducibility
        date (str): Current forecasting date
        
    Returns:
        pd.DataFrame: Training history with train/validation losses
        
    Note:
        - Uses early stopping based on validation loss
        - Saves best model weights during training
        - Returns loss history for convergence analysis
    """
    torch.manual_seed(seed)  # Set random seed for reproducibility
    print("------ Model %d Starts with Random Seed %d " % (model_index, seed))
    
    # Initialize model based on specified architecture
    if opt.model_name == 'HAR':
        model = HAR()
    elif opt.model_name == 'GHAR':
        model = GHAR(opt.n_hid)
    elif opt.model_name == 'GNNHAR1L':
        model = GNNHAR1L(opt.n_hid)
    elif opt.model_name == 'GNNHAR2L':
        model = GNNHAR2L(opt.n_hid)
    elif opt.model_name == 'GNNHAR3L':
        model = GNNHAR3L(opt.n_hid)
    else:
        print('Please choose the correct model')
        return
    
    if cuda:
        model.cuda()  # Move model to GPU if available

    # Print model parameters for debugging (can be commented out for cleaner output)
    for parameter in model.parameters():
        print(parameter)

    # Set up optimizer and loss function
    loss_function = Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    best_val_mse = 1e8  # Track best validation loss for model saving

    train_loss = []
    valid_loss = []
    
    # Training loop
    for epoch in range(opt.n_epochs):
        epoch_loss_train = []
        epoch_loss_valid = []

        # Training phase
        model.train()  # Set model to training mode
        for _, (train_X, train_y) in enumerate(train_loader):
            train_X, train_y = Variable(train_X), Variable(train_y)

            optimizer.zero_grad()  # Clear gradients from previous step

            # Forward pass
            forecast_y = model(train_X, adj_df)
            loss = loss_function(train_y, forecast_y)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            epoch_loss_train.append(loss.item())

        # Validation phase
        model.eval()  # Set model to evaluation mode
        for _, (val_X, val_y) in enumerate(valid_loader):
            val_X, val_y = Variable(val_X), Variable(val_y)

            val_out = model(val_X, adj_df)
            loss = loss_function(val_y, val_out)
            epoch_loss_valid.append(loss.item())

        # Calculate epoch losses
        train_loss_epoch = np.mean(epoch_loss_train)
        valid_loss_epoch = np.mean(epoch_loss_valid)
        train_loss.append(train_loss_epoch)
        valid_loss.append(valid_loss_epoch)

        # Print progress periodically (every 10% of total epochs)
        if epoch % int(opt.n_epochs/10) == 0:
            print("[Epoch %d] [Train Loss: %.4f] [Valid Loss: %.4f]" % (epoch, train_loss_epoch, valid_loss_epoch))

        # Save best model based on validation loss (early stopping)
        if loss.item() < best_val_mse:
            best_val_mse = loss.item()
            torch.save(model.state_dict(), join(model_save_path, 'Best_Model' + '_' + date + '_index%d' % model_index))

    # Save training history for analysis
    train_loss_arr = np.array(train_loss)
    valid_loss_arr = np.array(valid_loss)
    loss_arr = np.stack([train_loss_arr, valid_loss_arr], axis=1)
    loss_df = pd.DataFrame(loss_arr, columns=['Train', 'Valid'])
    loss_df.to_csv(join(model_save_path, 'loss_%s_index%d.csv' % (date, model_index)), index=False)
    return loss_df


def Train(dataset, adj_df, s_p, v_p, timestamp, f_p, targets, date):
    """
    Main training function that handles data splitting and model ensemble training.
    
    Args:
        dataset (TensorDataset): Complete dataset with features and targets
        adj_df (torch.Tensor): Adjacency matrix for current time period
        s_p (int): Start index for training period
        v_p (int): Start index for validation period
        timestamp (int): Current date index
        f_p (int): End index for forecasting period
        targets (pd.DataFrame): Target dataframe for indexing
        date (str): Current forecasting date
        
    Note:
        - Splits data temporally (no data leakage)
        - Trains ensemble of models with different random seeds
        - Generates forecasts for testing period
        - Implements restart mechanism for poorly converged models
    """
    # Split data into train/validation/test sets based on time indices
    # This ensures no look-ahead bias in the forecasting setup
    train_idx = range(s_p, v_p)      # Training period
    val_idx = range(v_p, timestamp)   # Validation period  
    test_idx = range(timestamp, f_p)  # Testing/forecasting period

    # Create dataset subsets
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_idx), shuffle=False)

    # Train multiple models with different random seeds for ensemble
    for iii in range(opt.ens*opt.numNN, (opt.ens+1)*opt.numNN):
        seed = np.random.randint(low=1, high=10000)
        loss_df = Train_Single(train_loader, valid_loader, model_index=iii, seed=seed, date=date)

        # Restart training if model doesn't converge properly
        # Check for: 1) Loss plateauing (no improvement), 2) Loss explosion
        while (np.abs(loss_df['Valid'].diff()) < 1e-6).mean() > 0.5 or loss_df['Valid'].iloc[-1] > 100:
            print(' * ' * 20)
            print('  Attention!!!   Restart Training!!!  ')
            print(' * ' * 20)
            seed = np.random.randint(low=1, high=10000)
            loss_df = Train_Single(train_loader, valid_loader, model_index=iii, seed=seed, date=date)

    # Generate forecasts for testing period using trained models
    for iii in range(opt.ens*opt.numNN, (opt.ens+1)*opt.numNN):
        with torch.no_grad():  # Disable gradient computation for inference
            # Initialize model architecture (same as training)
            if opt.model_name == 'HAR':
                model = HAR()
            elif opt.model_name == 'GHAR':
                model = GHAR(opt.n_hid)
            elif opt.model_name == 'GNNHAR1L':
                model = GNNHAR1L(opt.n_hid)
            elif opt.model_name == 'GNNHAR2L':
                model = GNNHAR2L(opt.n_hid)
            elif opt.model_name == 'GNNHAR3L':
                model = GNNHAR3L(opt.n_hid)
            else:
                print('Please choose the correct model')
                return
    
            # Load trained model weights
            model.load_state_dict(torch.load(join(model_save_path, 'Best_Model' + '_' + date + '_index%d' % iii)))
            model.eval()  # Set to evaluation mode

            if cuda:
                model.cuda()

            # Generate predictions on test set
            for _, (test_X, test_y) in enumerate(test_loader):
                test_X, test_y = Variable(test_X), Variable(test_y)
                forecast_test_y = model(test_X, adj_df)

        # Save predictions to file
        y_pred = forecast_test_y.cpu().detach().numpy()
        test_pred_df = pd.DataFrame(y_pred, index=targets.index[test_idx], columns=targets.columns)

        print('Min: %.3f' % test_pred_df.min().min())  # Check for negative predictions

        save_path = join(path, 'Var_Pred_Results', this_version)
        os.makedirs(save_path, exist_ok=True)

        test_pred_df.to_csv(join(save_path, 'Pred_%s_Ens%d.csv' % (date, iii)))


def Screen_Ensemble(date, thres_perc=50):
    """
    Screen ensemble models based on validation performance.
    Only use models with validation loss below the specified percentile threshold.
    
    Args:
        date (str): Current forecasting date
        thres_perc (int): Percentile threshold for model selection (default: 50th percentile)
        
    Returns:
        list: Indices of selected models that converged well
        
    Note:
        This prevents look-ahead bias by using only training/validation data for selection.
        Models with poor convergence (high validation loss) are excluded from ensemble.
    """
    loss_l = []
    # Collect validation losses from all ensemble models
    for j in range(opt.numNN):
        loss_df = pd.read_csv(join(model_save_path, 'loss_%s_index%d.csv' % (date, j)))
        loss_l.append(loss_df['Valid'].iloc[-1])  # Final validation loss

    # Set threshold based on percentile of validation losses
    threshold_loss = np.percentile(loss_l, thres_perc)
    select_l = []
    
    # Select models with validation loss below threshold
    for j in range(opt.numNN):
        if loss_l[j] <= threshold_loss:
            select_l.append(j)
        else:
            pass  # Exclude poorly converged models
    return select_l


def connect_pred():
    """
    Connect forecasts from different time periods and ensemble models.
    Creates final forecast variance matrix with shape T_t * N, where T_t is the 
    length of the entire testing period and N is the number of assets.
    
    Process:
    1. Load predictions from all time periods
    2. Screen ensemble models based on convergence
    3. Average predictions across selected models
    4. Concatenate predictions across time periods
    5. Scale by forecasting horizon
    
    Note:
        This creates the final out-of-sample forecast file that can be used
        for performance evaluation against realized volatilities.
    """
    save_path = join(path, 'Var_Pred_Results', this_version)
    files_l = os.listdir(save_path)
    
    # Extract unique dates from prediction files
    # File naming convention: 'Pred_{date}_Ens{index}.csv'
    dates_l = [i.split('_')[1] for i in files_l if 'Pred_' in i and '_Ens0.csv']
    dates_l = list(set(dates_l))
    dates_l.sort()  # Ensure chronological order

    test_pred_df_l = []
    
    # Process predictions for each forecasting date
    for date in dates_l:
        tmp_pred_df_l = []
        select_l = Screen_Ensemble(date)  # Get well-converged models
        
        # Load predictions from selected ensemble models
        for j in select_l:
            tmp_test_pred_df = pd.read_csv(join(save_path, '_'.join(['Pred', date, 'Ens%d.csv' % j])), index_col=0)
            tmp_pred_df_l.append(tmp_test_pred_df)

        # Average predictions across selected ensemble models
        # This reduces overfitting and improves robustness
        test_pred_df = pd.DataFrame(np.stack(tmp_pred_df_l).mean(0), index=tmp_test_pred_df.index, columns=tmp_test_pred_df.columns)
        test_pred_df_l.append(test_pred_df)

    # Concatenate all predictions and scale by forecasting horizon
    # Scaling accounts for the fact that longer horizons typically have higher volatility
    test_pred_df = pd.concat(test_pred_df_l) * opt.horizon
    print(test_pred_df)

    # Save final predictions
    sum_path = join(path, 'Var_Results_Sum')
    os.makedirs(sum_path, exist_ok=True)
    test_pred_df.to_csv(join(sum_path, this_version + '_pred.csv'))


if __name__ == '__main__':
    """
    Main execution pipeline for GNNHAR volatility forecasting.
    
    Pipeline:
    1. Load data (features, targets, returns)
    2. Create HAR features (daily, weekly, monthly components)
    3. Rolling window training and forecasting
    4. Combine predictions across time periods
    
    The rolling window approach ensures realistic out-of-sample evaluation
    where models are retrained periodically with expanding windows.
    """
    # Load all required data
    feature_df = load_feature_data(opt.universe)  # Feature data for HAR components
    vech_df = load_data(opt.universe, opt.horizon)  # Target variance data
    ret_df = load_ret(opt.universe)  # Return data for adjacency matrix computation

    n = vech_df.shape[1]  # Number of assets in the universe

    # Create HAR features (daily, weekly, monthly averages) based on forecasting horizon
    # Skip first 22 days to allow for monthly lag computation
    if opt.horizon == 1:
        # For 1-day ahead forecasting, use all available data
        lag1 = get_lag_avg(feature_df, 1).iloc[22:]    # Daily (1-day average)
        lag5 = get_lag_avg(feature_df, 5).iloc[22:]    # Weekly (5-day average)
        lag22 = get_lag_avg(feature_df, 22).iloc[22:]  # Monthly (22-day average)
    else:
        # For multi-day ahead forecasting, adjust end index to prevent look-ahead
        e_idx = -opt.horizon + 1
        lag1 = get_lag_avg(feature_df, 1).iloc[22:e_idx]
        lag5 = get_lag_avg(feature_df, 5).iloc[22:e_idx]
        lag22 = get_lag_avg(feature_df, 22).iloc[22:e_idx]

    targets = vech_df.iloc[22:]  # Target values (skip first 22 days for lag computation)

    # Convert to numpy arrays for processing
    Y, lag1, lag5, lag22 = np.array(targets), np.array(lag1), np.array(lag5), np.array(lag22)

    Y /= opt.horizon  # Normalize targets by forecasting horizon

    # Stack HAR features into input tensor
    # Shape: (T, N, 3) where T=time, N=assets, 3=HAR components
    X = [lag1, lag5, lag22]
    X = np.stack(X, axis=-1)  # Stack along last dimension
    
    # Convert to PyTorch tensors
    X, Y = Tensor(X), Tensor(Y)

    # Create dataset for PyTorch DataLoader
    dataset = TensorDataset(X, Y)

    print('Training Starts Now ...')
    date_l = targets.index.tolist()
    idx = date_l.index('2011-07-01')  # Start training from this date (out-of-sample period)

    # Rolling window training: train model for each date with specified window
    # This simulates realistic trading conditions where models are retrained periodically
    for date in date_l[idx::opt.window]:
        print(' * ' * 20 + date + ' * ' * 20)
        # Compute adjacency matrix and time indices for current date
        adj_df, s_p, v_p, timestamp, f_p = Compute_Adj(ret_df, vech_df, date, date_l)
        # Train models and generate forecasts
        Train(dataset, adj_df, s_p, v_p, timestamp, f_p, targets, date)

    # Combine all forecasts into final prediction file
    connect_pred()