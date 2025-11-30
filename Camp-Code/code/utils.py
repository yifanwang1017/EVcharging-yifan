
import torch 
import pandas as pd 
import numpy as np

import baselines
import frontlines
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score,mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import moe

class CreateDataset(Dataset):
    def __init__(self, args, occ, extra_feat, static_feat, device):
        lb = args.seq_len
        pt = args.pred_len
        self.pred_type = args.pred_type

        occ, label = create_rnn_data(occ, lb, pt)
        self.occ = torch.tensor(occ)
        self.label = torch.tensor(label)

        self.extra_feat = None
        if isinstance(extra_feat, np.ndarray) and extra_feat.size > 0:
            extra_feat, _ = create_rnn_data(extra_feat, lb, pt)
            self.extra_feat = torch.tensor(extra_feat)

        self.static_feat = torch.tensor(static_feat).unsqueeze(0)

        self.device = device

    def __len__(self):
        return len(self.occ)

    def __getitem__(self, idx):
        output_occ = torch.transpose(self.occ[idx], 0, 1).to(self.device)
        output_label = self.label[idx].to(self.device)
        output_static = self.static_feat[0].to(self.device)
        if self.extra_feat is not None:
            output_extra_feat = torch.transpose(self.extra_feat[idx], 0, 1).to(self.device)
            return output_occ, output_label, output_extra_feat, output_static
        else:
            return output_occ, output_label, output_static

def create_loaders(train_occ, valid_occ, test_occ, train_extra_feat, valid_extra_feat, test_extra_feat, train_static_feat, valid_static_feat, test_static_feat, args, device):
    train_dataset = CreateDataset(args, train_occ, train_extra_feat, train_static_feat, device)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, drop_last=True)

    valid_dataset = CreateDataset(args, valid_occ, valid_extra_feat, valid_static_feat, device)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_occ), shuffle=False)

    test_dataset = CreateDataset(args, test_occ, test_extra_feat, test_static_feat, device)
    test_loader = DataLoader(test_dataset, batch_size=len(test_occ), shuffle=False)

    return train_loader, valid_loader, test_loader



def read_data(args):
    """
    Read and preprocess the dataset for model input.
    """

    # Load datasets
    inf = pd.read_csv('Camp-Code/data/inf.csv', header=0, index_col=None)
    occ = pd.read_csv('Camp-Code/data/occupancy.csv', header=0, index_col=0)
    duration = pd.read_csv('Camp-Code/data/duration.csv', header=0, index_col=0)
    volume = pd.read_csv('Camp-Code/data/volume.csv', header=0, index_col=0)
    e_price = pd.read_csv('Camp-Code/data/e_price.csv', index_col=0, header=0).values
    s_price = pd.read_csv('Camp-Code/data/s_price.csv', index_col=0, header=0).values
    adj = pd.read_csv('Camp-Code/data/adj.csv', header=0, index_col=None)
    adj.index = adj.columns

    time = pd.to_datetime(occ.index)

    feat = occ
    if args.feat == 'duration':
        feat = duration
    elif args.feat == 'volume':
        feat = volume

    # Normalize
    charge_count_dict = dict(zip(inf['TAZID'].astype(str), inf['charge_count']))
    for col in occ.columns:
        charge_count = charge_count_dict[col]
        occ[col] = occ[col] / charge_count

    scaler = MinMaxScaler(feature_range=(0, 1))
    e_price = scaler.fit_transform(e_price)
    s_price = scaler.fit_transform(s_price)

    #Static feature
    static_feat = pd.read_csv(r'Camp-Code/data/static_feat.csv')
    static_feat = scaler.fit_transform(static_feat)

    # Load weather data
    weather = pd.read_csv(r'Camp-Code/data/weather_central.csv', header=0, index_col='time')

    extra_feat = 'None'
    if args.add_feat != 'None':
        extra_feat = np.zeros([occ.shape[0], occ.shape[1], 1])
        add_feat_list = args.add_feat.split('+')
        for add_feat in add_feat_list:
            if add_feat == 'all':
                extra_feat = np.concatenate([extra_feat, e_price[:, :, np.newaxis]], axis=2)
                extra_feat = np.concatenate([extra_feat, s_price[:, :, np.newaxis]], axis=2)
                extra_feat = np.concatenate([extra_feat,
                                             np.repeat(weather.values[:, np.newaxis, :], occ.shape[1], axis=1)], axis=2)
            elif add_feat == 'e':
                extra_feat = np.concatenate([extra_feat, e_price[:, :, np.newaxis]], axis=2)
            elif add_feat == 's':
                extra_feat = np.concatenate([extra_feat, s_price[:, :, np.newaxis]], axis=2)
            else:
                extra_feat = np.concatenate([extra_feat,
                                             np.repeat(weather[add_feat].values[:, np.newaxis, np.newaxis], occ.shape[1], axis=1)], axis=2)
        extra_feat = extra_feat[:, :, 1:]
    
    #print('Main feature shape:', np.array(feat).shape)
    #print('Extra featutre shape', extra_feat.shape)
    #print('Static featutre shape', static_feat.shape)
    return np.array(feat), np.array(adj), extra_feat, static_feat, time


def set_seed(seed, flag):
    if flag:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def division(data, train_rate=0.8, valid_rate=0.1, test_rate=0.1):
    data_length = len(data)
    train_division_index = int(data_length * train_rate)
    valid_division_index = int(data_length * (train_rate + valid_rate))
    test_division_index = int(data_length * (1 - test_rate))
    train_data = data[:train_division_index]
    valid_data = data[train_division_index:valid_division_index]
    test_data = data[test_division_index:]
    return train_data, valid_data, test_data


def load_net(args, adj, device, occ, extra_feat, static_feat):
    adj_dense  = torch.Tensor(adj).to(device)
    num_node = occ.shape[1] if args.pred_type =='region' else 1

    n_fea = 1
    if args.add_feat == 'all':
        n_fea = extra_feat.shape[-1] + 1 #9
    
    n_static_fea = 0
    if args.add_static_feat == 'all':
        n_static_fea = static_feat.shape[-1]

    if args.model == 'lstm':
        model = baselines.Lstm(args.seq_len, n_fea, node=num_node).to(device)
    elif args.model == 'lo':
        model = baselines.Lo(args)
    elif args.model == 'ar':
        model = baselines.Ar(pred_len=args.pred_len,lags=args.seq_len,args=args)
    elif args.model == 'arima':
        model = baselines.Arima(pred_len=args.pred_len,p=args.seq_len,args=args)
    elif args.model == 'fcnn':
        model = baselines.Fcnn(n_fea, node=num_node, seq=args.seq_len).to(device)
    elif args.model == 'gcnlstm':
        model = baselines.Gcnlstm(args.seq_len,adj_dense=adj_dense,n_fea=n_fea, node=num_node,gcn_out=32, gcn_layers=1,lstm_hidden_dim=32, lstm_layers=1
                 ,hidden_dim=32).to(device)
    elif args.model == 'gcn':
        model = baselines.Gcn(args.seq_len, n_fea=n_fea, n_static_fea=n_static_fea, adj_dense=adj_dense, gcn_hidden=32, gcn_layers=1).to(device)
    elif args.model == 'astgcn':
        model = baselines.Astgcn(adj_dense=adj_dense,nb_block=1,in_channels=n_fea, K=1, nb_chev_filter=32, nb_time_filter=32, time_strides=1,num_for_predict=1,len_input=12,num_of_vertices=num_node).to(device)
    elif args.model == 'gcn-moe':
        model = moe.MoE(args.seq_len, n_fea=n_fea, n_static_fea=n_static_fea, adj_dense=adj_dense, hidden_size=32, expert_layer=2, num_experts=3, expert_type='gcn', k=2).to(device)
    elif args.model == 'gcn-spmoe':
        model = frontlines.GNN_SpMoE(args.seq_len, n_fea=n_fea, n_static_fea=n_static_fea, adj_dense=adj_dense, gcn_hidden=32, gcn_layers=2, num_experts=3, gate_hidden=32, drop_ratio=0.1, gnn_type='gcn', k=2, coef=1e-2).to(device)
    return model


def create_rnn_data(dataset, lookback, predict_time):
    x = []
    y = []
    for i in range(len(dataset) - lookback - predict_time):
        x.append(dataset[i:i + lookback])
        y.append(dataset[i + lookback + predict_time - 1])
    return np.array(x), np.array(y)


def metrics(test_pre, test_real):
    eps = 2e-2
    MAPE_test_real = test_real.copy()
    MAPE_test_pre = test_pre.copy()
    MAPE_test_real[np.where(MAPE_test_real <= eps)] = np.abs(MAPE_test_real[np.where(MAPE_test_real <= eps)]) + eps
    MAPE_test_pre[np.where(MAPE_test_real <= eps)] = np.abs(MAPE_test_pre[np.where(MAPE_test_real <= eps)]) + eps

    MAPE = mean_absolute_percentage_error(MAPE_test_real, MAPE_test_pre)
    MAE = mean_absolute_error(test_real, test_pre)
    MSE = mean_squared_error(test_real, test_pre)
    RMSE = np.sqrt(MSE)
    RAE = np.sum(abs(MAPE_test_pre - MAPE_test_real)) / np.sum(abs(np.mean(MAPE_test_real) - MAPE_test_real))

    #print('MAPE: {}'.format(MAPE))
    #print('MAE:{}'.format(MAE))
    #print('MSE:{}'.format(MSE))
    #print('RMSE:{}'.format(RMSE))
    #print(('RAE:{}'.format(RAE)))
    output_list = [MSE, RMSE, MAPE, RAE, MAE]
    return output_list


def split_cv(args, time, feat, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1,
             extra_feat='None', static_feat='None'):
    """
    Split dataset based on time for time-series rolling cross-validation.
    static_feat: shape [num_nodes, static_dim], does NOT vary with time.
    """
    assert len(time) == len(feat)
    fold = args.fold
    month_list = list(time.month.unique())
    assert args.total_fold == len(month_list)
    fold_time = time.month.isin(month_list[0:fold]).sum()

    # ------- 时间切分 -------
    train_end = int(fold_time * train_ratio)
    valid_start = train_end
    valid_end = int(valid_start + fold_time * valid_ratio)
    test_start = valid_end
    test_end = int(fold_time)

    # ------- 时序主特征切分 -------
    train_feat = feat[:train_end]
    valid_feat = feat[valid_start:valid_end]
    test_feat = feat[test_start:test_end]

    # ------- 标准化 -------
    scaler = 'None'
    if args.pred_type == 'region':
        if args.feat != 'occ':
            scaler = StandardScaler()
            train_feat = scaler.fit_transform(train_feat)
            valid_feat = scaler.transform(valid_feat)
            test_feat = scaler.transform(test_feat)
    else:
        node_idx = int(args.pred_type)
        if args.feat != 'occ':
            scaler = StandardScaler()
            train_feat = scaler.fit_transform(train_feat[:, node_idx].reshape(-1, 1))
            valid_feat = scaler.transform(valid_feat[:, node_idx].reshape(-1, 1))
            test_feat = scaler.transform(test_feat[:, node_idx].reshape(-1, 1))
        else:
            train_feat = train_feat[:, node_idx].reshape(-1, 1)
            valid_feat = valid_feat[:, node_idx].reshape(-1, 1)
            test_feat = test_feat[:, node_idx].reshape(-1, 1)

    # ------- extra_feat 切分 -------
    train_extra_feat, valid_extra_feat, test_extra_feat = 'None', 'None', 'None'
    if isinstance(extra_feat, np.ndarray) and extra_feat.size > 0:
        train_extra_feat = extra_feat[:train_end]
        valid_extra_feat = extra_feat[valid_start:valid_end]
        test_extra_feat = extra_feat[test_start:test_end]

    # ------- static_feat 处理（不会切分） -------
    # static_feat: [num_nodes, static_dim], 不随时间变化
    train_static_feat = static_feat
    valid_static_feat = static_feat
    test_static_feat = static_feat

    return (
        train_feat, valid_feat, test_feat,
        train_extra_feat, valid_extra_feat, test_extra_feat,
        train_static_feat, valid_static_feat, test_static_feat,
        scaler
    )