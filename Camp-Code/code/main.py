import utils
import torch
from parse import parse_args

import train
import numpy as np
from utils import split_cv, create_loaders, metrics
from conformal_prediction import conformal_prediction,calculate_metrics
import pandas as pd
from pandas import ExcelWriter

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.8, 0.1, 0.1

if __name__ == "__main__":

    args = parse_args()
    # device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    utils.set_seed(seed=args.seed, flag=True)
    feat, adj, extra_feat, static_feat, time= utils.read_data(args)
    print(
        f"Running {args.model} with feat={args.feat}, pre_l={args.pred_len}, fold={args.fold}, add_feat={args.add_feat}, pred_type(node)={args.pred_type}")

    # Initialize and train model
    net = utils.load_net(args, np.array(adj), device, feat, extra_feat, static_feat)

    (train_feat, valid_feat, test_feat,
     train_extra_feat, valid_extra_feat, test_extra_feat,
     train_static_feat, valid_static_feat, test_static_feat,
     scaler) = split_cv(args, time, feat, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, extra_feat, static_feat)

    train_loader, valid_loader, test_loader = create_loaders(train_feat, valid_feat, test_feat,
                                                             train_extra_feat, valid_extra_feat, test_extra_feat,
                                                             train_static_feat, valid_static_feat, test_static_feat,
                                                             args, device)
    moe = False
    if args.model == 'gcn-moe' or args.model == 'gin-moe' or args.model == 'gcn-spmoe' or args.model == 'gin-spmoe':
        moe = True
    if args.model == 'lo' or args.model == 'ar' or args.model == 'arima':
        optim = None
        loss_func =None
        args.is_train = False
        args.stat_model = True
        train_valid_feat = np.vstack((train_feat, valid_feat,test_feat[:args.seq_len+args.pred_len,:]))
        test_loader = [train_valid_feat,test_feat[args.pred_len+args.seq_len:,:]]
    else:
        optim = torch.optim.Adam(net.parameters(), weight_decay=0.00001)
        args.stat_model = False
        loss_func = torch.nn.MSELoss()
        if args.is_train:
            valid_true, valid_pred, valid_pred_std, best_epoch, best_valid_loss = train.training(args, net, optim, loss_func, train_loader, valid_loader, args.fold, moe)
            print('The best epoch is:', best_epoch)
    
    test_true, test_pred, test_pred_std = train.test(args, test_loader, feat, net, scaler, static_feat, moe)

if moe:
    metric, test_pred_lower, test_pred_upper = conformal_prediction(valid_true, valid_pred, valid_pred_std, test_true, test_pred, test_pred_std)
    for score_name, score_result in metric.items():
        print(f"--- {score_name} ---")
        print(f"Coverage: {score_result['coverage']:.4f}")
        for metric_name, metric_value in score_result["metrics"].items():
            print(f"{metric_name}: {metric_value:.4f}")
        print("\n")
else:
    name = ['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE']
    metric = metrics(test_true, test_pred)
    metric = [float('{:.4f}'.format(i)) for i in metric]
    for i in range(len(name)):
         print(name[i]+':', metric[i])

with ExcelWriter('test.xlsx') as writer:
        pd.DataFrame(test_true).to_excel(writer, index=False, sheet_name='test_true')
        pd.DataFrame(test_pred).to_excel(writer, index=False, sheet_name='test_pred')
        pd.DataFrame(test_pred_lower).to_excel(writer, index=False, sheet_name='lower')
        pd.DataFrame(test_pred_upper).to_excel(writer, index=False, sheet_name='upper')

with ExcelWriter('valid.xlsx') as writer:
        pd.DataFrame(valid_true).to_excel(writer, index=False, sheet_name='valid_true')
        pd.DataFrame(valid_pred).to_excel(writer, index=False, sheet_name='valid_pred')