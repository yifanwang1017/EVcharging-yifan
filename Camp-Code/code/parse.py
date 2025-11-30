import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go Spatio-temporal EV Charging Demand Prediction!")

    parser.add_argument('--device', type=int, default=0, help="CUDA.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--seq_len', type=int, default=12, help="The sequence length of input data.")
    parser.add_argument('--bs', type=int, default=32, help="The batch size of fine-tuning.")
    parser.add_argument('--epoch', type=int, default=20, help="The max epoch of the training process.")
    parser.add_argument('--total_fold', type=int, default=6, help="The fold used for spliting data in cross-validation")

    parser.add_argument('--model', type=str, default='gcn', help="The used model")
    parser.add_argument('--pred_len', type=int, default=3, help="The length of prediction interval.")
    parser.add_argument('--add_feat', type=str, default='all', help="Whether to use additional features for prediction")
    parser.add_argument('--add_static_feat', type=str, default='all', help="Whether to use additional static features for prediction")
    parser.add_argument('--fold', type=int, default=6, help="The current fold number for training data")
    parser.add_argument('--pred_type', type=str, default='region', help="Prediction at node or regional level")
    parser.add_argument('--feat', type=str, default='occ', help="Which feature to use for prediction")

    parser.add_argument('--patience', type=int, default=20, help="No improvement during 10 epochs")
    parser.add_argument('--min_delta', type=float, default=1e-4, help="Minimum improvement")
    parser.add_argument('--early_stop_threshold', type=float, default=1e-3, help="Loss function")

    parser.add_argument('--is_train', action='store_true', default=True)

    return parser.parse_args()