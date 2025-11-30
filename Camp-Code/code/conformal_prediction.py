import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

def conformal_prediction(valid_true, valid_pred, valid_pred_std, test_true, test_pred, test_pred_std, alpha=0.1):
    
    valid_true = np.array(valid_true)
    valid_pred = np.array(valid_pred)
    valid_pred_std = np.array(valid_pred_std)
    test_true = np.array(test_true)
    test_pred = np.array(test_pred)
    test_pred_std = np.array(test_pred_std)

    def variables_flatten(true, pred, pred_std):
        if true.shape != pred.shape or pred.shape != pred_std.shape:
            raise ValueError("All inputs must have the same shape")

        true_flat = true.flatten()
        pred_flat = pred.flatten()
        pred_std_flat = pred_std.flatten()

        non_nan_mask = ~np.isnan(true_flat)
        ture_1d = true_flat[non_nan_mask]
        pred_1d = pred_flat[non_nan_mask]
        pred_std_1d = pred_std_flat[non_nan_mask]

        return ture_1d, pred_1d, pred_std_1d, true.shape, ~non_nan_mask

    def variables_restore(true_1d, pred_1d, pred_std_1d, original_shape, nan_mask):
        total_elements = np.prod(original_shape)

        if len(true_1d) != len(pred_1d) or len(pred_1d) != len(pred_std_1d):
            raise ValueError("All inputs must have the same shape")

        if len(true_1d) != total_elements - np.sum(nan_mask):
            raise ValueError("Tensor length does not match the mask")

        def restore_single_variable(clean_data, original_shape, nan_mask):
            restored = np.empty(total_elements)
            restored[nan_mask] = np.nan
            restored[~nan_mask] = clean_data
            return restored.reshape(original_shape)

        true = restore_single_variable(true_1d, original_shape, nan_mask)
        pred = restore_single_variable(pred_1d, original_shape, nan_mask)
        pred_std = restore_single_variable(pred_std_1d, original_shape, nan_mask)

        true = np.nan_to_num(true, nan=0.0)
        pred = np.nan_to_num(pred, nan=0.0)
        pred_std = np.nan_to_num(pred_std, nan=0.0)

        return true, pred, pred_std
    
    valid_true_1d, valid_pred_1d, valid_pred_std_1d, valid_true_shape, valid_mask = variables_flatten(valid_true, valid_pred, valid_pred_std)
    test_true_1d, test_pred_1d, test_pred_std_1d, test_true_shape, test_mask = variables_flatten(test_true, test_pred, test_pred_std)

    def score_function_1(true, pred, pred_std):
        eps = 1e-5
        return np.abs(pred - true) / (pred_std + eps)

    def score_function_2(true, pred, pred_std):
        return (pred - true) ** 2 / (pred_std ** 2 + 1e-5)

    def score_function_3(true, pred, pred_std):
        return np.abs(pred - true)

    def score_function_4(true, pred, pred_std):
        return (pred - true) ** 2

    # Test different score functions
    score_functions = {
        "Score Function 1 (Standardized Residual)": score_function_1,
        "Score Function 2 (Squared Residual / Variance)": score_function_2,
        "Score Function 3 (Absolute Residual)": score_function_3,
        "Score Function 4 (Squared Residual)": score_function_4
    }

    results = {}

    for score_name, score_func in score_functions.items():
        scores = score_func(valid_true_1d, valid_pred_1d, valid_pred_std_1d)

        q = np.quantile(scores, 1 - alpha, method="higher")

        test_pred_1d_lower = test_pred_1d - q * test_pred_std_1d
        test_pred_1d_upper = test_pred_1d + q * test_pred_std_1d

        coverage = np.mean((test_true_1d >= test_pred_1d_lower) & (test_true_1d <= test_pred_1d_upper))

        valid_true, valid_pred, valid_pred_std = variables_restore(valid_true_1d, valid_pred_1d, valid_pred_std_1d, valid_true_shape, valid_mask)
        test_true, test_pred, test_pred_std = variables_restore(test_true_1d, test_pred_1d, test_pred_std_1d, test_true_shape, test_mask)
        test_true, test_pred_lower, test_pred_upper = variables_restore(test_true_1d, test_pred_1d_lower, test_pred_1d_upper, test_true_shape, test_mask)

        metrics = calculate_metrics(valid_true, valid_pred, test_true, test_pred, test_pred_lower, test_pred_upper)
        results[score_name] = {"coverage": coverage, "metrics": metrics}

    return results, test_pred_lower, test_pred_upper


def calculate_metrics(val_true, val_pred, test_true, test_pred, test_lower, test_upper):
    metrics = {}
    eps = 2e-2
    test_true_eps = test_true.copy()
    test_pred_eps = test_pred.copy()
    test_true_eps[np.where(test_true_eps <= eps)] = np.abs(test_true_eps[np.where(test_true_eps <= eps)]) + eps
    test_pred_eps[np.where(test_pred_eps <= eps)] = np.abs(test_pred_eps[np.where(test_pred_eps <= eps)]) + eps
    
    residuals = val_true - val_pred
    sigma = np.std(residuals)
    nll = -np.mean(norm.logpdf(test_true, loc=test_pred, scale=sigma))
    metrics['Negative Log Likelihood'] = nll
    
    mae = mean_absolute_error(test_true, test_pred)
    metrics['MAE'] = mae
    
    mape = mean_absolute_percentage_error(test_true_eps, test_pred_eps)
    metrics['MAPE'] = mape

    mse = mean_squared_error(test_true, test_pred)
    metrics['MSE'] = mse

    rmse = np.sqrt(mse)
    metrics['RMSE'] = rmse

    rae = np.sum(abs(test_pred_eps - test_true_eps)) / np.sum(abs(np.mean(test_true_eps) - test_true_eps))
    metrics['RAE'] = rae
    
    alpha_levels = np.linspace(0.1, 0.9, 9)
    cal_errors = []
    for alpha in alpha_levels:
        lower = np.percentile(val_pred - val_true, 100 * alpha/2)
        upper = np.percentile(val_pred - val_true, 100 * (1 - alpha/2))
        coverage = np.mean((test_true >= test_pred + lower) & (test_true <= test_pred + upper))
        cal_errors.append(np.abs(coverage - (1 - alpha)))
    metrics['CE'] = np.mean(cal_errors)
    
    pi_width = np.mean(test_upper - test_lower)
    metrics['MPIW'] = pi_width
    
    #coverage = np.mean((test_true >= test_lower) & (test_true <= test_upper))
    #metrics['Coverage'] = coverage
    
    return metrics
