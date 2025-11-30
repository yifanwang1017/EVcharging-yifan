import os
from tqdm import tqdm
import torch
import numpy as np
import utils
import pandas as pd

def training(args, net, optim, loss_func, train_loader, valid_loader, fold, moe):
    # 初始化变量
    best_valid_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_model_state = None
    
    # 提前停止参数
    early_stop_threshold = getattr(args, 'early_stop_threshold', 1e-6)  # 误差减少阈值
    patience = getattr(args, 'patience', 20)  # 容忍的epoch数
    min_delta = getattr(args, 'min_delta', 1e-4)  # 最小改进值
    
    net.train()
    
    for epoch in tqdm(range(args.epoch), desc='Training'):
        # 训练阶段
        net.train()
        train_loss = 0.0
        num_batches = 0
        
        for j, data in enumerate(train_loader):
            # 数据处理
            extra_feat = 'None'
            if args.add_feat != 'None':
                occupancy, label, extra_feat, static_feat = data
            else:
                occupancy, label = data

            optim.zero_grad()
            if moe:
                predict, loss = net(occupancy, extra_feat, static_feat)
            else:
                predict = net(occupancy, extra_feat, static_feat)
                loss = loss_func(predict, label)

            loss.backward()
            optim.step()
            
            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # 验证阶段
        net.eval()
        valid_loss = 0.0
        num_valid_batches = 0
        
        valid_true_list = []
        valid_pred_list = []
        valid_pred_std_list = []
        
        with torch.no_grad():
            for j, data in enumerate(valid_loader):
                extra_feat = 'None'
                if args.add_feat != 'None':
                    occupancy, label, extra_feat = data
                else:
                    occupancy, label = data

                if moe:
                    predict, loss = net(occupancy, extra_feat, static_feat)
                else:
                    predict = net(occupancy, extra_feat, static_feat)
                    loss = loss_func(predict, label)
                    
                valid_loss += loss.item()
                num_valid_batches += 1
                
                # 收集验证结果
                valid_true_list.append(label.cpu().numpy())
                valid_pred_list.append(predict.cpu().numpy())
                if moe:
                    valid_pred_std_list.append(predict_std.cpu().numpy())

        avg_valid_loss = valid_loss / num_valid_batches
        
        # 检查是否需要保存最佳模型
        if avg_valid_loss < best_valid_loss - min_delta:
            best_valid_loss = avg_valid_loss
            best_epoch = epoch
            patience_counter = 0
            
            # 保存最佳模型状态
            best_model_state = net.state_dict().copy()
            
            # 保存最佳模型到文件
            output_dir = 'Camp-Code/checkpoints/'
            os.makedirs(output_dir, exist_ok=True)
            path = (output_dir + args.model + '_' +
                    'feat-' + args.feat + '_' +
                    'pred_len-' + str(args.pred_len) + '_' +
                    'fold-' + str(args.fold) + '_' +
                    'node-' + str(args.pred_type) + '_' +
                    'add_feat-' + str(args.add_feat) + '_' +
                    'best_epoch-' + str(best_epoch) + '_' +
                    'val_loss-' + f"{best_valid_loss:.6f}" + '.pth')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optim.state_dict(),
                'valid_loss': best_valid_loss,
                'train_loss': avg_train_loss
            }, path)
            
            print(f"Epoch {epoch}: 最佳模型已保存，验证损失: {best_valid_loss:.6f}")
        else:
            patience_counter += 1
            print(f"Epoch {epoch}: 验证损失 {avg_valid_loss:.6f} 未改善，耐心计数: {patience_counter}/{patience}")

        # 检查提前停止条件
        if patience_counter >= patience:
            print(f"提前停止训练！在 epoch {epoch} 停止，最佳 epoch 为 {best_epoch}，最佳验证损失: {best_valid_loss:.6f}")
            break
            
        # 检查是否达到阈值
        if avg_valid_loss <= early_stop_threshold:
            print(f"达到损失阈值 {early_stop_threshold}，停止训练！")
            break

    # 训练结束后加载最佳模型
    if best_model_state is not None:
        net.load_state_dict(best_model_state)
        print(f"训练完成！加载最佳模型 (epoch {best_epoch}, 验证损失: {best_valid_loss:.6f})")
    else:
        print("训练完成！但未找到最佳模型，使用最终模型。")
        best_model_state = net.state_dict()

    # 最终验证以获取结果
    net.eval()
    valid_true = []
    valid_pred = []
    valid_pred_std = []
    
    with torch.no_grad():
        for j, data in enumerate(valid_loader):
            extra_feat = 'None'
            if args.add_feat != 'None':
                occupancy, label, extra_feat = data
            else:
                occupancy, label = data

            if moe:
                predict, loss = net(occupancy, extra_feat, static_feat)
            else:
                predict = net(occupancy, extra_feat, static_feat)

            valid_true.append(label.cpu().numpy())
            valid_pred.append(predict.cpu().numpy())

    # 合并结果
    valid_true = np.concatenate(valid_true, axis=0)
    valid_pred = np.concatenate(valid_pred, axis=0)
    valid_pred_std = np.concatenate(valid_pred_std, axis=0) if moe else np.array([])
    
    return valid_true, valid_pred, valid_pred_std, best_epoch, best_valid_loss

def test(args, test_loader, occ, net, scaler='None', static_feat=None, moe=True):
    # ----init---
    result_list = []
    predict_list = np.zeros([1, occ.shape[1]])
    predict_std_list = np.zeros([1, occ.shape[1]])
    label_list = np.zeros([1, occ.shape[1]])
    if args.pred_type != 'region':
        predict_list = np.zeros([1,1])
        predict_std_list = np.zeros([1,1])
        label_list = np.zeros([1,1])
    # ----init---
    
    if not args.stat_model:
        output_dir = 'Camp-Code/checkpoints/'
        os.makedirs(output_dir, exist_ok=True)
        
        # 寻找最佳模型文件
        model_files = []
        for file in os.listdir(output_dir):
            if (file.startswith(args.model + '_') and 
                f'fold-{args.fold}' in file and 
                f'node-{args.pred_type}' in file and
                'best_epoch' in file):
                model_files.append(file)
        
        if model_files:
            # 按验证损失排序，选择最佳模型
            best_file = None
            best_val_loss = float('inf')
            for file in model_files:
                try:
                    # 从文件名中提取验证损失
                    val_loss_str = file.split('val_loss-')[-1].replace('.pth', '')
                    val_loss = float(val_loss_str)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_file = file
                except:
                    continue
            
            if best_file:
                path = os.path.join(output_dir, best_file)
                print(f"加载最佳模型: {best_file}")
            else:
                # 如果没有找到最佳模型文件，使用原来的命名方式
                path = (output_dir + args.model + '_' +
                        'feat-' + args.feat + '_' +
                        'pred_len-' + str(args.pred_len) + '_' +
                        'fold-' + str(args.fold) + '_' +
                        'node-' + str(args.pred_type) + '_' +
                        'add_feat-' + str(args.add_feat) + '_' +
                        'epoch-' + str(args.epoch) + '.pth')
        else:
            # 如果没有找到最佳模型文件，使用原来的命名方式
            path = (output_dir + args.model + '_' +
                    'feat-' + args.feat + '_' +
                    'pred_len-' + str(args.pred_len) + '_' +
                    'fold-' + str(args.fold) + '_' +
                    'node-' + str(args.pred_type) + '_' +
                    'add_feat-' + str(args.add_feat) + '_' +
                    'epoch-' + str(args.epoch) + '.pth')
        
        # 加载模型
        checkpoint = torch.load(path, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'])
            print(f"加载模型完成，epoch: {checkpoint.get('epoch', 'N/A')}, "
                  f"验证损失: {checkpoint.get('valid_loss', 'N/A'):.6f}")
        else:
            net.load_state_dict(checkpoint)
            print("加载模型完成（旧格式）")
        
        net.eval()
        for j, data in enumerate(test_loader):
            extra_feat = 'None'
            if args.add_feat != 'None':
                occupancy, label, extra_feat = data
            else:
                occupancy, label = data
            with torch.no_grad():
                if moe:
                    predict, loss = net(occupancy, extra_feat, static_feat)
                else:
                    predict = net(occupancy, extra_feat, static_feat)
                    loss = loss_func(predict, label)

    else:
        train_valid_occ, test_occ = test_loader
        if moe:
            predict, loss = net(occupancy, extra_feat, static_feat)
        else:
            predict = net(train_valid_occ, test_occ)

        label = test_occ
    
    predict_list = np.concatenate((predict_list, predict), axis=0)
    if moe:
        predict_std_list = np.concatenate((predict_std_list, predict_std), axis=0)
    label_list = np.concatenate((label_list, label), axis=0)
    if scaler != 'None':
        predict_list = scaler.inverse_transform(predict_list)
        if moe:
            predict_std_list = scaler.inverse_transform(predict_std_list)
        label_list = scaler.inverse_transform(label_list)

    output_no_noise = utils.metrics(test_pre=predict_list[1:], test_real=label_list[1:])
    result_list.append(output_no_noise)

    # Adding model name, pre_l and metrics and so on to DataFrame
    result_df = pd.DataFrame(result_list, columns=['MSE', 'RMSE', 'MAPE', 'RAE', 'MAE'])
    result_df['model_name'] = args.model
    result_df['pred_len'] = args.pred_len
    result_df['fold'] = args.fold 

    # Save the results in a CSV file
    output_dir = 'Camp-Code/result' + '/' + 'main_exp' + '/' + 'region'
    os.makedirs(output_dir, exist_ok=True)
    csv_file = output_dir + '/' + f'results.csv'

    # Append the result if the file exists, otherwise create a new file
    if os.path.exists(csv_file):
        result_df.to_csv(csv_file, mode='a', header=False, index=False, encoding='gbk')
    else:
        result_df.to_csv(csv_file, index=False, encoding='gbk')
    
    return label_list[1:], predict_list[1:], predict_std_list[1:]