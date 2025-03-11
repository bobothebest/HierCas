'''
train.py 是整个实验的主脚本，负责数据加载、模型训练、验证和测试。

参数设置：通过 argparse 接收命令行参数，设置实验的超参数（如学习率、批次大小、层数等）。
日志记录：配置日志记录器，记录训练过程中的重要信息，便于调试和结果分析。
数据加载与预处理：
读取并处理级联数据，映射节点和边的ID，划分训练、验证和测试集。
构建邻接列表，并初始化 NeighborFinder 以便高效查询邻居信息。
模型初始化：实例化HierCas模型，配置优化器和损失函数（均方误差损失）。
训练循环：
前向传播：对于每个批次（即一个CAS），通过模型预测流行度。
损失计算与反向传播：计算预测值与真实值的均方误差，执行反向传播更新模型参数。
梯度裁剪：使用 clip_grad_norm_ 防止梯度爆炸。
学习率调度：使用 StepLR 调整学习率，提高训练稳定性。
评估函数：eval_one_epoch 在验证和测试集上评估模型性能，计算MSLE、SMAPE和R2等指标。
'''

import math
import logging
import time
import sys
import random
import argparse
import numpy as np
import torch
import pandas as pd
import math
from graph import NeighborFinder
from module import TGAN
from tqdm import tqdm
import os
import torch.nn as nn
import wandb
import torch.optim.lr_scheduler as lr_scheduler
import utils

exp_seed = 0
random.seed(exp_seed)
np.random.seed(exp_seed)
torch.manual_seed(exp_seed)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

### Argument and global variables
parser = argparse.ArgumentParser('Interface for experiments')
parser.add_argument('-d', '--data', type=str, help='data sources to use, try weibo or aps', default='weibo_1800')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=4)
parser.add_argument('--n_epoch', type=int, default=5, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
parser.add_argument('--gpu', type=int, default=1, help='idx for the gpu to use')
parser.add_argument('--feat_dim', type=int, default=64, help='Dimentions of the feature embedding')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
FEAT_DIM = args.feat_dim
BSIZE = args.bs
### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(time.ctime()))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)
'''
def eval_one_epoch(num_batch, cas_dict, s_idx, e_idx, tgan, cas, src, dst, ts, test_e_l, test_label_l):
    with torch.no_grad():
        tgan = tgan.eval()
        tr_loss = 0
        nb_tr_steps = 0
        scores = []
        labels = []
        batch_score = []
        batch_label = []
        for k in tqdm(range(0, num_batch), desc='Inference'):
            cas_l_cut = cas[s_idx:e_idx].to_numpy()
            src_l_cut = src[s_idx:e_idx].to_numpy()
            dst_l_cut = dst[s_idx:e_idx].to_numpy()
            ts_l_cut = ts[s_idx:e_idx].to_numpy()
            e_l_cut = test_e_l[s_idx:e_idx].to_numpy()
            label_l_cut = test_label_l[s_idx:e_idx].to_numpy()

            if k < num_batch - 1:
                s_idx = e_idx
                e_idx = s_idx + cas_dict[test_cas_l[s_idx]]

            score, att_score = tgan.forward(cas_l_cut, src_l_cut, dst_l_cut, ts_l_cut, e_l_cut, NUM_NEIGHBORS)
            label = label_l_cut[0] - max(e_l_cut)
            label = np.log2(label + 1)
            batch_score.append(score)  
            batch_label.append(label)

            scores.append(score.item())
            labels.append(float(label))

            if (k + 1) % BSIZE == 0 or k == num_batch - 1: 
                batch_score = torch.stack(batch_score).view(-1).to(device)
                batch_label = torch.as_tensor(batch_label, dtype=torch.float32).to(device)
                loss = criterion(batch_score, batch_label)

                batch_score = []
                batch_label = []
                tr_loss += loss.item()
                nb_tr_steps += 1

        test_loss = tr_loss / nb_tr_steps

        predictions = np.array(scores).reshape(-1, )
        test_labels = np.array(labels).reshape(-1, )
        MSLE = np.mean(np.square(predictions - test_labels))
    
        SMAPE = np.mean(np.abs(np.subtract(predictions, test_labels)) / ((np.abs(predictions) + np.abs(test_labels)) / 2))
        rss = np.mean(np.square(np.subtract(predictions, test_labels)))
        tss = np.mean(np.square(test_labels - np.mean(test_labels)))
        r2_score = 1 - rss / tss
        
    return test_loss, MSLE, SMAPE, r2_score
'''

def eval_one_epoch(num_batch, cas_dict, s_idx, e_idx, tgan, cas, src, dst, ts, test_e_l, test_label_l):
    # 打印模型第一层权重的前几个值，确认模型在更新
    for name, param in tgan.named_parameters():
        if 'node_embed.0.weight' in name:
            print(f"Model weights sample: {param[0][0:5].detach().cpu().numpy()}")
            break
    
    with torch.no_grad():
        tgan = tgan.eval()
        tr_loss = 0
        nb_tr_steps = 0
        scores = []
        labels = []
        batch_score = []
        batch_label = []
        for k in tqdm(range(0, num_batch), desc='Inference'):
            cas_l_cut = cas[s_idx:e_idx].to_numpy()
            src_l_cut = src[s_idx:e_idx].to_numpy()
            dst_l_cut = dst[s_idx:e_idx].to_numpy()
            ts_l_cut = ts[s_idx:e_idx].to_numpy()
            e_l_cut = test_e_l[s_idx:e_idx].to_numpy()
            label_l_cut = test_label_l[s_idx:e_idx].to_numpy()

            if k < num_batch - 1:
                s_idx = e_idx
                e_idx = s_idx + cas_dict[test_cas_l[s_idx]]

            score, att_score = tgan.forward(cas_l_cut, src_l_cut, dst_l_cut, ts_l_cut, e_l_cut, NUM_NEIGHBORS)
            
            # 修改标签计算方式
            raw_label = float(label_l_cut[0]) - float(max(e_l_cut))
            # 确保标签为正值
            label = max(raw_label, 1e-8)
            # 使用安全的对数计算
            label = np.log2(label + 1)
            
            batch_score.append(score)  
            batch_label.append(label)

            scores.append(score.item())
            labels.append(float(label))

            if (k + 1) % BSIZE == 0 or k == num_batch - 1: 
                batch_score = torch.stack(batch_score).view(-1).to(device)
                batch_label = torch.as_tensor(batch_label, dtype=torch.float32).to(device)
                
                # 在这里添加调试代码
                if len(batch_score) > 0 and len(batch_label) > 0:
                    print(f"Sample prediction: {batch_score[0].item()}, true label: {batch_label[0].item()}")
 
                
                loss = criterion(batch_score, batch_label)

                batch_score = []
                batch_label = []
                tr_loss += loss.item()
                nb_tr_steps += 1

        test_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else float('inf')

        predictions = np.array(scores).reshape(-1, )
        test_labels = np.array(labels).reshape(-1, )
        
        # 添加安全检查
        valid_indices = ~np.isnan(predictions) & ~np.isnan(test_labels)
        predictions = predictions[valid_indices]
        test_labels = test_labels[valid_indices]
        
        if len(predictions) > 0:
            MSLE = np.mean(np.square(predictions - test_labels))
            SMAPE = np.mean(np.abs(np.subtract(predictions, test_labels)) / ((np.abs(predictions) + np.abs(test_labels)) / 2 + 1e-8))
            rss = np.mean(np.square(np.subtract(predictions, test_labels)))
            tss = np.mean(np.square(test_labels - np.mean(test_labels)))
            r2_score = 1 - rss / tss if tss != 0 else 0
        else:
            MSLE = float('nan')
            SMAPE = float('nan')
            r2_score = float('nan')
        
    return test_loss, MSLE, SMAPE, r2_score
    
### Load data and train val test split
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
g_df.rename(columns={'size':'e_idx'}, inplace = True)
cas_list = g_df['cas'].unique().tolist()
num_graphs = len(cas_list)
node_list = pd.concat([g_df['src'], g_df['target']]).unique().tolist()
edge_list = g_df['e_idx'].unique().tolist()
# map id
casid_dict = {old_cas: new_cas for new_cas, old_cas in enumerate(cas_list)}
node_dict = {old_node: new_node for new_node, old_node in enumerate(node_list)}
#print("different node num: ", len(node_list))
edge_dict = {old_edge: new_edge for new_edge, old_edge in enumerate(edge_list)}
g_df['cas'] = g_df['cas'].apply(lambda x: casid_dict[x])
g_df['src'] = g_df['src'].apply(lambda x: node_dict[x])
g_df['target'] = g_df['target'].apply(lambda x: node_dict[x])
g_df['e_idx'] = g_df['e_idx'].apply(lambda x: edge_dict[x])
cas_num = len(cas_list)
# spilt train, val, test by cas_id to 7:1.5:1.5
val_cas_split = int(cas_num * 0.7)
test_cas_split = int(cas_num * 0.85)

grouped = g_df.groupby('cas')

np.random.seed(exp_seed)
group_order = np.random.permutation(list(grouped.groups.keys()))
shuffled_groups = [grouped.get_group(key) for key in group_order]

shuffled_g_df = pd.concat(shuffled_groups, ignore_index=True)

cas_l = shuffled_g_df.cas
src_l = shuffled_g_df.src
dst_l = shuffled_g_df.target
e_l = shuffled_g_df.e_idx 
ts_l = shuffled_g_df.ts
label_l = g_df.label
EDGE_NUM = max(e_l) + 1
NODE_NUM = max(max(src_l), max(dst_l)) + 1
cas_dict = dict()

for idx, cas in enumerate(cas_l):
    if cas not in cas_dict.keys():
        cas_dict[cas] = 1
    else:
        cas_dict[cas] += 1
#print(len(cas_dict.keys()), min(cas_dict.values()), np.mean(list(cas_dict.values())), max(cas_dict.values()))

train_cas = group_order[:val_cas_split]
test_cas = group_order[test_cas_split:]

train_df = pd.concat([grouped.get_group(cas) for cas in train_cas]).reset_index(drop=True)
test_df = pd.concat([grouped.get_group(cas) for cas in test_cas]).reset_index(drop=True)

train_cas_l = train_df.cas
train_src_l = train_df.src
train_dst_l = train_df.target
train_ts_l = train_df.ts
train_e_l = train_df.e_idx
train_label_l = train_df.label

test_cas_l = test_df.cas
test_src_l = test_df.src
test_dst_l = test_df.target
test_ts_l = test_df.ts
test_e_l = test_df.e_idx
test_label_l = test_df.label

train_cas_num = len(train_cas)
test_cas_num = len(test_cas)
val_cas_num = cas_num - train_cas_num - test_cas_num
print("total_cas_num: {}, train_cas_num: {}, val_cas_num: {}, test_cas_num: {}".format(cas_num, train_cas_num, val_cas_num, test_cas_num))

### Initialize the data structure for graph and edge sampling

train_max_idx = max(max(train_src_l), max(train_dst_l))
train_adj_list = [[] for _ in range(train_max_idx + 1)]
for cas, src, dst, eidx, ts in zip(train_cas_l, train_src_l, train_dst_l, train_e_l, train_ts_l):
    train_adj_list[src].append((cas, dst, eidx, ts))
    train_adj_list[dst].append((cas, src, eidx, ts))
train_ngh_finder = NeighborFinder(train_adj_list, uniform=UNIFORM)
print("train ngh_finder finish.")

test_max_idx = max(max(test_src_l), max(test_dst_l))
test_adj_list = [[] for _ in range(test_max_idx + 1)]
for cas, src, dst, eidx, ts in zip(test_cas_l, test_src_l, test_dst_l, test_e_l, test_ts_l):
    test_adj_list[src].append((cas, dst, eidx, ts))
    test_adj_list[dst].append((cas, src, eidx, ts))
test_ngh_finder = NeighborFinder(test_adj_list, uniform=UNIFORM)
print("test ngh_finder finish.")

device = torch.device('cuda:{}'.format(GPU))
#device = torch.device('cpu')
tgan = TGAN(train_ngh_finder, node_num=NODE_NUM, edge_num=EDGE_NUM, feat_dim=FEAT_DIM, device=device,
            num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT)
tgan = tgan.to(device)

optimizer = tgan.configure_optimizers(args)
#criterion = MeanSquaredLogError().to(device)
#criterion = nn.HuberLoss().to(device)
criterion = nn.MSELoss().to(device)
num_instance = len(train_src_l)
num_batch = train_cas_num
logger.info('num of training instances: {}'.format(num_instance))
logger.info('num of batches per epoch: {}'.format(num_batch))
#early_stopper = EarlyStopMonitor()


# wandb.init(project="GRLPP")
# wandb.config.update(args)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
'''
for epoch in range(NUM_EPOCH):
    # use a cas graph as a batch
    tgan.ngh_finder = train_ngh_finder
    s_idx, e_idx = 0, cas_dict[train_cas_l[0]]
    logger.info('start {} epoch'.format(epoch))
    tr_loss = 0
    nb_tr_steps = 0
    batch_score = []
    batch_label = []
    for k in tqdm(range(num_batch), desc='Train'):
        cas_l_cut = train_cas_l[s_idx:e_idx].to_numpy()
        src_l_cut = train_src_l[s_idx:e_idx].to_numpy()
        dst_l_cut = train_dst_l[s_idx:e_idx].to_numpy()
        ts_l_cut = train_ts_l[s_idx:e_idx].to_numpy()
        e_l_cut = train_e_l[s_idx:e_idx].to_numpy()
        label_l_cut = train_label_l[s_idx:e_idx].to_numpy()
        # print(train_cas_l[s_idx], cas_l_cut.shape)
        if k < num_batch - 1:
            s_idx = e_idx
            e_idx = s_idx + cas_dict[train_cas_l[s_idx]]
        tgan = tgan.train()
        score, att_score = tgan.forward(cas_l_cut, src_l_cut, dst_l_cut, ts_l_cut, e_l_cut, NUM_NEIGHBORS)
        #if epoch == NUM_EPOCH-1 and cas_l_cut.shape[0] == 99:
            #utils.plot_graph(k, src_l_cut, dst_l_cut, att_score)
        label = label_l_cut[0] - max(e_l_cut)
        label = np.log2(label + 1)
        batch_score.append(score)
        batch_label.append(label)
        if (k + 1) % BSIZE == 0 or k == num_batch - 1: 
            batch_score = torch.stack(batch_score).view(-1).to(device)
            batch_label = torch.as_tensor(batch_label, dtype=torch.float32).to(device)
            loss = criterion(batch_score, batch_label)
            #print(loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=tgan.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            tr_loss += loss.item()
            nb_tr_steps += 1
            batch_score = []
            batch_label = []

    train_loss = tr_loss / nb_tr_steps
   '''

# 主训练循环修改
for epoch in range(NUM_EPOCH):
    tgan.ngh_finder = train_ngh_finder
    s_idx, e_idx = 0, cas_dict[train_cas_l[0]]
    logger.info('start {} epoch'.format(epoch))
    tr_loss = 0
    nb_tr_steps = 0
    batch_score = []
    batch_label = []
    
    for k in tqdm(range(num_batch), desc='Train'):
        cas_l_cut = train_cas_l[s_idx:e_idx].to_numpy()
        src_l_cut = train_src_l[s_idx:e_idx].to_numpy()
        dst_l_cut = train_dst_l[s_idx:e_idx].to_numpy()
        ts_l_cut = train_ts_l[s_idx:e_idx].to_numpy()
        e_l_cut = train_e_l[s_idx:e_idx].to_numpy()
        label_l_cut = train_label_l[s_idx:e_idx].to_numpy()
        
        if k < num_batch - 1:
            s_idx = e_idx
            e_idx = s_idx + cas_dict[train_cas_l[s_idx]]
            
        tgan = tgan.train()
        score, att_score = tgan.forward(cas_l_cut, src_l_cut, dst_l_cut, ts_l_cut, e_l_cut, NUM_NEIGHBORS)
        
        # 修改标签计算方式
        raw_label = float(label_l_cut[0]) - float(max(e_l_cut))
        # 确保标签为正值
        label = max(raw_label, 1e-8)
        # 使用安全的对数计算
        label = np.log2(label + 1)
        
        batch_score.append(score)
        batch_label.append(label)
        
        if (k + 1) % BSIZE == 0 or k == num_batch - 1: 
            batch_score = torch.stack(batch_score).view(-1).to(device)
            batch_label = torch.as_tensor(batch_label, dtype=torch.float32).to(device)
            
            # 添加数值稳定性检查
            if not torch.isnan(batch_score).any() and not torch.isnan(batch_label).any():
                loss = criterion(batch_score, batch_label)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=tgan.parameters(), max_norm=5, norm_type=2)
                optimizer.step()
                tr_loss += loss.item()
                nb_tr_steps += 1
            
            batch_score = []
            batch_label = []

    train_loss = tr_loss / nb_tr_steps if nb_tr_steps > 0 else float('inf')
    # validation phase
    tgan.ngh_finder = test_ngh_finder
    test_s_idx, test_e_idx = 0, cas_dict[test_cas_l[0]]
    
    # 在这里添加你的调试代码
    print(f"Test labels stats - mean: {np.mean(test_label_l)}, std: {np.std(test_label_l)}, min: {np.min(test_label_l)}, max: {np.max(test_label_l)}")

    
    test_loss, MSLE, SMAPE, R2 = eval_one_epoch(test_cas_num, cas_dict, test_s_idx, test_e_idx, tgan, test_cas_l, test_src_l, test_dst_l, test_ts_l, test_e_l, test_label_l)
    scheduler.step()
    logger.info('end {} epoch, train loss is {}, test loss is {}, MSLE is {}, SMAPE is {}, R2 is {}'.format(epoch, train_loss, test_loss, MSLE, SMAPE, R2))
    # wandb.log(
    #         (
    #             {
    #                 "train_loss": train_loss,
    #                 "test_loss": test_loss,
    #                 "MSLE": MSLE,
    #                 "SMAPE": SMAPE,
    #                 "r2": R2
    #             }
    #         )
    #    )
