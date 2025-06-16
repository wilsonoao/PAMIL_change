import pandas as pd
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn.functional as F

import os
# import re
# import csv
# import yaml
# import json
# import glob
import shutil
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score,accuracy_score, confusion_matrix
import heapq
import statistics


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='tcga',type=str)  
    parser.add_argument('--mode', default='rlselect',type=str)
    parser.add_argument('--seed', default=2021,type=int)
    parser.add_argument('--num_epochs', default=300,type=int)
    parser.add_argument('--lr', default=0.00001,type=int)
    

    parser.add_argument('--in_chans', default=1024,type=int)
  
    parser.add_argument('--embed_dim', default=768,type=int)
    parser.add_argument('--attn', default='normal',type=str)
    parser.add_argument('--gm', default='cluster',type=str)
    parser.add_argument('--cls', default=True,type=bool)
    parser.add_argument('--num_msg', default=1,type=int)
    parser.add_argument('--ape', default=True,type=bool)
    parser.add_argument('--n_classes', default=2,type=int)
    parser.add_argument('--num_layers', default=2,type=int) 

    parser.add_argument('--instaceclass', default=True,type=bool,help='') 
    parser.add_argument('--CE_CL', default=True,type=bool,help='')
    parser.add_argument('--ape_class', default=False,type=bool,help='') 


    parser.add_argument('--test_h5', default='/n/scratch/users/f/fas994/wilson/data/patch_coordinate/LUAD',type=str)
    parser.add_argument('--train_h5',default='/n/scratch/users/f/fas994/wilson/data/patch_coordinate/LUAD',type=str)
    parser.add_argument('--csv', default='/n/scratch/users/f/fas994/wilson/RL/ppo_chief/PAMIL/data_csv/dataset_0.csv',type=str)
    parser.add_argument('--feature_dir', default='/n/scratch/users/f/fas994/wilson/data/CHIEF/LUAD',type=str)
 
    
    parser.add_argument('--policy_hidden_dim', type=int, default=768)
    parser.add_argument('--feature_dim', type=int, default=768)
    parser.add_argument('--state_dim', type=int, default=768)
    parser.add_argument('--action_size', type=int, default=10) 
    parser.add_argument('--policy_conv', action='store_true', default=False)
    parser.add_argument('--action_std', type=float, default=0.5)
    parser.add_argument('--ppo_lr', type=float, default=0.00001)
    parser.add_argument('--ppo_gamma', type=float, default=0.1)
    parser.add_argument('--K_epochs', type=int, default=3)
    
    parser.add_argument('--test_total_T', type=int, default=3)
    parser.add_argument('--train_total_T', type=int, default=3)

    parser.add_argument('--reward_rule', type=str, default="cl",help=' ')
    parser.add_argument('--save_dir', type=str, default="/n/scratch/users/f/fas994/wilson/RL/ppo_chief/PAMIL/result",help='')
    parser.add_argument('--patience', type=int, default=20, help = '')

    parser.add_argument('--test_dir', default="/n/scratch/users/f/fas994/wilson/RL/ppo_chief/PAMIL/result",help='')

    
    
    args = parser.parse_args()
    return args

import torch
import torch.nn.functional as F

def pred_label_process(memory, true_label, cls_pred):

    """
    Args:
        token_preds: Tensor of shape [T], 每輪 token 預測值（如 sigmoid 後機率）
        cls_pred: float or Tensor scalar, CLS token 的預測值（如 sigmoid output）
        true_label: int or scalar Tensor, ground truth label (0 or 1)
    
    Returns:
        final_pred: float, 最終融合預測值
    """
    T = len(memory.results_dict)
    pred = [results_dict['Y_prob'][0][true_label].item() for results_dict in memory.results_dict]
    topk = heapq.nlargest(min(T, 5), pred)  # 前 K 大值
    max_pred = topk[0]
    avg_top3 = statistics.mean(topk[:3])
    avg_top5 = statistics.mean(topk[:5])

    if true_label == 1:
        final_pred = (max_pred + avg_top3 + avg_top5 + cls_pred) / 4
    else:
        final_pred = cls_pred

    return final_pred

def compute_pamil_reward(cls_token, sub_tokens, pred_label, true_label, reward_value=1.0):
    """
    計算 PAMIL reward:
        R = r* - penalty  if prediction is correct
          = 0 - penalty   otherwise

    Args:
        cls_token: Tensor, [B, D] → 最終分類的 CLS token (h_cls)
        sub_tokens: List[Tensor] or Tensor [T, B, D] → 每輪 token (u^s_t)
        pred_label: Tensor, [B] → 預測類別
        true_label: Tensor, [B] → 真實類別
        reward_value: float → 若預測正確給的正向獎勵值（如 1 or 2）

    Returns:
        reward: Tensor, [B] → 每個樣本的 reward 值
    """
    if isinstance(sub_tokens, list):
        sub_tokens = torch.stack(sub_tokens, dim=0)  # [T, B, D]

    # normalize
    cls_token = F.normalize(cls_token.view(1, 1, 1, 768).expand_as(sub_tokens), dim=-1)          # [B, D]
    sub_tokens = F.normalize(sub_tokens, dim=-1)        # [T, B, D]

    # 計算 cosine similarity → [T, B]
    cosine_sims = F.cosine_similarity(cls_token, sub_tokens, dim=-1)
    
    # 平均所有輪次的 similarity 當作 penalty（越高表示越穩定，懲罰越小）
    penalty = cosine_sims.mean(dim=0)  # [B]

    # 預測正確給正向獎勵，錯誤就是 - penalty
    reward = torch.where(pred_label == true_label,
                         reward_value - penalty,
                         -penalty)

    #reward = torch.where(pred_label == true_label,
    #                     reward_value,
    #                     0)

    return reward  # shape: [B]


def simsiam_sia_loss(merge_msg_states, FusionHisF):
    """
    計算 SIA 對比學習損失（雙向 cosine similarity）
    
    Args:
        merge_msg_states: List[Tensor] or Tensor of shape [T, D] or [T, 1, D]
                          每輪的 CLS token 語意表示
        projector: nn.Module，用來投影語意 token 得到 p^s_t

    Returns:
        scalar 損失值（float）
    """
    T = len(merge_msg_states)
    total_loss = 0.0

    for t in range(1, T):
        u_t = merge_msg_states[t]           # CLS token at t
        u_t_minus_1 = merge_msg_states[t-1] # CLS token at t-1

        p_t = FusionHisF.mlp_projector(u_t)
        p_t_minus_1 = FusionHisF.mlp_projector(u_t_minus_1)

        # normalize
        p_t = F.normalize(p_t, dim=-1)
        p_t_minus_1 = F.normalize(p_t_minus_1, dim=-1)
        u_t = F.normalize(u_t, dim=-1)
        u_t_minus_1 = F.normalize(u_t_minus_1, dim=-1)

        # cosine similarity (negated)
        loss_1 = - (p_t * u_t_minus_1.detach()).sum(dim=-1).mean()
        loss_2 = - (p_t_minus_1 * u_t.detach()).sum(dim=-1).mean()

        total_loss += 0.5 * (loss_1 + loss_2)

    return total_loss / (T - 1)


 

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    schedule_per_epoch = []
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            value = np.linspace(start_warmup_value, base_value, warmup_epochs)[epoch]
        else:
            iters_passed = epoch * niter_per_ep
            iters_left = epochs * niter_per_ep - iters_passed
            alpha = 0.5 * (1 + np.cos(np.pi * iters_passed / (epochs * niter_per_ep)))
            value = final_value + (base_value - final_value) * alpha
        schedule_per_epoch.append(value)
    return schedule_per_epoch




def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()
	return error



def calculate_metrics(targets, probs):
    threshold = 0.5  # You can adjust this threshold as needed
    predictions = (probs[:, 1] >= threshold).astype(int)
 
    precision = precision_score(targets, predictions)
    recall = recall_score(targets, predictions)
    f1 = f1_score(targets, predictions) 
    auc = roc_auc_score(targets, probs[:, 1])  
    accuracy = accuracy_score(targets, predictions)
    print(f"precision: {precision}")
    print(f"recall: {recall}")
    print(f"auc: {auc}")
    print(f"accuracy: {accuracy}")
    print(f"f1: {f1}")
    print("confusion matrix:")
    print(confusion_matrix(targets, predictions))

    return precision, recall, f1, auc, accuracy

 
 

def cat_msg2cluster_group(x_groups,msg_tokens):
    x_groups_cated = []
    for x in x_groups:  
        x = x.unsqueeze(dim=0)  
        try:
            temp = torch.cat((msg_tokens,x),dim=2)
        except Exception as e:
            print('Error when cat msg tokens to sub-bags')
        x_groups_cated.append(temp)

    return x_groups_cated



def split_array(array, m):
    n = len(array)
    indices = np.random.choice(n, n, replace=False)  
    split_indices = np.array_split(indices, m)   

    result = []
    for indices in split_indices:
        result.append(array[indices])

    return result



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.flag = False

    def __call__(self, epoch, val_loss, model, args, ckpt_name = ''):
        ckpt_name = './ckp/{}_checkpoint_{}_{}.pt'.format(str(args.type),str(args.seed),str(epoch))
        score = -val_loss
        self.flag = False
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name, args)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name, args)
            self.counter = 0
        

    def save_checkpoint(self, val_loss, model, ckpt_name, args):
        '''Saves model when validation loss decrease.'''
        if self.verbose and not args.overfit:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',ckpt_name)
        elif self.verbose and args.overfit:
            print(f'Training loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...',ckpt_name)           
        torch.save(model.state_dict(), ckpt_name)
        print(ckpt_name)
        self.val_loss_min = val_loss
        self.flag = True

def save_checkpoint(state,best_acc, auc,checkpoint, filename='checkpoint.pth.tar'):
    best_acc = f"{best_acc:.4f}"
    auc = f"{auc:.4f}"
    filepath = os.path.join(checkpoint, best_acc+"_"+auc+"_"+filename)
    torch.save(state, filepath)
 

