import torchmetrics
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
from utilmodule.utils import calculate_error,calculate_metrics,f1_score,split_array,save_checkpoint,cosine_scheduler, simsiam_sia_loss, compute_pamil_reward, pred_label_process

import numpy as np
from sklearn.cluster import KMeans
 
from tqdm import tqdm
import torch.nn.functional as F
from utilmodule.environment import expand_data
import torch.optim as optim
import wandb
import datetime
import os
from tqdm import tqdm
from models.CHIEF import CHIEF
from CHIEF_network import ClfNet

def test(args,basedmodel,ppo,classifymodels,FusionHisF,memory_space,test_loader, chief_model, run_type="test", epoch=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        label_list = []
        Y_prob_list = []
        for idx, (coords, data, label) in enumerate (tqdm(test_loader)):
            correct = 0
            total = 0
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()

           
            if args.type == 'camelyon16':
                update_data = basedmodel.fc1(update_data) 
            else:
                update_data = update_data.float() 
            #if args.ape: 
            #    update_data = update_data + basedmodel.absolute_pos_embed.expand(1,update_data.shape[1],basedmodel.args.embed_dim)  

            update_coords, update_data ,total_T = expand_data(update_coords, update_data, action_size = args.action_size, total_steps=args.test_total_T)
            grouping_instance = grouping(action_size=args.action_size)  
            
            for patch_step in range(0, total_T):
                restart = False
                restart_batch = False
                if patch_step == 0:
                    restart = True
                    restart_batch = True
            
                action_index_pro,memory = grouping_instance.rlselectindex_grouping(ppo,memory_space,update_coords,sigma=0.02,restart=restart)  
                features_group , update_coords,update_data ,memory = grouping_instance.action_make_subbags(ppo,memory,action_index_pro,update_coords,update_data,action_size = args.action_size ,restart = restart,delete_begin=True) 
                features_group = features_group.squeeze(0)
                memory.select_feature_pool.append(features_group)
                memory.merge_msg_states.append(cheif_wsi_embedding(chief_model, torch.cat(memory.select_feature_pool, dim=0)))
                _ = ppo.select_action(None,memory,restart_batch=restart_batch,training=True)

            wsi_embedding = cheif_wsi_embedding(chief_model, torch.cat(memory.select_feature_pool, dim=0))
            classifier_reward = []
            classifier_logits = []
            for i in range(len(classifymodels)):
                output = classifymodels[i](wsi_embedding)
                pre = torch.argmax(output, dim=1)
                classifier_logits.append(output)

            # Soft Voting
            avg_logits = torch.stack(classifier_logits).mean(dim=0)  # shape: [B, C]
            pred = torch.argmax(avg_logits, dim=1)
            probs = F.softmax(avg_logits, dim=1)

            # 計算正確率
            correct += (pred == label).sum().item()
            total += label.size(0)
            memory.clear_memory()
            label_list.append(label)
            Y_prob_list.append(probs)

        targets = np.asarray(torch.cat(label_list, dim=0).cpu().numpy()).reshape(-1)  
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).cpu().numpy())  
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        print(args.csv.split("/")[-1])
        #print(f'[Epoch {epoch+1}/{args.num_epochs}] {run_type} Accuracy: {accuracy:.4f} " {run_type} Precision: {precision:.4f}, {run_type} Recall: {recall:.4f}, {run_type} F1 Score: {f1:.4f}, {run_type} AUC: {auc:.4f}')
    return precision, recall, f1, auc, accuracy

def test_baseline(args,basedmodel,ppo,classifymodel,FusionHisF,memory_space,test_loader, run_type="test", epoch=0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        label_list = []
        Y_prob_list = []
        for idx, (coords, data, label) in enumerate (tqdm(test_loader)):
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()


            #if args.type == 'camelyon16':
            #    update_data = basedmodel.fc1(update_data)
            #else:
            #    update_data = update_data.float()
            #if args.ape:
            #    update_data = update_data + basedmodel.absolute_pos_embed.expand(1,update_data.shape[1],basedmodel.args.embed_dim)

            #update_coords, update_data ,total_T = expand_data(update_coords, update_data, action_size = args.action_size, total_steps=args.test_total_T)
            #grouping_instance = grouping(action_size=args.action_size)

            W_logits = classifymodel (update_data).squeeze(1)
            W_Y_prob = F.softmax(W_logits, dim=1)
            W_Y_hat = torch.argmax(W_Y_prob, dim=1)

            label_list.append(label)
            Y_prob_list.append(W_Y_prob)

        targets = np.asarray(torch.cat(label_list, dim=0).cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        #print(f'[Epoch {epoch+1}/{args.num_epochs}] {run_type} Accuracy: {accuracy:.4f} " {run_type} Precision: {precision:.4f}, {run_type} Recall: {recall:.4f}, {run_type} F1 Score: {f1:.4f}, {run_type} AUC: {auc:.4f}')
    return precision, recall, f1, auc, accuracy

def cheif_wsi_embedding(chief_model, feature):
    anatomical=13
    with torch.no_grad():
        x,tmp_z = feature,anatomical
        result = chief_model(x, torch.tensor([tmp_z]))
        wsi_feature_emb = result['WSI_feature']  ###[1,768]
        # print(wsi_feature_emb.size())

    return wsi_feature_emb


def train(args,basedmodel,ppo,classifymodels,FusionHisF,memory_space,train_loader, validation_loader, test_loader=None):
    
    run_name = f"{args.csv.split('/')[-1].split('.')[0]}"
    save_dir = os.path.join(args.save_dir, run_name)
    wandb.login(key="6c2e984aee5341ab06b1d26cefdb654ffea09bc7")
    wandb.init(
        project=args.save_dir.split("/")[-1],      # 可以在網站上看到
        name=run_name,      # optional，可用於區分實驗
        config=vars(args)                    # optional，紀錄一些超參數
    )

    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_list = []
    Y_prob_list = []
    lamda_SIA = cosine_scheduler(0.5, 0, args.num_epochs, 1, warmup_epochs=0, start_warmup_value=0)
    lamda_STL = cosine_scheduler(0, 0.1, args.num_epochs, 1, warmup_epochs=0, start_warmup_value=0)
    lamda_WSI = 1
    optimizers_list = []
    for i in range(len(classifymodels)):
        optimizers_list.append(torch.optim.Adam(list(classifymodels[i].parameters()),lr=1e-4, weight_decay=1e-5))

    chief_model = CHIEF(size_arg="small", dropout=True, n_classes=2)
    td = torch.load(r'../CHIEF/model_weight/CHIEF_pretraining.pth', map_location=device)
    chief_model.load_state_dict(td, strict=True)
    chief_model.to(device)
    chief_model.eval()

    none_epoch = 0
    best_auc = 0
    
    for idx, epoch in enumerate(range(args.num_epochs)):
        basedmodel.train()
        for classifymodel in classifymodels:
            classifymodel.train()
        FusionHisF.train()

        epoch_loss = [0] * len(classifymodels)
        correct = 0
        total = 0

        for _, (coords, data, label) in enumerate(tqdm(train_loader)):
            for i in range(len(classifymodels)):
                optimizers_list[i].zero_grad()
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()

            # 預處理
            if args.type == 'camelyon16':
                update_data = basedmodel.fc1(update_data)
            else:
                update_data = update_data.float()

            if args.ape:
                update_data += basedmodel.absolute_pos_embed.expand(1, update_data.shape[1], basedmodel.args.embed_dim)

            # 補齊資料到固定輪數與大小
            update_coords, update_data, total_T = expand_data(
                update_coords, update_data, 
                action_size=args.action_size, 
                total_steps=args.train_total_T
            )
            #print(total_T)

            grouping_instance = grouping(action_size=args.action_size)

            for patch_step in range(0, total_T):
                restart = (patch_step == 0)
                restart_batch = restart

                action_index_pro, memory = grouping_instance.rlselectindex_grouping(
                    ppo, memory_space, update_coords, sigma=0.02, restart=restart
                )

                features_group, update_coords, update_data, memory = grouping_instance.action_make_subbags(
                    ppo, memory, action_index_pro, update_coords, update_data, 
                    action_size=args.action_size, restart=restart, delete_begin=True
                )
                features_group = features_group.squeeze(0) 
                # CHIEF model replace the basemodel
                memory.select_feature_pool.append(features_group)                                            
                memory.merge_msg_states.append(cheif_wsi_embedding(chief_model, torch.cat(memory.select_feature_pool, dim=0)))    

                _ = ppo.select_action(
                    None, memory, restart_batch=restart_batch, training=True
                )

                memory.rewards.append(torch.zeros(1, device='cuda'))

            # 最終分類（WSI level）
            wsi_embedding = cheif_wsi_embedding(chief_model, torch.cat(memory.select_feature_pool, dim=0))
            classifier_reward = []
            classifier_logits = []
            for i in range(len(classifymodels)):
                optimizers_list[i].zero_grad()
                output = classifymodels[i](wsi_embedding)
                loss = F.cross_entropy(output, label)
                loss.backward()
                optimizers_list[i].step()
                pre = torch.argmax(output, dim=1)
                classifier_reward.append(F.softmax(output, dim=1)[0,label.item()].detach())
                
                # record loss for every cls
                epoch_loss[i] += loss.item()
                classifier_logits.append(output)
            
            # Soft Voting
            avg_logits = torch.stack(classifier_logits).mean(dim=0)  # shape: [B, C]
            pred = torch.argmax(avg_logits, dim=1)
            probs = F.softmax(avg_logits, dim=1)

            # reward
            #print(classifier_reward)
            memory.rewards[-1] = (torch.stack(classifier_reward).mean(dim=0)).unsqueeze(0)
            record_reward = memory.rewards[-1]
            ppo.update(memory)

            # 計算正確率
            correct += (pred == label).sum().item()
            total += label.size(0)
            memory.clear_memory()
            label_list.append(label)
            Y_prob_list.append(probs)

        targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)
        #print(f'[Epoch {epoch+1}/{args.num_epochs}] train Loss: {epoch_loss:.4f}, train Accuracy: {accuracy:.4f}, train Precision: {precision:.4f},train Recall: {recall:.4f},train F1 Score: {f1:.4f},train AUC: {auc:.4f}')
        
        for i in range(len(classifymodels)):
            wandb.log({
                f"classifier_{i}": epoch_loss[i]/len(train_loader),
            }, step=idx)


        wandb.log({
            "epoch": epoch,
            "reward": record_reward,
            "train/precision": precision,
            "train/recall": recall,
            "train/f1": f1,
            "train/auc": auc,
            "train/acc": accuracy
        }, step=idx)

        #acc = correct / total
        #print(f"[Epoch {epoch+1}/{args.num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")
        precision, recall, f1, val_auc, val_accuracy = test(args,basedmodel,ppo,classifymodels,FusionHisF,memory_space,validation_loader, chief_model, "val", epoch)
        wandb.log({
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1,
            "val/auc": val_auc,
            "val/acc": val_accuracy
        }, step=idx)
        precision, recall, f1, auc, accuracy = test(args,basedmodel,ppo,classifymodels,FusionHisF,memory_space,test_loader, chief_model, "test", epoch)
        wandb.log({
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1,
            "test/auc": auc,
            "test/acc": accuracy
        }, step=idx)
        if val_auc >= best_auc:
            best_auc = val_auc
            # save model
            print(f'Save model at epoch {epoch}.')
            print(f'val auc: {val_auc}')
            torch.save(basedmodel.state_dict(), os.path.join(save_dir, "basedmodel.pth"))
            torch.save(classifymodel.state_dict(), os.path.join(save_dir, "classifymodel.pth"))
            torch.save(FusionHisF.state_dict(), os.path.join(save_dir, "FusionHisF.pth"))
            ppo.save(save_dir)
            none_epoch = 0
        elif none_epoch >= args.patience:
            print(f"Break at epoch {epoch}.")
            break

        none_epoch += 1

def train_baseline(args,basedmodel,ppo,classifymodel,FusionHisF,memory_space,train_loader, validation_loader, test_loader=None):
    run_name = f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.login(key="6c2e984aee5341ab06b1d26cefdb654ffea09bc7")
    wandb.init(
        project="WSI_baseline",      # 可以在網站上看到
        name=run_name,      # optional，可用於區分實驗
        config=vars(args)                    # optional，紀錄一些超參數
    )

    save_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifymodel.to(device)
    optimizer = torch.optim.Adam(
        list(classifymodel.parameters()),
        lr=1e-4, weight_decay=1e-5
    )
    
    none_epoch = 0
    best_auc = 0

    for idx, epoch in enumerate(range(args.num_epochs)):
        classifymodel.train()

        epoch_loss = 0.0
        correct = 0
        total = 0

        label_list = []
        Y_prob_list = []

        for _, (coords, data, label) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            update_coords, update_data, label = coords.to(device), data.to(device), label.to(device).long()

            # 預處理
            #if args.type == 'camelyon16':
            #    update_data = basedmodel.fc1(update_data)
            #else:
            #    update_data = update_data.float()
            
            W_logits = classifymodel(update_data).squeeze(1)
            W_Y_prob = F.softmax(W_logits, dim=1)
            W_Y_hat = torch.argmax(W_Y_prob, dim=1)

            # 計算 loss（以交叉熵為例）
            #print(W_logits.shape)
            #print(label.shape)
            loss_WSI = F.cross_entropy(W_logits, label)
            #print(loss_SIA.item(), loss_WSI.item())
            loss = loss_WSI
            loss.backward()
            optimizer.step()

            #print(f'Idx: {idx},loss: {loss.item()}')

            # reward

            # 計算正確率
            epoch_loss += loss.item()
            correct += (W_Y_hat == label).sum().item()
            total += label.size(0)
            label_list.append(label)
            Y_prob_list.append(W_Y_prob)

        targets = np.asarray(torch.cat(label_list, dim=0).detach().cpu().numpy()).reshape(-1)
        probs = np.asarray(torch.cat(Y_prob_list, dim=0).detach().cpu().numpy())
        precision, recall, f1, auc, accuracy = calculate_metrics(targets, probs)


        wandb.log({
            "loss": epoch_loss,
            "epoch": epoch,
            "train/precision": precision,
            "train/recall": recall,
            "train/f1": f1,
            "train/auc": auc,
            "train/acc": accuracy
        }, step=idx)

        #acc = correct / total
        #print(f"[Epoch {epoch+1}/{args.num_epochs}] Loss: {epoch_loss:.4f}, Accuracy: {acc:.4f}")
        precision, recall, f1, val_auc, val_accuracy = test_baseline(args,basedmodel,ppo,classifymodel,FusionHisF,memory_space,validation_loader, "val", epoch)
        wandb.log({
            "val/precision": precision,
            "val/recall": recall,
            "val/f1": f1,
            "val/auc": val_auc,
            "val/acc": val_accuracy
        }, step=idx)
        precision, recall, f1, auc, accuracy = test_baseline(args,basedmodel,ppo,classifymodel,FusionHisF,memory_space,test_loader, "test", epoch)
        wandb.log({
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1,
            "test/auc": auc,
            "test/acc": accuracy
        }, step=idx)
        if val_auc >= best_auc:
            best_auc = val_auc
            # save model
            print(f'Save model at epoch {epoch}.')
            print(f'val auc: {val_auc}')
            torch.save(classifymodel.state_dict(), os.path.join(save_dir, "classifymodel.pth"))
            none_epoch = 0
        elif none_epoch >= args.patience:
            print(f"Break at epoch {epoch}.")
            break

        none_epoch += 1



    
    


def interpolate_probs(probs, new_length,action_size):
    b,c = probs.shape
    x = np.linspace(0, 1, c)
    x_new = np.linspace(0, 1, new_length)
    new_probs = np.interp(x_new, x, probs.view(-1).cpu().numpy())
    interpolate_action_probs = new_probs / new_probs.sum()   
    index = np.random.choice(np.arange(new_length), size=action_size, p=interpolate_action_probs)
    return index


def seed_torch(seed=2021):
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 


    
class grouping:

    def __init__(self,action_size = 128):
        self.action_size = action_size 
        self.action_std = 0.1
           
     
    def rlselectindex_grouping(self,ppo,memory,coords,sigma=0.02,restart=False): 
        B, N, C = coords.shape
        if restart  : 
            if self.action_size < N: 
                action = torch.distributions.dirichlet.Dirichlet(torch.ones(self.action_size)).sample((1,)).to(coords.device)
            else: 
                random_values = torch.rand(1, N).to(coords.device)
                indices = torch.randint(0, N, (self.action_size,)).to(coords.device)
                action = random_values[0, indices].unsqueeze(0)
            memory.actions.append(action) 
            memory.logprobs.append(action) 
            return action.detach(), memory
        else:  
            action = memory.actions[-1] 
            
            return action.detach(), memory
          
    
    def action_make_subbags(self,ppo,memory, action_index_pro, update_coords,update_features,action_size= None,restart= False,delete_begin=False): 
        
        B, N, C = update_coords.shape
        idx = interpolate_probs(action_index_pro, new_length = N ,action_size = action_size)
        idx = torch.tensor(idx)
        features_group = update_features[:, idx[:], :]
        action_group = update_coords[:, idx[:], :] 

        if restart and delete_begin:
            memory.coords_actions.append(action_group)
            return features_group, update_coords, update_features ,memory
        else:
            idx = torch.unique(idx)
            mask = torch.ones(update_features.size(1), dtype=torch.bool) 
            mask[idx] = False  
            updated_features = update_features[:, mask, :] 
            updated_coords = update_coords[:, mask, :]
            memory.coords_actions.append(action_group)
            return features_group, updated_coords, updated_features ,memory

    
