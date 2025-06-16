import os
import sys
 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from utilmodule.utils import make_parse
from utilmodule.core import train ,seed_torch
from torch.utils.data import DataLoader
from datasets.load_datasets import h5file_Dataset
import torch
import numpy as np
from utilmodule.createmode import create_model
import pandas as pd
import torch.nn as nn
import torch.nn.init as init

class TwoLayerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, a=0.01)  # 可換成其他如 normal_
        if m.bias is not None:
            init.constant_(m.bias, 0)



def main(args):
 
    seed_torch(2021)
    res_list = []
    
    basedmodel,ppo,_,memory,FusionHisF = create_model(args)
    data_csv_dir = args.csv
    feature_dir = args.feature_dir
    h5file_dir = args.train_h5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 5 classifier
    classifymodels = []
    for i in range(5):
        torch.manual_seed(2021 + i)
        clf = TwoLayerClassifier().to(device)
        clf.apply(init_weights)  # 每個都初始化不同
        classifymodels.append(clf)

    torch.manual_seed(2021)

    train_dataset = h5file_Dataset(data_csv_dir,h5file_dir,feature_dir,'train')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    validation_dataset = h5file_Dataset(data_csv_dir,h5file_dir,feature_dir,'val')
    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=True)
    test_dataset = h5file_Dataset(data_csv_dir,h5file_dir,feature_dir,'test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        
    train(args,basedmodel,ppo,classifymodels,FusionHisF,memory,train_dataloader, validation_dataloader, test_dataloader)



if __name__ == "__main__":

    args = make_parse()
    main(args)
