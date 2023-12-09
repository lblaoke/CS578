from __future__ import print_function

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from math import *
from sklearn.metrics import roc_auc_score

def _ece_score(y_true,y_pred,bins:int=10):
    ece = 0.
    for i in range(bins):
        c_start,c_end = i/bins,(i+1)/bins
        mask = (c_start<=y_pred)&(y_pred<c_end)

        ni = mask.count_nonzero().item()
        if ni<=0: continue

        acc,conf = y_true[mask].sum()/ni,y_pred[mask].mean()
        ece += ni*(acc-conf).abs()

    return float(ece)/len(y_true)

def score(output,uncertainty,target):
    pred = torch.argmax(output,dim=1)
    correct = (pred==target).to(torch.float32)
    acc = correct.mean()
    acc0 = correct[target==0].mean()
    acc1 = correct[target==1].mean()

    confidence = 1-uncertainty

    nll = torch.nn.functional.cross_entropy(output,target)
    auc = roc_auc_score(correct,confidence)
    ece = _ece_score(correct,confidence)

    print(f'ACC : {acc:.4f}')
    print(f'ACC0: {acc0:.4f}')
    print(f'ACC1: {acc1:.4f}')
    print(f'NLL : {nll:.4f}')
    print(f'AUC : {auc:.4f}')
    print(f'ECE : {ece:.4f}')

    return acc

def noise(net,coeff):
    _noise = 0
    for param in net.parameters():
        _noise += torch.sum(param*torch.randn_like(param.data)*coeff)
    return _noise

class LT_Dataset(torch.utils.data.Dataset):
    def __init__(self,data):
        self.data = data
        self.labels = self.data['TARGET'].to_numpy(dtype=np.int64)
        fe_cols = self.data.columns
        self.features = self.data[fe_cols.drop('TARGET')].to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,index):
        label = self.labels[index]
        feature = self.features[index]
        feature_tensor = torch.tensor(feature,dtype=torch.float32)
        label_tensor = torch.tensor(label,dtype=torch.int64)

        return feature_tensor,label_tensor

# df = pd.read_csv("processed_data.csv",index_col=0)
# df_test = df.sample(frac=0.1)
# df_train = df.drop(df_test.index)

df_test = pd.read_csv("balanced_test.csv")
df_train = pd.read_csv("balanced_train.csv",index_col=0)

_train_set = LT_Dataset(df_train)
_test_set = LT_Dataset(df_test)

# hyper-parameter
learning_rate = 0.01
batch_size = 1000
num_epoch = 1

# load data
train_loader = torch.utils.data.DataLoader(_train_set,batch_size=batch_size,shuffle=True,drop_last=False,num_workers=0)
test_loader = torch.utils.data.DataLoader(_test_set,batch_size=2*batch_size,shuffle=False,drop_last=False,num_workers=0)

# init model
net = torch.nn.Sequential(
    torch.nn.Linear(45,64),
    torch.nn.ReLU(),
    torch.nn.Linear(64,2)
)

# trainer
criterion = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(net.parameters(),lr=learning_rate)

# test
def test(net,testloader):
    net.eval()
    val_targets = torch.tensor(testloader.dataset.labels,dtype=torch.int64)
    output,uncertainty = [],[]

    with torch.no_grad():
        for _,(inputs,_) in enumerate(testloader):
            o = net(inputs)
            u = -torch.sum(F.softmax(o,dim=1)*F.log_softmax(o,dim=1),dim=1)
            output.append(o.detach())
            uncertainty.append(u.detach())

    output = torch.cat(output,dim=0)
    uncertainty = torch.cat(uncertainty)

    score(output,uncertainty,val_targets)

def multi_test(net,testloader,w_list):
    val_targets = torch.tensor(testloader.dataset.labels,dtype=torch.int64)
    prob_list = []

    for w in w_list:
        net.load_state_dict(w)
        net.eval()
        prob = []

        for _,(inputs,_) in enumerate(testloader):
            with torch.no_grad():
                o = net(inputs)
                p = F.softmax(o,dim=1)
            prob.append(p.detach())

        prob_list.append(torch.cat(prob,dim=0))

    final_prob = sum(prob_list)/len(prob_list)
    uncertainty = -torch.sum(final_prob*torch.log(final_prob),dim=1)

    score(final_prob,uncertainty,val_targets)

# training
w_list = []
data_size = _train_set.data.shape[0]
num_batch = data_size//batch_size

for epoch in range(1,num_epoch+1):
    global_loss,global_acc = 0,0

    for batch_id,(inputs,targets) in enumerate(train_loader):
        outputs = net(inputs)

        loss = criterion(outputs,targets)
        if num_batch-batch_id<40:
            for param_group in opt.param_groups:
                lr = param_group['lr']
            noise_coeff = sqrt(2/lr/data_size*1)
            loss += noise(net,noise_coeff)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if num_batch-batch_id<40: w_list.append(net.state_dict())

        global_loss += loss.item()*len(targets)
        pred = torch.argmax(outputs,dim=1)
        global_acc += (pred==targets).sum().item()

    data_size = _train_set.data.shape[0]
    print('\nEpoch %d: Train Loss = %.4f, Train ACC = %.4f' % (epoch,global_loss/data_size,global_acc/data_size))

    print(len(w_list))
    multi_test(net,test_loader,w_list)