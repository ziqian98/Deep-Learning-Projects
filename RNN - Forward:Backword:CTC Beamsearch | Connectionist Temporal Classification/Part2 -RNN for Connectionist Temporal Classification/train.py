#!/usr/bin/env python
# coding: utf-8


from torch.utils import data
import torch
import torch.nn as nn 
import sys 
import torch.optim as optim
import time
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import *

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(cuda)


train  = np.load("wsj0_train",allow_pickle=True,encoding="latin1")
print("train.shape:",train.shape)
print("train[0].shape:",train[0].shape)

dev = np.load("wsj0_dev.npy",allow_pickle=True,encoding="latin1")

test  = np.load("wsj0_test",allow_pickle=True,encoding="latin1")

print("test.shape:",test.shape)
print("test[0].shape:",test[0].shape)

train_labels  = np.load("wsj0_train_merged_labels.npy",allow_pickle=True,encoding="latin1")
dev_labels  = np.load("wsj0_dev_merged_labels.npy",allow_pickle=True,encoding="latin1")

print(train_labels.shape)
print(train_labels[0].shape)

for idx in range(train.shape[0]):
    mean = np.mean(train[idx], axis = 0)
    std = np.std(train[idx], axis = 0)
    train[idx] = (train[idx] - mean) / std


for idx in range(dev.shape[0]):
    mean = np.mean(dev[idx], axis = 0)
    std = np.std(dev[idx], axis = 0)
    dev[idx] = (dev[idx] - mean) / std


def collate_lines(seq_list):
    inputs, targets = zip(*seq_list)
    xlens = torch.LongTensor([len(seq) for seq in inputs])
    ylens = torch.LongTensor([len(seq) for seq in targets])
    
    padinp = pad_sequence(inputs)
    padtar = pad_sequence(targets, batch_first=True)
    
    return padinp, xlens, padtar, ylens


class MyDataset(data.Dataset):
    def __init__ (self,x,y):
        self.x = x
        self.y = y + 1
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        x = torch.Tensor(self.x[index])
        y = torch.Tensor(self.y[index])
        
        return x, y
        


train_dataset = MyDataset(train, train_labels)
train_loader_args = dict(shuffle=True, batch_size=32, num_workers=8, pin_memory=True, collate_fn = collate_lines) if cuda else dict(shuffle=True, batch_size=64,collate_fn = collate_lines)
train_loader = data.DataLoader(train_dataset, **train_loader_args)


dev_dataset = MyDataset(dev, dev_labels)
dev_loader_args = dict(shuffle=False, batch_size=32, num_workers=8, pin_memory=True,collate_fn = collate_lines, drop_last=True) if cuda else dict(shuffle=False, batch_size=64,collate_fn = collate_lines, drop_last=True)
dev_loader = data.DataLoader(dev_dataset, **dev_loader_args)


class My_LSTM(nn.Module):
    def __init__(self,size_list):
        super(My_LSTM, self).__init__()
        input_size = size_list[0]
        output_size = size_list[1]
        num_layers = size_list[2]
        hidden_size = size_list[3]
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, dropout = 0.2)
        self.linear1 = nn.Linear(hidden_size*2, hidden_size*4)  
        self.linear2 = nn.Linear(hidden_size*4, output_size) 
        
    def forward(self, X, lengths):
        packed_X = pack_padded_sequence(X, lengths, enforce_sorted=False)
        packed_out = self.lstm(packed_X)[0]
        out, out_lens = pad_packed_sequence(packed_out)
        
        out = self.linear1(out).relu()
        out = self.linear2(out).log_softmax(2)
        
        return out, out_lens


model = My_LSTM([40, 47, 4, 256])
#model.load_state_dict(torch.load("15epochnorm4lstmdp.t7"))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.8,patience=1)
criterion = nn.CTCLoss()
model = model.to(device)
criterion = criterion.to(device)



def train_epoch(model,optimizer,train_loader,criterion):
    model.train()
    running_loss = 0.0
    start_time = time.time()
    
    print("training", len(train_loader), "number of batches")
    
    for batch_idx, (padinp, xlens, padtar, ylens) in enumerate(train_loader):

        optimizer.zero_grad()
        padinp = padinp.to(device)
        padtar = padtar.to(device)
        
        out, out_lens = model(padinp,xlens)
        
        loss = criterion(out, padtar, out_lens, ylens)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        
        if (batch_idx % 100 == 0):
            print("batch_idx",batch_idx)
            print("Training Loss Current Batch: ", loss.item())
            print("Training Perplexity Current Batch:", np.exp(loss.item()))
        
    end_time = time.time()
    running_loss = running_loss / len(train_loader)
    
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')


def dev_epoch(model,optimizer,dev_loader,criterion):
    with torch.no_grad():
        model.eval()
        
        running_loss = 0.0
    
    for batch_idx, (padinp, xlens, padtar, ylens) in enumerate(dev_loader):
        padinp = padinp.to(device)
        padtar = padtar.to(device)
        
        out, out_lens = model(padinp,xlens)
        
        loss = criterion(out, padtar, out_lens, ylens)
        
        running_loss += loss.item()
        
    running_loss = running_loss / len(dev_loader)
    
    print("\nValidation loss per batch:",running_loss)
    print("Validation perplexity :",np.exp(running_loss),"\n")
    
    return running_loss

n_epochs = 100

for i in range(n_epochs):
    print("epoch", i+1)

    train_epoch(model,optimizer,train_loader,criterion)

    val_loss = dev_epoch(model,optimizer,dev_loader,criterion)
    scheduler.step(val_loss)

    print('='*20)
    modelname = str(i+1) + "epochnorm4lstmdp.t7"  
    torch.save(model.state_dict(),modelname)


# 0.001 15 epoch
# 0.0001 1 epoch valloss = 0.33  best

