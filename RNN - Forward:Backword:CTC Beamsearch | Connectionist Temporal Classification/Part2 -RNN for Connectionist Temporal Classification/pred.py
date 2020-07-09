#!/usr/bin/env python
# coding: utf-8

import csv
from torch.utils import data
import torch
import torch.nn as nn 
import sys 
import torch.optim as optim
import time
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import *
from phoneme_list import *
from ctcdecode import CTCBeamDecoder

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(cuda)

test  = np.load("wsj0_test",allow_pickle=True,encoding="latin1")



for idx in range(test.shape[0]):
    mean = np.mean(test[idx], axis = 0)
    std = np.std(test[idx], axis = 0)
    test[idx] = (test[idx] - mean) / std



def collate_lines(seq_list):
    inputs = seq_list
    xlens = torch.LongTensor([len(seq) for seq in inputs])
    padinp = pad_sequence(inputs)
    return padinp, xlens


class MyDataset(data.Dataset):
    def __init__ (self,x):
        self.x = x
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self,index):
        x = torch.Tensor(self.x[index])
        
        return x
        

test_dataset = MyDataset(test)
test_loader_args = dict(shuffle=False, batch_size=32, num_workers=8, pin_memory=True, collate_fn = collate_lines) if cuda else dict(shuffle=False, batch_size=64,collate_fn = collate_lines)
test_loader = data.DataLoader(test_dataset, **test_loader_args)



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
model.load_state_dict(torch.load("16epochnorm4lstmdp.t7"))
model = model.to(device)



def pred_model(model, test_loader):
    with torch.no_grad():
        model.eval()

        predLabel= []


        for batch_idx, (padinp,xlens) in enumerate(test_loader):
            padinp = padinp.to(device)

            batchlabel = []

            out, out_lens = model(padinp,xlens)

            phonemes = [" "] + PHONEME_MAP

            decoder = CTCBeamDecoder(phonemes, beam_width=10, log_probs_input=True)
            out_lens = torch.LongTensor(out_lens)

            pred, _, _, pred_lens = decoder.decode(out.transpose(0,1),out_lens)

            for i in range(len(pred)):
                seq = ""
                for j in range(pred_lens[i,0]):
                    seq += phonemes[int(pred[i,0,j])]

                batchlabel.append(seq)

            predLabel = predLabel + batchlabel
    
    return predLabel


print("ppp")

predres = pred_model(model, test_loader)
print(len(predres))

with open ("sub.csv", mode="w") as wfile:
    mywriter = csv.writer(wfile,delimiter = ',')
    mywriter.writerow(["id","Predicted"])

    for i in range(len(predres)):
        mywriter.writerow([i,predres[i]])

print("done")



