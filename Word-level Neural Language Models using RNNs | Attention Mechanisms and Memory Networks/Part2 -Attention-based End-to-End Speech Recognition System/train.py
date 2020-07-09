#!/usr/bin/env python
# coding: utf-8

import csv
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.optim as optim
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.nn import functional as F
import torch
import importlib
#from dataloader import *
#import models
import random
import torch.nn.utils as utils
from torch.nn.utils.rnn import pad_sequence

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
print(cuda)


def random_search(model, inputs, inputs_lens, rs_sample_size, gumbel_noise=True):
    predlist = []
    for idx, single_input in enumerate(inputs):
        key, value, enc_lens = model.encoder(single_input.unsqueeze(0), [inputs_lens[idx]])
        single_prediction = randomhelp(model, key.repeat(rs_sample_size, 1, 1), value.repeat(rs_sample_size, 1, 1), enc_lens.repeat(rs_sample_size), rs_sample_size, gumbel_noise=True)
        predlist.append(single_prediction)
    return predlist



def randomhelp(model, key, value, src_lens, rs_sample_size, gumbel_noise=True):
    predictions = model.decoder(key, value, src_lens=src_lens, text=None, isTrain=False, gumbel_noise=True, random_search=True)

    texts = transform_index_to_letter(predictions.argmax(dim=-1).detach().cpu().numpy(), index2letter=index2letter, stoppos=[letter2index['<eos>']])

    indices_list = []
    for text in texts:
        indices = []
        idx = 0
        while (idx < len(text)):
            letter = text[idx]
            if letter == '<':  
                special_tag = text[idx:idx+len('<eos>')]
                indices.append(letter2index[special_tag])
                idx += len(special_tag)
            else:
                indices.append(letter2index[letter])
                idx += 1
        indices_list.append([letter2index['<sos>']] + indices + [letter2index['<eos>']])

    outputs_text = [torch.tensor(text[:-1]) for text in indices_list]
    targets_text = [torch.tensor(text[1:]) for text in indices_list]

    output = pad_sequence(outputs_text, batch_first=True).to(device)
    target = pad_sequence(targets_text, batch_first=True).to(device)

    prediction = model.decoder(key, value, src_lens=src_lens, text=output, isTrain=True, gumbel_noise=True)

    lens = torch.tensor([len(text) for text in targets_text])

    loss_per_sample = torch.mean(criterion(prediction.view(-1, prediction.size(2)), target.view(-1)).view(prediction.size(0), -1) * (torch.arange(0, torch.max(lens)).repeat(lens.size(0), 1) < lens.unsqueeze(1).expand(lens.size(0), torch.max(lens))).int().to(device), dim=-1)
    
    res =  transform_index_to_letter(predictions[torch.argmin(loss_per_sample).item()].unsqueeze(0).argmax(-1).detach().cpu().numpy(),index2letter, [letter2index['<eos>'], letter2index['<pad>']])            
    return res[0]


def train(model, train_loader, criterion, optimizer, detect_anomaly=False, gumbel_noise = True): 
    model.train()

    running_loss = 0.0
    running_perplexity = 0.0

    start_time = time.time()
    
    for batch_idx, (inputs, outputs, targets, inputs_lens, outputs_lens, targets_lens) in enumerate(train_loader):
        if (batch_idx == 2 ):
            print("batch_idx: ", batch_idx)
            break
            
        optimizer.zero_grad()
        
        torch.autograd.set_detect_anomaly(detect_anomaly)
        
        inputs = inputs.to(device)
        outputs = outputs.to(device)
        targets = targets.to(device)
        
        predictions = model(inputs, inputs_lens, text_input=outputs, isTrain=True)
        
        lens = torch.tensor(targets_lens)
        mask = (torch.arange(0, torch.max(lens)).repeat(lens.size(0), 1) < lens.unsqueeze(1).expand(lens.size(0), torch.max(lens))).int().to(device)
        
        loss = torch.sum(criterion(predictions.view(-1, predictions.size(2)), targets.view(-1)) * mask.view(-1)) / torch.sum(mask)
        
        running_loss += loss.item()
        running_perplexity += torch.exp(loss).item()
        
        loss.backward()
        
        optimizer.step()
        
    end_time = time.time()
    running_loss /= len(train_loader)
    running_perplexity /= len(train_loader)
    
    print('Train Loss: ',running_loss,'Train Perplexity: ',running_perplexity,'Time: ',end_time - start_time, 's')
    
    return running_loss, running_perplexity



def evaluate(model, eval_loader, criterion, rs_sample_size=0, gumbel_noise = True):
    with torch.no_grad():
        model.eval()
        running_loss = 0.0
        running_perplexity = 0.0

        
        for batch_idx, (inputs, outputs, targets, inputs_lens, outputs_lens, targets_lens) in enumerate(eval_loader):
            if (batch_idx == 2):
                print("batch_idx: ", batch_idx)
                break

            inputs, outputs, targets = inputs.to(device), outputs.to(device), targets.to(device)
            
            predictions = model(inputs, inputs_lens, text_input=outputs, isTrain=True, gumbel_noise=gumbel_noise)
            
            lens = torch.tensor(targets_lens)
            mask = (torch.arange(0, torch.max(lens)).repeat(lens.size(0), 1) < lens.unsqueeze(1).expand(lens.size(0), torch.max(lens))).int().to(device)

            loss = torch.sum(criterion(predictions.view(-1, predictions.size(2)), targets.view(-1)) * mask.view(-1)) / torch.sum(mask)
            
            running_loss += loss.item()
            running_perplexity += torch.exp(loss).item()
            
            if rs_sample_size == 0:
                predictions = model(inputs, inputs_lens, text_input=None, isTrain=False, gumbel_noise=gumbel_noise)
                pred = transform_index_to_letter(predictions.argmax(-1).detach().cpu().numpy(), index2letter, [letter2index['<eos>'], letter2index['<pad>']])
            else:
                pred = random_search(model, inputs, inputs_lens, rs_sample_size)
                

        running_loss /= len(eval_loader)
        running_perplexity /= len(eval_loader)

        
        print('Eval Loss: ', running_loss, 'Eval Perplexity: ', running_perplexity)

        return running_loss, running_perplexity
                


def training(model, epochs, train_loader, eval_loader, criterion, optimizer, 
                scheduler=None, detect_anomaly=False, rs_sample_size=0, gumbel_noise=True):
    model.to(device)

    for i in range(epochs):
        print("epoch: ",i+1)
        
        train_loss, train_perplexity = train(model, train_loader, criterion, optimizer, detect_anomaly, gumbel_noise=gumbel_noise)
        eval_loss, eval_perplexity= evaluate(model, eval_loader, criterion, rs_sample_size=rs_sample_size, gumbel_noise=gumbel_noise)
        scheduler.step(eval_loss)

        print('=' * 20)
        modelname = str(i+1) + "epoch.t7"  
        torch.save(model.state_dict(),modelname)

def testing(model, test_loader, save=False, filename="sub.csv", rs_sample_size=100):
    res = []
    
    with torch.no_grad():
        model.eval()
        model.to(device)

        for batch_idx, (inputs, inputs_lens) in enumerate(test_loader):
            inputs = inputs.to(device)
            print("test batch_idx: ", batch_idx)
            pred = random_search(model, inputs, inputs_lens, rs_sample_size)
            res.append(pred)
    return res


LETTER_LIST = ['<pad>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']


speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()


batch_size = 256
epochs = 1
vocab_size = len(LETTER_LIST)
value_size=128
key_size=128
rs_sample_size = 100
input_dim = 40
enc_hidden_dim = 256
dec_hidden_dim = 512
emb_dim = 256

num_workers = 8 if cuda else 0 



character_text_train = transform_letter_to_index(transcript_train, LETTER_LIST)
character_text_valid = transform_letter_to_index(transcript_valid, LETTER_LIST)


train_dataset = Speech2TextDataset(speech_train, character_text_train)
val_dataset = Speech2TextDataset(speech_valid, character_text_valid)
test_dataset = Speech2TextDataset(speech_test, None, False)


train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True,collate_fn = collate_train)
val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True,collate_fn = collate_train)
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True,collate_fn = collate_test)

letter2index, index2letter = create_dictionaries(LETTER_LIST)

model = models.Seq2Seq(input_dim=input_dim, vocab_size=vocab_size, enc_hidden_dim=enc_hidden_dim, dec_hidden_dim=dec_hidden_dim, emb_dim=emb_dim,isAttended=True, start=letter2index['<sos>'])


optimizer = optim.Adam(model.to(device).parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.8,patience=1)
criterion = nn.CrossEntropyLoss(reduction='none')


training(model, epochs, train_loader, val_loader, criterion, optimizer, scheduler=scheduler,detect_anomaly=False, rs_sample_size=0, gumbel_noise=False)


print("finished training")

predres = testing(model, test_loader, save=True, filename="sub.csv", rs_sample_size=100)

print(len(predres))

with open ("sub.csv", mode="w") as wfile:
    mywriter = csv.writer(wfile,delimiter = ',')
    mywriter.writerow(["Id","Predicted"])

    for i in range(len(predres)):
        mywriter.writerow([i,predres[i]])

print("finished predicting")

