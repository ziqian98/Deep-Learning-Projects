from torch.utils import data
import torch
import torch.nn as nn 
import sys 
import torch.optim as optim
import time

cuda = torch.cuda.is_available()
cuda
k=15

import numpy as np
train = np.load("train.npy",allow_pickle=True)

print(train.shape)
print("123")


dev = np.load("dev.npy",allow_pickle=True)
print(dev.shape)


train_labels = np.load("train_labels.npy",allow_pickle=True)
print(train_labels.shape)
dev_labels = np.load("dev_labels.npy",allow_pickle=True)
print(dev_labels.shape)


test = np.load("test.npy",allow_pickle=True)
print(test.shape)
print("aaa")

class MyDataset(data.Dataset):
    def __init__ (self,x,y,k):
        self.x = x
        self.y = y
        self.k = k
        
        self.eachPairList = []
        
        for dataInedx, data in enumerate(x):
            for phenomIndex in range(data.shape[0]):
                self.eachPairList.append((dataInedx,phenomIndex))
        
    def __len__(self):
        return len(self.eachPairList)
    
    def __getitem__(self,index):
        dataInedx = self.eachPairList[index][0]
        phenomIndex = self.eachPairList[index][1]
        
        xspand = self.x[dataInedx].take(range(phenomIndex-self.k,phenomIndex+self.k+1), 
                                mode = "clip",axis=0).flatten()


        
        xspand = torch.Tensor(xspand).float()
        
        
        yphenom= self.y[dataInedx][phenomIndex]
        #yphenom = torch.Tensor(yphenom).float()
        
        return xspand, yphenom


num_wokers = 8 if cuda else 0

train_dataset = MyDataset(train, train_labels, k)
train_loader_args = dict(shuffle=True, batch_size=256, num_workers=8, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
train_loader = data.DataLoader(train_dataset, **train_loader_args)


test_dataset = MyDataset(dev, dev_labels, k)
test_loader_args = dict(shuffle=False, batch_size=256, num_workers=8, pin_memory=True) if cuda else dict(shuffle=False, batch_size=64)
test_loader = data.DataLoader(test_dataset, **test_loader_args)



class My_MLP(nn.Module):
    def __init__ (self, size_list):
        super(My_MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list)-2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.BatchNorm1d(size_list[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)
        
    def forward(self,x):
        return self.net(x)


model = My_MLP([(2*k+1)*40,2048,2048,1024,1024,512,512,138])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
device = torch.device("cuda" if cuda else "cpu")
model.to(device)


def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    
    running_loss = 0.0
    
    start_time = time.time()
    print("in train")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if (batch_idx % 1000 == 0) :
            print("batch_idx",batch_idx)
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        
        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss = running_loss + loss.item()
        
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    
    running_loss = running_loss / len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    
    return running_loss


def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            
            outputs = model(data)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()
            
        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        return running_loss, acc
            

n_epochs = 8
Train_loss = []
Test_loss = []
Test_acc = []

print("cuda is: ",cuda)
print("new666")

for i in range(n_epochs):
    print("epoch", i)
    train_loss = train_epoch(model, train_loader, criterion, optimizer)
    print("test")
    test_loss, test_acc = test_model(model, test_loader, criterion)
    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    print('='*20)
    modelname = str(i+1) + "epochdeep.t7"  
    torch.save(model.state_dict(),modelname)

