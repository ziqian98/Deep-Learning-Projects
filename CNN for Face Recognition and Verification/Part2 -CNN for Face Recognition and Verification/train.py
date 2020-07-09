#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import torch
import torchvision
from torch.utils import data
import torch.nn as nn 
import torch.nn.functional as F
import time
from sklearn.metrics import roc_auc_score
from PIL import Image

cuda = torch.cuda.is_available()

device = torch.device("cuda" if cuda else "cpu")

print("cuda is: ",cuda)



class verification_dataset(data.Dataset):
    def __init__(self, task, directory):
        self.task = task
        self.directory = directory
        if(task == "validation"):
            self.line_list = open("validation_trials_verification.txt","r").read().split()
        else:
            self.line_list = open("test_trials_verification_student.txt","r").read().split()

    def __len__(self):
        if(task == "validation"):
            return int(len(self.line_list)/3)
        else:
            return int(len(self.line_list))
    
    def __getitem__(self,index):
        if(task == "validation"):
            img0 = Image.open(os.path.join(self.directory, self.line_list[index*3]))
            img0 = torchvision.transforms.ToTensor()(img0)

            img1 = Image.open(os.path.join(self.directory, self.line_list[index*3+1]))
            img1 = torchvision.transforms.ToTensor()(img1)

            onePerson = int(self.line_list[index*3+2])

            return img0, img1, onePerson

        else:
            img0 = Image.open(os.path.join(self.directory, self.line_list[index*2]))
            img0 = torchvision.transforms.ToTensor()(img0)

            img1 = Image.open(os.path.join(self.directory, self.line_list[index*2+1]))
            img1 = torchvision.transforms.ToTensor()(img1)

            return img0, img1





#classification

transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=32,padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])


print("loading train_dataset_classification")
train_dataset_classification = torchvision.datasets.ImageFolder(root = "train_data/medium/",
                                                 transform=transforms)


print("loading val_dataset_classification")
val_dataset_classification = torchvision.datasets.ImageFolder(root = "validation_classification/medium/",
                                                 transform=transforms)


train_loader_args_classification = dict(shuffle=True, batch_size=256, num_workers=8, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
train_loader_classification = data.DataLoader(train_dataset_classification, **train_loader_args_classification)
    


val_loader_args_classification = dict(shuffle=False, batch_size=256, num_workers=8, pin_memory=True) if cuda else dict(shuffle=False, batch_size=64)
val_loader_classification = data.DataLoader(val_dataset_classification, **val_loader_args_classification)
    


#verification 
print("loading val_dataset_verification")
task = "validation"
directory = "validation_verification"
val_dataset_verification = verification_dataset(task,directory)

val_loader_args_verification = dict(shuffle=False, batch_size=256, num_workers=8, pin_memory=True) if cuda else dict(shuffle=False, batch_size=64)
val_loader_verification = data.DataLoader(val_dataset_verification, **val_loader_args_verification)


print("loading done")



class Block(nn.Module):
    def __init__(self, inp, outp, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * inp
        self.conv1 = nn.Conv2d(inp, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, outp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(outp)

        self.shortcut = nn.Sequential()
        if stride == 1 and inp != outp:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, outp, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(outp),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileV2(nn.Module):

    settings = [(1,  16, 1, 1),
                (6,  24, 2, 1),
                (6,  32, 3, 2),
                (6,  64, 4, 2),
                (6,  96, 3, 1),
                (6, 160, 3, 2),
                (6, 320, 1, 1)]

    def __init__(self, num_classes=2300):
        super(MobileV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)
        #self.linear_closs = nn.Linear(1280, 2300, bias=False)
        #self.relu_closs = nn.ReLU(inplace=True)

    def _make_layers(self, inp):
        layers = []
        for expansion, outp, num_blocks, stride in self.settings:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(inp, outp, expansion, stride))
                inp = outp
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        embedding = out
        #closs_output = self.linear(out)
        #closs_output = self.relu_closs(closs_output)
        out = self.linear(out)
        classification = out
        return embedding, classification



model = MobileV2()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=5e-5, momentum=0.9, nesterov = True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.8,patience=1)
model.to(device)



def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    
    running_loss = 0.0
    total_predictions = 0.0
    correct_predictions = 0.0
    
    start_time = time.time()
    print("in train")
    
    for batch_idx, (data, target) in enumerate(train_loader):
        if (batch_idx % 1000 == 0) :
            print("batch_idx",batch_idx)
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        
        tupleres = model(data)
        
        outputs = tupleres[1]

        _, predicted = torch.max(outputs.data, 1)
        total_predictions += target.size(0)
        correct_predictions += (predicted == target).sum().item()

        loss = criterion(outputs, target)
        running_loss = running_loss + loss.item()
        
        loss.backward()
        optimizer.step()
        #scheduler.step(loss)
        
    end_time = time.time()
    
    running_loss = running_loss / len(train_loader)
    acc = (correct_predictions/total_predictions)*100.0
    print('Training Accuracy: ', acc, '%')
    print('Training Loss: ', running_loss)
    print('Time: ',end_time - start_time, 's')
    
    return running_loss


#on val
def test_model_classification(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        
        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            
            tupleres = model(data)
        
            outputs = tupleres[1]
            
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
            


def test_model_verification(model, test_loader):
    with torch.no_grad():
        model.eval()

        trueres=[]
        scores = []
        
        for batch_idx, (img0,img1,onePerson) in enumerate(test_loader):

            img0 = img0.to(device)
            img1 = img1.to(device)
            onePerson = onePerson.to(device) 

            embedding0 = model(img0)[1]
            embedding1 = model(img1)[1]

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            score = cos(embedding0,embedding1).detach()


            trueres += onePerson.tolist() #onePerson is tensor
            scores += score.tolist() #score is tensor
            
        
        ret = roc_auc_score(trueres, scores)
        print('AUC: ', ret)

        return ret

n_epochs = 100
Train_loss = []
Test_loss = []
Test_acc = []
Score = []

print("start train")

for i in range(n_epochs):

    print("epoch", i)
    train_loss = train_epoch(model, train_loader_classification, criterion, optimizer)
    
    print("classification test")
    test_loss, test_acc = test_model_classification(model, val_loader_classification, criterion)

    print("verification test")
    score = test_model_verification(model, val_loader_verification)

    
    scheduler.step(test_loss)  #on val loss

    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    Score.append(score)

    print('='*20)
    modelname = str(i+1) + "newnew.t7"  
    torch.save(model.state_dict(),modelname)



#final submit for task 1
#10lun 0.01 factor 0.8+ 6 0.1 factor 0.8 + 7lun 0.01 factor0.5  + 2lun 0.001 factor 0.8
#25best.t7
#acc 72.3


#10lun 0.01, no schedular
# best auc
