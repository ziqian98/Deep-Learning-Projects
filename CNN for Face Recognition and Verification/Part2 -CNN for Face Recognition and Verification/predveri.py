#!/usr/bin/env python
# coding: utf-8
import csv
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
            return int(len(self.line_list)/2)
    
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

#for pred on classification task
class ImageDataset(data.Dataset):
    def __init__ (self, file_list):
        self.file_list = file_list

    def __len__ (self):
        return len(self.file_list)

    def __getitem__(self,index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        #remember to do the data preprocessing during pred!!!!!!!!!!

        return img



#classification

transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])



#verification 
print("loading pre_dataset_verification")
task = "pred"
directory = "test_verification"
pred_dataset_verification = verification_dataset(task,directory)

print("len: ",pred_dataset_verification.__len__()) # 899965
pred_loader_args_verification = dict(shuffle=False, batch_size=2048, num_workers=8, pin_memory=True) if cuda else dict(shuffle=False, batch_size=64)
pred_loader_verification = data.DataLoader(pred_dataset_verification, **pred_loader_args_verification)


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

# model.load_state_dict(torch.load( "25best.t7", map_location=torch.device('cpu') )) 
model.load_state_dict(torch.load("10newnew.t7")) #10epoch 0.01, no schedular!
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=5e-5, momentum=0.9, nesterov = True)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min",factor=0.8,patience=1)
model.to(device)


#pred verification

def pred_model_verification(model, pred_loader):
    with torch.no_grad():
        model.eval()

        scores = []
        
        for batch_idx, (img0,img1) in enumerate(pred_loader):
            if(batch_idx%50== 0):
                print(batch_idx)

            img0 = img0.to(device)
            img1 = img1.to(device) 

            embedding0 = model(img0)[1]
            embedding1 = model(img1)[1]

            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            score = cos(embedding0,embedding1).detach()

            scores += score.tolist() #score is tensor
            
        return scores
        



pic_list = open("test_trials_verification_student.txt","r").read().split()
print("start pred_model_verification")
predscore = pred_model_verification(model,pred_loader_verification)
print("end pred_model_verification")
print(len(pic_list)) #899965*2
print(len(predscore)) #899965

with open ("new_new_sub_verification.csv", mode="w") as wfile:
    mywriter = csv.writer(wfile,delimiter = ',')
    mywriter.writerow(["trial","score"])
    for i in range(len(predscore)):
        mywriter.writerow( [ str(pic_list[i*2])+ " " + str(pic_list[i*2+1]), predscore[i]] )



print("done")

