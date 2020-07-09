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
model.load_state_dict(torch.load( "25best.t7", map_location=torch.device('cpu') )) 
#model.load_state_dict(torch.load( "25best.t7"))
model.to(device)

#classification

transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])


#for pred on classification task
class ImageDataset(data.Dataset):
    def __init__ (self, file_list,transforms):
        self.file_list = file_list
        self.transforms = transforms

    def __len__ (self):
        return len(self.file_list)

    def __getitem__(self,index):
        img = Image.open(self.file_list[index])
        img = self.transforms(img)
        #remember to do transforms!!!!!!!!!!!!!!!!!!

        return img


# classification pred
file_list = []
pic_list = open("test_order_classification.txt", "r").read().split()
for pic in pic_list:
    file_list.append(os.path.join("test_classification/medium",pic))


pred_dataset_classification = ImageDataset(file_list,transforms)


pred_loader_args_classification = dict(shuffle=False, batch_size=256, num_workers=8, pin_memory=True) if cuda else dict(shuffle=False, batch_size=64)
pred_loader_classification = data.DataLoader(pred_dataset_classification, **pred_loader_args_classification )

#pred classification
def pred_model_classification(model, pred_loader):
    with torch.no_grad():  
        model.eval()

        predLabel= []

        for batch_idx, data in enumerate(pred_loader):

            data = data.to(device)
            tupleres = model(data)
            outputs = tupleres[1]

            _, predicted = torch.max(outputs.data, 1) 

            predLabel = predLabel + predicted.tolist() 

    return predLabel

predLabel = pred_model_classification(model, pred_loader_classification)

key=sorted([str(x) for x in range(2300)])
val=[i for i in range(2300)]
intkey = [int(x) for x in key]
mapping = dict(zip(val,intkey))

with open ("new_sub_classification.csv", mode="w") as wfile:
    mywriter = csv.writer(wfile,delimiter = ',')
    mywriter.writerow(["Id","Category"])

    for i in range(len(predLabel)):
        realclass = mapping[predLabel[i]]
        mywriter.writerow([pic_list[i],realclass])

print("done")