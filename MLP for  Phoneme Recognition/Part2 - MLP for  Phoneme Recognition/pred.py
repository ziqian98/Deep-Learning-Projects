import torch
import numpy as np
from torch.utils import data
import torch.nn as nn 
import csv
k=15

predx = np.load("test.npy",allow_pickle=True)

print(predx.shape)


class MyPredDataset(data.Dataset):
    def __init__ (self,x,k):
        self.x = x
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
        
        return xspand



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


pred_dataset = MyPredDataset(predx,k)
pred_loader_args = dict(shuffle=False, batch_size=32)
pred_loader = data.DataLoader(pred_dataset,**pred_loader_args)


model = My_MLP([(2*k+1)*40,2048,2048,1024,1024,512,512,138])

model.load_state_dict(torch.load("9epochnewdeep.t7", map_location=torch.device('cpu') ) )

def pred_model(model, pred_loader):
    with torch.no_grad():  
        model.eval()

        predLabel= []

        for batch_idx, data in enumerate(pred_loader):
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1) 

            predLabel = predLabel + predicted.tolist() 

    return predLabel

predLabel = pred_model(model, pred_loader)
print(len(predLabel))


with open ("sub.csv", mode="w") as wfile:
    mywriter = csv.writer(wfile,delimiter = ',')
    mywriter.writerow(["id","label"])

    for i in range(len(predLabel)):
        mywriter.writerow([i,predLabel[i]])

print("done")






