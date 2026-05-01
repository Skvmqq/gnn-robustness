import torch.nn as nn
import torch
from  torch_geometric.nn import GCNConv
from sklearn.linear_model import LogisticRegression

class GCNModel(nn.Module):
    def __init__(self,num_features,args,num_classes,dropout):
        super().__init__()
        self.conv1=GCNConv(num_features,args.hidden)
        self.conv2=GCNConv(args.hidden, num_classes)
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self,x,edge_index):
        x=self.conv1(x,edge_index)
        x=x.relu()
        x=self.conv2(x,edge_index)
        return x


class MLPModel(nn.Module):
    def __init__(self,num_features,args,num_classes,dropout):
        super().__init__()
        self.nn1= nn.Linear(num_features,args.hidden)
        self.nn2= nn.Linear(args.hidden, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def reset_parameters(self):
        self.nn1.reset_parameters()
        self.nn2.reset_parameters()

    def forward(self,x,edge_index=None):
        x=self.nn1(x)
        x=x.relu()
        x=self.nn2(x)

        return x 


class LogisticRegressionModel:
    def __init__(self, max_iter=1000, random_state=42):
        self.model = LogisticRegression(max_iter=max_iter, random_state=random_state)

    def fit(self, x_train, y_train):
        if torch.is_tensor(x_train):
            x_train = x_train.detach().cpu().numpy()
        if torch.is_tensor(y_train):
            y_train = y_train.detach().cpu().numpy()
        self.model.fit(x_train, y_train)
        return self

    def predict(self, x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return self.model.predict(x)






