import torch
from torch import nn 
from torch import optim
import torch.nn.functional as F  

class MLP(nn.Module):
    def __init__(self,model_conf):
        super(MLP, self).__init__()
        in_dim = model_conf['input_dimension']
        out_dim = model_conf['num_classes']

        
        
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 2000),
            nn.Tanh(),
            #nn.Linear(1000, 1000),
            #nn.Tanh(),
            #nn.Linear(1000, 1000),
            #nn.Tanh(),
            nn.Linear(2000, 1500),
            nn.Tanh(),
            #nn.Linear(1500, 1000),
            #nn.Tanh(),
            nn.Linear(1500, out_dim)
            #nn.Tanh(),d
            #nn.Linear(1000, 500),
            #nn.Tanh(),
            #nn.Linear(500, 100),
            #nn.Tanh(),
            
        )


        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        
        out = self.layers(x)
        
        probs = F.softmax(out)
        output = {}
        output['probs'] = probs 
        output['abs_logits'] =  torch.abs(out)
        output['logits'] = out 

        return output