
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import * 

class TextClassifierMLPHead(nn.Module):

  def __init__(self, model_conf):

    super(TextClassifierMLPHead, self).__init__()
    PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
    self.drop = nn.Dropout(p=0.3)
    self.head = nn.Linear(self.bert.config.hidden_size, model_conf['num_classes'])
    self.device = model_conf['device']
  
  def forward(self, x):
    
    input_ids = x["input_ids"].to(self.device)
    attention_mask = x["attention_mask"].to(self.device)
    input_ids, attention_mask_, pooled_output = self.bert( input_ids=input_ids, attention_mask=attention_mask )
    output = self.drop(pooled_output)
    out    = self.head(output)
    probs = F.softmax(out)
    output = {}
    output['probs'] = probs 
    output['abs_logits'] =  torch.abs(out)
    output['logits'] = out 
    
    return output 