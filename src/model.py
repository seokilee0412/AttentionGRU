import torch
import torch.nn as nn

class Attention(nn.Module):
  def __init__(self, hidden_dims):
    super(Attention,self).__init__()
    self.linaer1 = nn.Linear(5*hidden_dims, 5*hidden_dims)
  def forward(self,x):
    x = torch.reshape(x, shape=(-1, x.shape[1] * x.shape[2]))
    logit = self.linaer1(x)
    attention_score = nn.Softmax()(logit)
    output = torch.mul(attention_score, x)
    return output

class Attentionbased_GRU(nn.Module):
  def __init__(self,input_size,hidden_size,drop_prob=0,bidirectional=True,num_layers=1):
    super(Attentionbased_GRU, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.dropout = drop_prob
    self.bidirectional = bidirectional
    self.num_layers = num_layers

    self.gru = nn.GRU(input_size,hidden_size,dropout=drop_prob,bidirectional=True,batch_first=True)
    self.attention = Attention(2*hidden_size)
    self.layerdense1 = nn.Linear(5*2*hidden_size, 128)
    self.layerdense2 = nn.Linear(128,128)
    self.layerdense3 = nn.Linear(128,64)
    self.layerdense4 = nn.Linear(64,3)
    

    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax()

  def forward(self,x):
    out, _ = self.gru(x)
    out = self.tanh(out)
    out = self.attention(out)
    out = self.layerdense1(self.relu(out))
    out = self.layerdense2(self.relu(out))
    out = self.layerdense3(self.relu(out))
    out = self.layerdense4(self.relu(out))
    # out = self.softmax(out)
    return out
  
  # def init_hidden(self):
  #   if self.bidirectional:
  #     return torch.zeros(self.num_layers*2,self.batch_size,self.hidden_size)
  #   else:
  #     return torch.zeros(self.num_layers,self.batch_size,self.hidden_size)