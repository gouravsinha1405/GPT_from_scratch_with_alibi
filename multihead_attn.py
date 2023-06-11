import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import math

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dropout_prob = 0.2
class Embedding(nn.Module):

  def __init__(self, vocab_size, embed_size):
    super().__init__()
    self.token_embeddings = nn.Embedding(vocab_size, embed_size)
  
  def forward(self, input):
    out = self.token_embeddings(input)
    # print(out.shape)
    return out

class PositionalEmbedding(nn.Module):
  """For every token in context length we will generate a positional encoding"""

  def __init__(self, context_len, embed_size):
    super().__init__()
    pos_encoding = torch.zeros(context_len, embed_size)
    for pos in range(context_len):
      for i in range(0,embed_size,2):
        pos_encoding[pos,i] = math.sin(pos / (10000 ** ((2 * i)/embed_size)))
        pos_encoding[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1))/embed_size)))
    pos_encoding = pos_encoding.unsqueeze(dim=0)
    pos_encoding = pos_encoding
    # self.dropout = nn.Dropout(dropout_prob)
    self.register_buffer('pe',pos_encoding)
  
  def forward(self, input):
    context_len = input.shape[1]
    # print(context_len)
    # print(input.shape)
    # print(self.pe[:,:context_len].shape)
    input = input + self.pe[:,:context_len,:]
    # input = input + torch.autograd.Variable(self.pe[:,:context_len],requires_grad=False)
    return input
##-----implementing alibi-------##
def get_relative_token_pos_in_context(context_len):
  x = torch.arange(context_len).unsqueeze(0).to(device) # adds row dimenison i.e 1,3
  y = torch.arange(context_len).unsqueeze(1).to(device) # adds column dimension i.e 3,1
  return x-y

def get_alibi_slope(head_num):
  m = 2**(-8/head_num)
  return torch.tensor(m).to(device)
#------self attention with single head----------#
class Head(nn.Module):

  def __init__(self,head_size,embed_size,context_length,head_num,causal_mask=True):
    super().__init__()

    self.query = nn.Linear(embed_size, head_size, bias=False)
    self.key = nn.Linear(embed_size, head_size, bias=False)
    self.value = nn.Linear(embed_size, head_size, bias=False)
    self.dropout = nn.Dropout(dropout_prob)
    self.causal_mask = causal_mask
    # self.register_buffer("relative_position",get_relative_token_pos_in_context(context_length))
    self.register_buffer("alibi_slope",get_alibi_slope(head_num))
    
  
  def forward(self, q, k, v, mask=None):
    # batch_size, context_len, embed_size = q.shape
    batch_size = q.shape[0]
    output_context_len = q.shape[1]
    input_context_len = k.shape[1]
    embed_size = q.shape[2]
    tril = torch.tril(torch.ones(output_context_len, input_context_len)).to(device)
    # if self.causal_mask:
    #   self.register_buffer('tril',torch.tril(torch.ones(output_context_len, input_context_len)))
    q = self.query(q)
    k = self.key(k)
    v = self.value(v)
    # print(q.shape, k.shape, v.shape)
    bias = self.alibi_slope * get_relative_token_pos_in_context(input_context_len)
    bias = bias.unsqueeze(0)
    weights = q @ k.transpose(-2,-1) * embed_size**-0.5
    # print(weights.shape)
    weights += bias
    # print(weights.shape)
    # print(self.tril.shape)
    if mask is not None:
      weigths = weights.masked_fill(mask[:,None, None, :] == 0, float('-inf'))
    if self.causal_mask:
      # print(self.tril.shape)
      weights = weights.masked_fill(tril[:output_context_len,:input_context_len]==0, float('-inf'))
    # print(weights[0])
    weights = F.softmax(weights, dim=-1)
    weights = self.dropout(weights)
    out = weights @ v
    return out
#-----multihead attention------#
class MultiHeadAttention(nn.Module):

  def __init__(self, num_heads, head_size,embed_size,context_length,causal_mask=True):
    super().__init__()
    self.multiheads = nn.ModuleList([Head(head_size,embed_size,context_length,head_num+1,causal_mask) for head_num in range(num_heads)])
    self.projection = nn.Linear(embed_size, embed_size)
    self.dropout = nn.Dropout(dropout_prob)
  
  def forward(self,q,k,v,mask=None):
    out = torch.cat([h(q,k,v,mask) for h in self.multiheads], dim = -1)
    return self.dropout(self.projection(out))
#---------Feedforward--------------#
class FeedForward(nn.Module):
  """Self attention while calculates the interactions among the tokens the feedforward will train the model on
  individual tokens and try to extract the information individually"""
  def __init__(self,embed_size):
    super().__init__()
    self.neural_net = nn.Sequential(
        nn.Linear(embed_size, 4 * embed_size), # multiplying by 4 as per the paper 'attention is all you need', this expands the hidden layer
        nn.GELU(),
        nn.Linear(4 * embed_size, embed_size), #--projection layer 
        nn.Dropout(dropout_prob)
    )
  def forward(self, x):
    return self.neural_net(x)