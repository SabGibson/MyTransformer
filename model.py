import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model : int , vocab_size:int):
        """ 
        Input Embeddings convrts sentence to input embeddings 

        Parameters
        ----------
        d_model : int 
            - dimension of embedding vector
        
        vocab_size : int 
            - number of words in the vocabulary
        """
        super.__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    
    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self,d_model : int, seq_len:int , dropout:float=0.2) -> None:
        """ 
        Positinoal Encoding add vector of equal size to input embedding,
        to enrich encoding with positional information of a tokens location
        in a given sequence

        Parameters
        ----------
        d_model : int 
             - size of embedding dim 
        seq_len : int 
            - max length of sequence 

        dropout: float (0 - 1) default 0.2
            - dropout during training for reqularization
        """
        super.__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout

        # creaate matrix of shape (seq_len, d_model)
        pe = torch.zeros((seq_len, d_model))
        
        # create seq_len vector to represent position of ft in seq - logaritmic version 
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1) # pos 
        div_term = torch.exp(torch.arange(0,d_model,2)).float()*(-math.log(1000.0)/d_model) # 1/ 10000^(2i/d_model) with log - epx transform

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0) # (1 , seq_len,d_model)

        self.register_buffer('pe',pe) # saved when saved in file of model 

    def forward(self,x):
        x = x + (self.pe[:,x.shape[1],:]).requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self,eps : float = 10e-6) -> None:
        """ 
        Layer norm will normalize each feature with respect to itself 

        parameters
        ----------


        """
        super.__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(di = -1,keepdim=True)
        std = x.std(dim=-1,keepdim=True)
        return self.alpha * (x - mean)/ (std + self.eps) * self.bias
    

class PointwiseFeedForwardBlock(nn.Module):
    def __init__(self,d_model:int,d_ff:int,dropouot:float):
        super.__init()
        self.linear = nn.Linear(d_model,d_ff) #w1 and b1
        self.dropout = nn.Dropout(dropouot)
        self.linear_2 = nn.Linear(dff,d_model) # w2 and b2 


    def forward(self,x):
        # (Batch , seq_len , d_model) -> (Batch,seq_len,d_ff) -> (Batch , seq_len , d_model)
        return self.linear_2(self.dropout(F.relu(self.linear(x))))

