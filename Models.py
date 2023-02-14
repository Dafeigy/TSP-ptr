import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

import math


# Hierachy level
class Actor():
    def __init__(self) -> None:
        pass

class Critic():
    def __init__(self) -> None:
        pass


# Basic Modules
class Encoder():
    def __init__(self,embedding_size, hidden_size) -> None:
        self.RNN = RNN_Block(embedding_size, hidden_size)
    
    def forward(self,x):
        encoder_outputs, ht, ct = self.RNN(x)
        return encoder_outputs, ht, ct

class Decoder():
    def __init__(self,embedding_size, hidden_size) -> None:
        self.RNN = RNN_Block(embedding_size, hidden_size)

    def forward(self,x):
        decoder_outputs, ht, ct = self.RNN(x)
        return decoder_outputs, ht, ct

# Basic Units
class Embedding_Block(nn.Module):
    def __init__(self, input_size, embedding_size, use_cuda = False) -> None:
        super(Embedding_Block, self).__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        
        self.embedding = nn.Parameter(torch.FloatTensor(input_size,embedding_size))
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)),1. / math.sqrt(embedding_size))

    def forward(self, input_seq):
        
        # Input_seq shape : [batch_size x 2 x seq_len]
        
        batch_size, _ ,seq_len = input_seq.shape
        input_seq = input_seq.unsqueeze(1)

        # Input_seq shape : [batch_size x 1 x 2 x seq_len]
        print(input_seq.shape)
        embedding = self.embedding.repeat(batch_size, 1, 1) 
        embedding_output = \
             [torch.bmm(input_seq[:,:,:,i].float(), self.embedding.repeat(batch_size, 1, 1)) for i in range(seq_len)]
        for i in range(seq_len):
            embedding_output.append(torch.bmm(input_seq[:, :, :, i].float(), embedding))
        return torch.cat(embedding_output,1)



class RNN_Block(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layer = 1) -> None:
        super(RNN_Block).__init__()
        self.LSTM = nn.LSTM(embedding_size, hidden_size,num_layer = num_layer)

    def forward(self,x):
        x, ht, ct= self.LSTM(x)
        return x, ht,ct



class Attention_Block(nn.Module):
    def __init__(self) -> None:
        super(Attention_Block).__init__()

if __name__ == "__main__":
    from datagenerator import TSPdataset
    from torch.utils.data import DataLoader
    Embedding = Embedding_Block(input_size=2, embedding_size=128)
    data = TSPdataset(10,20)
    dataloader = DataLoader(data, batch_size = 4, shuffle = True)
    for batch_index, sample in enumerate(dataloader):
        print(Embedding(Variable(sample)))

