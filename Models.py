import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable

import math


def reward_func(route):
    batch_size = route[0].size(0) 
    n = len(route)
    tour_len = Variable(torch.zeros([batch_size])).cuda()

    for i in range(n):
        tour_len += torch.norm(route[i] - route[(i + 1) % n], dim=1)

    return tour_len

# Ptr network
class PTRNet(nn.Module):
    def __init__(self,
                embedding_size,
                hidden_size,
                seq_len,
                n_glimpse,
                C,
                use_tanh,
    ):
        super(PTRNet).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_glimpse = n_glimpse
        self.seq_len = seq_len


        self.embedding = Embedding_Block(2, embedding_size)
        self.encoder = Encoder(embedding_size, hidden_size)
        self.decoder = Decoder(embedding_size, hidden_size)
        self.pointer = Attention_Block(hidden_size, use_tanh = use_tanh, C = C )
        self.glimpse = Attention_Block(hidden_size, use_tanh=False, C = C)

        self.decoder_init = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_init.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

    def apply_mask(self, logits, mask ,idxs):
        batch_size = logits.size(0)
        clone_mask = mask.clone()

        if idxs is not None:
            clone_mask[[i for i in range(batch_size)], idxs.data] = 1
            logits[clone_mask] = -np.inf
        return logits, clone_mask

    def forward(self,input_seq):
        batch_size, _, seq_len = input_seq.shape()
        embedding_result = self.embedding(input_seq)
        encoder_outputs, (ht,ct) = self.encoder(embedding_result)

        prev_probs = []
        prev_idxs = []
        mask = torch.zeros(batch_size, seq_len).byte().cuda()
        idxs = None
        decoder_input = self.decoder_init.unsqueeze(0).repeat(batch_size, 1)

        for i in range(seq_len):
            _, ht, ct = self.decoder(decoder_input.unsqueeze(1),(ht, ct))

            query = ht.squeeze(0)
            for j in range(self.n_glimpse):
                ref, logits = self.glimpse(query, encoder_outputs)
                logits, mask = self.apply_mask(logits, mask, idxs)
                query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2)

            _, logits = self.pointer(query, encoder_outputs)
            logits, mask = self.apply(logits, mask, idxs)
            probs = F.softmax(logits)

            idxs = probs.multinomial().squezze(1)
            for old_idxs in prev_idxs:
                if old_idxs.eq(idxs).data.any():
                    idxs = probs.multinomial().squeeze(1)
                    break

            decoder_input = embedding_result[[i for i in range(batch_size)], idxs.data, :]

            prev_probs.append(probs)
            prev_idxs.append(idxs)

        return prev_probs, prev_idxs



# Hierachy level
class Actor(nn.Module):
    def __init__(self,
                 embedding_size,
                hidden_size,
                seq_len,
                n_glimpse,
                C,
                use_tanh,
                reward):
        super(Actor, self).__init__()
        self.reward = reward
        self.ptr = PTRNet(
            embedding_size,
                hidden_size,
                seq_len,
                n_glimpse,
                C,
                use_tanh,
        )
    
    def forward(self, x):
        batch_size, input_size, seq_len = x.shape()
        probs, action_idxs = self.ptr(x)

        actions = []
        inputs = x.transpose(1,2)

        for action_id in action_idxs:
            actions.append(inputs[[x for x in range(batch_size)], action_id.data, :])
        
        action_probs = []
        for prob, action_id in zip(probs, action_idxs):
            action_probs.append([prob[[x for x in range(batch_size)]]])

        R = self.reward(actions)
        return R, action_probs, actions, action_idxs

class Critic(nn.Module):
    def __init__(self) -> None:
        super(Critic).__init__()
        self.judge = reward_func


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
    def __init__(self, input_size, embedding_size) -> None:
        super(Embedding_Block, self).__init__()
        self.embedding_size = embedding_size
        self.embedding = nn.Parameter(torch.FloatTensor(input_size,embedding_size))
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))

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
    def __init__(self, hidden_size:int, use_tanh:bool = False, C=10) -> None:
        super(Attention_Block).__init__()
        self.C = C
        self.use_tanh = use_tanh
        self.W_query = nn.Linear(hidden_size, hidden_size)
        self.W_ref = nn.Conv1d(hidden_size, hidden_size, 1 ,1)
        self.V = nn.Parameter(torch.FloatTensor(hidden_size).cuda())
        self.V.data.uniform_(-(1. / math.sqrt(hidden_size)) , 1. / math.sqrt(hidden_size))

    def forward(self,query, ref):
        '''
        Args:
            query: [batch_size x hidden_size]
            ref:    [batch_size x seq_len x hidden_size]
        '''
        batch_size, seq_len, _ = ref.shape()
        ref = ref.permute(0,2,1)    # So it becomes [batch_size x hidden_size x seq_len]
        expand_query = query.repeat(1,1,seq_len)   # [2 x batch_size, 2 x btachsize, (seq_len + 1) x hidden_size]
        V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size x 1 x hidden_size]
        logits = torch.bmm(V, F.tanh(expand_query + ref)).squeeze(1)

        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        return ref, logits


if __name__ == "__main__":

    from datagenerator import TSPdataset
    from torch.utils.data import DataLoader
    Embedding = Embedding_Block(input_size=2, embedding_size=128)
    data = TSPdataset(10,20)
    dataloader = DataLoader(data, batch_size = 4, shuffle = True)
    for batch_index, sample in enumerate(dataloader):
        print(Embedding(Variable(sample)))

