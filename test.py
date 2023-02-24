import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from datagenerator import TSPdataset
from torch.utils.data import DataLoader

USE_CUDA = True

class GraphEmbedding(nn.Module):
    def __init__(self, input_size, embedding_size, use_cuda=USE_CUDA):
        super(GraphEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.use_cuda = use_cuda
        
        self.embedding = nn.Parameter(torch.FloatTensor(input_size, embedding_size)) 
        self.embedding.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len    = inputs.size(2)
        embedding = self.embedding.repeat(batch_size, 1, 1)  
        embedded = []
        inputs = inputs.unsqueeze(1)
        for i in range(seq_len):
            embedded.append(torch.bmm(inputs[:, :, :, i].float(), embedding))
        embedded = torch.cat(embedded, 1)
        return embedded
    
class Attention(nn.Module):
    def __init__(self, hidden_size, use_tanh=False, C=10, name='Bahdanau', use_cuda=USE_CUDA):
        super(Attention, self).__init__()
        
        self.use_tanh = use_tanh
        self.C = C
        self.name = name
        
        if name == 'Bahdanau':
            self.W_query = nn.Linear(hidden_size, hidden_size)
            self.W_ref   = nn.Conv1d(hidden_size, hidden_size, 1, 1)

            V = torch.FloatTensor(hidden_size)
            if use_cuda:
                V = V.cuda()  
            self.V = nn.Parameter(V)
            self.V.data.uniform_(-(1. / math.sqrt(hidden_size)) , 1. / math.sqrt(hidden_size))
            
        
    def forward(self, query, ref):
        """
        Args: 
            query: [batch_size x hidden_size]
            ref:   ]batch_size x seq_len x hidden_size]
        """
        
        batch_size = ref.size(0)
        seq_len    = ref.size(1)
        
        if self.name == 'Bahdanau':
            ref = ref.permute(0, 2, 1)
            query = self.W_query(query).unsqueeze(2)  # [batch_size x hidden_size x 1]
            ref   = self.W_ref(ref)  # [batch_size x hidden_size x seq_len] 
            expanded_query = query.repeat(1, 1, seq_len) # [batch_size x hidden_size x seq_len]
            V = self.V.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1) # [batch_size x 1 x hidden_size]
            logits = torch.bmm(V, F.tanh(expanded_query + ref)).squeeze(1)
            
        elif self.name == 'Dot':
            query  = query.unsqueeze(2)
            logits = torch.bmm(ref, query).squeeze(2) #[batch_size x seq_len x 1]
            ref = ref.permute(0, 2, 1)
        
        else:
            raise NotImplementedError
        
        if self.use_tanh:
            logits = self.C * F.tanh(logits)
        else:
            logits = logits  
        return ref, logits
    
class TSPModel(nn.Module):
    def __init__(self) -> None:
        super(TSPModel, self).__init__()

        self.embedding = GraphEmbedding(2, embedding_size=128, use_cuda=True)
        self.encoder = nn.LSTM(128, 128, batch_first = True)
        self.decoder = nn.LSTM(128, 128, batch_first = True)
        self.glimpse = Attention(128, use_tanh= True, C=10, name='Bahdanau')

        self.pointer = Attention(128, use_tanh= False, C=10, name='Bahdanau')

        self.decoder_start_input = nn.Parameter(torch.FloatTensor(128))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(128)), 1. / math.sqrt(128))

    def forward(self, sample):
        embedded = self.embedding(inputs)
        print(f"Embedded:\n{embedded.shape}")
        encoder_outs, (ht, ct) = self.encoder(embedded)
        print(f"ht:{ht.shape}\nct:{ct.shape}")
        decoder_input =self.decoder_start_input.unsqueeze(0).repeat(4, 1)
        for i in range(10):
            _, (ht,ct) = self.decoder(decoder_input.unsqueeze(1), (ht,ct))
            query = ht.squeeze(0)
            ref, logits = self.glimpse(query,encoder_outs)
            query = torch.bmm(ref, F.softmax(logits).unsqueeze(2)).squeeze(2)
            if i == 0 :
                print(query.shape)
            


if __name__ == "__main__":
    data = TSPdataset(10,100)
    loader = DataLoader(data, batch_size = 8, shuffle=True)
    for idx, sample in enumerate(loader):
        #print(f'sample:\n{sample}')
        break
    inputs = sample.cuda()
    model = TSPModel().cuda()
    print(model(inputs))
    