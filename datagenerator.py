# For Training
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader

class TSPdataset(Dataset):
    '''
    Generate `num_samples` of `num_nodes` cities.
    '''
    def __init__(self, num_nodes, num_samples) -> None:
        super(TSPdataset).__init__()
        self.data_set = []

        for _ in tqdm(range(num_samples)):
            x = torch.FloatTensor(2, num_nodes).uniform_(0, 1)
            self.data_set.append(x)
        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.data_set[index]

if __name__ == "__main__":
    data = TSPdataset(10,100)
    loader = DataLoader(data, batch_size = 4, shuffle=True)
    for idx, sample in enumerate(loader):
        print(sample)
        print(sample.shape)
        break