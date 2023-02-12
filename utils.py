import random
import itertools
import numpy as np
import math

# For Inference
def generate_cities(N)->list[int,int]:
    '''
    Generate N cities location [(x1,y1),...(xn,yn)]
    '''
    random_list = list(itertools.product(range(1, N), range(1, N)))
    return random.sample(random_list, N)

def calculate_distance(p1,p2,sqrt=False)->float:
    '''
    Calculate distance between p1,p2, set False on sqrt for faster calcuation.
    '''
    if sqrt:
        return math.sqrt(sum([(x - y) ** 2 for x, y in zip(p1, p2)]))
    else:
        return sum([(x - y) ** 2 for x, y in zip(p1, p2)])

def calculate_total_dis(cities:list[tuple], order:list):
    '''
    Calculate total distance^2 as loss function
    '''
    return sum([calculate_distance(cities[order[i]], cities[order[(i+1)%len(order)]]) for i in range(len(order))])


# For Training
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.utils.data import Dataset, dataloader

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

if __name__ == '__main__':
    trainset = TSPdataset(10,1000)
    print(torch.FloatTensor(2, 10).uniform_(0, 6))