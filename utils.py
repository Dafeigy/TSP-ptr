import random
import itertools
import numpy as np
import math


def generate_cities(N)->list[int,int]:
    '''
    Generate N cities location [(x1,y1),...(xn,yn)]
    '''
    random_list = list(itertools.product(range(1, N), range(1, N)))
    return random.sample(random_list, N)

def calculate_distance(p1,p2,sqrt=False)->float:
    '''
    Calculate distance between p1,p2
    '''
    if sqrt:
        return math.sqrt(sum([(x - y) ** 2 for x, y in zip(p1, p2)]))
    else:
        return sum([(x - y) ** 2 for x, y in zip(p1, p2)])

def calculate_total_dis(cities:list[tuple], order:list):
    '''
    Calculate
    '''
    return sum([calculate_distance(cities[order[i]], cities[order[(i+1)%len(order)]]) for i in range(len(order))])


if __name__ == '__main__':
    a = generate_cities(3)
    order = [0,1,2]
    test = calculate_distance((1,1),(2,2))
    print(test)
    print(a)
    loss = calculate_total_dis(a,order)
    print(loss)