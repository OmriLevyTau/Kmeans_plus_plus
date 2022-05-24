import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def initialize_centroids(S,k):
    np.random.seed(0)
    N = len(S)
    i = 1
    first_idx = np.random.randint(N)
    centroids = np.array(S[first_idx]).reshape(-1,len(S[0]))
    indices = [first_idx]
    while i<k:
        D = np.zeros(N)
        for l in range(N):
            D[l] = (np.linalg.norm(S[l]-centroids,axis=1)**2).min()
        D_SUM = D.sum()
        Prob = D/D_SUM
#         print(Prob)
        next_mu_index = np.random.choice(np.arange(N),p=Prob)
        centroids = np.append(centroids,[S[next_mu_index]],axis=0)
        indices.append(next_mu_index)
        i+=1
    return centroids,indices

def read_data(name1: str, name2: str):
    df1 = pd.read_csv(name1, sep=",", header=None)
    df2 = pd.read_csv(name2, sep=",", header=None)
    res = pd.merge(df1,df2,how='inner',on=0)
    res.sort_values(by=0,inplace=True)
    return res.drop(0,axis=1).to_numpy()
