import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

length: int
width: int
data_dict = {}



class dataLoader(Dataset):

    def __init__(self, path):
        dataFrame = pd.read_csv(path)
        pure_data = dataFrame.iloc[:, 2:]
        print(pure_data.shape)
        for ind_col in range(len(pure_data.columns)-1):
            max_value = np.max(pure_data.iloc[:,ind_col])
            min_value = np.min(pure_data.iloc[:,ind_col])

            pure_data.iloc[:,ind_col] = (pure_data.iloc[:,ind_col]-min_value)/(max_value-min_value)
            pure_data.iloc[:,ind_col] = (pure_data.iloc[:,ind_col]-min_value) / (pure_data.iloc[:,ind_col].std())



        for i in range(dataFrame.__len__()):
            tmp = list(pure_data.iloc[i, :])
            if tmp[-1] == 'Genuine':
                tmp[-1] = 1
            else:
                tmp[-1] = 0
            data_dict[i] = tmp
            # print(tmp)
        self.data_dict = data_dict



    def __getitem__(self, index):
        return torch.Tensor(self.data_dict[index])

    def __len__(self):
        return self.data_dict.__len__()