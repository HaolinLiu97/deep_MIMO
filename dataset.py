import os,sys
sys.path.append("../")
import time
from torch.utils.data import Dataset
import numpy as np
import math
import torch

class MIMO_dataset(Dataset):
    def __init__(self,num_bits=8,isTrain=True):
        self.num_bits=num_bits
        if isTrain==True:
            self.dataset_len=256
        else:
            self.dataset_len=256

    def __len__(self):
        return self.dataset_len*100

    def __getitem__(self,index):
        binary_input = np.random.rand((self.num_bits))
        binary_input = binary_input > 0.5
        binary_input=binary_input.astype(np.float)

        index=0
        for i in range(binary_input.shape[0]):
            binary_number=binary_input[i]
            if binary_number==1:
                index+=2**i
        target=np.zeros((2**self.num_bits))
        target[index]=1
        return binary_input,target


if __name__=="__main__":
    dataset=MIMO_dataset()
    binary_input,target=dataset.__getitem__(0)
    print(binary_input,target)