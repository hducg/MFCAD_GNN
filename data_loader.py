# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:38:16 2019

@author: 2624224
"""
import pickle
import numpy as np

import torch
from torch_geometric.data import Data, InMemoryDataset
    
    
def load_dataset(path_list, transform=None):
    with open(path_list, 'rb') as file:
        paths = pickle.load(file)
#    paths = glob.glob(graph_dir + '*.graph')
    
    data_list = []    
    for path in paths:
        with open(path, 'rb') as file:
            a_graph = pickle.load(file)            
                            
            x = torch.tensor(a_graph['x'], dtype=torch.float)               
            edge_index = torch.tensor(a_graph['edge_index'], dtype=torch.long)                
            y = torch.tensor(a_graph['y'], dtype=torch.long)                
            data = Data(x=x, edge_index=edge_index, y=y)
            if transform is not None:
                data = transform(data)
            data_list.append(data)
            
    return data_list, paths


class RandomTransform(object):
    def __init__(self, translate):
        self.translate = translate
    
    def __call__(self, data):
        np_random = np.random.RandomState(np.random.randint(1e6))
        m = np.eye(3,dtype='float32')
        m[0,0] *= np_random.randint(0,2)*2-1
        m = np.dot(m,np.linalg.qr(np_random.randn(3,3))[0])
        t = np_random.uniform(-self.translate,self.translate,(3,1)).astype('float32')
        
        x = data.x
        m = torch.tensor(m).to(x.dtype).to(x.device)
        t = torch.tensor(t).to(x.dtype).to(x.device)
        try:
            data.x[:,:3] = torch.mm(x[:,:3], m)
            data.x[:,3] += torch.mm(x[:,:3], t).view(x.size()[0])
        except:
            raise Exception('{}, {}'.format(x, x.size()))
        return data
    
        
class FaceNet(InMemoryDataset):
    def __init__(self, root, data_list_file, transform=None):        
        super(FaceNet, self).__init__(root, transform)
        
        data_list = []
        with open(data_list_file, 'rb') as file:
            paths = pickle.load(file)
            
        for path in paths:            
            with open(path, 'rb') as file:
                a_graph = pickle.load(file)
                x = torch.tensor(a_graph['x'], dtype=torch.float)                
                edge_index = torch.tensor(a_graph['edge_index'], dtype=torch.long)                
                y = torch.tensor(a_graph['y'], dtype=torch.long)                
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)
              
        self.data, self.slices = self.collate(data_list)
        
    @property    
    def raw_file_names(self):
        return ['simple']
    
    @property
    def processed_file_names(self):
        return ['simple']
    
    def download(self):
        pass
    
    def process(self):
        pass
    