import torch
from torch.nn.init import xavier_uniform_
import pandas as pd
from functools import reduce
from operator import concat
from torch_geometric.utils import to_undirected

def load_node_csv(path, encoder=None, **kwargs):
    u_feat_names = ['user_id','u_yelping_year-u_stars']
    i_feat_names = ['item_id','i_city-i_stars-i_is_open']
    df = pd.read_csv(path, **kwargs)
    df = df.set_index(u_feat_names)
    #print(df.index.unique())
    userf_map = {index: i for i, index in enumerate(df.index.unique())}
    df.reset_index(inplace=True)
    df = df.set_index(i_feat_names)
    itemf_map = {index: i for i, index in enumerate(df.index.unique())}
    df.reset_index(inplace=True)

    x = None
    if encoder is not None:
        #print(df.columns)
        user_feat = encoder(df[u_feat_names], u_feat_names)

        #user_feat = torch.stack(u, dim=0)
        #print('user_feat',user_feat.shape)

        item_feat = encoder(df[i_feat_names], i_feat_names)
        #item_feat = torch.stack(i, dim=0)
        #print('item_feat',item_feat)
    return user_feat, item_feat, userf_map, itemf_map

def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)
    #print(df[src_index_col])
    #print(df.columns)
    #print('src_index_col',src_index_col)
    #print(dst_index_col)
    #print(df[src_index_col])
    c_feat_name = ['c_city-c_year-c_month-c_day-c_hour-c_minute-c_DoW-c_last']

    df = df.set_index(src_index_col)
    src = [src_mapping[index] for index in df.index]
    df.reset_index(inplace=True)
    df = df.set_index(dst_index_col)
    dst = [dst_mapping[index] for index in df.index]
    df.reset_index(inplace=True)
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attr = encoders(df[c_feat_name], c_feat_name)
    
    # an edge is (2, num_edges) and this is one direction map
    # to make a hetero graph undirecct, we swap the edge_index rows
    reverse_map = torch.zeros_like(edge_index)
    index = torch.LongTensor([1, 0,])
    reverse_map[index] = edge_index
    return edge_index, edge_attr, reverse_map


class InitEncoder(object):
    def __init__(self):
        super().__init__()

    
    def __call__(self, df, col, hidden_dim=64):
        #print(col)
        col = [c.split('-') for c in col]
        col = reduce(concat, col)
        num_row = len(df.values)
        #print(col)
        num_feat = len(col) # caculate the number of features in a col
        #print(col.split('-'))
        x = torch.empty(num_row, num_feat, hidden_dim)
        #print((num_row, num_feat, hidden_dim))
        x = xavier_uniform_(x)
        return x


u_feat_names = ['user_id','u_yelping_year-u_stars']
i_feat_names = ['item_id','i_city-i_stars-i_is_open']
path = '/Users/kerryngan/neu/7500_dl/final_project/checkpoint1/GCM/dataset/Yelp-NC/train.dat'
user_feats, item_feats, userf_map, itemf_map = load_node_csv(path, InitEncoder())
edge_index, edge_attr, reverse_map = load_edge_csv(path,u_feat_names,userf_map,i_feat_names,itemf_map,InitEncoder())


from torch_geometric.data import HeteroData
data = HeteroData()

data['user'].x = item_feats  # Users do not have any features.
data['item'].x = user_feats
data['user','context','item'].edge_index = edge_index
data['user','context','item'].edge_attr = edge_attr



print(list(itemf_map.items())[:100])
#print(list(edge_index)[:edge_attr])