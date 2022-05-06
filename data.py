import torch
from torch.nn.init import xavier_uniform_
import pandas as pd
from functools import reduce
from operator import concat
from torch_geometric.utils import to_undirected
import scipy.sparse as sp
import numpy as np
from DataIterator import DataIterator

path = '/Users/kerryngan/neu/7500_dl/final_project/checkpoint1/GCM/dataset/Yelp-NC'

KEEP_CONTEXT = {
    'yelp-nc': ['c_city', 'c_month', 'c_hour', 'c_DoW', 'c_last'],
    'yelp-oh': ['c_city', 'c_month', 'c_hour', 'c_DoW', 'c_last'],
    'amazon-book': ['c_year', 'c_month', 'c_day', 'c_DOW', 'c_last']
}

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
        x = torch.empty(num_row, num_feat, hidden_dim,requires_grad=True, dtype=torch.float)
        #print((num_row, num_feat, hidden_dim))
        x = xavier_uniform_(x)
        return x


class DataInfo(object):
    def __init__(self):
        """
        Constructor
        """
        #self.separator = conf.data_separator # [',', '-'] #hardcoded
        #self.dataset_name = conf.dataset # 'Yelp-OH' 
        #self.dataset_folder = conf.data_path #'/Users/kerryngan/neu/7500_dl/final_project/checkpoint1/GCM/dataset/%s/' % args.dataset #hardcoded
        self.num_negatives = 4
        self.separator = [',', '-']
        self.dataset_name = 'Yelp-NC' 
        self.dataset_folder = '/Users/kerryngan/neu/7500_dl/final_project/checkpoint1/GCM/dataset/Yelp-NC/'
        data_format = 'UIC' #default
        #conf.model_name = 'GCM' #default
        #data_splitter = GivenData(self.dataset_name, self.dataset_folder, conf.data_format, self.separator, self.logger)

        self.load_data()
        # self.test_context_list = self.all_data_dict['test_data']['context_id'].tolist() if self.side_info is not None else None
        if self.side_info is None:
            self.test_context_dict = None
        else:
            self.test_context_dict = {}
            for user, context in zip(self.all_data_dict['test_data']['user_id'].tolist(), self.all_data_dict['test_data']['context_id'].tolist()):
                self.test_context_dict[user] = context

        self.num_users, self.num_valid_items = self.train_matrix.shape
        if self.side_info is not None:
            self.num_user_features = self.side_info['side_info_stats']['num_user_features']
            self.num_item_featuers = self.side_info['side_info_stats']['num_item_features']
            self.num_context_features = self.side_info['side_info_stats']['num_context_features']

    def load_node_csv(path, u_feat_names, i_feat_names, encoder=None):
        #u_feat_names = ['user_id','u_yelping_year-u_stars']
        #i_feat_names = ['item_id','i_city-i_stars-i_is_open']
        
        df = pd.read_csv(path)
        df = df.set_index(u_feat_names)
        #print('HERE-------',len(set(df['item_id'].to_list())))
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
        
       # print(df[u_feat_names].head)
        label = df[u_feat_names[-1]].to_list()
        label = [i.split('-')[-1] for i in label]
        #print(label)

        return user_feat, item_feat, userf_map, itemf_map, label

    def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,c_feat_name, encoders=None, **kwargs):
        df = pd.read_csv(path, **kwargs)
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
        # reverse_map = torch.zeros_like(edge_index,requires_grad=True,dtype=torch.float)
        # index = torch.LongTensor([1, 0,])
        # reverse_map[index] = edge_index
        return edge_index, edge_attr#, reverse_map



    # u_feat_names = ['user_id','u_yelping_year-u_stars']
    # i_feat_names = ['item_id','i_city-i_stars-i_is_open']
    # path = '/Users/kerryngan/neu/7500_dl/final_project/checkpoint1/GCM/dataset/Yelp-NC/train.dat'
    # user_feats, item_feats, userf_map, itemf_map = load_node_csv(path, InitEncoder())
    # edge_index, edge_attr, reverse_map = load_edge_csv(path,u_feat_names,userf_map,i_feat_names,itemf_map,InitEncoder())


    # from torch_geometric.data import HeteroData
    # data = HeteroData()

    # data['user'].x = item_feats  # Users do not have any features.
    # data['item'].x = user_feats
    # data['user','context','item'].edge_index = edge_index
    # data['user','context','item'].edge_attr = edge_attr



    #print(list(itemf_map.items())[:100])
    #print(list(edge_index)[:edge_attr])


    def load_dict_from_file(self,filename):
        f = open(filename,'r')
        data=f.read()
        f.close()
        return eval(data)


    def csr_to_user_dict(self,train_matrix):
        """convert a scipy.sparse.csr_matrix to a dict,
        where the key is row number, and value is the
        non-empty index in each row.
        """
        train_dict = {}
        for idx, value in enumerate(train_matrix):
            if len(value.indices):
                train_dict[idx] = value.indices.copy().tolist()
        return train_dict

    

    def load_data(self):
        side_info, all_data_dict = None, None
        train_data = pd.read_csv(self.dataset_folder + "train.dat", sep=self.separator[0])
        test_data = pd.read_csv(self.dataset_folder + "test.dat", sep=self.separator[0])
        userid_dict = self.load_dict_from_file(self.dataset_folder + 'userid_dict.txt')
        itemid_dict = self.load_dict_from_file(self.dataset_folder + 'itemid_dict.txt')
        # self.logger.info('Loading full testset')

        all_data = pd.concat([train_data, test_data])

        self.num_users = len(userid_dict)
        self.num_items = len(itemid_dict) 
        self.num_valid_items = all_data["item_id"].max() + 1

        self.num_train = len(train_data["user_id"])
        self.num_test = len(test_data["user_id"])
        
        train_matrix = sp.csr_matrix(([1] * self.num_train, (train_data["user_id"], train_data["item_id"])), shape=(self.num_users, self.num_valid_items))
        test_matrix = sp.csr_matrix(([1] * self.num_test, (test_data["user_id"], test_data["item_id"])),  shape=(self.num_users, self.num_valid_items))
        side_info, side_info_stats, all_data_dict = {}, {}, {}
        column = all_data.columns.values.tolist()
        context_column = column[2].split(self.separator[1])
        user_feature_column = column[3].split(self.separator[1]) if 'yelp' in self.dataset_name.lower() else None
        item_feature_column = column[-1].split(self.separator[1])
        print('user_feature_column', user_feature_column)
        keep_context = KEEP_CONTEXT[self.dataset_name.lower()]
        new_context_column = '-'.join(keep_context)
        all_data[context_column] = all_data[all_data.columns[2]].str.split(self.separator[1], expand=True)
        all_data[new_context_column] = all_data[keep_context].apply('-'.join, axis=1)
        print('all_data[new_context_column]', all_data[new_context_column])
        # map context to id
        unique_context = all_data[new_context_column].unique()
        context2id = pd.Series(data=range(len(unique_context)), index=unique_context)
        # contextids = context2id.to_dict()
        all_data["context_id"] = all_data[new_context_column].map(context2id)
        train_data = all_data.iloc[:self.num_train, :]
        test_data = all_data.iloc[self.num_train:, :]

        if user_feature_column:
            user_feature = all_data.drop_duplicates(["user_id", '-'.join(user_feature_column)])
            user_feature = user_feature[["user_id", '-'.join(user_feature_column)]]
            user_feature[user_feature_column] = user_feature[user_feature.columns[-1]].str.split(self.separator[1], expand=True)
            user_feature.drop(user_feature.columns[[1]], axis=1, inplace=True)
        else:
            user_feature = None
        item_feature = all_data.drop_duplicates(["item_id", '-'.join(item_feature_column)])
        item_feature = item_feature[["item_id", '-'.join(item_feature_column)]]
        item_feature[item_feature_column] = item_feature[item_feature.columns[-1]].str.split(self.separator[1], expand=True)
        item_feature.drop(item_feature.columns[[1]], axis=1, inplace=True)
        context_feature = all_data.drop_duplicates(["context_id", new_context_column])[["context_id", new_context_column]]
        context_feature[keep_context] = context_feature[context_feature.columns[-1]].str.split(self.separator[1], expand=True)
        context_feature.drop(context_feature.columns[[1]], axis=1, inplace=True)
        if user_feature_column:
            side_info['user_feature'] = user_feature.set_index('user_id').astype(int)
            side_info_stats['num_user_features'] = side_info['user_feature'][user_feature_column[-1]].max() + 1
            side_info_stats['num_user_fields'] = len(user_feature_column)
        else:
            side_info['user_feature'] = None
            side_info_stats['num_user_features'] = 0
            side_info_stats['num_user_fields'] = 0
        
        side_info['item_feature'] = item_feature.set_index('item_id').astype(int)
        side_info['context_feature'] = context_feature.set_index('context_id').astype(int)
        side_info_stats['num_item_features'] = side_info['item_feature'][item_feature_column[-1]].max() + 1
        side_info_stats['num_item_fields'] = len(item_feature_column)
        side_info_stats['num_context_features'] = side_info['context_feature'][keep_context[-2]].max() + 1 + self.num_items
        side_info_stats['num_context_fields'] = len(keep_context)
        #self.logger.info("\n" + "\n".join(["{}={}".format(key, value) for key, value in side_info_stats.items()]))
        #self.logger.info("context feature name: " + ",".join([f.replace('c_', '') for f in keep_context]))
        all_data_dict['train_data'] = train_data[['user_id', 'item_id', 'context_id']]
        all_data_dict['test_data'] = test_data[['user_id', 'item_id', 'context_id']]
        # all_data_dict['positive_dict'] = df_to_positive_dict(all_data_dict['train_data'])

        all_data_dict['positive_dict'] = self.load_dict_from_file(self.dataset_folder + '/user_pos_dict.txt')
        side_info['side_info_stats'] = side_info_stats
        
        num_ratings = len(train_data["user_id"]) + len(test_data["user_id"])
        
        self.train_matrix = train_matrix
        self.test_matrix = test_matrix
        self.all_data_dict = all_data_dict
        self.side_info = side_info

    def _get_pointwise_all_data_context(self, phase='train'):
        user_input, context_input, item_input, labels = [], [], [], []
        if phase == 'train':
            train_data = self.all_data_dict['train_data']
        else:
            train_data = self.all_data_dict['test_data']
            user_pos_test = self.csr_to_user_dict(self.test_matrix)
        user_insts = train_data['user_id'].tolist()
        context_insts = train_data['context_id'].tolist()
        item_insts_pos = train_data['item_id'].tolist()

        num_items = self.num_valid_items
        
        for idx in range(len(user_insts)):
            user_id = user_insts[idx]
            context_id = context_insts[idx]
            user_input.extend([user_id] * (self.num_negatives + 1))
            context_input.extend([context_id] * (self.num_negatives + 1))
            item_input.append(item_insts_pos[idx])
            labels.append(1)
            try:
                user_pos = self.all_data_dict['positive_dict'][user_id][context_id]
            except Exception:
                user_pos = []
            if phase != 'train':
                user_pos = user_pos + user_pos_test[user_id]
            for _ in range(self.num_negatives):
                neg_item_id = np.random.randint(num_items)
                while neg_item_id in user_pos:
                    neg_item_id = np.random.randint(num_items)
                item_input.append(neg_item_id)
                labels.append(0)
        return user_input, context_input, item_input, labels

    def get_feature_matrix(self, key_word):
        mat = self.side_info['%s_feature' % key_word]
        return mat.values if mat is not None else None



