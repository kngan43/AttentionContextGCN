
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import math
from dataset1 import *
from torch_geometric.data import HeteroData
#from torch_geometric.nn import HeteroConv, Linear, SAGEConv
from torch_geometric.nn import HeteroConv
import torch_geometric.nn as nn
import torch_geometric.transforms as gTrans
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.nn.inits import glorot, uniform
from temporal import RelTemporalEncoding
from torch_geometric.utils import softmax
import torch.nn as torchnn
import torch.nn.functional as F


class BipartiteEdgeConv(MessagePassing):
    '''
    Message passing layer for bipartite graphs
    '''
    def __init__(self,in_channels=64, out_channels=64, num_types=2, num_relations=1,n_heads=2, use_norm=True,dropout=0.2):
        super().__init__(aggr='add')
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
        self.lin3 = torch.nn.Linear(in_channels, out_channels)
        # self.x1_key, self.c_key, self.x2_key = keys
        # self.x_dict = x_dict
        # self.edge_index_dict = edge_index_dict
        # self.edge_attr_dict = edge_attr_dict

        self.in_dim        = in_channels
        self.out_dim       = out_channels
        self.num_types     = num_types
        self.num_relations = num_relations
        self.total_rel     = num_types * num_relations * num_types
        self.n_heads       = n_heads
        self.d_k           = out_channels // n_heads
        self.sqrt_dk       = math.sqrt(self.d_k)
        self.use_norm      = use_norm
        self.att           = None
        
        self.k_linears = torchnn.Linear(in_channels,   out_channels)
        self.q_linears = torchnn.Linear(in_channels,   out_channels)
        self.v_linears = torchnn.Linear(in_channels,   out_channels)
        self.a_linears = torchnn.Linear(out_channels,  out_channels)
        if use_norm:
            self.norms = torchnn.LayerNorm(out_channels)
        '''
            TODO: make relation_pri smaller, as not all <st, rt, tt> pair exist in meta relation list.
        '''
        self.relation_pri   = torchnn.Parameter(torch.ones(self.n_heads))
        self.relation_att   = torchnn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.relation_msg   = torchnn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.drop           = torchnn.Dropout(dropout)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)

    def forward(self, x_tuple, edge_attr, edge_index):
        
        dest, src = x_tuple    # destination is the node that recieves information from it's source neighbors during message passing

        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=dest.size(0))
        
        # Step 2: Multiply with weights
        dest = self.lin1(dest)
        src = self.lin2(src)
        edge_attr = self.lin3(edge_attr)

        # Step 3: Calculate the normalization       
        row, col = edge_index
        deg = degree(col, dest.size(0), dtype=dest.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] #* deg_inv_sqrt[col]
        # print("HERE1")
        # print(edge_attr.shape)
        # Step 4: Propagate the embeddings to the next layer
        # print('edge_index',edge_index.shape)
        # print('dest',dest.shape)
        # print('src',src.shape)
        edge_attr = torch.mean(edge_attr,1)
        # print('sdfsffff\n\n\n\n\n\n\n')
        # print('x_tuple',len(x_tuple))
        return self.propagate(edge_index, x=x_tuple, edge_attr=edge_attr, norm=norm)

    def message(self,edge_index_i,x_i, x_j, edge_attr_j, norm):
        '''
            j: source, i: target; <j, i>
        '''
        data_size = edge_index_i.size(0)
        '''
            Create Attention and Message tensor beforehand.
        '''
        res_att     = torch.zeros(data_size, self.n_heads).to(x_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(x_i.device)
        
       
        k_linear = self.k_linears
        v_linear = self.v_linears
        q_linear = self.q_linears

        '''
            Get the corresponding input node representations by idx.
            Add tempotal encoding to source representation (j)
        '''
        target_node_vec = x_i
        source_node_vec = x_j

        '''
            Step 1: Heterogeneous Mutual Attention
        '''
        q_mat = q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
        k_mat = k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
        k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att).transpose(1,0)
        res_att = (q_mat * k_mat).sum(dim=-1) * self.relation_pri/ self.sqrt_dk
        '''
            Step 2: Heterogeneous Message Passing
        '''
        v_mat = v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
        res_msg = torch.bmm(v_mat.transpose(1,0), self.relation_msg).transpose(1,0)   
        '''
            Softmax based on target node's id (edge_index_i). Store attention value in self.att for later visualization.
        '''
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        # print(type(res))
        del res_att, res_msg
        #return res.view(-1, self.out_dim)

        res = res.reshape(x_j.shape[0],-1) + edge_attr_j
        # print("TRANFORMSLDKFJ:LSDJFLSDKFJ")
        # print('res.shape',res.shape)
        # print('edge_attr_j',edge_attr_j.shape)
        # print('x_j',x_j.shape)
        # print('norm.view(-1, 1) * (x_j + edge_attr_j)',(norm.view(-1, 1) * (x_j + edge_attr_j)).shape)
        # print('res.view(-1, self.out_dim)',res.view(-1, self.out_dim).shape)

        # Normalize node features.     
        #return norm.view(-1, 1) * (x_j + edge_attr_j)
        return res.view(-1, self.out_dim)

    def update(self, aggr_out, x_i, x_j):
        '''
            Step 3: Target-specific Aggregation
            x = W[node_type] * gelu(Agg(x)) + x
        '''
        inputx = x_i#torch.stack((x_i,x_j))
        #print('aggr_out',aggr_out.shape)
        aggr_out = F.gelu(aggr_out)
        #print('aggr_out',aggr_out.shape)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(x_i.device)

        trans_out = self.drop(self.a_linears(aggr_out))
        '''
            Add skip connection with learnable weight self.skip[t_id]
        '''

        # print('trans_out',trans_out.shape)
        # print('inputx',inputx.shape)
        if self.use_norm:
            res = self.norms(trans_out + inputx[0])
        else:

            res = trans_out + inputx[0]
        return res


class HeteroGNN(torch.nn.Module):
    def __init__(self,hidden_channels=64, out_channels=64, num_layers=2,num_class=1):
        super().__init__()
        # self.x_dict = x_dict
        # self.edge_index_dict = edge_index_dict
        # self.edge_attr_dict = edge_attr_dict
        
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('user','context','item'): BipartiteEdgeConv(),
                ('item', 'rev_context', 'user'): BipartiteEdgeConv(),
            }, aggr='sum')
            self.convs.append(conv)

        self.lin = nn.Linear(hidden_channels, out_channels)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels,hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels,num_class)

        )

    def forward(self, x_dict,edge_attr_dict, edge_index_dict):
        #print(self.convs)
        for key, value in x_dict.items():
            x_dict[key] = torch.mean(value,1)
        
        user_layer_outputs = [x_dict['user']]
        item_layer_ouputs = [x_dict['item']]

        for conv in self.convs:
            #x_dict = {key: torch.mean(x,0) for key, x in x_dict.items()}
            #print("PASSED")
            output = conv(x_dict, edge_attr_dict, edge_index_dict)
            x_dict = output
            user_layer_outputs.append(x_dict['user'])
            item_layer_ouputs.append(x_dict['item'])
            #x_dict = {key: x.relu() for key, x in x_dict.items()}

        user_layer_outputs = torch.stack(user_layer_outputs)
        item_layer_ouputs = torch.stack(item_layer_ouputs)
        a_user = 1/len(user_layer_outputs+1)
        a_item = 1/len(item_layer_ouputs+1)

        user_final = torch.sum(user_layer_outputs*a_user,0).unsqueeze(1)
        item_final = torch.sum(item_layer_ouputs*a_item,0).unsqueeze(1)
        #print('user_final',user_final.shape)
        #print('item_final',item_final.shape)
        context_final = edge_attr_dict[('user','context','item')]
        
        final_cat = torch.cat((user_final,item_final,context_final),1)
        B,F,E = final_cat.shape
        #final_cat = final_cat.reshape(B,-1)
        final_cat = final_cat.sum(1)
        logits = self.fc(final_cat)
        #print('final_cat',final_cat.shape)
        #print(logits.shape)
        return logits


# model = HGT(hidden_channels=64, out_channels=dataset.num_classes,
#             num_heads=2, num_layers=2)


d = DataInfo()

# yelp
u_feat_names = ['user_id','u_yelping_year-u_stars']
i_feat_names = ['item_id','i_city-i_stars-i_is_open']
c_feat_name = ['c_city-c_year-c_month-c_day-c_hour-c_minute-c_DoW-c_last']
path = '/Users/kerryngan/neu/7500_dl/final_project/checkpoint1/GCM/dataset/Yelp-NC/train.dat'
user_feats, item_feats, userf_map, itemf_map, label= d.load_node_csv(path,u_feat_names,i_feat_names, encoder=InitEncoder())
edge_index, edge_attr = d.load_edge_csv(path,u_feat_names,userf_map,i_feat_names,itemf_map,c_feat_name,encoders=InitEncoder())


#anime
#uid,profile,anime_uid,gender,year,title,episodes,members,popularity,ranked,score,score,year_start,year_end,genra_pca_1,genra_pca_2,genra_pca_3,genra_pca_4,genra_pca_5,genra_pca_6,genra_pca_7,genra_pca_8,genra_pca_9,genra_pca_10,genra_pca_11,genra_pca_12,genra_pca_13,genra_pca_14,genra_pca_15,genra_pca_16,genra_pca_17,genra_pca_18,genra_pca_19,genra_pca_20,score,score







data = HeteroData()

data['user'].x = item_feats  # Users do not have any features.
data['item'].x = user_feats
data['user','context','item'].edge_index = edge_index
data['user','context','item'].edge_attr = edge_attr
data = gTrans.ToUndirected()(data)
# data['item','context','user'].edge_index = reverse_map
# data['item','context','user'].edge_attr = edge_attr



model = HeteroGNN(num_layers=2)
with torch.no_grad():  # Initialize lazy modules.
     out = model(data.x_dict, data.edge_attr_dict, data.edge_index_dict)








# '''
# Training
# '''
# #assert torch.cuda.is_available()
# # device = 'cuda'
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Hyperparameters
# import time
# epochs = 5 # epoch
# lr = 0.0005  # learning rate
# batch_size = 64 # batch size for training
  

# emsize = 64

# num_heads = 4
# num_trx_cells = 2

# gradient_norm_clip = 1

# model = HeteroGNN(num_layers=2).to(device)

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-8)
# # total_accu = None

# for epoch in range(1, epochs + 1):
#     epoch_start_time = time.time()
#     model.train()
#     total_acc, total_count = 0, 0
#     log_interval = 500
#     start_time = time.time()

#     for idx, (label, text) in enumerate(dataloader):
#         label = label.to(device)
#         text = text.to(device)
#         optimizer.zero_grad()
        
#         ###########################################################################
#         # TODO: compute the logits of the input, get the loss, and do the         #
#         # gradient backpropagation.
#         ###########################################################################
#         logits = model(text)
#         sig_logits = torch.nn.Sigmoid(logits)
#         loss_func = torch.optim.adamw()
#         loss = loss_func(logits, label)
#         loss.backward()
#         ###########################################################################
#         #                             END OF YOUR CODE                            #
#         ###########################################################################
        
#         torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
#         optimizer.step()
#         total_acc += (logits.argmax(1) == label).sum().item()
#         total_count += label.size(0)
#         if idx % log_interval == 0 and idx > 0:
#             elapsed = time.time() - start_time
#             print('| epoch {:3d} | {:5d}/{:5d} batches '
#                   '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
#                                               total_acc/total_count))
#             total_acc, total_count = 0, 0
#             start_time = time.time()
#     accu_val = evaluate(model, val_loader, loss_func, device)
#     if total_accu is not None and total_accu > accu_val:
#         scheduler.step()
#     else:
#         total_accu = accu_val
#     print('-' * 59)
#     print('| end of epoch {:3d} | time: {:5.2f}s | '
#           'valid accuracy {:8.3f} '.format(epoch,
#                                            time.time() - epoch_start_time,
#                                            accu_val))
#     print('-' * 59)
