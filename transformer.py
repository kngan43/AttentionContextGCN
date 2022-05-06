
from audioop import reverse
from numpy import dtype, float32
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import math
from data import *
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv
import torch_geometric.nn as nn
import torch_geometric.transforms as gTrans
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.nn.inits import glorot, uniform
from torch_geometric.utils import softmax
import torch.nn as torchnn
import torch.nn.functional as F
from torch.nn.init import xavier_normal


class BipartiteEdgeConv(MessagePassing):
    '''
    Message passing layer for bipartite graphs
    '''
    def __init__(self,in_channels=64, out_channels=64,n_heads=2, use_norm=True,dropout=0.2):
        super().__init__(aggr='add')
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
        self.lin3 = torch.nn.Linear(in_channels, out_channels)

        self.in_dim = in_channels
        self.out_dim = out_channels
        self.n_heads = n_heads
        self.d_k = out_channels // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm
        self.att = None
        
        self.k_linears = torchnn.Linear(in_channels,out_channels)
        self.q_linears = torchnn.Linear(in_channels,out_channels)
        self.v_linears = torchnn.Linear(in_channels,out_channels)
        self.a_linears = torchnn.Linear(out_channels,out_channels)
        if use_norm:
            self.norms = torchnn.LayerNorm(out_channels)

        self.relation_pri   = torchnn.Parameter(torch.ones(self.n_heads))
        self.relation_att   = torchnn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.relation_msg   = torchnn.Parameter(torch.Tensor(n_heads, self.d_k, self.d_k))
        self.drop           = torchnn.Dropout(dropout)
        
        glorot(self.relation_att)
        glorot(self.relation_msg)

    def forward(self, x_tuple, edge_attr, edge_index):
        
        dest, src = x_tuple    # destination is the node that recieves information from it's source neighbors during message passing
        print(dest.shape)
        print(dest[0])
        edge_index = torch.LongTensor(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=dest.size(0))
        

        # dest = self.lin1(dest)
        # src = self.lin2(src)
        # edge_attr = self.lin3(edge_attr)
      
        # row, col = edge_index
        # deg = degree(col, dest.size(0), dtype=dest.dtype)
        # deg_inv_sqrt = deg.pow(-0.5)
        # deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        # norm = deg_inv_sqrt[row] #* deg_inv_sqrt[col]



        
        # print("HERE1")
        # print(edge_attr.shape)
        # Step 4: Propagate the embeddings to the next layer
        # print('edge_index',edge_index.shape)
        # print('dest',dest.shape)
        # print('src',src.shape)
        #print(edge_attr.shape)
        #edge_attr = torch.mean(edge_attr,1)
        #print(edge_attr.shape)
        # print('sdfsffff\n\n\n\n\n\n\n')
        # print('x_tuple',len(x_tuple))
        print(edge_index.shape)
        print(x_tuple[0].shape)
        print(x_tuple[1].shape)
        print(edge_attr.shape)
        return self.propagate(edge_index, x=x_tuple, edge_attr=edge_attr)

    def message(self,edge_index_i,x_i, x_j, edge_attr_j):
        data_size = edge_index_i.size(0)
        res_att     = torch.zeros(data_size, self.n_heads).to(x_i.device)
        res_msg     = torch.zeros(data_size, self.n_heads, self.d_k).to(x_i.device)
        
        target_node_vec = x_i
        source_node_vec = x_j

        q_mat = self.q_linear(target_node_vec).view(-1, self.n_heads, self.d_k)
        k_mat = self.k_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
        k_mat = torch.bmm(k_mat.transpose(1,0), self.relation_att).transpose(1,0)
        res_att = (q_mat * k_mat).sum(dim=-1) * self.relation_pri/ self.sqrt_dk
        v_mat = self.v_linear(source_node_vec).view(-1, self.n_heads, self.d_k)
        res_msg = torch.bmm(v_mat.transpose(1,0), self.relation_msg).transpose(1,0)   
        self.att = softmax(res_att, edge_index_i)
        res = res_msg * self.att.view(-1, self.n_heads, 1)
        del res_att, res_msg

        res = res.reshape(x_j.shape[0],-1) + edge_attr_j

        return res.view(-1, self.out_dim)

    def update(self, aggr_out, x_i, x_j):

        inputx = x_i#torch.stack((x_i,x_j))
        #print('aggr_out',aggr_out.shape)
        aggr_out = F.gelu(aggr_out)
        #print('aggr_out',aggr_out.shape)
        res = torch.zeros(aggr_out.size(0), self.out_dim).to(x_i.device)

        trans_out = self.drop(self.a_linears(aggr_out))

        # print('trans_out',trans_out.shape)
        # print('inputx',inputx.shape)
        if self.use_norm:
            res = self.norms(trans_out + inputx[0])
        else:

            res = trans_out + inputx[0]
        return res


class HeteroGNN(torch.nn.Module):
    def __init__(self,num_users, num_user_features, num_items, num_item_features, num_context_features, 
        hidden_channels=64, out_channels=64, num_layers=2,num_class=1):
        super().__init__()
        print(num_context_features)
        print("HERE")
        batch = 30
        self.user_embeddings = torch.nn.Embedding(batch,(num_users,hidden_channels), dtype=torch.float)
        self.user_feature_embeddings = torch.nn.Embedding(batch,(num_user_features,hidden_channels), dtype=torch.float)
        self.item_embeddings = torch.nn.Embedding(batch,(num_items,hidden_channels), dtype=torch.float)
        self.item_feature_embeddings = torch.nn.Embedding(batch,(num_item_features,hidden_channels), dtype=torch.float)
        self.context_feature_embeddings = torch.nn.Embedding(batch,(num_context_features-num_items,hidden_channels), dtype=torch.float)
        
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
        self.non_lin = torch.nn.Sigmoid()
        self.loss_func = torchnn.CrossEntropyLoss()
        print('print(d.num_context_features)',d.num_context_features)

    def forward(self, x_dict,edge_attr_dict, edge_index_dict,feat_dict):
        #print(self.convs)
        
        user_id = torch.LongTensor(x_dict['user'])
        item_id = torch.LongTensor(x_dict['item'])
        user_feat = torch.LongTensor(feat_dict['user'])
        item_feat = torch.LongTensor(feat_dict['item'])
        print(edge_attr_dict.keys())
        context_feat1 = torch.LongTensor(edge_attr_dict[('user', 'context', 'item')])
        context_feat2 = torch.LongTensor(edge_attr_dict[('item', 'context', 'user')])
        uid_embed = self.user_embeddings(user_id)
        itemid_embed = self.item_embeddings(item_id)
        userfeat_embed = self.user_feature_embeddings(user_feat)
        itemfeat_embed = self.item_feature_embeddings(item_feat)
        #print('contextfeat',edge_attr_dict[('user', 'context', 'item')].shape)
        context_embed1 = self.context_feature_embeddings(context_feat1)
        context_embed2 = self.context_feature_embeddings(context_feat2)
        userfeat_embed = torch.mean(userfeat_embed,1)
        itemfeat_embed = torch.mean(itemfeat_embed,1)

        # print('uid_embed',uid_embed.shape)
        # print('userfeat_embed',userfeat_embed.shape)
        # print('itemid_embed',itemid_embed.shape)
        # print('itemfeat_embed',itemfeat_embed.shape)
        x_feature_dict = {
            'user':torch.cat((uid_embed,userfeat_embed),0),
            'item':torch.cat((itemid_embed,itemfeat_embed),0)
        }
        edge_attr_dict = {
            ('user', 'context', 'item'):context_embed1,
            ('item', 'context', 'user'):context_embed2
        }
        
        # for key, value in x_feature_dict.items():
        #     print('thing')
        #     print(x_feature_dict[key].shape)
        #     x_feature_dict[key] = torch.mean(value,1)
        
        user_layer_outputs = [x_feature_dict['user']]
        item_layer_ouputs = [x_feature_dict['item']]

        for conv in self.convs:
            output = conv(x_feature_dict, edge_attr_dict, edge_index_dict)
            x_feature_dict = output
            user_layer_outputs.append(x_feature_dict['user'])
            item_layer_ouputs.append(x_feature_dict['item'])
            #x_feature_dict = {key: x.relu() for key, x in x_feature_dict.items()}

        user_layer_outputs = torch.stack(user_layer_outputs)
        item_layer_ouputs = torch.stack(item_layer_ouputs)
        a_user = 1/len(user_layer_outputs+1)
        a_item = 1/len(item_layer_ouputs+1)

        user_final = torch.sum(user_layer_outputs*a_user,0).unsqueeze(1)
        item_final = torch.sum(item_layer_ouputs*a_item,0).unsqueeze(1)

        context_final = edge_attr_dict[('user','context','item')]
        
        final_cat = torch.cat((user_final,item_final,context_final),1)
        B,F,E = final_cat.shape
        final_cat = final_cat.sum(1)
        logits = self.fc(final_cat)

        return logits
    
    # def loss(self,labels, logits, embedding):
    #         nonlin_logits = self.non_lin(logits)
    #         log_loss = self.loss_func(labels, logits)

    #         u_embeddings_pre = tf.nn.embedding_lookup(self.user_embeddings, self.user_input)
    #         i_embeddings_pre = tf.nn.embedding_lookup(self.item_embeddings, self.item_input)
    #         feature_embeddings_pre = tf.concat([self.user_feature_embeddings,
    #                                             self.item_feature_embeddings,
    #                                             self.context_feature_embeddings], 0)
    #         self.emb_loss = self.reg * Tool.l2_loss(u_embeddings_pre, i_embeddings_pre, feature_embeddings_pre)
    #         # self.emb_loss = self.regs[0] * Tool.l2_loss(self.all_init_embedding, self.all_init_bias)
    #         self.loss = self.log_loss + self.emb_loss


# model = HGT(hidden_channels=64, out_channels=dataset.num_classes,
#             num_heads=2, num_layers=2)








#anime
#uid,profile,anime_uid,gender,year,title,episodes,members,popularity,ranked,score,score,year_start,year_end,genra_pca_1,genra_pca_2,genra_pca_3,genra_pca_4,genra_pca_5,genra_pca_6,genra_pca_7,genra_pca_8,genra_pca_9,genra_pca_10,genra_pca_11,genra_pca_12,genra_pca_13,genra_pca_14,genra_pca_15,genra_pca_16,genra_pca_17,genra_pca_18,genra_pca_19,genra_pca_20,score,score









# # yelp
# d = DataInfo()
# u_feat_names = ['user_id','u_yelping_year-u_stars']
# i_feat_names = ['item_id','i_city-i_stars-i_is_open']
# c_feat_name = ['c_city-c_year-c_month-c_day-c_hour-c_minute-c_DoW-c_last']
# path = '/Users/kerryngan/neu/7500_dl/final_project/checkpoint1/GCM/dataset/Yelp-NC/train.dat'
# user_feats, item_feats, userf_map, itemf_map, label= d.load_node_csv(path,u_feat_names,i_feat_names, encoder=InitEncoder())
# edge_index, edge_attr = d.load_edge_csv(path,u_feat_names,userf_map,i_feat_names,itemf_map,c_feat_name,encoders=InitEncoder())

data = HeteroData()

# data['user'].x = item_feats  # Users do not have any features.
# data['item'].x = user_feats
# data['user','context','item'].edge_index = edge_index
# data['user','context','item'].edge_attr = edge_attr
# data = gTrans.ToUndirected()(data)

d = DataInfo()
user_input_val, context_input_val, item_input_val, labels_val = d._get_pointwise_all_data_context( phase='valid')
user_input_val = user_input_val
context_input_val = context_input_val[:30]
item_input_val = item_input_val[:30]
labels_val = labels_val[:30]
min = len(user_input_val)
for i in (user_input_val, context_input_val, item_input_val, labels_val):
    if min > len(i):
        min = i


edge = batch,(num_users,hidden_channels), dtype=torch.float)
self.user_feature_embeddings = torch.nn.Embedding(batch,(num_user_features,hidden_channels), dtype=torch.float)
self.item_embeddings = torch.nn.Embedding(batch,(num_items,hidden_channels), dtype=torch.float)
self.item_feature_embeddings = torch.nn.Embedding(batch,(num_item_features,hidden_channels), dtype=torch.float)
self.context_feature_embeddings = torch.nn.Embedding(batch,(num_context_features-num_items,hidden_channels), dtype=torch.float)

edge_index = [user_input_val,item_input_val]
data['user'].x = user_input_val  # Users do not have any features.
data['item'].x = item_input_val
data['user','context','item'].edge_index = np.asarray(edge_index)
data['item','context','item'].edge_index = np.asarray(list(reversed(edge_index)))
data['user','context','item'].edge_attr = context_input_val
data['item','context','user'].edge_attr = context_input_val
print(data)
#print(data.is_undirected())
print('UPPPPPPPP')
#data = gTrans.ToUndirected()(data)
data['item'].label = labels_val
data['item'].feat = d.get_feature_matrix('item')
data['user'].feat = d.get_feature_matrix('user')


# print(labels_val[:2])
# print(user_input_val[:2])
# print(context_input_val[:2])
# print('trainmatrix', d.train_matrix.shape)
# data_iter_val = DataIterator(user_input_val, context_input_val, item_input_val, labels_val,batch_size=32, shuffle=False)

placeholder = 200000
model = HeteroGNN(d.num_users,d.num_user_features,d.num_items, d.num_item_featuers, placeholder,num_layers=2)
data_iter_val = DataIterator(user_input_val[:30], context_input_val[:30], item_input_val[:30], labels_val[:30],batch_size=32, shuffle=False)
with torch.no_grad():  # Initialize lazy modules.
     out = model(data.x_dict, data.edge_attr_dict, data.edge_index_dict, data.feat_dict)








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




