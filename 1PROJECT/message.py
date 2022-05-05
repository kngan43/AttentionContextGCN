
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
GCNConv
class BipartiteEdgeConv(MessagePassing):
    '''
    Message passing layer for bipartite graphs
    '''
    def __init__(self,in_channels=64, out_channels=64):
        super().__init__(aggr='add')
        self.lin1 = torch.nn.Linear(in_channels, out_channels)
        self.lin2 = torch.nn.Linear(in_channels, out_channels)
        self.lin3 = torch.nn.Linear(in_channels, out_channels)
        # self.x1_key, self.c_key, self.x2_key = keys
        # self.x_dict = x_dict
        # self.edge_index_dict = edge_index_dict
        # self.edge_attr_dict = edge_attr_dict

    def forward(self, x_tuple, edge_attr, edge_index):
        '''
        src: node 1 embedding
        dst: node 2 embedding
        e: edge embedding

        '''
        
        src, dst = x_tuple    # src is source, dst is destination
        #src = torch.mean(src,1) #torch.Size([179072, 4, 64]) --> torch.Size([179072, 64])
        #dst = torch.mean(dst,1)
        # print('src',src.shape)
        # print('dst',dst.shape)

        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=src.size(0))
        
        # Step 2: Multiply with weights
        src = self.lin1(src)
        dst = self.lin2(dst)
        edge_attr = self.lin3(edge_attr)

        # Step 3: Calculate the normalization       
        row, col = edge_index
        deg = degree(col, src.size(0), dtype=src.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] #* deg_inv_sqrt[col]
        # print("HERE1")
        # print(edge_attr.shape)
        # Step 4: Propagate the embeddings to the next layer
        # print('edge_index',edge_index.shape)
        # print('src',src.shape)
        # print('dst',dst.shape)
        edge_attr = torch.mean(edge_attr,1)
        return self.propagate(edge_index, size=(src.size(0), dst.size(0)), x=(src,dst), edge_attr=edge_attr, norm=norm)

    def message(self, x_j, edge_attr_j, norm):
        # Normalize node features.  
        # print("HERE")
        # print(edge_attr_j.shape)
        # print(x_j.shape)
        
        return norm.view(-1, 1) * (x_j + edge_attr_j)




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
            torch.nn.LeakyReLU(),
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





u_feat_names = ['user_id','u_yelping_year-u_stars']
i_feat_names = ['item_id','i_city-i_stars-i_is_open']
path = '/Users/kerryngan/neu/7500_dl/final_project/checkpoint1/GCM/dataset/Yelp-NC/train.dat'
user_feats, item_feats, userf_map, itemf_map = load_node_csv(path, InitEncoder())
edge_index, edge_attr, reverse_map = load_edge_csv(path,u_feat_names,userf_map,i_feat_names,itemf_map,InitEncoder())


data = HeteroData()

data['user'].x = item_feats  # Users do not have any features.
data['item'].x = user_feats
data['user','context','item'].edge_index = edge_index
data['user','context','item'].edge_attr = edge_attr
data = gTrans.ToUndirected()(data)
# data['item','context','user'].edge_index = reverse_map
# data['item','context','user'].edge_attr = edge_attr

# model = HeteroGNN(num_layers=2)
# with torch.no_grad():  # Initialize lazy modules.
#      out = model(data.x_dict, data.edge_attr_dict, data.edge_index_dict)




'''
Training
'''
#assert torch.cuda.is_available()
# device = 'cuda'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
import time
epochs = 5 # epoch
lr = 0.0005  # learning rate
batch_size = 64 # batch size for training
  

emsize = 64

num_heads = 4
num_trx_cells = 2

gradient_norm_clip = 1

model = HeteroGNN(num_layers=2).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-8)
# total_accu = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()

    for idx, (label, text) in enumerate(dataloader):
        label = label.to(device)
        text = text.to(device)
        optimizer.zero_grad()
        
        ###########################################################################
        # TODO: compute the logits of the input, get the loss, and do the         #
        # gradient backpropagation.
        ###########################################################################
        logits = model(text)
        sig_logits = torch.nn.Sigmoid(logits)
        loss_func = torch.optim.adamw()
        loss = loss_func(logits, label)
        loss.backward()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()
        total_acc += (logits.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()
    accu_val = evaluate(model, val_loader, loss_func, device)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
