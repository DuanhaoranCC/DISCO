# -*- coding: utf-8 -*-
# @Time    :
# @Author  :
# @Email   :
# @File    : model.py
# @Software: PyCharm
# @Note    :
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Parameter
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.nn import global_mean_pool, global_add_pool, TemporalEncoding, GINConv, GCNConv, BatchNorm, \
    global_max_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, to_undirected, dropout_edge, mask_feature
from torch_geometric.data import Data
import numpy as np
import copy
from functools import partial


class GCNConvWithEdgeFeatures(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # 使用加法聚合
        self.linear = torch.nn.Linear(in_channels, out_channels)
        self.edge_linear = torch.nn.Linear(100, out_channels)
        self.time_encoder = TemporalEncoding(100)

    def forward(self, x, edge_index, edge_attr):
        # x: [num_nodes, in_channels]
        # edge_index: [2, num_edges]
        # edge_attr: [num_edges, feature_dim]

        # 添加自环
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        loop_attr = self.time_encoder(torch.tensor([1.0], device=x.device))
        loop_attr = loop_attr.repeat(x.size(0), 1)
        edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        x = self.linear(x)  # 节点特征线性变换
        edge_attr = self.edge_linear(edge_attr)  # 边特征线性变换
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # 将邻居的特征与边特征结合
        return x_j + edge_attr


class textprompt(nn.Module):
    def __init__(self, hid_units, type='mul'):
        super(textprompt, self).__init__()
        self.act = nn.ELU()
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units))
        self.prompttype = type
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding):
        # print("weight",self.weight)
        if self.prompttype == 'add':
            weight = self.weight.repeat(graph_embedding.shape[0], 1)
            graph_embedding = weight + graph_embedding
        if self.prompttype == 'mul':
            graph_embedding = self.weight * graph_embedding

        return graph_embedding


class TDrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats, Prompt):
        super(TDrumorGCN, self).__init__()
        self.Prompt = Prompt
        self.conv1 = GCNConv(in_feats, 128)
        self.conv2 = GCNConv(128, out_feats)

    def forward(self, x, edge_index, batch):

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)

        h = self.conv2(h, edge_index)
        # h = F.relu(h)
        # h = F.dropout(h, training=self.training)

        if self.Prompt:
            hs = global_add_pool(h[:-1], batch) + h[-1]
            return hs, h[:-1]
        else:
            hs = global_add_pool(h, batch)
            return hs, h


class BUrumorGCN(torch.nn.Module):
    def __init__(self, in_feats, out_feats, Prompt):
        super(BUrumorGCN, self).__init__()
        self.Prompt = Prompt
        self.conv1 = GCNConv(in_feats, 128)
        self.conv2 = GCNConv(128, out_feats)

    def forward(self, x, edge_index, batch):

        edge_index = torch.flip(edge_index, dims=[0])

        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)

        h = self.conv2(h, edge_index)
        # h = F.relu(h)
        # h = F.dropout(h, training=self.training)

        if self.Prompt:
            hs = global_add_pool(h[:-1], batch) + h[-1]
            return hs, h[:-1]
        else:
            hs = global_add_pool(h, batch)
            return hs, h


class HierarchicalPrompt(nn.Module):
    def __init__(self, num_sub_prompts, in_feats):
        super(HierarchicalPrompt, self).__init__()
        self.sub_prompts = nn.ParameterList([
            nn.Parameter(torch.FloatTensor(1, in_feats)) for _ in range(num_sub_prompts)
        ])
        self.gating = nn.Linear(in_feats, num_sub_prompts)  # 动态选择子 prompt
        self.reset_parameters()
        self.num_sub_prompts = num_sub_prompts

    def reset_parameters(self):
        for sub_prompt in self.sub_prompts:
            torch.nn.init.xavier_uniform_(sub_prompt)

    def orthogonality_loss(self):
        # 将所有子Prompt堆叠成矩阵 Q
        Q = torch.stack([p for p in self.sub_prompts], dim=0).squeeze(1)  # (num_sub_prompts, in_feats)

        # 计算 Q Q^T
        Q_tQ = torch.matmul(Q, Q.T)  # (in_feats, in_feats)

        # 创建单位矩阵 I
        I = torch.eye(Q.size(0), device=Q.device)  # (in_feats, in_feats)

        # 计算 Q^T Q - I
        orthogonal_loss = torch.norm(Q_tQ - I, p='fro')  # Frobenius norm

        return orthogonal_loss

    def forward(self, graph_features):
        # 计算每个子 prompt 的权重
        sub_prompt_weights = torch.softmax(self.gating(graph_features), dim=-1)
        print(sub_prompt_weights[0])
        combined_prompt = torch.mm(
            sub_prompt_weights, torch.stack([p for p in self.sub_prompts], dim=0).squeeze(1))  # (batch_size, in_feats)
        loss = self.orthogonality_loss()
        return combined_prompt, loss


class BiGCN_graphcl(torch.nn.Module):
    def __init__(self, in_feats, out_feats, t):
        super(BiGCN_graphcl, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, out_feats, True)
        self.BUrumorGCN = BUrumorGCN(in_feats, out_feats, True)
        self.proj_head = nn.Sequential(nn.Linear(out_feats * 2, 256), nn.ReLU(inplace=True),
                                       nn.Linear(256, 128))

        self.time_encoder = TemporalEncoding(100)
        self.t = t
        self.embedding = nn.Embedding(1, in_feats)
        # self.prompts = nn.ParameterDict({
        #     "root_prompt": nn.Parameter(torch.FloatTensor(1, in_feats)),
        #     "ch_prompt": nn.Parameter(torch.FloatTensor(1, in_feats)),
        #     "en_prompt": nn.Parameter(torch.FloatTensor(1, in_feats))
        # })

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight.data)
        # for param in self.prompts.values():
        #     torch.nn.init.xavier_uniform_(param)

    def add_virtual_node(self, x, data):
        """
        Adds a virtual node to the graph and connects it to all root nodes.

        Args:
            data: Graph data (torch_geometric.data.Data)

        Returns:
            Updated graph data (torch_geometric.data.Data)
        """
        device = data.x.device
        root_nodes = data.x.size(0)

        # Identify root nodes (assumed to be node 0 for each graph in the batch)
        root_indices = []
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            root_indices.append(torch.nonzero(data.batch == num_batch, as_tuple=False)[0].item())

        # Add the virtual node feature
        # virtual_node_feature = self.prompts["root_prompt"]  # Shape: (1, feature_dim)
        virtual_node_feature = self.embedding(torch.tensor([0]).to(device))
        new_x = torch.cat([x, virtual_node_feature], dim=0)

        # Create edges connecting the virtual node to all root nodes
        root_edges = torch.stack([
            torch.full((batch_size,), root_nodes, dtype=torch.long, device=device),  # Virtual node index
            torch.tensor(root_indices, dtype=torch.long, device=device)  # Root node indices
        ], dim=0)

        # Combine the original and new edges
        new_edge_index = torch.cat([data.edge_index, root_edges], dim=1)

        # Add default edge attributes if edge_attr exists
        if data.edge_attr is not None:
            virtual_edge_attr = torch.ones(root_edges.size(1), dtype=data.edge_attr.dtype, device=device)
            new_edge_attr = torch.cat([data.edge_attr, virtual_edge_attr], dim=0)
        else:
            new_edge_attr = None

        # Create a new Data object
        new_data = Data(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            y=data.y,  # Copy labels if they exist
            batch=data.batch  # Copy batch information
        )

        # Add root indices to the new Data object
        new_data.root = torch.tensor(root_indices, dtype=torch.long, device=device)

        return new_data

    def process_data(self, data):
        """
        Process a single dataset with the corresponding prompt.
        :param data: Input graph data.
        :return: Combined TD and BU features.
        """
        data = self.add_virtual_node(data.x, data)
        x = data.x
        # x = self.prompts[prompt_key] * data.x
        # x = self.virtualnode_embedding(torch.tensor([prompt_key]).to(data.x.device)) * data.x
        TD_x, _ = self.TDrumorGCN(x, data.edge_index, data.batch)
        BU_x, _ = self.BUrumorGCN(x, data.edge_index, data.batch)

        return torch.cat((BU_x, TD_x), 1)

    def forward(self, *data_list):
        """
        Forward pass for multiple datasets.
        :param data_list: List of input graph data objects.
        :return: Projected feature representation.
        """

        hs = []
        for data in data_list:
            h = self.process_data(data)
            hs.append(h)

        h = torch.cat(hs, dim=0)
        # h = self.process_data(data_list)

        h = self.proj_head(h)
        return h

    def loss_graphcl(self, x1, x2, mean=True):

        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / self.t)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss

    def get_embeds(self, data):

        data = self.add_virtual_node(data.x, data)
        # x = self.prompts[prompt_key] * data.x
        # x = self.prompts["en_prompt"] * data.x
        x = data.x
        # edge_attr = torch.log1p(torch.abs(data.edge_attr))
        # # 计算 Bottom-Up 边缘特征(时间差的倒数)
        # epsilon = 1e-6  # 避免除零
        # edge_attr_bottomup = 1 / (data.edge_attr + epsilon)
        # edge_attr_bottomup = self.time_encoder(edge_attr_bottomup)

        TD_x1, _ = self.TDrumorGCN(x, data.edge_index, data.batch)
        BU_x1, _ = self.BUrumorGCN(x, data.edge_index, data.batch)
        h = torch.cat((BU_x1, TD_x1), 1)

        return h


class GNN_graphpred(torch.nn.Module):
    def __init__(self, out_feats):
        super(GNN_graphpred, self).__init__()

        # Initialize GNN module
        self.gnn = BiGCN_graphcl(768, out_feats, 0.5)

        # Initialize prompt parameter
        self.prompt = nn.Parameter(torch.FloatTensor(1, 768))
        self.reset_parameters()

        # Freeze GNN parameters
        self.freeze_gnn_parameters()

    def reset_parameters(self):
        # Xavier initialization for prompt parameter
        torch.nn.init.xavier_uniform_(self.prompt)

    def freeze_gnn_parameters(self):
        # Freeze all parameters in GNN module (no gradients)
        for param in self.gnn.parameters():
            param.requires_grad = False

    def from_pretrained(self, model_file):
        # Load pretrained GNN weights
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, data):
        data = self.gnn.add_virtual_node(data.x, data)
        # Forward pass: multiply prompt with input
        x = self.prompt * data.x

        # Apply GNN modules for different graph parts
        TD_x1, _ = self.gnn.TDrumorGCN(x, data.edge_index, data.batch)
        BU_x1, _ = self.gnn.BUrumorGCN(x, data.edge_index, data.batch)

        # Concatenate outputs from both parts
        h = torch.cat((BU_x1, TD_x1), 1)

        return h


class BiGCN_individual(torch.nn.Module):
    def __init__(self, in_feats, out_feats, num_class):
        super(BiGCN_individual, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, out_feats, False)
        self.BUrumorGCN = BUrumorGCN(in_feats, out_feats, False)
        self.proj_head = nn.Sequential(nn.Linear(out_feats * 2, 256), nn.ReLU(inplace=True),
                                       nn.Linear(256, 128))

        self.time_encoder = TemporalEncoding(100)
        self.fc = nn.Linear(out_feats * 2, num_class)

    def forward(self, data):
        # edge_attr = np.log(1 + np.abs(data.edge_attr))
        # edge_attr = self.time_encoder(edge_attr)

        x = data.x

        TD_x1, _ = self.TDrumorGCN(x, data.edge_index, data.batch)
        BU_x1, _ = self.BUrumorGCN(x, data.edge_index, data.batch)
        h = torch.cat((BU_x1, TD_x1), 1)

        h = self.proj_head(h)
        return h

    def loss_graphcl(self, x1, x2, mean=True):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss

    def bigcn(self, data):
        TD_x, _ = self.TDrumorGCN(data.x, data.edge_index, data.batch)
        BU_x, _ = self.BUrumorGCN(data.x, data.edge_index, data.batch)
        x = torch.cat((BU_x, TD_x), 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

    def get_embeds(self, data):
        TD_x1, _ = self.TDrumorGCN(data.x, data.edge_index, data.batch)
        BU_x1, _ = self.BUrumorGCN(data.x, data.edge_index, data.batch)
        h = torch.cat((BU_x1, TD_x1), 1)

        return h


class UPFD_Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_class, concat=True):
        super().__init__()
        self.concat = concat

        self.conv1 = GCNConv(in_channels, out_channels)
        # self.conv1 = SAGEConv(in_channels, hidden_channels)
        # self.conv1 = GATConv(in_channels, hidden_channels)

        if self.concat:
            self.lin0 = Linear(in_channels, out_channels)
            self.lin1 = Linear(2 * out_channels, out_channels)

        self.lin2 = Linear(out_channels, num_class)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index = to_undirected(edge_index)
        h = self.conv1(x, edge_index).relu()
        h = global_max_pool(h, batch)

        if self.concat:
            # Get the root node (tweet) features of each graph:
            root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
            root = torch.cat([root.new_zeros(1), root + 1], dim=0)
            news = x[root]

            news = self.lin0(news).relu()
            h = self.lin1(torch.cat([news, h], dim=-1)).relu()

        h = self.lin2(h)
        return h.log_softmax(dim=-1)


def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss = - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def mask(x, mask_rate=0.5):
    """
    Function to mask a subset of nodes in a graph.

    Args:
        x (torch.Tensor): Node feature matrix.
        mask_rate (float): The rate of nodes to be masked.

    Returns:
        torch.Tensor: Indices of the masked nodes.
    """
    num_nodes = x.size(0)  # Number of nodes in the graph
    perm = torch.randperm(num_nodes, device=x.device)  # Random permutation of node indices
    num_mask_nodes = int(mask_rate * num_nodes)  # Number of nodes to mask
    mask_nodes = perm[:num_mask_nodes]  # Select the indices of masked nodes

    return mask_nodes


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, num_layers=2):
        super(Encoder, self).__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.act = nn.ModuleList()

        # Initialize GIN layers with MLPs
        for layer in range(num_layers):
            if layer == 0:
                mlp = MLP(in_dim, hidden, out_dim)
            else:
                mlp = MLP(out_dim, hidden, out_dim)

            self.ginlayers.append(GINConv(mlp))
            self.batch_norms.append(BatchNorm(out_dim))
            self.act.append(nn.PReLU())

    def forward(self, x, edge_index, batch):
        output = []

        for i, layer in enumerate(self.ginlayers):
            x = layer(x, edge_index)  # Message passing
            x = self.batch_norms[i](x)  # Batch normalization
            x = F.relu(x)  # Activation function
            pooled = global_add_pool(x, batch)  # Global pooling (sum pooling)
            output.append(pooled)

        return x, torch.cat(output, dim=1)


class GAMC(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden=512):
        super().__init__()

        self.encoder = Encoder(in_dim, out_dim, hidden, 2)
        self.decoder = Encoder(out_dim, in_dim, hidden, 1)
        self.criterion = self.setup_loss_fn("sce")

    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=1)
        else:
            raise NotImplementedError
        return criterion

    def forward(self, *data_list):
        loss = 0
        for data in data_list:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            edge_index = to_undirected(edge_index)
            edge_index, _ = dropout_edge(edge_index, p=0.2)
            mask_nodes = mask(x, mask_rate=0.5)

            x1 = x.clone()
            x1[mask_nodes] = 0.0
            h, gh = self.encoder(x1, edge_index, batch)

            re_h = h.clone()
            re_h[mask_nodes] = 0.0
            re_x1, _ = self.decoder(re_h, edge_index, batch)
            loss1 = self.criterion(re_x1[mask_nodes], x[mask_nodes].detach())

            ################################################################################
            x, edge_index, batch = data.x, data.edge_index, data.batch
            edge_index = to_undirected(edge_index)
            edge_index, _ = dropout_edge(edge_index, p=0.2)
            mask_nodes = mask(x, mask_rate=0.5)

            x1 = x.clone()
            x1[mask_nodes] = 0.0
            h, gh = self.encoder(x1, edge_index, batch)

            re_h = h.clone()
            re_h[mask_nodes] = 0.0
            re_x2, _ = self.decoder(re_h, edge_index, batch)
            loss2 = self.criterion(re_x2[mask_nodes], x[mask_nodes].detach())
            ############################################################################
            # Contrastive
            cl_loss = self.criterion(re_x2, re_x1)
            loss += loss1 + loss2 + cl_loss * 0.1

        # return loss1 + loss2 + cl_loss*0.1
        return loss

    def get_embeds(self, data):
        h, gh = self.encoder(data.x, data.edge_index, data.batch)

        return gh


class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))


class Encoder1(torch.nn.Module):
    def __init__(self, in_feats, out_feats):
        super(Encoder1, self).__init__()
        self.TDrumorGCN = TDrumorGCN(in_feats, out_feats, True)
        self.BUrumorGCN = BUrumorGCN(in_feats, out_feats, True)

    def forward(self, data, x):

        TD_x1, h1 = self.TDrumorGCN(x, data.edge_index, data.batch)
        BU_x1, h2 = self.BUrumorGCN(x, data.edge_index, data.batch)
        hs = torch.cat((BU_x1, TD_x1), 1)
        h = torch.cat((h1, h2), 1)

        return h, hs

    def loss_graphcl(self, x1, x2, mean=True):
        T = 0.5
        batch_size, _ = x1.size()

        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)
        if mean:
            loss = loss.mean()
        return loss

def mask1(x, batch, mask_rate=0.5):
    """
    Mask a subset of nodes in a batched graph, ensuring the 0th node in each graph is not masked.

    Args:
        x (torch.Tensor): Node feature matrix.
        batch (torch.Tensor): Batch vector indicating the graph each node belongs to.
        mask_rate (float): The rate of nodes to be masked.

    Returns:
        torch.Tensor: Indices of the masked nodes.
    """
    mask_nodes = []  # Store indices of masked nodes
    num_graphs = batch.max().item() + 1  # Number of graphs in the batch

    for graph_idx in range(num_graphs):
        # Get the nodes belonging to the current graph
        graph_nodes = (batch == graph_idx).nonzero(as_tuple=True)[0]
        # Exclude the 0th node of the graph
        non_zero_nodes = graph_nodes[1:] if len(graph_nodes) > 1 else graph_nodes
        # Compute the number of nodes to mask
        num_mask_nodes = int(mask_rate * len(non_zero_nodes))
        # Randomly select nodes to mask
        perm = torch.randperm(len(non_zero_nodes), device=x.device)
        masked_nodes = non_zero_nodes[perm[:num_mask_nodes]]
        mask_nodes.append(masked_nodes)

    # Concatenate masked node indices from all graphs
    return torch.cat(mask_nodes)

class CFOP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rate, alpha, hidden=64):
        super().__init__()

        # self.online_encoder = Encoder(in_dim, out_dim, hidden, 2)
        # self.target_encoder = Encoder(in_dim, out_dim, hidden, 2)
        self.online_encoder = Encoder1(in_dim, out_dim)
        self.target_encoder = Encoder1(in_dim, out_dim)
        self.criterion = self.setup_loss_fn("sce")
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))

        self.rate = rate
        self.alpha = alpha

        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.embedding = nn.Embedding(1, in_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight.data)

    def add_virtual_node(self, x, data):
        """
        Adds a virtual node to the graph and connects it to all root nodes.

        Args:
            data: Graph data (torch_geometric.data.Data)

        Returns:
            Updated graph data (torch_geometric.data.Data)
        """
        device = data.x.device
        root_nodes = data.x.size(0)

        # Identify root nodes (assumed to be node 0 for each graph in the batch)
        root_indices = []
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            root_indices.append(torch.nonzero(data.batch == num_batch, as_tuple=False)[0].item())

        # Add the virtual node feature
        # virtual_node_feature = self.prompts["root_prompt"]  # Shape: (1, feature_dim)
        virtual_node_feature = self.embedding(torch.tensor([0]).to(device))
        new_x = torch.cat([x, virtual_node_feature], dim=0)

        # Create edges connecting the virtual node to all root nodes
        root_edges = torch.stack([
            torch.full((batch_size,), root_nodes, dtype=torch.long, device=device),  # Virtual node index
            torch.tensor(root_indices, dtype=torch.long, device=device)  # Root node indices
        ], dim=0)

        # Combine the original and new edges
        new_edge_index = torch.cat([data.edge_index, root_edges], dim=1)

        # Add default edge attributes if edge_attr exists
        if data.edge_attr is not None:
            virtual_edge_attr = torch.ones(root_edges.size(1), dtype=data.edge_attr.dtype, device=device)
            new_edge_attr = torch.cat([data.edge_attr, virtual_edge_attr], dim=0)
        else:
            new_edge_attr = None

        # Create a new Data object
        new_data = Data(
            x=new_x,
            edge_index=new_edge_index,
            edge_attr=new_edge_attr,
            y=data.y,  # Copy labels if they exist
            batch=data.batch  # Copy batch information
        )

        # Add root indices to the new Data object
        new_data.root = torch.tensor(root_indices, dtype=torch.long, device=device)

        return new_data

    def setup_loss_fn(self, loss_fn):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=1)
        else:
            raise NotImplementedError
        return criterion

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.
        Args:
            mm (float): Momentum used in moving average update.
        """
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)

    def forward(self, *data_list):
        loss = 0
        for data in data_list:
            data = self.add_virtual_node(data.x, data)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            mask_nodes = mask1(x, batch, mask_rate=self.rate)
            x1 = x.clone()
            x1[mask_nodes] = 0.0
            x1[mask_nodes] += self.enc_mask_token

            h1, gh1 = self.online_encoder(data, x1)
            with torch.no_grad():
                h2, gh2 = self.target_encoder(data, x)

            loss += self.criterion(h1[mask_nodes], h2[mask_nodes].detach()) + \
                   self.criterion(gh1, gh2.detach())
        ##################################################################################
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        # # edge_index = to_undirected(edge_index)
        # mask_nodes = mask1(x, batch, mask_rate=self.rate)
        # x1 = x.clone()
        # x1[mask_nodes] = 0.0
        # x1[mask_nodes] += self.enc_mask_token
        #
        # h1, gh1 = self.online_encoder(data, x1)
        # with torch.no_grad():
        #     h2, gh2 = self.target_encoder(data, x)
        #
        # loss = self.criterion(h1[mask_nodes], h2[mask_nodes].detach()) + \
        #        self.criterion(gh1, gh2.detach())

        return loss

    def get_embeds(self, data):
        data = self.add_virtual_node(data.x, data)
        h, gh = self.online_encoder(data, data.x)

        return gh
