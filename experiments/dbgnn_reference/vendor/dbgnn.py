import torch
import torch_geometric


class BipartiteGraphOperator(torch_geometric.nn.MessagePassing):
    def __init__(self, in_ch, out_ch):
        super(BipartiteGraphOperator, self).__init__('add')
        self.lin1 = torch.nn.Linear(in_ch, out_ch)
        self.lin2 = torch.nn.Linear(in_ch, out_ch)

    def forward(self, x, bipartite_index, N, M):
        x = (self.lin1(x[0]), self.lin2(x[1]))
        return self.propagate(bipartite_index, size=(N, M), x=x)

    def message(self, x_i, x_j):
         return x_i + x_j


class HO_GCN(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        num_features=[60, 32],
        hidden_dims=[16, 32, 32],
        p_dropout=0.4
    ):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.p_dropout = p_dropout

        self.layer_2_2_0 = torch_geometric.nn.GCNConv(self.num_features[0], self.hidden_dims[0])
        self.layer_2_2_1 = torch_geometric.nn.GCNConv(self.hidden_dims[0], self.hidden_dims[1])

        self.layer_1_1_0 = torch_geometric.nn.GCNConv(self.num_features[1], self.hidden_dims[0])
        self.layer_1_1_1 = torch_geometric.nn.GCNConv(self.hidden_dims[0], self.hidden_dims[1])

        self.layer_2_1 = BipartiteGraphOperator(self.hidden_dims[1], self.hidden_dims[2])
        self.mlp = torch.nn.Linear(self.hidden_dims[2], num_classes)

    def forward(self, data, device):
        import torch.nn.functional as F
        ho_index_list = data.edge_index
        ho_weights = data.edge_weight
        ho_index_to_fo_index = data.edge_index_hon_to_fon

        x = data.x_ho
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = torch.nn.functional.elu(self.layer_2_2_0(x, ho_index_list, ho_weights))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = torch.nn.functional.elu(self.layer_2_2_1(x, ho_index_list, ho_weights))
        x = F.dropout(x, p=self.p_dropout, training=self.training)

        fo_index_list = data.edge_index_fo
        fo_weights = data.edge_weight_fo

        x1 = data.x_fo
        x1 = F.dropout(x1, p=self.p_dropout, training=self.training)
        x1 = torch.nn.functional.elu(self.layer_1_1_0(x1, fo_index_list, fo_weights))
        x1 = F.dropout(x1, p=self.p_dropout, training=self.training)
        x1 = torch.nn.functional.elu(self.layer_1_1_1(x1, fo_index_list, fo_weights))
        x1 = F.dropout(x1, p=self.p_dropout, training=self.training)

        num_fo_nodes = data.num_nodes
        num_ho_nodes = data.num_ho_nodes
        x = torch.nn.functional.elu(
            self.layer_2_1(
                (x, x1), ho_index_to_fo_index, N=num_ho_nodes, M=num_fo_nodes
            )
        )
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.mlp(x)
        return x
