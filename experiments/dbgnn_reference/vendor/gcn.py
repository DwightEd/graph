import torch
import torch_geometric


class GCN(torch.nn.Module):
    def __init__(
        self,
        num_features,
        num_classes,
        hidden_dims=[16, 32, 32],
        p_dropout=0.4
    ):
        super(GCN, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.p_dropout = p_dropout

        self.layer_2_2_0 = torch_geometric.nn.GCNConv(self.num_features, self.hidden_dims[0])
        self.layer_2_2_1 = torch_geometric.nn.GCNConv(self.hidden_dims[0], self.hidden_dims[1])
        self.layer_2_2_2 = torch_geometric.nn.GCNConv(self.hidden_dims[1], self.hidden_dims[2])
        self.mlp = torch.nn.Linear(self.hidden_dims[2], num_classes)

    def forward(self, data):
        import torch.nn.functional as F
        index_list = data.edge_index
        weights = data.edge_weight

        x = data.x
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = torch.nn.functional.elu(self.layer_2_2_0(x, index_list, weights))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = torch.nn.functional.elu(self.layer_2_2_1(x, index_list, weights))
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = torch.nn.functional.elu(self.layer_2_2_2(x, index_list, weights))
        x = F.dropout(x, p=self.p_dropout, training=self.training)

        x = self.mlp(x)
        return x
