import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GINConv, GATConv, GCNConv, NNConv, MFConv, GINEConv
from torch_geometric.utils import get_laplacian, add_self_loops
from torch_geometric.nn.conv import MessagePassing
from torch import ones_like, sparse_coo_tensor, svd_lowrank
from params import N_CHEM_NODE_FEAT, N_PROT_NODE_FEAT, N_PROT_EDGE_FEAT, N_CHEM_ECFP
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


class FCLayers(torch.nn.Module):
    def __init__(self, trial, prefix, in_features, layers_range=(2, 3), n_units_list=(128, 256, 512, 1024, 2048, 4096),
                 dropout_range=(0.1, 0.7), **kwargs):
        super(FCLayers, self).__init__()
        self.trial = trial
        self.prefix = prefix
        self.in_features = in_features
        self.layers = None
        self.n_out = None

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list, dropout_range=dropout_range)

    def get_layers(self, layers_range, n_units_list, dropout_range):

        in_features = self.in_features
        fc_layers = []
        n_fc_layers = self.trial.suggest_int(self.prefix + "_n_fc_layers", layers_range[0], layers_range[1])
        activation = nn.ReLU()
        use_batch_norm = self.trial.suggest_categorical(self.prefix + "_fc_use_bn", (True, False))
        out_features = None
        for i in range(n_fc_layers):
            out_features = self.trial.suggest_categorical(self.prefix + f"_fc_n_out_{i}", n_units_list)
            dropout = self.trial.suggest_float(self.prefix + f"_fc_dropout_{i}", dropout_range[0], dropout_range[1])

            fc_layers.append(nn.Linear(in_features, out_features))
            if use_batch_norm:
                fc_layers.append(nn.BatchNorm1d(out_features))
            fc_layers.append(activation)
            fc_layers.append(nn.Dropout(dropout))

            in_features = out_features

        self.layers = nn.Sequential(*fc_layers)
        self.n_out = out_features

    def forward(self, x):
        return self.layers(x)


class GraphPool:
    def __init__(self, trial, prefix,
                 pool_types=("mean", "add", "max", "mean_add", "mean_max", "add_max", "mean_add_max")):
        self.coef_dict = {"mean": 1, "add": 1, "max": 1, "mean_add": 2, "mean_max": 2, "add_max": 2, "mean_add_max": 3}
        self.type_ = trial.suggest_categorical(prefix + "_graph_pool_type", pool_types)
        self.coef = self.coef_dict[self.type_]

    def __call__(self, _graph_out, _graph_batch):
        out = None
        if self.type_ == "mean":
            out = global_mean_pool(_graph_out, _graph_batch)
        elif self.type_ == "add":
            out = global_add_pool(_graph_out, _graph_batch)
        elif self.type_ == "max":
            out = global_max_pool(_graph_out, _graph_batch)
        elif self.type_ == "mean_add":
            out = torch.cat([global_mean_pool(_graph_out, _graph_batch), global_add_pool(_graph_out, _graph_batch)],
                            dim=1)
        elif self.type_ == "mean_max":
            out = torch.cat([global_mean_pool(_graph_out, _graph_batch), global_max_pool(_graph_out, _graph_batch)],
                            dim=1)
        elif self.type_ == "add_max":
            out = torch.cat([global_add_pool(_graph_out, _graph_batch), global_max_pool(_graph_out, _graph_batch)],
                            dim=1)
        elif self.type_ == "mean_add_max":
            out = torch.cat([global_mean_pool(_graph_out, _graph_batch), global_add_pool(_graph_out, _graph_batch),
                             global_max_pool(_graph_out, _graph_batch)], dim=1)
        return out


class GNNLayers(torch.nn.Module):
    def __init__(self, trial, prefix, _node_features_len=None, _edge_features_len=None, use_edges_features=False):
        super(GNNLayers, self).__init__()
        self.trial = trial
        self.prefix = prefix
        self._node_features_len = _node_features_len
        self._edge_features_len = _edge_features_len
        self.use_edges_features = use_edges_features

        self.activation = None
        self.layers_list = None
        self.bn_list = None
        self.n_out = None

    def forward(self, data, **kwargs):
        _graph_out = data.x
        _edges_index = data.edge_index
        _edges_features = data.edge_attr if self.use_edges_features else None

        for _nn, _bn in zip(self.layers_list, self.bn_list):
            _graph_out = _nn(_graph_out, _edges_index, edge_attr=_edges_features) if self.use_edges_features else _nn(
                _graph_out, _edges_index)
            if _bn is not None:
                _graph_out = _bn(_graph_out)
            if self.activation is not None:
                _graph_out = self.activation(_graph_out)

        return _graph_out


class GATLayers(GNNLayers):
    def __init__(self, trial, prefix, gnn, _node_features_len, _edge_features_len, use_edges_features, layers_range,
                 heads_range, dropout_range,  **kwargs):
        super(GATLayers, self).__init__(trial, prefix, _node_features_len=_node_features_len,
                                        _edge_features_len=_edge_features_len, use_edges_features=use_edges_features)
        self.gnn = gnn
        self.get_layers(layers_range=layers_range, heads_range=heads_range, dropout_range=dropout_range)


    def get_layers(self, layers_range, heads_range, dropout_range):
        _node_features_len = self._node_features_len
        _edge_features_len = self._edge_features_len

        self.use_edges_features = self.trial.suggest_categorical(self.prefix + "_use_edges_features",
                                                                 (True, False)) if self.use_edges_features else False
        if self.use_edges_features:
            _edge_features_fill = self.trial.suggest_categorical(self.prefix + "_edge_features_fill",
                                                                 ("mean", "add", "max", "mul", "min"))
        else:
            _edge_features_len = None
            _edge_features_fill = "mean"

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []
        _all_heads = 1

        for i in range(_n_layers):
            _heads = self.trial.suggest_int(self.prefix + f"_heads_{i}", heads_range[0], heads_range[1])
            _dropout = self.trial.suggest_float(self.prefix + f"_dropout_{i}", dropout_range[0], dropout_range[1])

            if self.use_edges_features:
                _gnn = self.gnn(_node_features_len * _all_heads, _node_features_len * _all_heads, heads=_heads,
                                dropout=_dropout, edge_dim=_edge_features_len, fill_value=_edge_features_fill)
            else:
                _gnn = self.gnn(_node_features_len * _all_heads, _node_features_len * _all_heads, heads=_heads,
                                dropout=_dropout)
            _layers_list.append(_gnn)

            _all_heads = _all_heads * _heads

            if use_bn:
                bn = nn.BatchNorm1d(_node_features_len * _all_heads)
                _bn_layers_list.append(bn)

        self.n_out = _node_features_len * _all_heads
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


class GATv1Layers(GATLayers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 3), heads_range=(1, 5),
                 dropout_range=(0.0, 0.3), **kwargs):
        super(GATv1Layers, self).__init__(trial, prefix=prefix + "_gatv1", gnn=GATConv,
                                          _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                          use_edges_features=True,
                                          layers_range=layers_range, heads_range=heads_range,
                                          dropout_range=dropout_range, **kwargs)


class GCNLayers(GNNLayers):
    def __init__(self, trial, prefix, _node_features_len, layers_range=(1, 4), n_units_list=(64, 128, 256, 512, 1024,),
                 **kwargs):
        super(GCNLayers, self).__init__(trial, prefix + "_gcn", _node_features_len=_node_features_len)

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list)

    def get_layers(self, layers_range, n_units_list):
        _n_out = None
        _n_in = self._node_features_len

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []

        for i in range(_n_layers):
            _n_out = self.trial.suggest_categorical(self.prefix + f"_n_out_{i}", n_units_list)

            _gnn = GCNConv(_n_in, _n_out)
            _layers_list.append(_gnn)

            if use_bn:
                bn = nn.BatchNorm1d(_n_out)
                _bn_layers_list.append(bn)

            _n_in = _n_out

        self.n_out = _n_out
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


class GIN_Layers(GNNLayers):
    def __init__(self, trial, prefix, gnn, _node_features_len, _edge_features_len, use_edges_features, layers_range,
                 n_units_list, **kwargs):
        super(GIN_Layers, self).__init__(trial, prefix, _node_features_len=_node_features_len,
                                         _edge_features_len=_edge_features_len, use_edges_features=use_edges_features)
        self.gnn = gnn

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list)

    def get_layers(self, layers_range, n_units_list):
        _n_out = None
        _n_in = self._node_features_len

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []

        for i in range(_n_layers):
            _n_out = self.trial.suggest_categorical(self.prefix + f"_n_out_{i}", n_units_list)

            _nn = nn.Sequential(nn.Linear(_n_in, _n_out), nn.ReLU(), nn.Linear(_n_out, _n_out))
            _gnn = self.gnn(_nn, edge_dim=self._edge_features_len) if self.use_edges_features else self.gnn(_nn)
            _layers_list.append(_gnn)
            if use_bn:
                bn = nn.BatchNorm1d(_n_out)
                _bn_layers_list.append(bn)

            _n_in = _n_out

        self.n_out = _n_out
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


class GINLayers(GIN_Layers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 4),
                 n_units_list=(64, 128, 256, 512, 1024,), **kwargs):
        super(GINLayers, self).__init__(trial, prefix=prefix + "_gin", gnn=GINConv,
                                        _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                        use_edges_features=False,
                                        layers_range=layers_range, n_units_list=n_units_list, **kwargs)


class GINELayers(GIN_Layers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 4),
                 n_units_list=(64, 128, 256, 512, 1024,), **kwargs):
        super(GINELayers, self).__init__(trial, prefix=prefix + "_gine", gnn=GINEConv,
                                         _node_features_len=_node_features_len, _edge_features_len=_edge_features_len,
                                         use_edges_features=True,
                                         layers_range=layers_range, n_units_list=n_units_list, **kwargs)


class QGNNLayers(GNNLayers):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len, layers_range=(1, 3),
                 n_units_list=(16, 32, 64), **kwargs):
        super(QGNNLayers, self).__init__(trial, prefix + "_qgnn", _node_features_len=_node_features_len,
                                        _edge_features_len=_edge_features_len, use_edges_features=True)

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list)

    def get_layers(self, layers_range, n_units_list):
        _n_out = None
        _n_in = self._node_features_len

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []

        for i in range(_n_layers):
            _n_out = self.trial.suggest_categorical(self.prefix + f"_n_out_{i}", n_units_list)
            hidden_size = round((_n_in * _n_out) / 2)
            nn_ = nn.Sequential(nn.Linear(self._edge_features_len, hidden_size), nn.ReLU(),
                                nn.Linear(hidden_size, _n_in * _n_out))
            _gnn = NNConv(_n_in, _n_out, nn_)
            _layers_list.append(_gnn)

            if use_bn:
                bn = torch.nn.BatchNorm1d(_n_out)
                _bn_layers_list.append(bn)

            _n_in = _n_out

        self.n_out = _n_out
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


class GMFLayers(GNNLayers):
    def __init__(self, trial, prefix, _node_features_len, layers_range=(1, 4), n_units_list=(64, 128, 256, 512, 1024,),
                 **kwargs):
        super(GMFLayers, self).__init__(trial, prefix + "_gmf", _node_features_len=_node_features_len)

        self.get_layers(layers_range=layers_range, n_units_list=n_units_list)

    def get_layers(self, layers_range, n_units_list):
        _n_out = None
        _n_in = self._node_features_len

        use_activation = self.trial.suggest_categorical(self.prefix + "_use_activation", (True, False))
        if use_activation:
            activation_name = self.trial.suggest_categorical(self.prefix + "_activation",
                                                             ("ReLU", "LeakyReLU", "Sigmoid"))
            self.activation = getattr(torch.nn, activation_name)()

        use_bn = self.trial.suggest_categorical(self.prefix + "_use_bn", (True, False))
        _n_layers = self.trial.suggest_int(self.prefix + "_n_layers", layers_range[0], layers_range[1])
        _layers_list = []
        _bn_layers_list = []

        for i in range(_n_layers):
            _n_out = self.trial.suggest_categorical(self.prefix + f"_n_out_{i}", n_units_list)

            _gnn = MFConv(_n_in, _n_out)
            _layers_list.append(_gnn)

            if use_bn:
                bn = nn.BatchNorm1d(_n_out)
                _bn_layers_list.append(bn)

            _n_in = _n_out

        self.n_out = _n_out
        self.layers_list = nn.ModuleList(_layers_list)
        self.bn_list = nn.ModuleList(_bn_layers_list) if use_bn else [None] * _n_layers


class MP(MessagePassing):
    def __init__(self):
        super(MP, self).__init__()

    def forward(self, x, edge_index, norm=None):
        return self.propagate(edge_index=edge_index, x=x, norm=None)

    def message(self, x_j, norm=None):
        if norm != None:
            return norm.view(-1, 1) * x_j
        else:
            return x_j


class Basis_Generator(nn.Module):
    def __init__(self, nx, nlx, nl, k, operator, low_x=False, low_lx=False, norm1=False):
        super(Basis_Generator, self).__init__()

        self.nx = nx
        self.nlx = nlx
        self.nl = nl
        self.norm1 = norm1
        self.k = k
        self.operator = operator
        self.low_x = low_x
        self.low_lx = low_lx
        self.mp = MP()

    def get_x_basis(self, x):
        x = F.normalize(x, dim=1)
        x = F.normalize(x, dim=0)

        if self.low_x:
            U, S, V = svd_lowrank(x, q=self.nx)
            low_x = U @ torch.diag(S)
            return low_x
        else:
            return x

    def get_lx_basis(self, x, edge_index):
        lxs = []
        num_nodes = x.shape[0]
        edge_index_lap, edge_weight_lap = get_laplacian(edge_index=edge_index, normalization='sym', num_nodes=num_nodes)
        h = F.normalize(x, dim=1)

        if self.operator == 'gcn':
            lxs = [h]
            edge_index, edge_weight = add_self_loops(edge_index=edge_index_lap,
                                                     edge_attr=-edge_weight_lap,
                                                     fill_value=2.0,
                                                     num_nodes=num_nodes)
            edge_index, edge_weight = get_laplacian(edge_index=edge_index,
                                                    edge_weight=edge_weight,
                                                    normalization='sym',
                                                    num_nodes=num_nodes)
            edge_index, edge_weight = add_self_loops(edge_index=edge_index,
                                                    edge_attr=-edge_weight,
                                                    fill_value=1.,
                                                    num_nodes=num_nodes)
            for k in range(self.k + 1):
                h = self.mp.propagate(edge_index=edge_index, x=h, norm=edge_weight)
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.operator == 'gpr':
            lxs = [h]
            edge_index, edge_weight = add_self_loops(edge_index=edge_index_lap,
                                                     edge_attr=-edge_weight_lap,
                                                     fill_value=1.0,
                                                     num_nodes=num_nodes)
            for k in range(self.k):
                h = self.mp.propagate(edge_index=edge_index, x=h, norm=edge_weight)
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.operator == 'cheb':
            edge_index, edge_weight = add_self_loops(edge_index=edge_index_lap,
                                                     edge_attr=edge_weight_lap,
                                                     fill_value=-1.0,
                                                     num_nodes=num_nodes)
            for k in range(self.k + 1):
                if k == 0:
                    pass
                elif k == 1:
                    h = self.mp.propagate(edge_index=edge_index, x=h, norm=edge_weight)
                else:
                    h = self.mp.propagate(edge_index=edge_index, x=h, norm=edge_weight) * 2
                    h = h - lxs[-1]
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        elif self.operator == 'ours':
            lxs = [h]
            edge_index, edge_weight = add_self_loops(edge_index=edge_index_lap,
                                                     edge_attr=edge_weight_lap,
                                                     fill_value=-1.0,
                                                     num_nodes=num_nodes)
            for k in range(self.k):
                h = self.mp.propagate(edge_index=edge_index, x=h, norm=edge_weight)
                h = h - lxs[-1]
                if self.norm1:
                    h = F.normalize(h, dim=1)
                lxs.append(h)

        norm_lxs = []
        low_lxs = []
        for lx in lxs:
            if self.low_lx:
                U, S, V = svd_lowrank(lx, q=self.nlx)
                low_lx = U @ torch.diag(S)
                low_lxs.append(low_lx)
                norm_lxs.append(F.normalize(low_lx, dim=1))
            else:
                norm_lxs.append(F.normalize(lx, dim=1))

        final_lxs = [F.normalize(lx, dim=0) for lx in lxs]
        return final_lxs

    def get_l_basis(self, edge_index, num_nodes):
        edge_index, edge_weight = get_laplacian(edge_index=edge_index, normalization='sym', num_nodes=num_nodes)

        adj = sparse_coo_tensor(indices=edge_index,
                                values=ones_like(edge_index[0]),
                                size=(num_nodes, num_nodes),
                                device=edge_index.device,
                                dtype=torch.float32).to_dense()

        adj = F.normalize(adj, dim=1)

        U, S, V = svd_lowrank(adj, q=self.nl, niter=2)
        adj = U @ torch.diag(S)

        adj = F.normalize(adj, dim=0)

        return adj


class FE_GNN(nn.Module):
    def __init__(self, ninput = N_CHEM_NODE_FEAT):
        super(FE_GNN, self).__init__()

        self.nx = ninput
        self.nlx = ninput
        self.nl = 50
        self.k = 2
        self.operator = 'gcn'
        self.nhid = 64
        self.basis_generator = Basis_Generator(nx=self.nx, nlx=self.nlx, nl=self.nl, k=self.k, operator=self.operator,
                                               low_x=False, low_lx=False, norm1=False)

        self.share_lx = 'True'
        self.thetas = nn.Parameter(torch.ones(self.k + 1), requires_grad=True)

        self.lin_lxs = nn.ModuleList()
        for i in range(self.k + 1):
            self.lin_lxs.append(nn.Linear(self.nlx, self.nhid, bias=True))

        self.lin_x = nn.Linear(self.nx, self.nhid, bias=True)
        self.lin_lx = nn.Linear(self.nlx, self.nhid, bias=True)
        self.lin_l = nn.Linear(self.nl, self.nhid, bias=True)

        self.regression_output = nn.Linear(self.nhid, 1024, bias=True)
        self.relu = nn.ReLU()
        self.n_out = 1024

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x_basis = self.basis_generator.get_x_basis(x)
        lx_basis = self.basis_generator.get_lx_basis(x, edge_index)
        l_basis = self.basis_generator.get_l_basis(edge_index, x.shape[0])

        if self.nx > 0:
            x_mat = self.lin_x(x_basis)
            feature_mat = x_mat.clone()
            feature_mat += x_mat
        else:
            feature_mat = 0

        if self.nlx > 0:
            lxs_mat = 0
            for k in range(self.k + 1):
                if self.share_lx:
                    lx_mat = self.lin_lx(lx_basis[k]) * self.thetas[k]
                else:
                    lx_mat = self.lin_lxs[k](lx_basis[k])
                lxs_mat = lxs_mat + lx_mat
            feature_mat += lxs_mat

        if self.nl > 0:
            l_mat = self.lin_l(l_basis)
            feature_mat += l_mat
        feature_mat = self.relu(feature_mat)

        feature_mat = pyg_nn.global_mean_pool(feature_mat, batch)
        output = self.regression_output(feature_mat)

        return output


class InteractiveMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, output_dim=2048, l2_reg=0.01):
        super(InteractiveMultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.l2_reg = l2_reg
        self.output_dim = output_dim

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        self.fc = nn.Linear(6144, output_dim)

        self.head_dim = input_dim // num_heads

        self.regularization = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=False),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=False)
        )

    def forward(self, chem_features, prot_features):
        # q = self.query(chem_features)
        # k = self.key(prot_features)
        # v = self.value(prot_features)
        q = self.query(prot_features)
        k = self.key(chem_features)
        v = self.value(chem_features)


        batch_size = q.size(0)
        q = q.view(batch_size, self.num_heads, -1, self.head_dim).permute(1, 0, 2, 3)
        k = k.view(batch_size, self.num_heads, -1, self.head_dim).permute(1, 0, 2, 3)
        v = v.view(batch_size, self.num_heads, -1, self.head_dim).permute(1, 0, 2, 3)

        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim).float())
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.permute(1, 0, 2, 3).contiguous().view(batch_size, -1)

        concatenated_output = torch.cat([chem_features, prot_features, attention_output], dim=-1)

        output = self.regularization(self.fc(concatenated_output))

        output = nn.functional.layer_norm(output, output.size()[1:])

        return output


class ProteinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(ProteinLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = out[:, -1, :]

        out = self.fc(out)
        return out


class Graph:
    pass


class GraphModel(torch.nn.Module):
    def __init__(self, trial, prefix, _node_features_len, _edge_features_len):
        super(GraphModel, self).__init__()
        _n_out = None

        self._gnn_arch = trial.suggest_categorical(prefix + "_gnn_arch", ("staked"))
        self._use_post_fc = trial.suggest_categorical(prefix + "_gnn_post_fc", (True,))
        prefix = prefix + "_" + self._gnn_arch

        if self._gnn_arch == "staked":
            _n_out = 0
            self.gat = GATv1Layers(trial, prefix, _node_features_len, _edge_features_len, heads_range=(1, 7),
                                   layers_range=(1, 3))
            self.gcn = GCNLayers(trial, prefix, self.gat.n_out)
            self.pool_gat_gcn = GraphPool(trial, prefix=prefix+"_gat_gcn")
            _n_out += self.gcn.n_out * self.pool_gat_gcn.coef
            self.use_gine = trial.suggest_categorical(prefix+"_use_gine", (True,))
            if self.use_gine:
                self.gine = GINELayers(trial, prefix, _node_features_len, _edge_features_len)
                self.pool_gine = GraphPool(trial, prefix=prefix+"_gine")
                _n_out += self.gine.n_out * self.pool_gine.coef

            self.use_qgnn = trial.suggest_categorical(prefix+"_use_qgnn", (True,))
            if self.use_qgnn:
                self.qgnn = QGNNLayers(trial, prefix, _node_features_len, _edge_features_len)
                self.pool_qgnn = GraphPool(trial, prefix=prefix+"_qgnn")
                _n_out += self.qgnn.n_out * self.pool_qgnn.coef

            self.use_gmf = trial.suggest_categorical(prefix+"_use_gmf", (True,))
            if self.use_gmf:
                self.gmf = GMFLayers(trial, prefix, _node_features_len)
                self.pool_gmf = GraphPool(trial, prefix=prefix+"_gmf")
                _n_out += self.gmf.n_out * self.pool_gmf.coef

        if self._use_post_fc:        # 如果 _use_post_fc 为 True
            self._post_fc = FCLayers(trial, prefix+"_post", _n_out, layers_range=(1, 1),
                                     n_units_list=(1024, 256, 512,  2048))
            _n_out = self._post_fc.n_out

        self.n_out = _n_out

    def forward(self, graph):
        x = None

        if self._gnn_arch == "staked":
            gat_out = self.gat(graph)
            graph_ = Graph()
            graph_.x = gat_out
            graph_.edge_index = graph.edge_index
            graph_.edge_attr = graph.edge_attr
            gat_gcn_out = self.gcn(graph_)
            x = self.pool_gat_gcn(gat_gcn_out, graph.batch)

            if self.use_gine:
                gine_out = self.gine(graph)
                gine_out = self.pool_gine(gine_out, graph.batch)
                x = torch.cat([x, gine_out], dim=1)
            if self.use_qgnn:
                qgnn_out = self.qgnn(graph)
                qgnn_out = self.pool_qgnn(qgnn_out, graph.batch)
                x = torch.cat([x, qgnn_out], dim=1)
            if self.use_gmf:
                gmf_out = self.gmf(graph)
                gmf_out = self.pool_gmf(gmf_out, graph.batch)
                x = torch.cat([x, gmf_out], dim=1)

        if self._use_post_fc:
            x = self._post_fc(x)

        return x


class DTIProtGraphChemGraphECFP(torch.nn.Module):
    def __init__(self, trial, prot_node_features_len=N_PROT_NODE_FEAT, prot_edge_features_len=N_PROT_EDGE_FEAT,
                 chem_ecfp_len=N_CHEM_ECFP):

        super(DTIProtGraphChemGraphECFP, self).__init__()

        self.use_chem_ecfp_post_fc = trial.suggest_categorical("chem_ecfp_post_fc", (True,))
        chem_ecfp_n_out = chem_ecfp_len
        if self.use_chem_ecfp_post_fc:
            self.chem_ecfp_post_fc = FCLayers(trial, "chem_ecfp_post", chem_ecfp_n_out, layers_range=(1, 1),
                                              n_units_list=(256, 512, 1024, 2048))

        self.prot_graph_encoder = GraphModel(trial, "prot", prot_node_features_len, prot_edge_features_len)
        self.chem_graph_encoder = FE_GNN()

        self.protein_lstm = ProteinLSTM(input_size=1, hidden_size=512, num_layers=2, output_size=1024)

        self.fc = FCLayers(trial, "final", 2048, layers_range=(2, 3), n_units_list=(256, 512, 1024, 2048, 4096))
        self.out = torch.nn.Linear(self.fc.n_out, 1)
        self.MultiHeadAttention = InteractiveMultiHeadAttention(2048, 32)

    def forward(self, data):
        chem_ecfp, chem_graph, prot_graph, proteins_series = data["e1_fp"], data["e1_graph"], data["e2_graph"], data["series"]
        proteins_series = proteins_series.unsqueeze(-1)
        chem_ecfp_out = chem_ecfp
        if self.use_chem_ecfp_post_fc:
            chem_ecfp_out = self.chem_ecfp_post_fc(chem_ecfp_out)

        chem_graph_out = self.chem_graph_encoder(chem_graph)
        prot_graph_out = self.prot_graph_encoder(prot_graph)

        lstm_output = self.protein_lstm(proteins_series)

        chem_features = torch.cat([chem_graph_out, chem_ecfp_out], dim=1)

        prot_features = torch.cat([prot_graph_out, lstm_output], dim=1)

        x = self.MultiHeadAttention(chem_features, prot_features)
        x = self.out(x)
        return x
