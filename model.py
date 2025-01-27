import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DropEdge
from dgl import function as fn
import dgl.nn as dglnn
from dgl.base import DGLError
from dgl.utils import expand_as_pair

## GCN
class GCNLayer(nn.Module):
    def __init__(
            self,
            in_feats,
            out_feats,
            norm="both",
            weight=True,
            bias=True,
            activation=None,
            allow_zero_in_degree=False,
            batch_norm=False,
    ):
        super(GCNLayer, self).__init__()
        if norm not in ("none", "both", "right", "left"):
            raise KeyError(
                'Invalid norm value. Must be either "none", "both", "right" or "left".'
                ' But got "{}".'.format(norm)
            )
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree
        self.if_batch_norm = batch_norm

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self._activation = activation
        self.batch_norm = nn.BatchNorm1d((out_feats))

    def reset_parameters(self):

        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None, edge_weight=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise KeyError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise KeyError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
            else:
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
                if weight is not None:
                    rst = th.matmul(rst, weight)

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self.if_batch_norm:
                rst = self.batch_norm(rst)

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(in_size, hid_size, activation=F.relu))
        for _ in range(n_layers - 2):
            self.layers.append(GCNLayer(hid_size, hid_size, activation=F.relu))
        self.layers.append(GCNLayer(hid_size, out_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)

        return F.log_softmax(h, dim=1)


## GAT
class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads, n_layers, dropout=0.6):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=dropout,
                attn_drop=dropout,
                activation=F.elu,
            )
        )
        for i in range(1, n_layers - 1):
            self.gat_layers.append(
                dglnn.GATConv(
                    hid_size * heads[i - 1],
                    hid_size,
                    heads[i],
                    feat_drop=0.6,
                    attn_drop=0.6,
                    activation=F.elu,
                )
            )
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[n_layers - 2],
                out_size,
                heads[n_layers - 1],
                feat_drop=dropout,
                attn_drop=dropout,
                activation=None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h_res = h
            h = layer(g, h)
            if i == len(self.gat_layers) - 1:
                h = h.mean(1)
            else:
                h = h.flatten(1)
            if h.shape == h_res.shape:
                h = h + h_res

        return F.log_softmax(h, dim=1)


## APPNP
class GraphConv(nn.Module):
    def __init__(
            self,
            norm="both",
            allow_zero_in_degree=False,
    ):
        super(GraphConv, self).__init__()
        if norm not in ("none", "both", "right", "left"):
            raise KeyError(
                'Invalid norm value. Must be either "none", "both", "right" or "left".'
                ' But got "{}".'.format(norm)
            )
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def forward(self, graph, feat):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise KeyError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )
            aggregate_fn = fn.copy_u("h", "m")

            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["left", "both"]:
                degs = graph.out_degrees().to(feat_src).clamp(min=1)
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = th.reshape(norm, shp)
                feat_src = feat_src * norm

            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
            rst = graph.dstdata["h"]

            if self._norm in ["right", "both"]:
                degs = graph.in_degrees().to(feat_dst).clamp(min=1)
                if self._norm == "both":
                    norm = th.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = th.reshape(norm, shp)
                rst = rst * norm

            return rst

class MLP(nn.Module):
    def __init__(self, in_size, hid_size, out_size, dropout=0.5):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(in_size, hid_size, bias=False))
        self.linears.append(nn.Linear(hid_size, out_size, bias=False))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        h = x
        h = F.relu(self.linears[0](self.dropout(h)))
        return self.linears[1](self.dropout(h))

class APPNP(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers, alpha, dropout=0.5):
        super().__init__()
        self.mlp = MLP(in_size, hid_size, out_size, dropout)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(GraphConv())
        self.alpha = alpha

    def forward(self, g, features):
        h = self.mlp(g, features)
        initial_residual = h
        for i, layer in enumerate(self.layers):
            h =  self.alpha * initial_residual + (1 - self.alpha) * layer(g, h)

        return F.log_softmax(h, dim=1)


## GCNII
class GCNIILayer(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        residual=False,
        allow_zero_in_degree = False
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.residual = residual
        self._allow_zero_in_degree = allow_zero_in_degree
        self.weight = nn.Parameter(th.FloatTensor(self.in_size, self.out_size))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.out_size)
        self.weight.data.uniform_(-std, std)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, feat_0, alpha, lamda, layer):
        residual_feat = feat
        beta = math.log(lamda / layer + 1)

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )


            degs = graph.in_degrees().to(feat).clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(feat.device).unsqueeze(1)

            feat = feat * norm

            graph.ndata["h"] = feat
            msg_func = fn.copy_u("h", "m")

            graph.update_all(msg_func, fn.sum(msg="m", out="h"))
            feat = graph.ndata.pop("h")

            feat = feat * norm

            feat_sum = feat * (1 - alpha) + feat_0 * alpha
            rst = (1 - beta) * feat_sum + beta * (feat_sum @ self.weight)

            if self.residual:
                rst = rst + residual_feat

            return rst

class GCNII(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        n_layers,
        alpha,
        lamda,
        dropout=0.5,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GCNIILayer(hidden_size, hidden_size))
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(in_size, hidden_size))
        self.linear.append(nn.Linear(hidden_size, out_size))
        self.params1 = list(self.layers.parameters())
        self.params2 = list(self.linear.parameters())
        self.activation = nn.ReLU()
        self.alpha = alpha
        self.lamda = lamda
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feature):
        h = feature
        h = self.activation(self.linear[0](self.dropout(h)))
        initial_residual = h
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(g, self.dropout(h), initial_residual, self.alpha, self.lamda, i + 1))
        h = self.linear[-1](self.dropout(h))

        return F.log_softmax(h, dim=1)

class GCNII_inductive(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        n_layers,
        alpha,
        lamda,
        dropout=0.5,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GCNIIILayer(hidden_size, hidden_size, residual=True))
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(in_size, hidden_size))
        self.linear.append(nn.Linear(hidden_size, out_size))
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha
        self.lamda = lamda
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feature):
        h = feature
        h = self.activation(self.linear[0](self.dropout(h)))
        initial_residual = h
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(g, self.dropout(h), initial_residual, self.alpha, self.lamda, i + 1))
        h = self.linear[1](self.dropout(h))

        return self.sigmoid(h)


## GCNIII
class IMAttention(nn.Module):
    def __init__(self):
        super(IMAttention, self).__init__()

    def forward(self, graph, feat):
        with graph.local_scope():
            degs_o = graph.out_degrees().to(feat).clamp(min=1)
            norm_l = th.pow(degs_o, -0.5)
            shp = norm_l.shape + (1,) * (feat.dim() - 1)
            norm_l = th.reshape(norm_l, shp)
            feat = feat * norm_l

            graph.srcdata["h"] = feat
            graph.update_all(fn.copy_u("h", "m"), fn.sum(msg="m", out="h"))
            rst = graph.dstdata["h"]

            degs_i = graph.in_degrees().to(feat).clamp(min=1)
            norm_r = th.pow(degs_i, -0.5)
            shp = norm_r.shape + (1,) * (feat.dim() - 1)
            norm_r = th.reshape(norm_r, shp)
            rst = rst * norm_r

            return rst

class GCNIIILayer(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        residual=False,
        allow_zero_in_degree = False
    ):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.residual = residual
        self._allow_zero_in_degree = allow_zero_in_degree
        self.weight = nn.Parameter(th.FloatTensor(self.in_size, self.out_size))
        self.reset_parameters()

    def reset_parameters(self):
        std = 1. / math.sqrt(self.out_size)
        self.weight.data.uniform_(-std, std)


    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, feat_0, alpha, lamda, layer, res, mapping):
        residual_feat = feat
        beta = math.log(lamda / layer + 1)

        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )


            degs = graph.in_degrees().to(feat).clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(feat.device).unsqueeze(1)

            feat = feat * norm

            graph.ndata["h"] = feat
            msg_func = fn.copy_u("h", "m")

            graph.update_all(msg_func, fn.sum(msg="m", out="h"))
            feat = graph.ndata.pop("h")

            feat = feat * norm

            if res:
                feat_sum = feat * (1 - alpha) + feat_0 * alpha
            else:
                feat_sum = feat

            if mapping:
                rst = (1 - beta) * feat_sum + beta * (feat_sum @ self.weight)
            else:
                rst = feat_sum

            if self.residual:
                rst = rst + residual_feat

            return rst

class GCNIII(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        n_layers,
        alpha,
        lamda,
        gamma,
        dropout=0.5,
        dropedge=0.0,
        intersect_memory=True,
        initial_residual=True,
        identity_mapping=True,
        batchnorm_wide=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GCNIIILayer(hidden_size, hidden_size))
        self.linear = nn.ModuleList()
        self.IM = IMAttention()
        self.wide = nn.Linear(in_size, out_size)
        self.linear.append(nn.Linear(in_size, hidden_size))
        self.linear.append(nn.Linear(hidden_size, out_size))
        self.linear.append(nn.Linear(in_size, out_size))
        self.params1 = list(self.layers.parameters())
        self.params2 = list(self.linear.parameters())
        self.activation = nn.ReLU()
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma
        self.dropout = nn.Dropout(dropout)
        self.dropedge = dropedge
        self.intersect_memory = intersect_memory
        self.initial_residual = initial_residual
        self.identity_mapping = identity_mapping
        self.batchnorm_wide = batchnorm_wide
        self.batch_norm = nn.BatchNorm1d((out_size))

    def forward(self, g, feature):
        h = feature
        h = self.activation(self.linear[0](self.dropout(h)))
        initial_residual = h
        if self.dropedge > 0 and self.training:
            g = DropEdge(self.dropedge)(g)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(g, self.dropout(h), initial_residual, self.alpha, self.lamda, i + 1, self.initial_residual, self.identity_mapping))
        wide = self.linear[-1](feature)
        if self.batchnorm_wide:
            wide = self.batch_norm(wide)
        if self.intersect_memory:
            wide = self.IM(g, wide)
        h = self.gamma * wide + (1 - self.gamma) * self.linear[1](self.dropout(h))

        return F.log_softmax(h, dim=1)

class GCNIII_inductive(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        hidden_size,
        n_layers,
        alpha,
        lamda,
        gamma,
        dropout=0.5,
        dropedge=0.0,
        intersect_memory=True,
        initial_residual=True,
        identity_mapping=True,
        batchnorm_wide=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(GCNIIILayer(hidden_size, hidden_size, residual=True))
        self.linear = nn.ModuleList()
        self.IM = IMAttention()
        self.linear.append(nn.Linear(in_size, hidden_size))
        self.linear.append(nn.Linear(hidden_size, out_size))
        self.linear.append(nn.Linear(in_size, out_size))
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.alpha = alpha
        self.lamda = lamda
        self.gamma = gamma
        self.dropout = nn.Dropout(dropout)
        self.dropedge = dropedge
        self.intersect_memory = intersect_memory
        self.initial_residual = initial_residual
        self.identity_mapping = identity_mapping
        self.batchnorm_wide = batchnorm_wide
        self.batch_norm = nn.BatchNorm1d((out_size))

    def forward(self, g, feature):
        h = feature
        h = self.activation(self.linear[0](self.dropout(h)))
        initial_residual = h
        if self.dropedge > 0 and self.training:
            g = DropEdge(self.dropedge)(g)
        for i, layer in enumerate(self.layers):
            h = self.activation(layer(g, self.dropout(h), initial_residual, self.alpha, self.lamda, i + 1, self.initial_residual, self.identity_mapping))
        wide = self.linear[-1](feature)
        if self.batchnorm_wide:
            wide = self.batch_norm(wide)
        if self.intersect_memory:
            wide = self.IM(g, wide)
        h = self.gamma * wide + (1 - self.gamma) * self.linear[1](self.dropout(h))

        return self.sigmoid(h)