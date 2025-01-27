import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair

## Linear
class Linear(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_size, out_size, bias=False))

    def forward(self, g, x):
        h = x
        return self.linears[0](h)

class Linear_BN(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(in_size, out_size, bias=False))
        self.batch_norm = nn.BatchNorm1d((out_size))

    def forward(self, g, x):
        h = x
        return self.batch_norm(self.linears[0](h))

## IMLinear: Linear with Intersect memory
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

class IMLinear(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(1):
            self.layers.append(IMAttention())
        self.linear = Linear(in_size, out_size)

    def forward(self, g, features):
        h = self.linear(g, features)
        for i, layer in enumerate(self.layers):
            h = layer(g, h)

        return h

class IMLinear_BN(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(1):
            self.layers.append(IMAttention())
        self.linear = Linear(in_size, out_size)
        self.batch_norm = nn.BatchNorm1d((out_size))

    def forward(self, g, features):
        h = self.batch_norm(self.linear(g, features))
        for i, layer in enumerate(self.layers):
            h = layer(g, h)

        return h


## MLP
class MLP(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(in_size, hid_size, bias=False))
        self.linears.append(nn.Linear(hid_size, out_size, bias=False))

    def forward(self, g, x):
        h = x
        h = F.relu(self.linears[0](h))
        return self.linears[1](h)

class MLP_BN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(in_size, hid_size, bias=False))
        self.linears.append(nn.Linear(hid_size, out_size, bias=False))
        self.batch_norm = nn.BatchNorm1d((hid_size))

    def forward(self, g, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)

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
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = th.matmul(feat_src, weight)
                graph.srcdata["h"] = feat_src
                graph.update_all(aggregate_fn, fn.sum(msg="m", out="h"))
                rst = graph.dstdata["h"]
            else:
                # aggregate first then mult W
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
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(GCNLayer(in_size, hid_size, activation=F.relu))
        self.layers.append(GCNLayer(hid_size, out_size))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)

        return h

class GCN_BN(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GCN
        self.layers.append(GCNLayer(in_size, hid_size, activation=F.relu, batch_norm=True))
        self.layers.append(GCNLayer(hid_size, out_size))

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            h = layer(g, h)

        return h