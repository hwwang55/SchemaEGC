import dgl
import torch
import numpy as np
from dgl.nn import GraphConv, GATConv, SAGEConv, SGConv, TAGConv


class GNN(torch.nn.Module):
    def __init__(self, gnn, n_layers, feature_len, dim):
        super(GNN, self).__init__()
        self.gnn = gnn
        self.gnn_layers = torch.nn.ModuleList([])
        if gnn in ['gcn', 'gat', 'sage', 'tag']:
            for i in range(n_layers):
                if gnn == 'gcn':
                    self.gnn_layers.append(GraphConv(in_feats=feature_len if i == 0 else dim,
                                                     out_feats=dim,
                                                     activation=None if i == n_layers - 1 else torch.relu))
                elif gnn == 'gat':
                    num_heads = 16  # make sure that dim is dividable by num_heads
                    self.gnn_layers.append(GATConv(in_feats=feature_len if i == 0 else dim,
                                                   out_feats=dim // num_heads,
                                                   activation=None if i == n_layers - 1 else torch.relu,
                                                   num_heads=num_heads))
                elif gnn == 'sage':
                    agg = 'pool'
                    self.gnn_layers.append(SAGEConv(in_feats=feature_len if i == 0 else dim,
                                                    out_feats=dim,
                                                    activation=None if i == n_layers - 1 else torch.relu,
                                                    aggregator_type=agg))
                elif gnn == 'tag':
                    hops = 2
                    self.gnn_layers.append(TAGConv(in_feats=feature_len if i == 0 else dim,
                                                   out_feats=dim,
                                                   activation=None if i == n_layers - 1 else torch.relu,
                                                   k=hops))
        elif gnn == 'sgc':
            self.gnn_layers.append(SGConv(in_feats=feature_len, out_feats=dim, k=n_layers))
        else:
            raise ValueError('unknown GNN model')

    def forward(self, graph):
        h = graph.ndata['feature']
        for layer in self.gnn_layers:
            h = layer(graph, h)
            if self.gnn == 'gat':
                h = torch.reshape(h, [h.size()[0], -1])
        return h


def get_dgl_schema(args, schema):
    # add edges
    edge_list = []
    node2neighbors, node2type, n_events = schema
    for k in node2neighbors.keys():
        for v in node2neighbors[k]:
            if not args.context_event_only or (k < n_events and v[1] < n_events):
                edge_list.append((k, v[1]))
    src = [s for (s, _) in edge_list]
    dst = [t for (_, t) in edge_list]
    graph = dgl.graph((src, dst), num_nodes=n_events if args.context_event_only else len(node2type))

    # add node features
    type2index = {}
    for k, v in node2type.items():
        if k < n_events:  # event nodes
            if args.event_type_ontology:
                for w in v.split('.'):
                    if w != 'Unspecified' and w not in type2index:
                        type2index[w] = len(type2index)
            else:
                if v not in type2index:
                    type2index[v] = len(type2index)
        else:  # entity nodes
            if not args.context_event_only:
                for w in v.split('/'):
                    if w not in type2index:
                        type2index[w] = len(type2index)

    node_features = np.zeros([n_events if args.context_event_only else len(node2type), len(type2index)])
    for k, v in node2type.items():
        if k < n_events:  # event nodes
            if args.event_type_ontology:
                for w in v.split('.'):
                    if w != 'Unspecified':
                        node_features[k][type2index[w]] = 1
            else:
                node_features[k][type2index[v]] = 1
        else:  # entity nodes
            if not args.context_event_only:
                for w in v.split('/'):
                    node_features[k][type2index[w]] = 1

    node_features = torch.tensor(node_features, dtype=torch.float)
    graph.ndata['feature'] = node_features
    graph = dgl.add_self_loop(graph)

    return graph, len(type2index)


def transform_data_wrt_context(data, n_events):
    all_data = []
    for s in data:
        for i in range(n_events):
            res = [0] * (n_events + 1)
            res[0] = i
            for j in s:
                if j != i:
                    res[j + 1] = 1
            all_data.append(res)
    all_data = torch.tensor(all_data)
    return all_data
