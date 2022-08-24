import torch
from collections import defaultdict


class MLP(torch.nn.Module):
    def __init__(self, n_layers, dim, feature_len):
        super(MLP, self).__init__()
        self.mlp_layers = torch.nn.ModuleList([])
        for i in range(n_layers):
            self.mlp_layers.append(torch.nn.Linear(feature_len if i == 0 else dim,
                                                   1 if i == n_layers - 1 else dim))

    def forward(self, x):
        for i in range(len(self.mlp_layers)):
            x = self.mlp_layers[i](x)
            if i != len(self.mlp_layers) - 1:
                x = torch.relu(x)
        x = x.squeeze()
        return x


def get_path_dict(args, schema):
    ht2paths = {}
    path2id = {}
    node2neighbors, _, n_events = schema
    for head in range(n_events):
        res = bfs(args, head, node2neighbors, n_events, path2id)
        ht2paths.update(res)
    return ht2paths, len(path2id)


def bfs(args, head, node2neighbors, n_events, path2id):
    # add length-1 paths
    if args.path_event_only:
        all_paths = [[i] for i in node2neighbors[head] if i[1] < n_events]
    else:
        all_paths = [[i] for i in node2neighbors[head]]

    p = 0
    for _ in range(args.max_path_length - 1):
        current_queue_length = len(all_paths)
        while p < current_queue_length:
            path = all_paths[p]
            last_node_in_path = path[-1][1]
            if not args.path_event_only or last_node_in_path < n_events:
                nodes_in_path = set([head] + [i[1] for i in path])
                for edge in node2neighbors[last_node_in_path]:
                    # append (relation, node) to the path if the new node does not appear in this path before
                    if edge[1] not in nodes_in_path:
                        all_paths.append(path + [edge])
            p += 1

    ht2paths = defaultdict(list)
    for path in all_paths:
        tail = path[-1][1]
        if tail < n_events:  # if the tail is also an entity
            relational_path = tuple([i[0] for i in path])
            if relational_path not in path2id:
                path2id[relational_path] = len(path2id)
            ht2paths[(head, tail)].append(path2id[relational_path])

    return ht2paths


def transform_data_wrt_path(data, ht2paths, path_feature_len, n_events):
    all_data = []
    for s in data:
        for head in range(n_events):
            res = [0] * path_feature_len
            for tail in s:
                if tail != head and (head, tail) in ht2paths:
                    for path_type_id in ht2paths[(head, tail)]:
                        res[path_type_id] = 1
            all_data.append(res)
    all_data = torch.tensor(all_data, dtype=torch.float)
    return all_data
