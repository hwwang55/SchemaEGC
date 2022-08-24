import torch
from path import MLP


class SchemaPredModel(torch.nn.Module):
    def __init__(self, args, schema, n_events, context_model, path_model):
        super(SchemaPredModel, self).__init__()
        self.args = args
        if args.use_context:
            self.schema = schema
            self.n_events = n_events
            self.context_model = context_model
            if args.pred_fn == 'mlp':
                self.context_mlp = MLP(n_layers=2, dim=256, feature_len=2 * self.args.gnn_dim)
        if args.use_path:
            self.path_model = path_model

    def forward(self, context, path):
        pred = 0
        if self.args.use_context:
            event_indices = context[:, 0]
            subgraph_indices = context[:, 1:].float()
            all_event_embeddings = self.context_model(self.schema)[:self.n_events]
            node_embeddings = all_event_embeddings[event_indices]
            # subgraph pooling
            if self.args.subgraph_pooling == 'sum':
                subgraph_embeddings = torch.mm(subgraph_indices, all_event_embeddings)
            elif self.args.subgraph_pooling == 'avg':
                subgraph_embeddings = torch.mm(subgraph_indices, all_event_embeddings)
                subgraph_embeddings /= torch.sum(subgraph_indices, dim=1, keepdim=True)
            elif self.args.subgraph_pooling == 'att':
                attention = torch.mm(node_embeddings, torch.transpose(all_event_embeddings, 0, 1))
                attention *= subgraph_indices
                attention = attention.masked_fill(attention == 0, -1e9)
                attention = torch.softmax(attention, dim=-1)
                subgraph_embeddings = torch.mm(attention, all_event_embeddings)
            else:
                raise ValueError('unknown subgraph pooling function')
            # final prediction function for context
            if self.args.pred_fn == 'dot':
                pred_context = torch.sum(node_embeddings * subgraph_embeddings, dim=1)
            elif self.args.pred_fn == 'mlp':
                pred_context = self.context_mlp(torch.cat([node_embeddings, subgraph_embeddings], dim=-1))
            else:
                raise ValueError('unknown final prediction function')
            pred += pred_context

        if self.args.use_path:
            pred_path = self.path_model(path)
            pred += pred_path

        return pred
