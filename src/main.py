import os
import argparse
import process_schema_IED
import process_schema_pandemic
import process_instances_IED
import process_instances_pandemic
import train


def print_setting(args):
    print('\n===========================')
    for k, v in args.__dict__.items():
        print('%s: %s' % (k, v))
    print('===========================\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pandemic', help='dataset name')
    parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')

    parser.add_argument('--use_context', type=bool, default=True, help='whether use context')
    parser.add_argument('--context_event_only', type=bool, default=False, help='only consider events for context')
    parser.add_argument('--event_type_ontology', type=bool, default=False, help='consider event type ontology')
    parser.add_argument('--gnn', type=str, default='gcn', help='name of the GNN model: gcn, gat, sage, tag, sgc')
    parser.add_argument('--gnn_layers', type=int, default=3, help='number of GNN layers')
    parser.add_argument('--gnn_dim', type=int, default=256, help='dimension of hidden GNN layers')
    parser.add_argument('--subgraph_pooling', type=str, default='sum', help='subgraph pooling function: sum, avg, att')
    parser.add_argument('--pred_fn', type=str, default='mlp', help='final prediction function: dot, mlp')

    parser.add_argument('--use_path', type=bool, default=True, help='whether use paths')
    parser.add_argument('--path_event_only', type=bool, default=False, help='only consider events for paths')
    parser.add_argument('--max_path_length', type=int, default=4, help='maximum length of paths')
    parser.add_argument('--path_mlp_layers', type=int, default=2, help='number of path MLP layers')
    parser.add_argument('--path_mlp_dim', type=int, default=256, help='dimension of path MLP hidden layers')

    args = parser.parse_args()
    print_setting(args)
    print('current working directory: ' + os.getcwd() + '\n')
    assert args.use_context or args.use_path

    if args.dataset in ['car_bombings', 'ied_bombings', 'suicide_ied']:
        schema = process_schema_IED.read_schema(args)
        data = process_instances_IED.read_instances(args, schema)
    elif args.dataset == 'pandemic':
        schema, node2idx = process_schema_pandemic.read_schema()
        data = process_instances_pandemic.read_instances(node2idx)
    else:
        raise ValueError('unknown dataset')
    train.train(args, schema, data)


if __name__ == '__main__':
    main()
