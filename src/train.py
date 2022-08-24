import torch
import numpy as np
from copy import deepcopy
from model import SchemaPredModel
from sklearn.metrics import roc_auc_score, f1_score
from neighbor import GNN, get_dgl_schema, transform_data_wrt_context
from path import MLP, get_path_dict, transform_data_wrt_path


def train(args, schema, data):
    train_data, val_data, test_data = data
    n_events = schema[2]

    if args.use_context:
        dgl_schema, context_feature_len = get_dgl_schema(args, schema)
        context_gnn = GNN(args.gnn, args.gnn_layers, context_feature_len, args.gnn_dim)
        train_data_context = transform_data_wrt_context(train_data, n_events)
        val_data_context = transform_data_wrt_context(val_data, n_events)
        test_data_context = transform_data_wrt_context(test_data, n_events)
    else:
        dgl_schema, context_gnn, train_data_context, val_data_context, test_data_context = [None] * 5

    if args.use_path:
        ht2paths, path_feature_len = get_path_dict(args, schema)
        path_mlp = MLP(args.path_mlp_layers, args.path_mlp_dim, path_feature_len)
        train_data_path = transform_data_wrt_path(train_data, ht2paths, path_feature_len, n_events)
        val_data_path = transform_data_wrt_path(val_data, ht2paths, path_feature_len, n_events)
        test_data_path = transform_data_wrt_path(test_data, ht2paths, path_feature_len, n_events)
    else:
        path_mlp, train_data_path, val_data_path, test_data_path = [None] * 4

    train_labels, val_labels, test_labels = get_labels(data, n_events)
    model = SchemaPredModel(args, dgl_schema, n_events, context_gnn, path_mlp)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_model_params = None
    best_val_auc = 0

    print('start training')
    for i in range(args.n_epochs):
        print('epoch %d:' % i, end='    ')

        # shuffle training data
        index = np.arange(len(train_labels))
        np.random.shuffle(index)
        if args.use_context:
            train_data_context = train_data_context[index]
        if args.use_path:
            train_data_path = train_data_path[index]
        train_labels = train_labels[index]

        # training
        model.train()
        s = 0
        while s < len(train_labels):
            pred = model(train_data_context[s:s + args.batch_size] if args.use_context else None,
                         train_data_path[s:s + args.batch_size] if args.use_path else None)
            loss = loss_fn(pred, train_labels[s:s + args.batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            s += args.batch_size

        # evaluation
        model.eval()
        with torch.no_grad():
            _, _, _ = evaluate(model, 'train', train_data_context, train_data_path, train_labels)
            _, val_auc, _ = evaluate(model, 'val', val_data_context, val_data_path, val_labels)
            _, _, _ = evaluate(model, 'test', test_data_context, test_data_path, test_labels)
        if val_auc > best_val_auc:
            best_model_params = deepcopy(model.state_dict())
            best_val_auc = val_auc
        print()

    model.load_state_dict(best_model_params)
    model.eval()
    print('final ', end='')
    evaluate(model, 'test', test_data_context, test_data_path, test_labels)
    print()


def evaluate(model, mode, data_context, data_path, label):
    pred = torch.sigmoid(model(data_context, data_path)).cpu().detach().numpy()
    label = label.cpu().detach().numpy()
    acc = float(np.mean((pred >= 0.5) == label))
    auc = roc_auc_score(y_score=pred, y_true=label)
    f1 = f1_score(y_pred=(pred >= 0.5), y_true=label)
    print('%s acc: %.4f  auc: %.4f  f1: %.4f' % (mode, acc, auc, f1), end='    ')
    return acc, auc, f1


def get_labels(data, n_events):
    train_data, val_data, test_data = data
    train_labels = calculate_labels(train_data, n_events)
    val_labels = calculate_labels(val_data, n_events)
    test_labels = calculate_labels(test_data, n_events)
    return train_labels, val_labels, test_labels


def calculate_labels(data, n_events):
    labels_all = []
    for s in data:
        labels = [0] * n_events
        for i in s:
            labels[i] = 1
        labels_all.extend(labels)
    labels_all = torch.tensor(labels_all, dtype=torch.float)
    return labels_all
