import json
import random


def read_data(filename, schema):
    print('processing instance graphs from ' + filename)
    _, s_node2type, n_events = schema
    s_next_event_types, s_prev_event_types = get_schema_event_neighbor_type(schema)
    data = []
    for line in open(filename):
        mapped_nodes_in_schema = set()
        line = line.rstrip('\n')
        instance_json = json.loads(line)['schemas'][0]
        i_next_event_types, i_prev_event_types, i_event2type = json_to_instance_graphs(instance_json)
        for i in i_event2type.keys():
            # the matched schema nodes for nodes i in the instance graph based solely on event type
            res = [s for s in s_node2type.keys() if i_event2type[i] == s_node2type[s]]
            if len(res) >= 1:  # if this node in the instance graph can be matched to the schema
                # matching scores
                scores = [jaccard_similarity(s_next_event_types[s], i_next_event_types[i]) +
                          jaccard_similarity(s_prev_event_types[s], i_prev_event_types[i]) for s in res]
                max_score = max(scores)
                # randomly select a node in the schema with the highest matching score as the matching result
                candidates = [res[i] for i in range(len(res)) if scores[i] == max_score]
                mapped_nodes_in_schema.add(random.choice(candidates))
        # the number of matched nodes should be at least 2
        if len(mapped_nodes_in_schema) >= 2:
            data.append(list(mapped_nodes_in_schema))
    return data


def get_schema_event_neighbor_type(schema):
    s_node2neighbors, s_node2type, s_n_events = schema
    s_next_event_types = {}
    s_prev_event_types = {}
    for k in s_node2neighbors.keys():
        if k < s_n_events:
            s_next_event_types[k] = set()
            s_prev_event_types[k] = set()
            for edge_type, node_idx in s_node2neighbors[k]:
                if edge_type == 'temporal':
                    s_next_event_types[k].add(s_node2type[node_idx])
                elif edge_type == 'temporal_rev':
                    s_prev_event_types[k].add(s_node2type[node_idx])

    return s_next_event_types, s_prev_event_types


def json_to_instance_graphs(instance_graph_json):
    i_next_event_types = {}
    i_prev_event_types = {}
    i_event2type = {}
    for event in instance_graph_json['steps']:
        event_id = event['@id']
        event_type = event['@type'].split('/')[-1]
        i_next_event_types[event_id] = set()
        i_prev_event_types[event_id] = set()
        i_event2type[event_id] = event_type

    for order in instance_graph_json['order']:
        b_name = order['before']
        a_name = order['after']
        i_next_event_types[b_name].add(i_event2type[a_name])
        i_prev_event_types[a_name].add(i_event2type[b_name])

    return i_next_event_types, i_prev_event_types, i_event2type


# jaccard similarity for set A and set B:
# J(A, B) = 0 if both A and B are empty set,
# otherwise J(A, B) = |intersection of A and B| / |union of A and B|
def jaccard_similarity(a, b):
    if len(a) == 0 and len(b) == 0:
        return 1
    else:
        return len(a & b) / len(a | b)


def read_instances(args, schema):
    # get_statistics(args)
    train_data = read_data('../data/' + args.dataset + '/train.json', schema)
    val_data = read_data('../data/' + args.dataset + '/val.json', schema)
    test_data = read_data('../data/' + args.dataset + '/test.json', schema)
    return train_data, val_data, test_data
