import json


node2index = {}
coreference = {}
node2neighbors = {}
node2type = {}


def read_schema(args):
    print('processing schema')
    if args.dataset == 'car_bombings':
        schema_obj = json.load(open('../data/schema_IED.json'))['schemas'][6]
    elif args.dataset == 'ied_bombings' or args.dataset == 'suicide_ied':
        schema_obj = json.load(open('../data/schema_IED.json'))['schemas'][1]
    else:
        raise ValueError('unknown dataset')

    n_events = add_event_nodes(schema_obj, args)
    add_event_event_edges(schema_obj)
    get_coreferential_nodes(schema_obj)
    add_event_entity_edges(schema_obj)
    add_entity_entity_edges(schema_obj)

    return node2neighbors, node2type, n_events


def add_event_nodes(schema_graph, args):
    for event in schema_graph['steps']:
        # this is to remove 'Transaction.ExchangeBuySell.Unspecified:2' and 'Movement.Transportation.Unspecified:2'
        # from Car-IED schema, since they are almost equivalent to 'Transaction.ExchangeBuySell.Unspecified:1'
        # and 'Movement.Transportation.Unspecified:1'
        if args.dataset != 'car_bombings' or (
                not event['@id'].endswith('Transaction.ExchangeBuySell.Unspecified:2')
                and not event['@id'].endswith('Movement.Transportation.Unspecified:2')):
            node2index[event['@id']] = len(node2index)
            event_index = node2index[event['@id']]
            event_type = event['@type'].split('/')[-1]
            node2neighbors[event_index] = set()
            node2type[event_index] = event_type
    return len(node2type)


def add_event_event_edges(schema_graph):
    for order in schema_graph['order']:
        for b_name in order['before'] if type(order['before']) is list else [order['before']]:
            for a_name in order['after'] if type(order['after']) is list else [order['after']]:
                if b_name in node2index and a_name in node2index:
                    b_index = node2index[b_name]
                    a_index = node2index[a_name]
                    node2neighbors[b_index].add(('temporal', a_index))
                    node2neighbors[a_index].add(('temporal_rev', b_index))


# use union-find set to solve coreference
def get_coreferential_nodes(schema_graph):
    for event in schema_graph['steps']:
        if event['@id'] in node2index:
            for p in event['participants']:
                coreference[p['@id']] = p['@id']

    for relation in schema_graph['entityRelations']:
        sub_name = relation['relationSubject']
        if sub_name in coreference:
            for r in relation['relations']:
                r_type = r['relationPredicate'].split('/')[-1]
                if r_type == 'Physical.SameAs.SameAs':
                    for obj_name in r['relationObject'] if type(r['relationObject']) is list else [r['relationObject']]:
                        if obj_name in coreference:
                            union(sub_name, obj_name)
    for k in coreference.keys():
        coreference[k] = find(k)


def find(e):
    if coreference[e] != e:
        coreference[e] = find(coreference[e])
    return coreference[e]


def union(e0, e1):
    coreference[find(e1)] = find(e0)


def add_event_entity_edges(schema_graph):
    for event in schema_graph['steps']:
        if event['@id'] in node2index:
            event_index = node2index[event['@id']]
            for p in event['participants']:
                if coreference[p['@id']] not in node2index:
                    node2index[coreference[p['@id']]] = len(node2index)
                entity_index = node2index[coreference[p['@id']]]
                entity_type = '/'.join([i.split('/')[-1] for i in p['entityTypes']])
                role_type = p['role'].split('/')[-1]
                node2neighbors[event_index].add((role_type, entity_index))
                if entity_index not in node2neighbors:
                    node2neighbors[entity_index] = {(role_type, event_index)}
                else:
                    node2neighbors[entity_index].add((role_type, event_index))
                if entity_index not in node2type:
                    node2type[entity_index] = entity_type
                else:
                    intersection = set(node2type[entity_index].split('/')) & set(entity_type.split('/'))
                    node2type[entity_index] = '/'.join(intersection)


def add_entity_entity_edges(schema_graph):
    for relation in schema_graph['entityRelations']:
        if relation['relationSubject'] in coreference:
            sub_index = node2index[coreference[relation['relationSubject']]]
            for r in relation['relations']:
                r_type = r['relationPredicate'].split('/')[-1]
                if r_type != 'Physical.SameAs.SameAs':
                    for obj_name in r['relationObject'] if type(r['relationObject']) is list else [r['relationObject']]:
                        if obj_name in coreference:
                            object_index = node2index[coreference[obj_name]]
                            node2neighbors[sub_index].add((r_type, object_index))
                            node2neighbors[object_index].add((r_type, sub_index))
