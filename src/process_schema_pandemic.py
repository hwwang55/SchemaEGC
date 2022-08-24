import json
from collections import defaultdict


def read_schema():
    node2neighbors = defaultdict(set)
    node2type = {}
    node2idx = {}
    leaf_event_list = []
    nonleaf_event_list = []
    entity_list = []

    schema_obj = json.load(open('../data/schema_pandemic.json'))

    # assign an index to each node
    for event in schema_obj['events']:
        if 'children' in event:
            nonleaf_event_list.append((event['@id'], event['name']))
        else:
            leaf_event_list.append((event['@id'], event['name']))
    for entity in schema_obj['entities']:
        entity_list.append((entity['@id'], entity['name']))

    for event_id, event_type in leaf_event_list:
        node2idx[event_id] = len(node2idx)
        node2type[node2idx[event_id]] = event_type
    for event_id, event_type in nonleaf_event_list:
        node2idx[event_id] = len(node2idx)
        node2type[node2idx[event_id]] = event_type
    for entity_id, entity_type in entity_list:
        node2idx[entity_id] = len(node2idx)
        node2type[node2idx[entity_id]] = entity_type

    # construct the schema graph
    for event in schema_obj['events']:
        event_idx = node2idx[event['@id']]
        if 'participants' in event:
            for participant in event['participants']:
                entity_idx = node2idx[participant['entity']]
                role_type = participant['roleName']
                node2neighbors[event_idx].add((role_type, entity_idx))
                node2neighbors[entity_idx].add((role_type, event_idx))
        if 'children' in event:
            for c in event['children']:
                child_idx = node2idx[c['child']]
                node2neighbors[event_idx].add(('has_child', child_idx))
                node2neighbors[child_idx].add(('has_parent', event_idx))
                if 'outlinks' in c:
                    for next_event_id in c['outlinks']:
                        node2neighbors[child_idx].add(('temp', node2idx[next_event_id]))
                        node2neighbors[node2idx[next_event_id]].add(('temp_rev', child_idx))

    for relation in schema_obj['relations']:
        sub_idx = node2idx[relation['relationSubject']]
        obj_idx = node2idx[relation['relationObject']]
        edge_type = relation['relationPredicate']
        node2neighbors[sub_idx].add((edge_type, obj_idx))
        node2neighbors[obj_idx].add((edge_type, sub_idx))

    return [node2neighbors, node2type, len(leaf_event_list)], node2idx
