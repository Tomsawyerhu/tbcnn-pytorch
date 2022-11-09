import os
import pickle
import re

import torch
from torch.utils.data import Dataset

from tbcnn.vocab_dict import VocabDict


def collate(batch):
    return [x['bug_nodes'] for x in batch], [x['bug_children'] for x in batch], [x['bug_children_nodes'] for x in
                                                                                 batch], [x['fix_nodes'] for x in
                                                                                          batch], [
               x['fix_children'] for x in batch], [x['fix_children_nodes'] for x in batch], [x['tgt'] for x in batch]


class RawTreeDataset(Dataset):
    def __init__(self, data=None, config=None):
        self.vocab_dict = VocabDict(
            file_name=config['vocabulary_dictionary_path'],
            name="JavaTokenVocabDictionary")
        self.vocab_dict.load()
        self.data = data if data is not None else []
        self.config = config if config is not None else dict()

    def __getitem__(self, index):
        max_tree_size = self.config['max_tree_size']
        max_children_size = self.config['max_children_size']

        result = dict()
        patch = self.data[index]
        result['tgt'] = int(patch['target'])

        result['bug_nodes'] = [self.vocab_dict.get_w2i(x[0]) for x in patch['jsgraph1']['node_features'].values()]
        bug_nodes_len = len(result['bug_nodes'])
        result['bug_nodes'].extend([0] * (max_tree_size - bug_nodes_len))

        result['bug_children'] = [[x[1] for x in patch['jsgraph1']['graph'] if x[0] == i] for i in
                                  range(patch['graph_size1'])]
        [x.extend([0] * (max_children_size - len(x))) for x in result['bug_children']]
        result['bug_children'].extend(
            [[0] * max_children_size for _ in range(max_tree_size - bug_nodes_len)])

        result['fix_nodes'] = [self.vocab_dict.get_w2i(x[0]) for x in patch['jsgraph2']['node_features'].values()]
        fix_nodes_len = len(result['fix_nodes'])
        result['fix_nodes'].extend([0] * (max_tree_size - fix_nodes_len))

        result['fix_children'] = [[x[1] for x in patch['jsgraph2']['graph'] if x[0] == i] for i in
                                  range(patch['graph_size2'])]
        [x.extend([0] * (max_children_size - len(x))) for x in result['fix_children']]
        result['fix_children'].extend(
            [[0] * max_children_size for _ in range(max_tree_size - fix_nodes_len)])
        result = {k: torch.tensor(result[k]) for k in result.keys()}
        result['fix_children_nodes'] = children_tensor(result['fix_nodes'], result['fix_children'])
        result['bug_children_nodes'] = children_tensor(result['bug_nodes'], result['bug_children'])
        return result

    def __len__(self):
        return len(self.data)


class TreeDataset(Dataset):
    def __init__(self, dir, mode="train"):
        self.dir = dir
        self.mode = mode

    def __getitem__(self, index):
        name = "{}/{}{}.pkl".format(self.dir, self.mode, index)
        f = open(name, "rb")
        data = pickle.loads(f.read())
        f.close()
        return data

    def __len__(self):
        l = 0
        for file in os.listdir(self.dir):
            if re.match(self.mode + "[0-9]+\\.pkl", file) is not None:
                l += 1
        return l


def children_tensor(nodes, children):
    num_nodes = nodes.shape[0]
    num_children = children.shape[1]
    placeholder = torch.zeros((num_nodes, num_children), dtype=torch.int32)
    for j in range(num_nodes):
        index = children[j]
        t = nodes.index_select(0, index)
        placeholder[j] = t
    return placeholder
