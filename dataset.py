import os
import pickle
import re

import torch
from torch.utils.data import Dataset

from code_tokenizer import CodeTokenizer
from vocab_dict import VocabDict


def collate(batch):
    return [x['bug_nodes'] for x in batch], [x['bug_children'] for x in batch], [x['bug_children_nodes'] for x in
                                                                                 batch], [x['fix_nodes'] for x in
                                                                                          batch], [
               x['fix_children'] for x in batch], [x['fix_children_nodes'] for x in batch], [x['tgt'] for x in batch]


class RawTreeDataset(Dataset):
    def __init__(self, data=None, config=None):
        self.node_vocab_dict = VocabDict(
            file_name=config['node_vocabulary_dictionary_path'],
            name="JavaTokenVocabDictionary")
        self.node_vocab_dict.load()
        self.token_vocab_dict = VocabDict(
            file_name=config['token_vocabulary_dictionary_path'],
            name="JavaTokenVocabDictionary")
        self.token_vocab_dict.load()
        self.data = data if data is not None else []
        self.config = config if config is not None else dict()
        self.tokenizer = CodeTokenizer(data=[], lang="C", tlevel='t3')

    def __getitem__(self, index):
        max_tree_size = self.config['max_tree_size']
        max_children_size = self.config['max_children_size']

        result = dict()
        patch = self.data[index]
        result['tgt'] = int(patch['target'])

        if self.config['embedding_type'] == "textual" or self.config['embedding_type'] == "both":
            result['bug_nodes'] = [
                [self.token_vocab_dict.get_w2i(y) for y in
                 self.tokenizer.tokenize(x[1] if len(x[1]) > 0 else "<EMPTY_CODE>").split(" ")]
                for x in patch['jsgraph1']['node_features'].values()]
            result['bug_nodes'] = [
                x[:self.config['max_token_length']] if len(x) >= self.config['max_token_length'] else (
                        x + [self.token_vocab_dict.get_w2i("EOS")] * (self.config['max_token_length'] - len(x)))
                for x in result['bug_nodes']]
            result['fix_nodes'] = [
                [self.token_vocab_dict.get_w2i(y) for y in
                 self.tokenizer.tokenize(x[1] if len(x[1]) > 0 else "<EMPTY_CODE>").split(" ")]
                for x in patch['jsgraph2']['node_features'].values()]
            result['fix_nodes'] = [
                x[:self.config['max_token_length']] if len(x) >= self.config['max_token_length'] else (
                        x + [self.token_vocab_dict.get_w2i("EOS")] * (self.config['max_token_length'] - len(x)))
                for x in result['fix_nodes']]
        elif self.config['embedding_type'] == "structure":
            result['bug_nodes'] = [self.node_vocab_dict.get_w2i(x[0]) for x in
                                   patch['jsgraph1']['node_features'].values()]
            result['fix_nodes'] = [self.node_vocab_dict.get_w2i(x[0]) for x in
                                   patch['jsgraph2']['node_features'].values()]
        if self.config['embedding_type'] == "both":
            for i in range(len(result['bug_nodes'])):
                result['bug_nodes'][i][-1] = self.node_vocab_dict.get_w2i(
                    patch['jsgraph1']['node_features'][i][0])
            for i in range(len(result['fix_nodes'])):
                result['fix_nodes'][i][-1] = self.node_vocab_dict.get_w2i(
                    patch['jsgraph2']['node_features'][i][0])

        bug_nodes_len = len(result['bug_nodes'])
        if self.config['embedding_type'] == "textual" or self.config['embedding_type'] == "both":
            if bug_nodes_len < max_tree_size:
                result['bug_nodes'].extend(
                    [[0] * self.config['max_token_length'] for _ in range(max_tree_size - bug_nodes_len)])
            else:
                # truncate
                result['bug_nodes'] = result['bug_nodes'][:max_tree_size]

        elif self.config['embedding_type'] == "structure":
            if bug_nodes_len < max_tree_size:
                result['bug_nodes'].extend([0] * (max_tree_size - bug_nodes_len))
            else:
                # truncate
                result['bug_nodes'] = result['bug_nodes'][:max_tree_size]

        result['bug_children'] = [[x[1] for x in patch['jsgraph1']['graph'] if x[0] == i] for i in
                                  range(patch['graph_size1'])]
        for i in range(len(result['bug_children'])):
            if max_children_size > len(result['bug_children'][i]):
                result['bug_children'][i].extend([0] * (max_children_size - len(result['bug_children'][i])))
            else:
                result['bug_children'][i] = result['bug_children'][i][:max_children_size]
        # [x.extend([0] * (max_children_size - len(x))) for x in result['bug_children']]
        if bug_nodes_len < max_tree_size:
            result['bug_children'].extend(
                [[0] * max_children_size for _ in range(max_tree_size - bug_nodes_len)])
        else:
            result['bug_children'] = result['bug_children'][:max_tree_size]

        fix_nodes_len = len(result['fix_nodes'])
        if self.config['embedding_type'] == "textual" or self.config['embedding_type'] == "both":
            if fix_nodes_len < max_tree_size:
                result['fix_nodes'].extend(
                    [[0] * self.config['max_token_length'] for _ in range(max_tree_size - fix_nodes_len)])
            else:
                # truncate
                result['fix_nodes'] = result['fix_nodes'][:max_tree_size]
        elif self.config['embedding_type'] == "structure":
            if fix_nodes_len < max_tree_size:
                result['fix_nodes'].extend([0] * (max_tree_size - fix_nodes_len))
            else:
                # truncate
                result['fix_nodes'] = result['fix_nodes'][:max_tree_size]

        result['fix_children'] = [[x[1] for x in patch['jsgraph2']['graph'] if x[0] == i] for i in
                                  range(patch['graph_size2'])]

        for i in range(len(result['fix_children'])):
            if max_children_size > len(result['fix_children'][i]):
                result['fix_children'][i].extend([0] * (max_children_size - len(result['fix_children'][i])))
            else:
                result['fix_children'][i] = result['fix_children'][i][:max_children_size]
        # [x.extend([0] * (max_children_size - len(x))) for x in result['fix_children']]
        if fix_nodes_len < max_tree_size:
            result['fix_children'].extend(
                [[0] * max_children_size for _ in range(max_tree_size - fix_nodes_len)])
        else:
            result['fix_children'] = result['fix_children'][:max_tree_size]

        result = {k: torch.tensor(result[k]) for k in result.keys()}
        result['fix_children_nodes'] = children_tensor(result['fix_nodes'], result['fix_children'],
                                                       third_dimension=None if self.config[
                                                                                   'embedding_type'] == "structure" else
                                                       self.config['max_token_length'])
        result['bug_children_nodes'] = children_tensor(result['bug_nodes'], result['bug_children'],
                                                       third_dimension=None if self.config[
                                                                                   'embedding_type'] == "structure" else
                                                       self.config['max_token_length'])
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


def children_tensor(nodes, children, third_dimension=None):
    num_nodes = nodes.shape[0]
    num_children = children.shape[1]
    if third_dimension is None:
        placeholder = torch.zeros((num_nodes, num_children), dtype=torch.int32)
    else:
        placeholder = torch.zeros((num_nodes, num_children, third_dimension), dtype=torch.int32)
    shape = [1] + list(nodes.shape)[1:]
    new_nodes = torch.cat((torch.zeros(shape, dtype=torch.int32), nodes), dim=0)

    for j in range(num_nodes):
        index = children[j].tolist()
        index = torch.tensor([x + 1 if x < num_nodes else 0 for x in index])

        t = new_nodes.index_select(0, index)
        placeholder[j] = t
    return placeholder
