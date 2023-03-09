import torch
from torch import nn

from sublayer import ConvolutionLayer, MaxPoolingLayer, MeanPoolingLayer
import torch.nn.functional as F

from vocab_dict import VocabDict


class TBCnnModel(nn.Module):
    def __init__(self, config):
        super(TBCnnModel, self).__init__()
        self.config = config
        # vocab dict
        self.node_vocab_dict = VocabDict(
            file_name=config['node_vocabulary_dictionary_path'],
            name="JavaTokenVocabDictionary")
        self.node_vocab_dict.load()
        self.token_vocab_dict = VocabDict(
            file_name=config['token_vocabulary_dictionary_path'],
            name="JavaTokenVocabDictionary")
        self.token_vocab_dict.load()
        # Embedding
        if self.config['embedding_type'] == "structure":
            self.node_emb_layer = nn.Embedding(num_embeddings=len(self.node_vocab_dict),
                                               embedding_dim=config['embedding_dim'],
                                               padding_idx=0)
        elif self.config['embedding_type'] == "textual":
            self.node_emb_layer = nn.Embedding(num_embeddings=len(self.token_vocab_dict),
                                               embedding_dim=config['embedding_dim'],
                                               padding_idx=0)
        elif self.config['embedding_type'] == "both":
            self.node_emb_layer = nn.Embedding(num_embeddings=len(self.token_vocab_dict) + len(self.node_vocab_dict),
                                               embedding_dim=config['embedding_dim'],
                                               padding_idx=0)

        if self.config['embedding_type'] == "textual" or self.config['embedding_type'] == "both":
            self.node_emb_pooling_layer = MeanPoolingLayer()
        self.conv_layer1 = ConvolutionLayer(config)
        self.conv_layer2 = ConvolutionLayer(config)
        self.pooling_layer1 = MaxPoolingLayer()
        self.pooling_layer2 = MaxPoolingLayer()
        self.fforward = nn.Linear(self.config['conv_output'] * self.config['conv_layer_num'] * 2,
                                  self.config['class_num'])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, bug_nodes, bug_children, bug_children_nodes, fix_nodes, fix_children, fix_children_nodes, label):
        bug_nodes_embedding = self.node_emb_layer(bug_nodes)
        fix_nodes_embedding = self.node_emb_layer(fix_nodes)
        bug_children_embedding = self.node_emb_layer(bug_children_nodes.int())
        fix_children_embedding = self.node_emb_layer(fix_children_nodes)
        if self.config['embedding_type'] == "textual" or self.config['embedding_type'] == "both":
            bug_nodes_embedding = self.node_emb_pooling_layer(bug_nodes_embedding)
            fix_nodes_embedding = self.node_emb_pooling_layer(fix_nodes_embedding)
            bug_children_embedding = self.node_emb_pooling_layer(bug_children_embedding)
            fix_children_embedding = self.node_emb_pooling_layer(fix_children_embedding)

        bug_conv_result = self.conv_layer1(bug_nodes_embedding, bug_children, bug_children_embedding)
        fix_conv_result = self.conv_layer2(fix_nodes_embedding, fix_children, fix_children_embedding)

        bug_pooling_result = self.pooling_layer1(bug_conv_result)
        fix_pooling_result = self.pooling_layer2(fix_conv_result)
        concat_result = torch.cat((bug_pooling_result, fix_pooling_result), dim=1)
        fforward_result = self.fforward(concat_result)
        dense_output = F.leaky_relu(fforward_result)
        mask_output = F.softmax(dense_output, dim=-1)
        loss = self.loss(dense_output, label)
        return mask_output, loss, label
