import torch
from torch import nn

from tbcnn.sublayer import ConvolutionLayer, PoolingLayer
import torch.nn.functional as F

from tbcnn.vocab_dict import VocabDict


class TBCnnModel(nn.Module):
    def __init__(self, config):
        super(TBCnnModel, self).__init__()
        self.config = config
        # vocab dict
        self.vocab_dict = VocabDict(
            file_name=config['vocabulary_dictionary_path'],
            name="JavaTokenVocabDictionary")
        self.vocab_dict.load()
        # Embedding
        self.node_emb_layer = nn.Embedding(num_embeddings=len(self.vocab_dict), embedding_dim=config['embedding_dim'],
                                           padding_idx=0)
        self.conv_layer1 = ConvolutionLayer(config)
        self.conv_layer2 = ConvolutionLayer(config)
        self.pooling_layer1 = PoolingLayer()
        self.pooling_layer2 = PoolingLayer()
        self.fforward = nn.Linear(self.config['conv_output'] * self.config['conv_layer_num'] * 2,
                                  self.config['class_num'])
        self.loss = nn.CrossEntropyLoss()

    def forward(self, bug_nodes, bug_children, bug_children_nodes, fix_nodes, fix_children, fix_children_nodes, label):
        bug_nodes_embedding = self.node_emb_layer(bug_nodes)
        fix_nodes_embedding = self.node_emb_layer(fix_nodes)
        bug_children_embedding = self.node_emb_layer(bug_children_nodes)
        fix_children_embedding = self.node_emb_layer(fix_children_nodes)

        bug_conv_result = self.conv_layer1(bug_nodes_embedding, bug_children,bug_children_embedding)
        fix_conv_result = self.conv_layer2(fix_nodes_embedding, fix_children,fix_children_embedding)

        bug_pooling_result = self.pooling_layer1(bug_conv_result)
        fix_pooling_result = self.pooling_layer2(fix_conv_result)
        concat_result = torch.cat((bug_pooling_result, fix_pooling_result), dim=1)
        fforward_result = self.fforward(concat_result)
        dense_output = F.leaky_relu(fforward_result)
        mask_output = F.softmax(dense_output, dim=-1)
        loss = self.loss(dense_output, label)
        return mask_output, loss, label
