"""
dataset的格式转换
"""
import pickle

import torch
from tqdm import tqdm

from tbcnn.dataset import RawTreeDataset

config = {
    'max_tree_size': 5000,
    'max_children_size': 200,
    'conv_layer_num': 1,
    'conv_output': 128,
    'class_num': 2,
    'max_epoch': 50,
    'batch_size': 16,
    'embedding_dim': 128,
    'lr': 0.001,
    'use_cuda': torch.cuda.is_available(),
    'vocabulary_dictionary_path': "./data/node_vocab_dict.pkl"
}

train_data_file_path = "./data/train.pkl"
val_data_file_path = "./data/val.pkl"
train_data_save_path = "./train/train{}.pkl"
val_data_save_path = "./val/val{}.pkl"
train_data_file = open(train_data_file_path, 'rb')
train_data = pickle.loads(train_data_file.read())
val_data_file = open(val_data_file_path, 'rb')
val_data = pickle.loads(val_data_file.read())
train_dataset = RawTreeDataset(data=train_data, config=config)
val_dataset = RawTreeDataset(data=val_data, config=config)
train_dataset_length = len(train_dataset)
val_dataset_length = len(val_dataset)

for idx in tqdm(range(train_dataset_length)):
    one = train_dataset[idx]
    f1 = open(train_data_save_path.format(idx), 'wb')
    pickle.dump(one, f1)

for idx in tqdm(range(val_dataset_length)):
    one = val_dataset[idx]
    f2 = open(val_data_save_path.format(idx), 'wb')
    pickle.dump(one, f2)

