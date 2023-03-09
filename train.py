import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader

import cuda_utils
from dataset import collate, TreeDataset
from model import TBCnnModel

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

config = {
    'max_tree_size': 3000,
    'max_children_size': 200,
    'conv_layer_num': 1,
    'conv_output': 128,
    'class_num': 2,
    'max_epoch': 50,
    'batch_size': 4,
    'embedding_dim': 128,
    'embedding_type': "structure",
    'lr': 0.001,
    'use_cuda': torch.cuda.is_available(),
    'node_vocabulary_dictionary_path': "./data/node_vocab_dict.pkl",
    'token_vocabulary_dictionary_path': "./data/token_vocab_dict.pkl",
    'max_token_length': 8,
}


def pretty_format(metrics: dict,
                  metric_headers=('accuracy', 'recall', 'precision', 'f1', 'auc', 'TP', 'FP', 'TN', 'FN')):
    return '|'.join(["{}: {}".format(x, metrics[x] if type(metrics[x])==int else round(metrics[x],4)) for x in metric_headers])


def evaluate_metrics(preds, labels):
    """
            Get binary score e.g., accuracy, f1, precision and recall
            :return:
            """
    metrics = dict()
    acc = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    prec = precision_score(labels, preds)
    auc = roc_auc_score(labels, preds)
    f1 = f1_score(labels, preds)

    metrics['accuracy'] = acc
    metrics['recall'] = recall
    metrics['precision'] = prec
    metrics['f1'] = f1
    metrics['auc'] = auc
    TP, FP, TN, FN, i = 0, 0, 0, 0, 0
    for i in range(len(preds)):
        if preds[i] == 1 and labels[i] == 1:
            TP += 1
        elif preds[i] == 1 and labels[i] == 0:
            FP += 1
        elif preds[i] == 0 and labels[i] == 0:
            TN += 1
        elif preds[i] == 0 and labels[i] == 1:
            FN += 1
    metrics['TP'], metrics['FP'], metrics['TN'], metrics['FN'] = TP, FP, TN, FN

    return metrics


if __name__ == '__main__':

    model_save_path = "./trained_models/model.pth"
    train_data_file_path = "./train"
    val_data_file_path = "./val"
    train_dataset = TreeDataset(dir=train_data_file_path)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True,
                                  collate_fn=collate)
    val_dataset = TreeDataset(dir=val_data_file_path, mode="val")
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True,
                                collate_fn=collate)

    model = TBCnnModel(config=config)
    print("Params:")
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    print("\n")
    if config['use_cuda']:
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    # total = sum([param.nelement() for param in model.parameters()])
    # print(total)

    print("Configuration:")
    for c in config:
        print("{}:{}".format(c, config[c]))

    best_val_acc = 0
    for epoch_num in range(config['max_epoch']):
        print("-------------------------- epoch {} --------------------------".format(epoch_num + 1))
        train_preds_all = []
        train_labels_all = []
        train_loss_all = []
        val_preds_all = []
        val_labels_all = []
        val_loss_all = []
        for idx, batch_dict in enumerate(train_dataloader):
            batch_dict = tuple([torch.stack(x, dim=0) for x in batch_dict])
            batch_dict = tuple([cuda_utils.to_cuda(x, use_cuda=config['use_cuda']) for x in batch_dict])
            model.train()
            optimizer.zero_grad()
            train_probs, train_loss, train_labels = model(*batch_dict)
            train_loss_all.append(train_loss.item())
            # print(train_loss)
            train_best_preds = np.asarray([np.argmax(line) for line in train_probs.cpu().tolist()])
            train_preds_all.extend(train_best_preds)
            train_labels_all.extend(train_labels.cpu().tolist())
            # backward
            train_loss.backward()
            optimizer.step()

        train_metrics = evaluate_metrics(train_preds_all, train_labels_all)
        train_metrics["loss"] = sum(train_loss_all) / len(train_loss_all)
        print("train evaluation: ")
        print(pretty_format(train_metrics))

        for idx, batch_dict in enumerate(val_dataloader):
            batch_dict = tuple([torch.stack(x, dim=0) for x in batch_dict])
            batch_dict = tuple([cuda_utils.to_cuda(x, use_cuda=config['use_cuda']) for x in batch_dict])
            # 关梯度
            with torch.no_grad():
                model.eval()
                val_probs, val_loss, val_labels = model(*batch_dict)
                val_loss_all.append(val_loss.item())
                val_best_preds = np.asarray([np.argmax(line) for line in val_probs.cpu().tolist()])
                val_preds_all.extend(val_best_preds)
                val_labels_all.extend(val_labels.cpu().tolist())

        val_metrics = evaluate_metrics(val_preds_all, val_labels_all)
        val_metrics["loss"] = sum(val_loss_all) / len(val_loss_all)
        print("val evaluation: ")
        if best_val_acc <= val_metrics['accuracy']:
            print("------------------------> best accuracy from {} to {} <------------------------".format(best_val_acc,
                                                                                                           val_metrics[
                                                                                                               'accuracy']))
            best_val_acc = val_metrics['accuracy']
            torch.save(model, model_save_path)
        print(pretty_format(val_metrics))
