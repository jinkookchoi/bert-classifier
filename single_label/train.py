import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import torch_optimizer as custom_optim

from bert_trainer import BertTrainer as Trainer
from data_loader import BertDataset, TokenizerWrapper

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--pretrained_model_name', type=str, default='beomi/kcbert-base')

    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--n_epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)

    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--warmup_ratio', type=float, default=.1)
    p.add_argument('--adam_epsilon', type=float, default=1e-8)
    p.add_argument('--use_radam', action='store_true')

    p.add_argument('--max_length', type=int, default=100)

    p.add_argument('--train_ratio', type=float, default=.8)

    config = p.parse_args()

    return config


def read_text(fn):
    with open(fn, 'r', encoding='utf-8') as f:
        lines = f.readlines()

        labels, texts = [], []
        for line in lines:
            if line.strip() != '':
                label, text = line.strip().split('\t')
                labels += [label]
                texts += [text]

    return labels, texts
    

def get_loaders(fn, tokenizer, train_ratio):
    # Get list of labels and list of texts
    labels, texts = read_text(fn)

    # Generate label to index map.
    uinique_labels = list(set(labels))
    label_to_index = {}
    index_to_label = {}
    for i, label in enumerate(uinique_labels):
        label_to_index[label] = i
        index_to_label[i] = label

    # Convert label text to integer value.
    labels = list(map(label_to_index.get, labels)) 

    # Shuffle before split into train and validation set.
    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * train_ratio)

    # Get dataloaders using given tokenizer as collate_fn.
    train_loader = DataLoader(
       BertDataset(texts[:idx], labels[:idx]),
       batch_size=config.batch_size,
       shuffle=True,
       collate_fn=TokenizerWrapper(tokenizer, config.max_length).collate,
    )

    valid_loader = DataLoader(
       BertDataset(texts[idx:], labels[idx:]),
       batch_size=config.batch_size,
       collate_fn=TokenizerWrapper(tokenizer, config.max_length).collate,
    )

    return train_loader, valid_loader, index_to_label


def main(config):
    # Get pretrained tokenizer.
    tokenizer  = AutoTokenizer.from_pretrained(config.pretrained_model_name)

    # Get dataloaders using tokenizer from untokenized corpus.
    train_loader, valid_loader, index_to_label = get_loaders(config.train_fn, tokenizer, config.train_ratio)
    
    print(
        '|train|=', len(train_loader) * config.batch_size,
        '|valid|=', len(valid_loader) * config.batch_size,
    )

    # Get pretrained model with specific softmax layer.
    model = BertForSequenceClassification.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(index_to_label)
    )

    if config.use_radam: 
        optimizer = custom_optim.RAdam(model.parameters(), lr=config.lr) 
    else:
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ] 

        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.lr,
            eps=config.adam_epsilon
        )

    crit = nn.CrossEntropyLoss()
    
    n_total_iterations = len(train_loader) * config.n_epochs
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=n_warmup_steps,
        num_training_steps=n_total_iterations
    ) 

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)    

    trainer = Trainer(config)
    model = trainer.train(
        model=model,
        crit=crit,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader
    )

    torch.save({
        'bert': model.state_dict(),
        'config': config,
        'classes': index_to_label,
        'tokenizer': tokenizer
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)