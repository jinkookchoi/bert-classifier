import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data

from transformers import AutoTokenizer
from transformers import BertForSequenceClassification

from sklearn.metrics import confusion_matrix, classification_report 

def define_argparse():

    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=32)

    config = p.parse_args()

    return config
    

def read_text():

    lines = []
    y_trues = []

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split('\t')[1]]
            y_trues += [line.strip().split('\t')[0]]

    return lines, y_trues
    

def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else f'cuda:{config.gpu_id}',
    )

    train_config = saved_data['config']    
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    lines, y_trues = read_text()

    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(train_config.pretrained_model_name)
        model = BertForSequenceClassification.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
        model.load_state_dict(bert_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        model.eval()
        
        y_hats = []
        for idx in range(0, len(lines), config.batch_size):
            mini_batch = tokenizer(
                lines[idx:idx+config.batch_size],
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask']
            mask = mask.to(device)
            
            y_hat = F.softmax(model(x, attention_mask=mask)[0], dim=-1)
            y_hats += [y_hat]

        y_hats = torch.cat(y_hats, dim=0)

        probs, indices = y_hats.cpu().topk(1)
        y_hat_labels = [index_to_label[idx] for idx in indices.squeeze().numpy()]

        print(classification_report(y_trues, y_hat_labels, digits=4))

if __name__ == "__main__":
    config = define_argparse()
    main(config)
