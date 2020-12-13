import torch
from torch.utils.data import Dataset

from torchtext import data

# DataLoader class

# TokenizerWrapper class
class TokenizerWrapper():

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def collate(self, samples):
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length
        )

        return {
            'text': texts,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long)
        }

class BertDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        return {
            'text': text,
            'label': label,
        }