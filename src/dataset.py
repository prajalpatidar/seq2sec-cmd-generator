"""
Dataset utilities for loading and preprocessing the command dataset.
"""

import json
import torch
from torch.utils.data import Dataset
from collections import Counter


class Vocabulary:
    """
    Vocabulary class for managing token-to-index and index-to-token mappings.
    """
    def __init__(self):
        self.token2idx = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2token = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.token_count = Counter()
        
    def add_token(self, token):
        """Add a token to the vocabulary."""
        self.token_count[token] += 1
        if token not in self.token2idx:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token
    
    def add_sentence(self, sentence):
        """Add all tokens in a sentence to the vocabulary."""
        for token in sentence.split():
            self.add_token(token)
    
    def __len__(self):
        return len(self.token2idx)
    
    def encode(self, sentence, add_sos=False, add_eos=False):
        """
        Convert sentence to list of indices.
        
        Args:
            sentence: string to encode
            add_sos: whether to add <SOS> token at beginning
            add_eos: whether to add <EOS> token at end
        Returns:
            list of token indices
        """
        tokens = sentence.split()
        indices = []
        
        if add_sos:
            indices.append(self.token2idx['<SOS>'])
        
        for token in tokens:
            indices.append(self.token2idx.get(token, self.token2idx['<UNK>']))
        
        if add_eos:
            indices.append(self.token2idx['<EOS>'])
        
        return indices
    
    def decode(self, indices, remove_special=True):
        """
        Convert list of indices back to sentence.
        
        Args:
            indices: list of token indices or tensor
            remove_special: whether to remove special tokens
        Returns:
            decoded sentence string
        """
        if torch.is_tensor(indices):
            indices = indices.tolist()
        
        tokens = []
        for idx in indices:
            if idx in self.idx2token:
                token = self.idx2token[idx]
                if remove_special and token in ['<PAD>', '<SOS>', '<EOS>']:
                    continue
                if token == '<EOS>':
                    break
                tokens.append(token)
        
        return ' '.join(tokens)
    
    def save(self, filepath):
        """Save vocabulary to file."""
        data = {
            'token2idx': self.token2idx,
            'idx2token': {int(k): v for k, v in self.idx2token.items()},
            'token_count': dict(self.token_count)
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """Load vocabulary from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        vocab = cls()
        vocab.token2idx = data['token2idx']
        vocab.idx2token = {int(k): v for k, v in data['idx2token'].items()}
        vocab.token_count = Counter(data['token_count'])
        return vocab


class CommandDataset(Dataset):
    """
    Dataset class for Natural Language to Linux Command pairs.
    """
    def __init__(self, data_path, input_vocab=None, output_vocab=None, build_vocab=False):
        """
        Args:
            data_path: path to JSON dataset file
            input_vocab: Vocabulary object for input (natural language)
            output_vocab: Vocabulary object for output (commands)
            build_vocab: whether to build vocabularies from data
        """
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        if build_vocab:
            self.input_vocab = Vocabulary()
            self.output_vocab = Vocabulary()
            
            for item in self.data:
                self.input_vocab.add_sentence(item['input'].lower())
                self.output_vocab.add_sentence(item['output'])
        else:
            self.input_vocab = input_vocab
            self.output_vocab = output_vocab
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            input_seq: encoded input sequence
            output_seq: encoded output sequence with <SOS> and <EOS>
            input_text: original input text
            output_text: original output text
        """
        item = self.data[idx]
        input_text = item['input'].lower()
        output_text = item['output']
        
        input_seq = self.input_vocab.encode(input_text)
        output_seq = self.output_vocab.encode(output_text, add_sos=True, add_eos=True)
        
        return {
            'input_seq': torch.tensor(input_seq, dtype=torch.long),
            'output_seq': torch.tensor(output_seq, dtype=torch.long),
            'input_text': input_text,
            'output_text': output_text,
            'input_len': len(input_seq),
            'output_len': len(output_seq)
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable length sequences.
    Pads sequences to the maximum length in the batch.
    """
    input_seqs = [item['input_seq'] for item in batch]
    output_seqs = [item['output_seq'] for item in batch]
    input_texts = [item['input_text'] for item in batch]
    output_texts = [item['output_text'] for item in batch]
    input_lens = torch.tensor([item['input_len'] for item in batch], dtype=torch.long)
    output_lens = torch.tensor([item['output_len'] for item in batch], dtype=torch.long)
    
    # Pad sequences
    input_seqs_padded = torch.nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=0)
    output_seqs_padded = torch.nn.utils.rnn.pad_sequence(output_seqs, batch_first=True, padding_value=0)
    
    return {
        'input_seq': input_seqs_padded,
        'output_seq': output_seqs_padded,
        'input_text': input_texts,
        'output_text': output_texts,
        'input_len': input_lens,
        'output_len': output_lens
    }
