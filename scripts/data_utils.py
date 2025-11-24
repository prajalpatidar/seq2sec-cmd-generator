"""
Data preprocessing and tokenization utilities.
"""

import re
from collections import Counter
from typing import List, Tuple, Dict
import pickle


class Tokenizer:
    """Simple character-level and word-level tokenizer."""
    
    def __init__(self, level='word'):
        """
        Args:
            level: 'word' or 'char' level tokenization
        """
        self.level = level
        self.vocab = {}
        self.reverse_vocab = {}
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        
        # Reserve special tokens
        self.special_tokens = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]
        
    def fit(self, texts: List[str], max_vocab_size: int = 5000):
        """
        Build vocabulary from texts.
        
        Args:
            texts: List of text strings
            max_vocab_size: Maximum vocabulary size
        """
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            all_tokens.extend(self._tokenize(text))
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Build vocabulary with special tokens first
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens)}
        
        # Add most common tokens
        most_common = token_counts.most_common(max_vocab_size - len(self.special_tokens))
        for token, _ in most_common:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        # Build reverse vocabulary
        self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}
        
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text based on level.
        
        Args:
            text: Input text string
        Returns:
            List of tokens
        """
        text = text.lower().strip()
        
        if self.level == 'char':
            return list(text)
        else:  # word level
            # Simple word tokenization
            tokens = re.findall(r'\w+|[^\w\s]', text)
            return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token indices.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add START and END tokens
        Returns:
            List of token indices
        """
        tokens = self._tokenize(text)
        indices = []
        
        if add_special_tokens:
            indices.append(self.vocab[self.START_TOKEN])
        
        for token in tokens:
            indices.append(self.vocab.get(token, self.vocab[self.UNK_TOKEN]))
        
        if add_special_tokens:
            indices.append(self.vocab[self.END_TOKEN])
        
        return indices
    
    def decode(self, indices: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token indices to text.
        
        Args:
            indices: List of token indices
            skip_special_tokens: Whether to skip special tokens
        Returns:
            Decoded text string
        """
        tokens = []
        for idx in indices:
            if idx in self.reverse_vocab:
                token = self.reverse_vocab[idx]
                if skip_special_tokens and token in self.special_tokens:
                    continue
                tokens.append(token)
        
        if self.level == 'char':
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def save(self, filepath: str):
        """Save tokenizer to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'level': self.level,
                'vocab': self.vocab,
                'reverse_vocab': self.reverse_vocab,
                'special_tokens': self.special_tokens
            }, f)
    
    def load(self, filepath: str):
        """Load tokenizer from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.level = data['level']
            self.vocab = data['vocab']
            self.reverse_vocab = data['reverse_vocab']
            self.special_tokens = data['special_tokens']
    
    def __len__(self):
        return len(self.vocab)


class Dataset:
    """Dataset class for sequence-to-sequence data."""
    
    def __init__(self, input_texts: List[str], output_texts: List[str], 
                 input_tokenizer: Tokenizer, output_tokenizer: Tokenizer):
        """
        Args:
            input_texts: List of input text strings
            output_texts: List of output text strings
            input_tokenizer: Tokenizer for input texts
            output_tokenizer: Tokenizer for output texts
        """
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.input_tokenizer = input_tokenizer
        self.output_tokenizer = output_tokenizer
        
    def __len__(self):
        return len(self.input_texts)
    
    def __getitem__(self, idx):
        """
        Get a single data sample.
        
        Returns:
            Tuple of (input_indices, output_indices)
        """
        input_indices = self.input_tokenizer.encode(self.input_texts[idx])
        output_indices = self.output_tokenizer.encode(self.output_texts[idx])
        return input_indices, output_indices


def collate_fn(batch: List[Tuple[List[int], List[int]]], pad_idx: int = 0):
    """
    Collate function for DataLoader.
    Pads sequences to the same length in a batch.
    
    Args:
        batch: List of (input_indices, output_indices) tuples
        pad_idx: Padding index
    Returns:
        Tuple of (input_tensor, output_tensor)
    """
    import torch
    
    inputs, outputs = zip(*batch)
    
    # Find max lengths
    max_input_len = max(len(seq) for seq in inputs)
    max_output_len = max(len(seq) for seq in outputs)
    
    # Pad sequences
    padded_inputs = []
    padded_outputs = []
    
    for inp, out in zip(inputs, outputs):
        padded_inputs.append(inp + [pad_idx] * (max_input_len - len(inp)))
        padded_outputs.append(out + [pad_idx] * (max_output_len - len(out)))
    
    return torch.tensor(padded_inputs), torch.tensor(padded_outputs)
