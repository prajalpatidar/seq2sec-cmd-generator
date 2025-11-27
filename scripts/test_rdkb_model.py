#!/usr/bin/env python3
"""Test the model with RDKB dmcli commands."""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.seq2seq import Seq2SeqModel
from scripts.data_utils import Tokenizer

def main():
    """Test model with RDKB commands."""
    
    # Load tokenizers
    input_tok = Tokenizer()
    output_tok = Tokenizer()
    input_tok.load(os.path.join('models', 'checkpoints', 'input_tokenizer.pkl'))
    output_tok.load(os.path.join('models', 'checkpoints', 'output_tokenizer.pkl'))
    
    # Load model
    checkpoint = torch.load('models/checkpoints/best_model.pth', weights_only=False, map_location='cpu')
    config = checkpoint['config']
    
    model = Seq2SeqModel(
        input_vocab_size=len(input_tok),
        output_vocab_size=len(output_tok),
        **config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: {config}")
    print(f"Vocabulary: Input={len(input_tok)}, Output={len(output_tok)}")
    print(f"Best Val Loss: {checkpoint.get('val_loss', 'N/A')}")
    print()
    
    # Test RDKB dmcli commands
    rdkb_tests = [
        'get device model name',
        'show wifi ssid',
        'display wifi password',
        'get lan ip address',
        'show dhcp clients',
        'enable wifi radio',
        'set wifi channel',
        'get cm status',
        'show docsis version',
        'enable moca',
        'factory reset device',
        'reboot device',
        'get firmware version',
        'show parental control status',
        'display acs url',
        'get ipv6 enable',
        'show dns servers',
        'enable bridge mode',
        'get qos enable',
        'show guest wifi status',
    ]
    
    # Also test regular Linux commands
    linux_tests = [
        'show network interfaces',
        'list all files',
        'display memory usage',
        'get system information',
    ]
    
    print("="*80)
    print("RDKB dmcli Command Tests (val loss 1.35):")
    print("="*80)
    print()
    
    for text in rdkb_tests:
        tokens = input_tok.encode(text)
        src = torch.tensor([tokens]).long()
        pred = model.predict(src, max_len=50, repetition_penalty=1.5)
        output = output_tok.decode(pred)
        print(f"{text:45s} → {output}")
    
    print()
    print("="*80)
    print("Regular Linux Command Tests:")
    print("="*80)
    print()
    
    for text in linux_tests:
        tokens = input_tok.encode(text)
        src = torch.tensor([tokens]).long()
        pred = model.predict(src, max_len=50, repetition_penalty=1.5)
        output = output_tok.decode(pred)
        print(f"{text:45s} → {output}")

if __name__ == "__main__":
    os.chdir('/home/prajal/work/seq2seq-model/seq2sec-cmd-generator')
    main()
