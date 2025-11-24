"""
Training script for the sequence-to-sequence model.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import yaml

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.seq2seq import Seq2SeqModel
from scripts.data_utils import Tokenizer, Dataset, collate_fn


def load_data(data_path: str, train_split: float = 0.8):
    """
    Load and split dataset.
    
    Args:
        data_path: Path to CSV file
        train_split: Fraction of data to use for training
    Returns:
        Tuple of (train_inputs, train_outputs, val_inputs, val_outputs)
    """
    df = pd.read_csv(data_path)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split data
    split_idx = int(len(df) * train_split)
    train_df = df[:split_idx]
    val_df = df[split_idx:]
    
    return (
        train_df['input'].tolist(),
        train_df['output'].tolist(),
        val_df['input'].tolist(),
        val_df['output'].tolist()
    )


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    """
    Train for one epoch.
    
    Args:
        model: Seq2SeqModel
        dataloader: DataLoader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        teacher_forcing_ratio: Teacher forcing ratio
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for src, tgt in tqdm(dataloader, desc="Training"):
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, tgt, teacher_forcing_ratio)
        
        # Reshape for loss calculation
        output = output[:, 1:].reshape(-1, output.size(-1))
        tgt = tgt[:, 1:].reshape(-1)
        
        # Calculate loss
        loss = criterion(output, tgt)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation set.
    
    Args:
        model: Seq2SeqModel
        dataloader: DataLoader
        criterion: Loss function
        device: Device to evaluate on
    Returns:
        Average loss for the validation set
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)
            
            # Forward pass (no teacher forcing during evaluation)
            output = model(src, tgt, teacher_forcing_ratio=0)
            
            # Reshape for loss calculation
            output = output[:, 1:].reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = criterion(output, tgt)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    """Main training function."""
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'train_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'commands_dataset.csv')
    train_inputs, train_outputs, val_inputs, val_outputs = load_data(
        data_path, 
        config['train_split']
    )
    
    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(val_inputs)}")
    
    # Create tokenizers
    input_tokenizer = Tokenizer(level='word')
    output_tokenizer = Tokenizer(level='word')
    
    # Fit tokenizers
    input_tokenizer.fit(train_inputs + val_inputs, max_vocab_size=config['max_vocab_size'])
    output_tokenizer.fit(train_outputs + val_outputs, max_vocab_size=config['max_vocab_size'])
    
    print(f"Input vocabulary size: {len(input_tokenizer)}")
    print(f"Output vocabulary size: {len(output_tokenizer)}")
    
    # Save tokenizers
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints'), exist_ok=True)
    input_tokenizer.save(os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints', 'input_tokenizer.pkl'))
    output_tokenizer.save(os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints', 'output_tokenizer.pkl'))
    
    # Create datasets
    train_dataset = Dataset(train_inputs, train_outputs, input_tokenizer, output_tokenizer)
    val_dataset = Dataset(val_inputs, val_outputs, input_tokenizer, output_tokenizer)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    model = Seq2SeqModel(
        input_vocab_size=len(input_tokenizer),
        output_vocab_size=len(output_tokenizer),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers']
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, 
            train_dataloader, 
            optimizer, 
            criterion, 
            device,
            config['teacher_forcing_ratio']
        )
        
        # Evaluate
        val_loss = evaluate(model, val_dataloader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(), 
                os.path.join(os.path.dirname(__file__), '..', 'models', 'checkpoints', 'best_model.pth')
            )
            print("Saved best model!")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
