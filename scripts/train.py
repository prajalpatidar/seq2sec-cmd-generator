"""
Training script for Seq2Seq model with attention mechanism.
Optimized for low memory footprint and embedded deployment.
"""

import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.model import Seq2SeqWithAttention
from src.dataset import CommandDataset, collate_fn


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        src = batch['input_seq'].to(device)
        trg = batch['output_seq'].to(device)
        src_len = batch['input_len']
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(src, src_len, trg, teacher_forcing_ratio)
        
        # Reshape for loss calculation (ignore first token which is <SOS>)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        
        # Calculate loss
        loss = criterion(output, trg)
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            src = batch['input_seq'].to(device)
            trg = batch['output_seq'].to(device)
            src_len = batch['input_len']
            
            # Forward pass without teacher forcing
            output = model(src, src_len, trg, teacher_forcing_ratio=0)
            
            # Reshape for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            
            # Calculate loss
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)


def test_inference(model, dataset, device, num_examples=5):
    """Test inference with some examples."""
    model.eval()
    
    print("\n" + "="*50)
    print("Testing Inference")
    print("="*50)
    
    with torch.no_grad():
        for i in range(min(num_examples, len(dataset))):
            sample = dataset[i]
            src = sample['input_seq'].unsqueeze(0).to(device)
            src_len = torch.tensor([sample['input_len']])
            
            # Generate output
            output = model.generate(src, src_len, max_length=30, sos_token=1, eos_token=2)
            
            # Decode
            predicted_command = dataset.output_vocab.decode(output[0])
            
            print(f"\nInput: {sample['input_text']}")
            print(f"Expected: {sample['output_text']}")
            print(f"Predicted: {predicted_command}")


def main():
    parser = argparse.ArgumentParser(description="Train Seq2Seq model for NL to Linux command translation")
    parser.add_argument('--data_path', type=str, default='data/commands_dataset.json',
                      help='Path to dataset JSON file')
    parser.add_argument('--embedding_dim', type=int, default=128,
                      help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                      help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=1,
                      help='Number of GRU layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                      help='Teacher forcing ratio')
    parser.add_argument('--output_dir', type=str, default='models',
                      help='Directory to save models and vocabularies')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*50)
    print("Seq2Seq Training for NL to Linux Command")
    print("="*50)
    print(f"Device: {args.device}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Load dataset and build vocabularies
    print("\nBuilding vocabularies...")
    full_dataset = CommandDataset(args.data_path, build_vocab=True)
    
    input_vocab = full_dataset.input_vocab
    output_vocab = full_dataset.output_vocab
    
    print(f"Input vocabulary size: {len(input_vocab)}")
    print(f"Output vocabulary size: {len(output_vocab)}")
    
    # Save vocabularies
    input_vocab.save(os.path.join(args.output_dir, 'input_vocab.json'))
    output_vocab.save(os.path.join(args.output_dir, 'output_vocab.json'))
    print("Vocabularies saved!")
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"\nTrain size: {train_size}")
    print(f"Validation size: {val_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    print("\nInitializing model...")
    model = Seq2SeqWithAttention(
        encoder_vocab_size=len(input_vocab),
        decoder_vocab_size=len(output_vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(args.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                args.device, args.teacher_forcing_ratio)
        
        # Evaluate
        val_loss = evaluate(model, val_loader, criterion, args.device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'encoder_vocab_size': len(input_vocab),
                    'decoder_vocab_size': len(output_vocab),
                    'embedding_dim': args.embedding_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'dropout': args.dropout
                }
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"Model saved! (Val Loss: {val_loss:.4f})")
        
        # Test inference every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_inference(model, full_dataset, args.device, num_examples=5)
    
    # Final testing
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    test_inference(model, full_dataset, args.device, num_examples=10)
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'encoder_vocab_size': len(input_vocab),
            'decoder_vocab_size': len(output_vocab),
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_layers,
            'dropout': args.dropout
        }
    }, os.path.join(args.output_dir, 'final_model.pt'))
    
    print(f"\nModels saved to {args.output_dir}/")
    print("You can now export to ONNX using export_onnx.py")


if __name__ == '__main__':
    main()
