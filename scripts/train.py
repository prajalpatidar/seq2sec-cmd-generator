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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.seq2seq import Seq2SeqModel
from scripts.data_utils import Tokenizer, Dataset, collate_fn


def load_data(data_path: str, train_split: float = 0.8):
    """
    Load and split dataset.
    
    This function reads the dataset (JSON or CSV), shuffles it to ensure randomness,
    and splits it into two parts:
    1. Training set (used to teach the model)
    2. Validation set (used to test how well the model is learning on unseen data)

    Args:
        data_path: Path to CSV or JSON file
        train_split: Fraction of data to use for training (e.g., 0.8 means 80%)
    Returns:
        Tuple of (train_inputs, train_outputs, val_inputs, val_outputs)
    """
    import json
    
    # Check file extension and load accordingly
    if data_path.endswith('.json'):
        with open(data_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(data_path)

    # Shuffle data (randomize order) to prevent the model from learning order-based biases
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split data into training and validation sets
    split_idx = int(len(df) * train_split)
    train_df = df[:split_idx]
    val_df = df[split_idx:]

    return (
        train_df["input"].tolist(),
        train_df["output"].tolist(),
        val_df["input"].tolist(),
        val_df["output"].tolist(),
    )


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio=0.5):
    """
    Train for one epoch (one complete pass through the training dataset).

    Args:
        model: The Seq2Seq neural network model
        dataloader: Feeds data in batches
        optimizer: The algorithm that updates model weights (e.g., Adam)
        criterion: The loss function (calculates error)
        device: CPU or GPU
        teacher_forcing_ratio: Probability of using the real target output as the next input
                             during training, instead of the model's own previous prediction.
                             Helps model learn faster.
    Returns:
        Average loss for the epoch
    """
    # Set model to training mode (enables features like Dropout)
    model.train()
    total_loss = 0

    for src, tgt in tqdm(dataloader, desc="Training"):
        # Move data to the active device (GPU or CPU)
        src, tgt = src.to(device), tgt.to(device)

        # Clear gradients from the previous batch (buffers must be reset)
        optimizer.zero_grad()

        # Forward pass: The model processes input and predicts output
        output = model(src, tgt, teacher_forcing_ratio)

        # Reshape for loss calculation
        # The Loss function expects a flat list of predictions and targets
        # We skip the first token (usually <SOS> start-of-sequence)
        output = output[:, 1:].reshape(-1, output.size(-1))
        tgt = tgt[:, 1:].reshape(-1)

        # Calculate loss (how different was the prediction from the actual target?)
        loss = criterion(output, tgt)

        # Backward pass (Backpropagation): Calculate gradients
        # This figures out how much each weight contributed to the error
        loss.backward()
        
        # Gradient Clipping: Prevents "exploding gradients" where numbers get too large
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update model weights based on the calculated gradients
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation set.
    
    This checks the model's performance on data it hasn't seen during training.
    We don't update weights here.

    Args:
        model: Seq2SeqModel
        dataloader: DataLoader
        criterion: Loss function
        device: Device to evaluate on
    Returns:
        Average loss for the validation set
    """
    # Set model to evaluation mode (disables Dropout, etc.)
    model.eval()
    total_loss = 0

    # Disable gradient calculation (saves memory and computation since we aren't training)
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)

            # Forward pass (no teacher forcing during evaluation - model must rely on itself)
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

    # Load configuration (hyperparameters like learning rate, batch size, etc.)
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "train_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set device: Use GPU (cuda) if available for faster training, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "commands-dataset.json")
    train_inputs, train_outputs, val_inputs, val_outputs = load_data(
        data_path, config["train_split"]
    )

    print(f"Training samples: {len(train_inputs)}")
    print(f"Validation samples: {len(val_inputs)}")

    # Create tokenizers
    # Tokenizers convert text (words) into numbers (IDs) that the model can understand
    input_tokenizer = Tokenizer(level="word")
    output_tokenizer = Tokenizer(level="word")

    # Fit tokenizers: Learn the vocabulary (list of all unique words) from the data
    input_tokenizer.fit(train_inputs + val_inputs, max_vocab_size=config["max_vocab_size"])
    output_tokenizer.fit(train_outputs + val_outputs, max_vocab_size=config["max_vocab_size"])

    print(f"Input vocabulary size: {len(input_tokenizer)}")
    print(f"Output vocabulary size: {len(output_tokenizer)}")

    # Save tokenizers so we can use them later for inference (prediction)
    os.makedirs(
        os.path.join(os.path.dirname(__file__), "..", "models", "checkpoints"), exist_ok=True
    )
    input_tokenizer.save(
        os.path.join(
            os.path.dirname(__file__), "..", "models", "checkpoints", "input_tokenizer.pkl"
        )
    )
    output_tokenizer.save(
        os.path.join(
            os.path.dirname(__file__), "..", "models", "checkpoints", "output_tokenizer.pkl"
        )
    )

    # Create datasets: Wraps data and tokenizers to provide easy access
    train_dataset = Dataset(train_inputs, train_outputs, input_tokenizer, output_tokenizer)
    val_dataset = Dataset(val_inputs, val_outputs, input_tokenizer, output_tokenizer)

    # Create dataloaders: Handles batching and shuffling of data
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn
    )

    # Create model
    # embedding_dim: Size of the vector representation for each word
    # hidden_dim: Size of the internal memory of the LSTM/GRU
    # num_layers: Number of stacked RNN layers
    model = Seq2SeqModel(
        input_vocab_size=len(input_tokenizer),
        output_vocab_size=len(output_tokenizer),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    # CrossEntropyLoss: Measures difference between predicted word and actual word
    # ignore_index=0: Don't calculate loss for padding tokens (0)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Adam Optimizer: Algorithm to update weights (standard choice for deep learning)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training loop
    best_val_loss = float("inf")
    patience = 30  # Early stopping patience - increased for more training
    patience_counter = 0

    for epoch in range(config["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")

        # Train
        train_loss = train_epoch(
            model, train_dataloader, optimizer, criterion, device, config["teacher_forcing_ratio"]
        )

        # Evaluate
        val_loss = evaluate(model, val_dataloader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        # Save best model
        # We only save the model if it performs better on the validation set
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), "..", "models", "checkpoints", "best_model.pth"
            )
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'input_vocab_size': len(input_tokenizer),
                    'output_vocab_size': len(output_tokenizer),
                    'config': {
                        'embedding_dim': config["embedding_dim"],
                        'hidden_dim': config["hidden_dim"],
                        'num_layers': config["num_layers"]
                    }
                },
                checkpoint_path
            )
            print("Saved best model!")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
            
            # Early stopping: Stop training if model stops improving
            # This prevents "overfitting" (memorizing training data instead of learning patterns)
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
