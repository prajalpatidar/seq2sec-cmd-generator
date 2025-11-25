"""
Simple Sequence-to-Sequence model for natural language to Linux command translation.
Optimized for embedded systems with minimal parameters.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Simple GRU-based encoder."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=False,  # Keep unidirectional for smaller model
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len)
        Returns:
            outputs: Output tensor of shape (batch_size, seq_len, hidden_dim)
            hidden: Hidden state of shape (num_layers, batch_size, hidden_dim)
        """
        embedded = self.embedding(x)
        outputs, hidden = self.gru(embedded)
        return outputs, hidden


class Decoder(nn.Module):
    """Simple GRU-based decoder with attention."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            embedding_dim + hidden_dim, hidden_dim, num_layers, batch_first=True  # Input + context
        )
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        """
        Args:
            x: Input tensor of shape (batch_size, 1)
            hidden: Hidden state of shape (num_layers, batch_size, hidden_dim)
            encoder_outputs: Encoder outputs of shape (batch_size, seq_len, hidden_dim)
        Returns:
            output: Output tensor of shape (batch_size, vocab_size)
            hidden: Hidden state of shape (num_layers, batch_size, hidden_dim)
        """
        embedded = self.embedding(x)

        # Simple attention mechanism
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # Repeat hidden state for all encoder outputs
        hidden_repeated = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)

        # Compute attention scores
        attention_input = torch.cat([encoder_outputs, hidden_repeated], dim=2)
        attention_scores = self.attention(attention_input)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # Compute context vector
        context = torch.sum(attention_weights * encoder_outputs, dim=1, keepdim=True)

        # Combine embedded input with context
        rnn_input = torch.cat([embedded, context], dim=2)

        # Pass through GRU
        output, hidden = self.gru(rnn_input, hidden)

        # Final prediction
        output = self.fc(output.squeeze(1))

        return output, hidden


class Seq2SeqModel(nn.Module):
    """Complete Sequence-to-Sequence model."""

    def __init__(
        self, input_vocab_size, output_vocab_size, embedding_dim=64, hidden_dim=128, num_layers=1
    ):
        super(Seq2SeqModel, self).__init__()
        self.encoder = Encoder(input_vocab_size, embedding_dim, hidden_dim, num_layers)
        self.decoder = Decoder(output_vocab_size, embedding_dim, hidden_dim, num_layers)
        self.hidden_dim = hidden_dim

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            tgt: Target tensor of shape (batch_size, tgt_seq_len)
            teacher_forcing_ratio: Probability of using teacher forcing
        Returns:
            outputs: Output tensor of shape (batch_size, tgt_seq_len, output_vocab_size)
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc.out_features

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(src.device)

        # Encode input
        encoder_outputs, hidden = self.encoder(src)

        # First input to decoder is <START> token (index 1)
        decoder_input = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            # Decode one step
            output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t] = output

            # Teacher forcing: use actual next token as input
            # Otherwise: use predicted token
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            decoder_input = tgt[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)

        return outputs

    def predict(self, src, max_len=50, start_token=1, end_token=2):
        """
        Predict output sequence for given input.

        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            max_len: Maximum length of output sequence
            start_token: Start token index
            end_token: End token index
        Returns:
            predictions: List of predicted token indices
        """
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)

            # Encode input
            encoder_outputs, hidden = self.encoder(src)

            # Start with <START> token
            decoder_input = torch.tensor([[start_token]] * batch_size).to(src.device)

            predictions = []

            for _ in range(max_len):
                output, hidden = self.decoder(decoder_input, hidden, encoder_outputs)
                top1 = output.argmax(1)
                predictions.append(top1.item() if batch_size == 1 else top1)

                # Stop if <END> token is predicted
                if batch_size == 1 and top1.item() == end_token:
                    break

                decoder_input = top1.unsqueeze(1)

            return predictions
