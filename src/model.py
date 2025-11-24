"""
Seq2Seq model with attention mechanism for Natural Language to Linux Command translation.
Designed for low memory footprint suitable for embedded systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder network that processes the input sequence (natural language).
    Uses GRU for efficiency over LSTM.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_seq, input_lengths):
        """
        Args:
            input_seq: (batch_size, seq_len)
            input_lengths: (batch_size,)
        Returns:
            outputs: (batch_size, seq_len, hidden_dim)
            hidden: (num_layers, batch_size, hidden_dim)
        """
        embedded = self.dropout(self.embedding(input_seq))
        
        # Pack padded sequences for efficiency
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        outputs, hidden = self.gru(packed)
        
        # Unpack sequences
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        
        return outputs, hidden


class Attention(nn.Module):
    """
    Bahdanau attention mechanism for focusing on relevant parts of input.
    """
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        """
        Args:
            hidden: (batch_size, hidden_dim) - current decoder hidden state
            encoder_outputs: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len) - mask for padded positions
        Returns:
            attention_weights: (batch_size, seq_len)
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)
        
        # Repeat hidden state for all encoder outputs
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, hidden_dim)
        
        # Calculate attention scores
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)  # (batch_size, seq_len)
        
        # Apply mask if provided
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    """
    Decoder network with attention that generates the output sequence (Linux command).
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(Decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(embedding_dim + hidden_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.out = nn.Linear(hidden_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_token, hidden, encoder_outputs, mask=None):
        """
        Args:
            input_token: (batch_size, 1)
            hidden: (num_layers, batch_size, hidden_dim)
            encoder_outputs: (batch_size, seq_len, hidden_dim)
            mask: (batch_size, seq_len)
        Returns:
            output: (batch_size, vocab_size)
            hidden: (num_layers, batch_size, hidden_dim)
            attention_weights: (batch_size, seq_len)
        """
        embedded = self.dropout(self.embedding(input_token))  # (batch_size, 1, embedding_dim)
        
        # Calculate attention weights using the last layer hidden state
        attention_weights = self.attention(hidden[-1], encoder_outputs, mask)
        
        # Calculate context vector
        context = attention_weights.unsqueeze(1).bmm(encoder_outputs)  # (batch_size, 1, hidden_dim)
        
        # Concatenate embedded input and context
        gru_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embedding_dim + hidden_dim)
        
        # Pass through GRU
        output, hidden = self.gru(gru_input, hidden)
        
        # Concatenate GRU output and context for final prediction
        output = torch.cat((output.squeeze(1), context.squeeze(1)), dim=1)  # (batch_size, hidden_dim * 2)
        prediction = self.out(output)  # (batch_size, vocab_size)
        
        return prediction, hidden, attention_weights


class Seq2SeqWithAttention(nn.Module):
    """
    Complete Seq2Seq model with attention for Natural Language to Linux Command translation.
    Optimized for low memory footprint.
    """
    def __init__(self, encoder_vocab_size, decoder_vocab_size, 
                 embedding_dim=128, hidden_dim=256, num_layers=1, dropout=0.1):
        super(Seq2SeqWithAttention, self).__init__()
        
        self.encoder = Encoder(encoder_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(decoder_vocab_size, embedding_dim, hidden_dim, num_layers, dropout)
        
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        """
        Args:
            src: (batch_size, src_seq_len)
            src_lengths: (batch_size,)
            trg: (batch_size, trg_seq_len)
            teacher_forcing_ratio: probability of using teacher forcing
        Returns:
            outputs: (batch_size, trg_seq_len, vocab_size)
        """
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.vocab_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(src.device)
        
        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Create mask for padded positions
        mask = (src != 0).float()
        
        # First input to decoder is <SOS> token
        decoder_input = trg[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len):
            # Pass through decoder
            output, hidden, attention = self.decoder(decoder_input, hidden, encoder_outputs, mask)
            
            # Store output
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            
            # Use teacher forcing or predicted token
            decoder_input = trg[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
        
        return outputs
    
    def generate(self, src, src_lengths, max_length=50, sos_token=1, eos_token=2):
        """
        Generate output sequence without teacher forcing (for inference).
        
        Args:
            src: (batch_size, src_seq_len)
            src_lengths: (batch_size,)
            max_length: maximum length of generated sequence
            sos_token: start of sequence token id
            eos_token: end of sequence token id
        Returns:
            outputs: (batch_size, generated_seq_len)
        """
        batch_size = src.size(0)
        
        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        
        # Create mask for padded positions
        mask = (src != 0).float()
        
        # Start with <SOS> token
        decoder_input = torch.full((batch_size, 1), sos_token, dtype=torch.long).to(src.device)
        
        outputs = []
        
        for _ in range(max_length):
            # Pass through decoder
            output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, mask)
            
            # Get the highest predicted token
            top1 = output.argmax(1)
            outputs.append(top1.unsqueeze(1))
            
            # Use predicted token as next input
            decoder_input = top1.unsqueeze(1)
            
            # Stop if all sequences generated <EOS> token
            if (top1 == eos_token).all():
                break
        
        return torch.cat(outputs, dim=1)
