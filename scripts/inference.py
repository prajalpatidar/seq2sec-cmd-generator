"""
CLI Inference tool using ONNX Runtime for Natural Language to Linux Command translation.
Optimized for embedded deployment with quantized models.
"""

import os
import argparse
import json
import numpy as np
import onnxruntime as ort

from src.dataset import Vocabulary


class ONNXInferenceEngine:
    """
    ONNX Runtime inference engine for Seq2Seq model.
    """
    def __init__(self, encoder_path, decoder_path, input_vocab_path, output_vocab_path):
        """
        Initialize ONNX inference engine.
        
        Args:
            encoder_path: path to encoder ONNX model
            decoder_path: path to decoder ONNX model
            input_vocab_path: path to input vocabulary JSON
            output_vocab_path: path to output vocabulary JSON
        """
        # Load vocabularies
        self.input_vocab = Vocabulary.load(input_vocab_path)
        self.output_vocab = Vocabulary.load(output_vocab_path)
        
        # Create ONNX Runtime sessions
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.encoder_session = ort.InferenceSession(encoder_path, sess_options)
        self.decoder_session = ort.InferenceSession(decoder_path, sess_options)
        
        print("ONNX Inference Engine initialized successfully!")
        print(f"  Input vocab size: {len(self.input_vocab)}")
        print(f"  Output vocab size: {len(self.output_vocab)}")
    
    def preprocess_input(self, text):
        """
        Preprocess natural language input.
        
        Args:
            text: natural language string
        Returns:
            input_seq: numpy array of token indices
            input_len: length of input sequence
        """
        text = text.lower().strip()
        input_seq = self.input_vocab.encode(text)
        input_len = len(input_seq)
        
        # Convert to numpy arrays
        input_seq = np.array([input_seq], dtype=np.int64)
        input_lengths = np.array([input_len], dtype=np.int64)
        
        return input_seq, input_lengths
    
    def run_encoder(self, input_seq, input_lengths):
        """
        Run encoder inference.
        
        Args:
            input_seq: numpy array (1, seq_len)
            input_lengths: numpy array (1,)
        Returns:
            encoder_outputs: numpy array (1, seq_len, hidden_dim)
            hidden: numpy array (num_layers, 1, hidden_dim)
        """
        encoder_outputs, hidden = self.encoder_session.run(
            None,
            {
                'input_seq': input_seq,
                'input_lengths': input_lengths
            }
        )
        return encoder_outputs, hidden
    
    def run_decoder(self, input_token, hidden, encoder_outputs, mask):
        """
        Run decoder inference for one step.
        
        Args:
            input_token: numpy array (1, 1)
            hidden: numpy array (num_layers, 1, hidden_dim)
            encoder_outputs: numpy array (1, seq_len, hidden_dim)
            mask: numpy array (1, seq_len)
        Returns:
            output: numpy array (1, vocab_size)
            new_hidden: numpy array (num_layers, 1, hidden_dim)
        """
        output, new_hidden = self.decoder_session.run(
            None,
            {
                'input_token': input_token,
                'hidden': hidden,
                'encoder_outputs': encoder_outputs,
                'mask': mask
            }
        )
        return output, new_hidden
    
    def generate(self, input_seq, input_lengths, max_length=50, sos_token=1, eos_token=2):
        """
        Generate output sequence using ONNX models.
        
        Args:
            input_seq: numpy array (1, seq_len)
            input_lengths: numpy array (1,)
            max_length: maximum length of generated sequence
            sos_token: start of sequence token id
            eos_token: end of sequence token id
        Returns:
            output_indices: list of output token indices
        """
        # Run encoder
        encoder_outputs, hidden = self.run_encoder(input_seq, input_lengths)
        
        # Create mask (1 for non-padding tokens)
        mask = (input_seq != 0).astype(np.float32)
        
        # Start with <SOS> token
        decoder_input = np.array([[sos_token]], dtype=np.int64)
        
        output_indices = []
        
        for _ in range(max_length):
            # Run decoder
            output, hidden = self.run_decoder(decoder_input, hidden, encoder_outputs, mask)
            
            # Get predicted token
            predicted_token = np.argmax(output, axis=-1)[0]
            
            # Stop if <EOS> token is generated
            if predicted_token == eos_token:
                break
            
            output_indices.append(predicted_token)
            
            # Use predicted token as next input
            decoder_input = np.array([[predicted_token]], dtype=np.int64)
        
        return output_indices
    
    def translate(self, text, max_length=50):
        """
        Translate natural language text to Linux command.
        
        Args:
            text: natural language string
            max_length: maximum length of generated command
        Returns:
            command: generated Linux command string
        """
        # Preprocess input
        input_seq, input_lengths = self.preprocess_input(text)
        
        # Generate output
        output_indices = self.generate(input_seq, input_lengths, max_length)
        
        # Decode output
        command = self.output_vocab.decode(output_indices, remove_special=True)
        
        return command


def main():
    parser = argparse.ArgumentParser(description="CLI inference tool for NL to Linux command translation")
    parser.add_argument('--encoder_path', type=str, default='models/encoder.onnx',
                      help='Path to encoder ONNX model')
    parser.add_argument('--decoder_path', type=str, default='models/decoder.onnx',
                      help='Path to decoder ONNX model')
    parser.add_argument('--input_vocab', type=str, default='models/input_vocab.json',
                      help='Path to input vocabulary')
    parser.add_argument('--output_vocab', type=str, default='models/output_vocab.json',
                      help='Path to output vocabulary')
    parser.add_argument('--use_quantized', action='store_true',
                      help='Use quantized models')
    parser.add_argument('--input', type=str, default=None,
                      help='Input text (if not provided, interactive mode)')
    parser.add_argument('--max_length', type=int, default=50,
                      help='Maximum length of generated command')
    
    args = parser.parse_args()
    
    # Use quantized models if specified
    if args.use_quantized:
        encoder_path = args.encoder_path.replace('.onnx', '_quantized.onnx')
        decoder_path = args.decoder_path.replace('.onnx', '_quantized.onnx')
    else:
        encoder_path = args.encoder_path
        decoder_path = args.decoder_path
    
    # Check if models exist
    if not os.path.exists(encoder_path):
        print(f"Error: Encoder model not found at {encoder_path}")
        print("Please train and export the model first.")
        return
    
    if not os.path.exists(decoder_path):
        print(f"Error: Decoder model not found at {decoder_path}")
        print("Please train and export the model first.")
        return
    
    print("="*50)
    print("NL to Linux Command Translation CLI")
    print("="*50)
    print(f"Using {'quantized' if args.use_quantized else 'standard'} models")
    print(f"Encoder: {encoder_path}")
    print(f"Decoder: {decoder_path}")
    print()
    
    # Initialize inference engine
    try:
        engine = ONNXInferenceEngine(
            encoder_path,
            decoder_path,
            args.input_vocab,
            args.output_vocab
        )
    except Exception as e:
        print(f"Error initializing inference engine: {e}")
        return
    
    # Single inference mode
    if args.input:
        print(f"\nInput: {args.input}")
        command = engine.translate(args.input, args.max_length)
        print(f"Command: {command}")
        return
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive Mode (type 'quit' to exit)")
    print("="*50)
    print("\nExamples:")
    print("  - show network interfaces")
    print("  - list all files")
    print("  - check disk usage")
    print("  - display memory usage")
    print()
    
    while True:
        try:
            user_input = input("Enter command: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Translate
            command = engine.translate(user_input, args.max_length)
            print(f"→ {command}\n")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == '__main__':
    main()
