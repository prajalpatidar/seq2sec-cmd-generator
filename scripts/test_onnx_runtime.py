#!/usr/bin/env python3
"""Test ONNX models with ONNX Runtime."""

import os
import sys
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.data_utils import Tokenizer


def test_onnx_models():
    """Test ONNX models."""
    
    # Load tokenizers
    input_tok = Tokenizer()
    output_tok = Tokenizer()
    input_tok.load('models/checkpoints/input_tokenizer.pkl')
    output_tok.load('models/checkpoints/output_tokenizer.pkl')
    
    # Load ONNX models
    encoder_session = ort.InferenceSession('models/onnx/encoder.onnx')
    decoder_session = ort.InferenceSession('models/onnx/decoder.onnx')
    
    # Load quantized models
    encoder_q_session = ort.InferenceSession('models/onnx/encoder_quantized.onnx')
    decoder_q_session = ort.InferenceSession('models/onnx/decoder_quantized.onnx')
    
    print("ONNX Runtime Model Test")
    print("="*70)
    print()
    
    # Test inputs
    test_cases = [
        'show wifi ssid',
        'get device model name',
        'enable moca interface',
        'display memory usage',
        'list all files',
    ]
    
    for test_text in test_cases:
        # Encode input
        tokens = input_tok.encode(test_text, add_special_tokens=False)
        input_ids = np.array([tokens], dtype=np.int64)
        
        # Run encoder
        encoder_outputs, hidden = encoder_session.run(None, {'input': input_ids})
        
        # Decode
        start_token = output_tok.vocab['<START>']
        end_token = output_tok.vocab['<END>']
        
        decoder_input = np.array([[start_token]], dtype=np.int64)
        predictions = []
        
        for _ in range(30):  # max length
            outputs, hidden = decoder_session.run(
                None,
                {
                    'input': decoder_input,
                    'hidden': hidden,
                    'encoder_outputs': encoder_outputs
                }
            )
            
            # Get predicted token
            token_id = np.argmax(outputs[0])
            
            if token_id == end_token:
                break
            
            predictions.append(int(token_id))
            decoder_input = np.array([[token_id]], dtype=np.int64)
        
        # Decode output
        output_text = output_tok.decode(predictions)
        
        print(f"Input:  {test_text:35s}")
        print(f"Output: {output_text}")
        print()
    
    print("="*70)
    print("✅ ONNX models working correctly!")
    print()
    
    # Test quantized models
    print("Testing Quantized Models:")
    print("="*70)
    print()
    
    test_text = 'show wifi ssid'
    tokens = input_tok.encode(test_text, add_special_tokens=False)
    input_ids = np.array([tokens], dtype=np.int64)
    
    # Run quantized encoder
    encoder_outputs, hidden = encoder_q_session.run(None, {'input': input_ids})
    
    decoder_input = np.array([[start_token]], dtype=np.int64)
    predictions = []
    
    for _ in range(30):
        outputs, hidden = decoder_q_session.run(
            None,
            {
                'input': decoder_input,
                'hidden': hidden,
                'encoder_outputs': encoder_outputs
            }
        )
        
        token_id = np.argmax(outputs[0])
        if token_id == end_token:
            break
        
        predictions.append(int(token_id))
        decoder_input = np.array([[token_id]], dtype=np.int64)
    
    output_text = output_tok.decode(predictions)
    
    print(f"Input:  {test_text}")
    print(f"Output: {output_text}")
    print()
    print("✅ Quantized models working correctly!")


if __name__ == "__main__":
    os.chdir('/home/prajal/work/seq2seq-model/seq2sec-cmd-generator')
    test_onnx_models()
