"""
Export Python tokenizer vocabularies to text files for C++ usage
"""
import pickle
import os

def export_vocab(tokenizer_path, output_path):
    """Export tokenizer vocabulary to text file"""
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    # Extract vocab from the dictionary structure
    if isinstance(tokenizer, dict) and 'vocab' in tokenizer:
        token2id = tokenizer['vocab']
    elif hasattr(tokenizer, 'token2id'):
        token2id = tokenizer.token2id
    elif isinstance(tokenizer, dict):
        # The tokenizer itself might be the token2id dict
        token2id = tokenizer
    else:
        raise ValueError(f"Cannot extract vocabulary from tokenizer: {type(tokenizer)}")
    
    # Sort by ID and write to file
    sorted_tokens = sorted(token2id.items(), key=lambda x: x[1])
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for token, _ in sorted_tokens:
            f.write(f"{token}\n")
    
    print(f"Exported {len(sorted_tokens)} tokens to {output_path}")

def main():
    # Paths
    input_tokenizer_path = 'models/checkpoints/input_tokenizer.pkl'
    output_tokenizer_path = 'models/checkpoints/output_tokenizer.pkl'
    
    input_vocab_path = 'models/checkpoints/input_vocab.txt'
    output_vocab_path = 'models/checkpoints/output_vocab.txt'
    
    # Check if tokenizers exist
    if not os.path.exists(input_tokenizer_path):
        print(f"Error: Input tokenizer not found at {input_tokenizer_path}")
        print("Please train the model first: python scripts/train.py")
        return
    
    if not os.path.exists(output_tokenizer_path):
        print(f"Error: Output tokenizer not found at {output_tokenizer_path}")
        print("Please train the model first: python scripts/train.py")
        return
    
    # Export vocabularies
    print("Exporting vocabularies for C++ deployment...")
    export_vocab(input_tokenizer_path, input_vocab_path)
    export_vocab(output_tokenizer_path, output_vocab_path)
    
    print("\n✅ Vocabulary export complete!")
    print(f"   Input vocabulary: {input_vocab_path}")
    print(f"   Output vocabulary: {output_vocab_path}")
    print("\nThese files can now be used with the C++ application.")

if __name__ == '__main__':
    main()
