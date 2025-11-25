"""
CLI tool for Linux command generation using ONNX Runtime.
Optimized for embedded systems.
"""

import os
import sys
import click
import onnxruntime as ort
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.data_utils import Tokenizer


class CommandGenerator:
    """Command generator using ONNX Runtime."""

    def __init__(self, encoder_path, decoder_path, input_tokenizer_path, output_tokenizer_path):
        """
        Initialize command generator.

        Args:
            encoder_path: Path to encoder ONNX model
            decoder_path: Path to decoder ONNX model
            input_tokenizer_path: Path to input tokenizer
            output_tokenizer_path: Path to output tokenizer
        """
        # Load tokenizers
        self.input_tokenizer = Tokenizer(level="word")
        self.output_tokenizer = Tokenizer(level="word")

        self.input_tokenizer.load(input_tokenizer_path)
        self.output_tokenizer.load(output_tokenizer_path)

        # Load ONNX models
        self.encoder_session = ort.InferenceSession(encoder_path)
        self.decoder_session = ort.InferenceSession(decoder_path)

        print("Models loaded successfully!")

    def generate(self, input_text, max_len=50):
        """
        Generate Linux command from natural language input.

        Args:
            input_text: Natural language instruction
            max_len: Maximum length of generated command
        Returns:
            Generated Linux command
        """
        # Encode input
        input_indices = self.input_tokenizer.encode(input_text, add_special_tokens=True)
        input_tensor = np.array([input_indices], dtype=np.int64)

        # Encode with encoder
        encoder_outputs, hidden = self.encoder_session.run(None, {"input": input_tensor})

        # Decode step by step
        start_token = self.output_tokenizer.vocab[self.output_tokenizer.START_TOKEN]
        end_token = self.output_tokenizer.vocab[self.output_tokenizer.END_TOKEN]

        decoder_input = np.array([[start_token]], dtype=np.int64)
        predictions = []

        for _ in range(max_len):
            output, hidden = self.decoder_session.run(
                None, {"input": decoder_input, "hidden": hidden, "encoder_outputs": encoder_outputs}
            )

            # Get predicted token
            predicted_token = np.argmax(output, axis=1)[0]

            # Stop if end token is predicted
            if predicted_token == end_token:
                break

            predictions.append(predicted_token)
            decoder_input = np.array([[predicted_token]], dtype=np.int64)

        # Decode predictions
        command = self.output_tokenizer.decode(predictions, skip_special_tokens=True)
        return command


@click.group()
def cli():
    """Linux Command Generator CLI"""
    pass


@cli.command()
@click.argument("input_text")
@click.option("--quantized", is_flag=True, help="Use quantized models for embedded deployment")
def generate(input_text, quantized):
    """
    Generate Linux command from natural language instruction.

    Example:
        python cli.py generate "show network interfaces"
    """
    # Determine model paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    if quantized:
        encoder_path = os.path.join(models_dir, "onnx", "encoder_quantized.onnx")
        decoder_path = os.path.join(models_dir, "onnx", "decoder_quantized.onnx")
        print("Using quantized models (optimized for embedded systems)")
    else:
        encoder_path = os.path.join(models_dir, "onnx", "encoder.onnx")
        decoder_path = os.path.join(models_dir, "onnx", "decoder.onnx")
        print("Using standard models")

    input_tokenizer_path = os.path.join(models_dir, "checkpoints", "input_tokenizer.pkl")
    output_tokenizer_path = os.path.join(models_dir, "checkpoints", "output_tokenizer.pkl")

    # Check if models exist
    if not os.path.exists(encoder_path):
        click.echo(f"Error: Encoder model not found at {encoder_path}")
        click.echo("Please train the model first using: python scripts/train.py")
        click.echo("Then export to ONNX using: python scripts/export_onnx.py")
        return

    # Initialize generator
    generator = CommandGenerator(
        encoder_path, decoder_path, input_tokenizer_path, output_tokenizer_path
    )

    # Generate command
    click.echo(f"\nInput: {input_text}")
    command = generator.generate(input_text)
    click.echo(f"Generated Command: {command}")


@cli.command()
def interactive():
    """
    Run in interactive mode.

    Example:
        python cli.py interactive
    """
    # Determine model paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")

    encoder_path = os.path.join(models_dir, "onnx", "encoder.onnx")
    decoder_path = os.path.join(models_dir, "onnx", "decoder.onnx")
    input_tokenizer_path = os.path.join(models_dir, "checkpoints", "input_tokenizer.pkl")
    output_tokenizer_path = os.path.join(models_dir, "checkpoints", "output_tokenizer.pkl")

    # Check if models exist
    if not os.path.exists(encoder_path):
        click.echo(f"Error: Models not found. Please train and export models first.")
        return

    # Initialize generator
    generator = CommandGenerator(
        encoder_path, decoder_path, input_tokenizer_path, output_tokenizer_path
    )

    click.echo("\nLinux Command Generator (Interactive Mode)")
    click.echo("Type 'quit' or 'exit' to stop\n")

    while True:
        try:
            input_text = input("Enter instruction: ").strip()

            if input_text.lower() in ["quit", "exit"]:
                click.echo("Goodbye!")
                break

            if not input_text:
                continue

            command = generator.generate(input_text)
            click.echo(f"Command: {command}\n")

        except KeyboardInterrupt:
            click.echo("\nGoodbye!")
            break
        except Exception as e:
            click.echo(f"Error: {str(e)}")


if __name__ == "__main__":
    cli()
