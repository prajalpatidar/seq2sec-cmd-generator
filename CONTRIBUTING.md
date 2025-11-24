# Contributing to seq2sec-cmd-generator

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature request:

1. Check if the issue already exists in the [Issues](https://github.com/prajalpatidar/seq2sec-cmd-generator/issues)
2. If not, create a new issue with:
   - Clear title and description
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment details

### Contributing Code

1. **Fork the repository**
   ```bash
   git clone https://github.com/prajalpatidar/seq2sec-cmd-generator.git
   cd seq2sec-cmd-generator
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed

4. **Run tests**
   ```bash
   pytest tests/ -v
   ```

5. **Format your code**
   ```bash
   black .
   flake8 .
   ```

6. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

7. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your fork and branch
   - Provide a clear description of your changes

## Development Setup

### Prerequisites

- Python 3.8+
- pip
- (Optional) CUDA-capable GPU for training

### Installation

```bash
# Clone the repository
git clone https://github.com/prajalpatidar/seq2sec-cmd-generator.git
cd seq2sec-cmd-generator

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Code Style

We use:
- **Black** for code formatting
- **Flake8** for linting
- **Type hints** where appropriate

Before submitting:
```bash
black .
flake8 .
```

## Areas for Contribution

### 1. Dataset Expansion

Add more Linux command mappings to `data/commands_dataset.csv`:

```csv
input,output
your natural language command,linux command
```

### 2. Model Improvements

- Experiment with different architectures
- Add beam search for better generation
- Implement caching for faster inference

### 3. CLI Enhancements

- Add command history
- Implement command verification
- Add support for command parameters

### 4. Documentation

- Improve existing documentation
- Add tutorials and examples
- Translate documentation to other languages

### 5. Testing

- Add more unit tests
- Add integration tests
- Add performance benchmarks

### 6. Deployment

- Create Docker container
- Add CI/CD pipelines
- Create deployment scripts for various platforms

## Code Review Process

1. All submissions require review
2. Maintainers will review your PR within a few days
3. Address any feedback or requested changes
4. Once approved, maintainers will merge your PR

## Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow
- Follow the [Code of Conduct](CODE_OF_CONDUCT.md)

## Questions?

If you have questions:
- Open an issue with the "question" label
- Check existing documentation
- Review closed issues for similar questions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉
