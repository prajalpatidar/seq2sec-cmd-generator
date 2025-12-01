#include <iostream>
#include <string>
#include <vector>
#include "seq2seq_inference.h"

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <mode> [options]\n"
              << "\nModes:\n"
              << "  generate <text>           Generate command from natural language\n"
              << "  interactive               Interactive mode\n"
              << "  batch <file>              Process batch file\n"
              << "\nOptions:\n"
              << "  --encoder <path>          Encoder model path (default: depends on model-type)\n"
              << "  --decoder <path>          Decoder model path (default: depends on model-type)\n"
              << "  --model-type <type>       Model type: int8 (default) or float32\n"
              << "  --input-vocab <path>      Input vocabulary (default: models/checkpoints/input_vocab.txt)\n"
              << "  --output-vocab <path>     Output vocabulary (default: models/checkpoints/output_vocab.txt)\n"
              << "  --max-length <n>          Maximum output length (default: 50)\n"
              << "\nExamples:\n"
              << "  " << program_name << " generate \"show network interfaces\"\n"
              << "  " << program_name << " generate \"show wifi ssid\"\n"
              << "  " << program_name << " interactive\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Default paths
    std::string encoder_path = "";
    std::string decoder_path = "";
    std::string input_vocab_path = "models/checkpoints/input_vocab.txt";
    std::string output_vocab_path = "models/checkpoints/output_vocab.txt";
    std::string model_type = "int8"; // Default to quantized
    int max_length = 50;
    
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }
    
    std::string mode = argv[1];
    
    // Parse options
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--encoder" && i + 1 < argc) {
            encoder_path = argv[++i];
        } else if (arg == "--decoder" && i + 1 < argc) {
            decoder_path = argv[++i];
        } else if (arg == "--input-vocab" && i + 1 < argc) {
            input_vocab_path = argv[++i];
        } else if (arg == "--output-vocab" && i + 1 < argc) {
            output_vocab_path = argv[++i];
        } else if (arg == "--max-length" && i + 1 < argc) {
            max_length = std::stoi(argv[++i]);
        } else if (arg == "--model-type" && i + 1 < argc) {
            model_type = argv[++i];
        }
    }

    // Set default paths based on model type if not provided
    if (encoder_path.empty()) {
        if (model_type == "float32" || model_type == "f32") {
            encoder_path = "models/onnx/encoder.onnx";
        } else {
            encoder_path = "models/onnx/encoder_quantized.onnx";
        }
    }

    if (decoder_path.empty()) {
        if (model_type == "float32" || model_type == "f32") {
            decoder_path = "models/onnx/decoder.onnx";
        } else {
            decoder_path = "models/onnx/decoder_quantized.onnx";
        }
    }
    
    // Initialize inference engine
    std::cout << "Initializing seq2seq command generator..." << std::endl;
    std::cout << "Model Type: " << model_type << std::endl;
    std::cout << "Encoder: " << encoder_path << std::endl;
    std::cout << "Decoder: " << decoder_path << std::endl;

    Seq2SeqInference inference;
    
    if (!inference.initialize(encoder_path, decoder_path, input_vocab_path, output_vocab_path, true)) {
        std::cerr << "Failed to initialize inference engine" << std::endl;
        return 1;
    }
    
    if (mode == "generate") {
        // Find the input text (everything after "generate" that's not an option)
        std::string input_text;
        for (int i = 2; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.substr(0, 2) == "--") {
                ++i; // Skip option value
                continue;
            }
            if (!input_text.empty()) input_text += " ";
            input_text += arg;
        }
        
        if (input_text.empty()) {
            std::cerr << "Error: No input text provided" << std::endl;
            return 1;
        }
        
        std::cout << "\nInput: " << input_text << std::endl;
        std::string command = inference.generate(input_text, max_length);
        std::cout << "Command: " << command << std::endl;
        
    } else if (mode == "interactive") {
        std::cout << "\n=== Interactive Mode ===" << std::endl;
        std::cout << "Enter natural language instructions (type 'quit' to exit)\n" << std::endl;
        
        std::string input_text;
        while (true) {
            std::cout << "> ";
            std::getline(std::cin, input_text);
            
            if (input_text == "quit" || input_text == "exit") {
                std::cout << "Goodbye!" << std::endl;
                break;
            }
            
            if (input_text.empty()) {
                continue;
            }
            
            std::string command = inference.generate(input_text, max_length);
            std::cout << "Command: " << command << std::endl;
        }
        
    } else if (mode == "batch") {
        if (argc < 3) {
            std::cerr << "Error: Batch file not specified" << std::endl;
            return 1;
        }
        
        std::string batch_file = argv[2];
        std::ifstream file(batch_file);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open batch file: " << batch_file << std::endl;
            return 1;
        }
        
        std::string line;
        int count = 0;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            
            std::cout << "\n[" << ++count << "] Input: " << line << std::endl;
            std::string command = inference.generate(line, max_length);
            std::cout << "    Command: " << command << std::endl;
        }
        
        std::cout << "\nProcessed " << count << " commands" << std::endl;
        
    } else {
        std::cerr << "Error: Unknown mode '" << mode << "'" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    return 0;
}
