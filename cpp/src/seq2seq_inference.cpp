#include "seq2seq_inference.h"
#include <iostream>
#include <algorithm>
#include <numeric>

Seq2SeqInference::Seq2SeqInference() 
    : hidden_dim_(512), num_layers_(2) {
}

Seq2SeqInference::~Seq2SeqInference() {
}

bool Seq2SeqInference::initialize(
    const std::string& encoder_model_path,
    const std::string& decoder_model_path,
    const std::string& input_vocab_path,
    const std::string& output_vocab_path,
    bool use_quantized
) {
    try {
        // Initialize ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "seq2sec");
        
        // Create session options
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(1);
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        // Load encoder model
        encoder_session_ = std::make_unique<Ort::Session>(*env_, encoder_model_path.c_str(), *session_options_);
        std::cout << "Loaded encoder model: " << encoder_model_path << std::endl;
        
        // Load decoder model
        decoder_session_ = std::make_unique<Ort::Session>(*env_, decoder_model_path.c_str(), *session_options_);
        std::cout << "Loaded decoder model: " << decoder_model_path << std::endl;
        
        // Load tokenizers
        input_tokenizer_ = std::make_unique<Tokenizer>();
        if (!input_tokenizer_->loadVocabulary(input_vocab_path)) {
            std::cerr << "Failed to load input vocabulary" << std::endl;
            return false;
        }
        
        output_tokenizer_ = std::make_unique<Tokenizer>();
        if (!output_tokenizer_->loadVocabulary(output_vocab_path)) {
            std::cerr << "Failed to load output vocabulary" << std::endl;
            return false;
        }
        
        std::cout << "Initialization complete!" << std::endl;
        return true;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
}

std::string Seq2SeqInference::generate(const std::string& input_text, int max_length) {
    try {
        // Encode input text
        std::vector<int64_t> input_tokens = input_tokenizer_->encode(input_text);
        if (input_tokens.empty()) {
            return "";
        }
        
        // Prepare encoder input
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_tokens.size())};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, input_tokens.data(), input_tokens.size(), 
            input_shape.data(), input_shape.size()
        );
        
        // Run encoder
        const char* encoder_input_names[] = {"input"};
        const char* encoder_output_names[] = {"encoder_outputs", "hidden"};
        
        auto encoder_outputs = encoder_session_->Run(
            Ort::RunOptions{nullptr}, 
            encoder_input_names, &input_tensor, 1,
            encoder_output_names, 2
        );
        
        // Get encoder outputs and hidden state
        float* encoder_output_data = encoder_outputs[0].GetTensorMutableData<float>();
        float* hidden_data = encoder_outputs[1].GetTensorMutableData<float>();
        
        auto encoder_output_shape = encoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        auto hidden_shape = encoder_outputs[1].GetTensorTypeAndShapeInfo().GetShape();
        
        size_t encoder_output_size = std::accumulate(
            encoder_output_shape.begin(), encoder_output_shape.end(), 
            1, std::multiplies<int64_t>()
        );
        size_t hidden_size = std::accumulate(
            hidden_shape.begin(), hidden_shape.end(), 
            1, std::multiplies<int64_t>()
        );
        
        std::vector<float> encoder_outputs_vec(encoder_output_data, encoder_output_data + encoder_output_size);
        std::vector<float> hidden_vec(hidden_data, hidden_data + hidden_size);
        
        // Decode
        std::vector<int64_t> output_tokens;
        int64_t decoder_input = output_tokenizer_->getStartTokenId();
        
        for (int step = 0; step < max_length; ++step) {
            // Prepare decoder input
            std::vector<int64_t> decoder_input_vec = {decoder_input};
            std::vector<int64_t> decoder_input_shape = {1, 1};
            
            Ort::Value decoder_input_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, decoder_input_vec.data(), 1,
                decoder_input_shape.data(), decoder_input_shape.size()
            );
            
            // Prepare hidden state
            Ort::Value hidden_tensor = Ort::Value::CreateTensor<float>(
                memory_info, hidden_vec.data(), hidden_vec.size(),
                hidden_shape.data(), hidden_shape.size()
            );
            
            // Prepare encoder outputs
            Ort::Value encoder_outputs_tensor = Ort::Value::CreateTensor<float>(
                memory_info, encoder_outputs_vec.data(), encoder_outputs_vec.size(),
                encoder_output_shape.data(), encoder_output_shape.size()
            );
            
            // Run decoder
            const char* decoder_input_names[] = {"input", "hidden", "encoder_outputs"};
            const char* decoder_output_names[] = {"output", "new_hidden"};
            
            std::vector<Ort::Value> decoder_inputs;
            decoder_inputs.push_back(std::move(decoder_input_tensor));
            decoder_inputs.push_back(std::move(hidden_tensor));
            decoder_inputs.push_back(std::move(encoder_outputs_tensor));
            
            auto decoder_outputs = decoder_session_->Run(
                Ort::RunOptions{nullptr},
                decoder_input_names, decoder_inputs.data(), decoder_inputs.size(),
                decoder_output_names, 2
            );
            
            // Get output logits
            float* output_data = decoder_outputs[0].GetTensorMutableData<float>();
            auto output_shape = decoder_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            int64_t vocab_size = output_shape[output_shape.size() - 1];
            
            // Get argmax
            int64_t next_token = std::distance(
                output_data, 
                std::max_element(output_data, output_data + vocab_size)
            );
            
            // Check for end token
            if (next_token == output_tokenizer_->getEndTokenId()) {
                break;
            }
            
            output_tokens.push_back(next_token);
            decoder_input = next_token;
            
            // Update hidden state
            float* new_hidden_data = decoder_outputs[1].GetTensorMutableData<float>();
            std::copy(new_hidden_data, new_hidden_data + hidden_size, hidden_vec.begin());
        }
        
        // Decode output tokens
        std::string result = output_tokenizer_->decode(output_tokens);
        return result;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return "";
    }
}

std::vector<std::string> Seq2SeqInference::generateBatch(
    const std::vector<std::string>& input_texts, 
    int max_length
) {
    std::vector<std::string> results;
    for (const auto& text : input_texts) {
        results.push_back(generate(text, max_length));
    }
    return results;
}
