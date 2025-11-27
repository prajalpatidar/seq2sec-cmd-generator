#ifndef SEQ2SEQ_INFERENCE_H
#define SEQ2SEQ_INFERENCE_H

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "tokenizer.h"

class Seq2SeqInference {
public:
    Seq2SeqInference();
    ~Seq2SeqInference();
    
    // Initialize models and tokenizers
    bool initialize(
        const std::string& encoder_model_path,
        const std::string& decoder_model_path,
        const std::string& input_vocab_path,
        const std::string& output_vocab_path,
        bool use_quantized = true
    );
    
    // Generate command from natural language input
    std::string generate(const std::string& input_text, int max_length = 50);
    
    // Batch generation
    std::vector<std::string> generateBatch(const std::vector<std::string>& input_texts, int max_length = 50);
    
private:
    // ONNX Runtime session
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> encoder_session_;
    std::unique_ptr<Ort::Session> decoder_session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    
    // Tokenizers
    std::unique_ptr<Tokenizer> input_tokenizer_;
    std::unique_ptr<Tokenizer> output_tokenizer_;
    
    // Model dimensions
    int64_t hidden_dim_;
    int64_t num_layers_;
    
    // Helper functions
    std::vector<float> runEncoder(const std::vector<int64_t>& input_tokens);
    int64_t runDecoderStep(
        int64_t input_token,
        const std::vector<float>& hidden_state,
        const std::vector<float>& encoder_outputs,
        std::vector<float>& new_hidden_state
    );
};

#endif // SEQ2SEQ_INFERENCE_H
