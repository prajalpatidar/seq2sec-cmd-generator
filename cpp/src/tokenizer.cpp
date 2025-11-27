#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <regex>

Tokenizer::Tokenizer() 
    : start_token_id_(0), end_token_id_(1), pad_token_id_(2), unk_token_id_(3) {
}

Tokenizer::~Tokenizer() {
}

bool Tokenizer::loadVocabulary(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open vocabulary file: " << vocab_file << std::endl;
        return false;
    }
    
    std::string line;
    int64_t id = 0;
    
    while (std::getline(file, line)) {
        // Remove whitespace
        line.erase(std::remove_if(line.begin(), line.end(), ::isspace), line.end());
        if (!line.empty()) {
            token2id_[line] = id;
            id2token_[id] = line;
            id++;
        }
    }
    
    // Set special token IDs
    if (token2id_.find("<START>") != token2id_.end()) {
        start_token_id_ = token2id_["<START>"];
    }
    if (token2id_.find("<END>") != token2id_.end()) {
        end_token_id_ = token2id_["<END>"];
    }
    if (token2id_.find("<PAD>") != token2id_.end()) {
        pad_token_id_ = token2id_["<PAD>"];
    }
    if (token2id_.find("<UNK>") != token2id_.end()) {
        unk_token_id_ = token2id_["<UNK>"];
    }
    
    std::cout << "Loaded vocabulary: " << token2id_.size() << " tokens" << std::endl;
    return true;
}

std::vector<std::string> Tokenizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string lower_text = toLowerCase(text);
    
    // Simple word tokenization (split by whitespace and punctuation)
    std::regex word_regex("\\w+|[^\\w\\s]");
    auto words_begin = std::sregex_iterator(lower_text.begin(), lower_text.end(), word_regex);
    auto words_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
        std::smatch match = *i;
        tokens.push_back(match.str());
    }
    
    return tokens;
}

std::string Tokenizer::toLowerCase(const std::string& text) {
    std::string result = text;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

std::vector<int64_t> Tokenizer::encode(const std::string& text) {
    std::vector<std::string> tokens = tokenize(text);
    std::vector<int64_t> token_ids;
    
    for (const auto& token : tokens) {
        auto it = token2id_.find(token);
        if (it != token2id_.end()) {
            token_ids.push_back(it->second);
        } else {
            token_ids.push_back(unk_token_id_);
        }
    }
    
    return token_ids;
}

std::string Tokenizer::decode(const std::vector<int64_t>& tokens) {
    std::vector<std::string> words;
    
    for (int64_t token_id : tokens) {
        // Skip special tokens
        if (token_id == start_token_id_ || token_id == end_token_id_ || 
            token_id == pad_token_id_) {
            continue;
        }
        
        auto it = id2token_.find(token_id);
        if (it != id2token_.end()) {
            words.push_back(it->second);
        }
    }
    
    // Join tokens
    std::string result;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i > 0 && words[i] != "." && words[i] != "," && 
            words[i] != "-" && words[i-1] != "-") {
            result += " ";
        }
        result += words[i];
    }
    
    // Apply dmcli post-processing
    result = postProcessDmcli(result);
    
    return result;
}

std::string Tokenizer::postProcessDmcli(const std::string& text) {
    std::string result = text;
    
    // Check if this is a dmcli command
    if (result.find("dmcli") == std::string::npos) {
        return result;
    }
    
    // Remove spaces around dots, commas, and hyphens
    result = std::regex_replace(result, std::regex("\\s*\\.\\s*"), ".");
    result = std::regex_replace(result, std::regex("\\s*,\\s*"), ",");
    result = std::regex_replace(result, std::regex("\\s*-\\s*"), "-");
    
    // Capitalize Device path components
    if (result.find("device.") != std::string::npos || 
        result.find("device ") != std::string::npos) {
        size_t pos = result.find("device");
        if (pos != std::string::npos) {
            result[pos] = 'D';
            
            // Capitalize first letter after each dot
            for (size_t i = pos; i < result.length(); ++i) {
                if (result[i] == '.' && i + 1 < result.length()) {
                    result[i + 1] = std::toupper(result[i + 1]);
                }
            }
        }
    }
    
    return result;
}
