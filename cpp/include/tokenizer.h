#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();
    
    // Load vocabulary from file
    bool loadVocabulary(const std::string& vocab_file);
    
    // Encode text to token IDs
    std::vector<int64_t> encode(const std::string& text);
    
    // Decode token IDs to text
    std::string decode(const std::vector<int64_t>& tokens);
    
    // Get special token IDs
    int64_t getStartTokenId() const { return start_token_id_; }
    int64_t getEndTokenId() const { return end_token_id_; }
    int64_t getPadTokenId() const { return pad_token_id_; }
    int64_t getUnkTokenId() const { return unk_token_id_; }
    
    // Get vocabulary size
    size_t getVocabSize() const { return token2id_.size(); }
    
private:
    std::unordered_map<std::string, int64_t> token2id_;
    std::unordered_map<int64_t, std::string> id2token_;
    
    int64_t start_token_id_;
    int64_t end_token_id_;
    int64_t pad_token_id_;
    int64_t unk_token_id_;
    
    // Helper functions
    std::vector<std::string> tokenize(const std::string& text);
    std::string toLowerCase(const std::string& text);
    std::string postProcessDmcli(const std::string& text);
};

#endif // TOKENIZER_H
