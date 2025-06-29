#ifndef UTILITY_HPP_
#define UTILITY_HPP_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <cstddef>
#include <memory>
#include <algorithm>

#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// Global var group size for quantization of the weights
extern int GS;

// ----------------------------------------------------------------------------
// Utility function

inline long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    auto now = std::chrono::system_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
}

inline void safe_print(const std::string& piece) {
    if (piece.empty()) {
        return;
    }
    if (piece.size() == 1) {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    std::cout << piece;
}

inline void read_stdin(const std::string& guide, std::string& buffer, size_t max_len) {
    std::cout << guide;
    std::getline(std::cin, buffer);
    if(buffer.length() > max_len) {
        buffer.resize(max_len);
    }
}

inline void rmsnorm(float* o, float* x, float* weight, int size) {
    // calculate sum of squares
    // variance 
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    // 1 / sqrt(variance + epsilon)
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    // normalize and scale
    for (int j = 0; j < size; j++) {
        o[j] = weight[j] * (ss * x[j]);
    }
};

inline void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
};

// ----------------------------------------------------------------------------
// struct definitions

// store params of transformer model 
struct Config {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
};
using ConfigType = Config;

// ----------------------------------------------------------------------------
// Tokennizer

struct TokenIndex {
    std::string str;
    int id;
    // overload operator '<'
    bool operator<(const TokenIndex& other) const {
        return str < other.str;
    }
};
using TokenIndexType = TokenIndex;

struct TokenizerData {
    std::vector<std::unique_ptr<char[]>> vocab;
    std::vector<TokenIndexType> sorted_vocab;
    std::vector<float> vocab_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
};
using TokenizerDataType = TokenizerData;

inline int string_lookup(const std::string& str, 
                         const int vocab_size,
                         const std::vector<TokenIndexType>& sorted_vocab) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndexType tok = {.str = str, 
                          .id = -1};

    // cannot use std::lower_bound function ?
    // auto iter = std::lower_bound(sorted_vocab.begin(), sorted_vocab.end(), tok);
    auto cmp = [](const void *a, const void *b) {
        const TokenIndexType* ta = static_cast<const TokenIndexType*>(a);
        const TokenIndexType* tb = static_cast<const TokenIndexType*>(b);
        return strcmp(ta->str.c_str(), tb->str.c_str());
    };

    // If we didn't reach the end and the string matches
    TokenIndexType *matched_string = static_cast<TokenIndexType*>(
        bsearch(&tok, sorted_vocab.data(), vocab_size, sizeof(TokenIndexType), cmp)
    );
    return matched_string != nullptr ? matched_string->id : -1;
}

class Tokenizer {
private:
    std::unique_ptr<TokenizerDataType> tokenizer_data_;
    void build_tokenizer(std::string_view tokenizer_path, int vocab_size);

public:
    Tokenizer(std::string_view tokenizer_path, int vocab_size) {
        tokenizer_data_ = std::make_unique<TokenizerDataType>();
        build_tokenizer(tokenizer_path, vocab_size);
    };
    ~Tokenizer() {};

    void encode(const std::string &text, const int8_t &bos, const int8_t &eos, std::vector<int> &tokens, int &num_tokens);
    std::string decode(int prev_token, int token);
};

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

// struct used when sorting probabilities during top-p sampling
struct ProbaIndex {
    float proba;
    int index;
};
using ProbaIndexType = ProbaIndex;

class Sampler {
private:
    std::vector<ProbaIndexType> proba_index_; // buffer used in top-p sampling
    int vocab_size_;
    float temperature_;
    float topp_;
    unsigned long long rng_state_;

    int sample_argmax(const std::vector<float> &proba);
    int sample_mult(const std::vector<float> &proba, float coin);
    int sample_topp(const std::vector<float> &proba, float coin);
    
    unsigned int random_u32(unsigned long long *state) {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        return (*state * 0x2545F4914F6CDD1Dull) >> 32;
    }

    float random_f32(unsigned long long *state) { // random float32 in [0,1)
        return (random_u32(state) >> 8) / 16777216.0f;
    }
    
public:
    Sampler(int vocab_size, 
        float temperature, 
        float topp, 
        unsigned long long rng_state) : vocab_size_(vocab_size),
            temperature_(temperature), 
            topp_(topp), 
            rng_state_(rng_state) {
        proba_index_.resize(vocab_size);
    }

    ~Sampler() = default;

    // void build_sampler(int vocab_size, float temperature, float topp, unsigned long long rng_seed);
    int sample(std::vector<float> &logits);

};



#endif // UTILITY_HPP_