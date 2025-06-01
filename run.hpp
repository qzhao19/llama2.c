#ifndef RUN_HPP_
#define RUN_HPP_

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <cstddef>
#include <memory>
#include <algorithm>
#include <chrono>

#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

// Global var group size for quantization of the weights
extern int GS;

// ----------------------------------------------------------------------------
// Model loading

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

// all weights params of model
struct TransformerWeights {
    // embedding layer
    std::vector<float> token_embedding_table; // token embedding table
    
    // rmsnorms layer
    std::vector<float> rms_att_weight; // (layer, dim) rmsnorm weights
    std::vector<float> rms_ffn_weight; // (layer, dim)
    // final rmsnorm
    std::vector<float> rms_final_weight; // (dim,) for the last layer

    // attention block
    // weights for matmuls. note dim == n_heads * head_size
    std::vector<float> wq; // (layer, dim, n_heads * head_size)
    std::vector<float> wk; // (layer, dim, n_kv_heads * head_size)
    std::vector<float> wv; // (layer, dim, n_kv_heads * head_size)
    std::vector<float> wo; // (layer, n_heads * head_size, dim)
    
    // weights for ffn
    std::vector<float> w1; // (layer, hidden_dim, dim)
    std::vector<float> w2; // (layer, dim, hidden_dim)
    std::vector<float> w3; // (layer, hidden_dim, dim)

    // (optional) classifier weights for the logits, on the last layer
    std::vector<float> wcls;
};

// all activation buffers and intermediate states during forward propagation
struct RunState {
    // current wave of activations
    std::vector<float> x; // activation at current time stamp (dim,)
    std::vector<float> xb; // same, but inside a residual branch (dim,)
    std::vector<float> xb2; // an additional buffer just for convenience (dim,)
    std::vector<float> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::vector<float> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::vector<float> q; // query (dim,)
    std::vector<float> k; // key (dim,)
    std::vector<float> v; // value (dim,)
    std::vector<float> att; // buffer for scores/attention values (n_heads, seq_len)
    std::vector<float> logits; // output logits
    // kv cache
    std::vector<float> key_cache;   // (layer, seq_len, dim)
    std::vector<float> value_cache; // (layer, seq_len, dim)
};

using ConfigType = Config;
using RunStateType = RunState;
using TransformerWeightsType = TransformerWeights;

// manager model loading
class ModelManager {
private:
    bool shared_weights_;
    std::unique_ptr<ConfigType> config_;
    std::unique_ptr<RunStateType> state_;
    std::unique_ptr<TransformerWeightsType> weight_;
    void malloc_weights();
    void malloc_run_state();
    // private
    ModelManager(): shared_weights_(true) {
        config_ = std::make_unique<ConfigType>();
        state_ = std::make_unique<RunStateType>();
        weight_ = std::make_unique<TransformerWeightsType>();
    };
    ~ModelManager() = default;

public:
    static ModelManager& get_instance() {
        static ModelManager model_manager;
        return model_manager;
    }

    ModelManager(const ModelManager&) = delete;
    ModelManager& operator=(const ModelManager&) = delete;
    
    // move ModelManager owner priority to caller 
    std::unique_ptr<ConfigType> config() { return std::move(config_); }
    std::unique_ptr<RunStateType> state() { return std::move(state_); }
    std::unique_ptr<TransformerWeightsType> weight() { return std::move(weight_); }
    
    // ptrs to visit data, but not use   
    ConfigType* config_ptr() { return config_.get(); }
    RunStateType* state_ptr() { return state_.get(); }
    TransformerWeightsType* weights_ptr() { return weight_.get(); }
    const ConfigType* config_ptr() const { return config_.get(); }
    const RunStateType* state_ptr() const { return state_.get(); }
    const TransformerWeightsType* weights_ptr() const { return weight_.get(); }
    
    // load model
    void load(std::string_view ckpt_path);
    void reset() {
        config_ = std::make_unique<ConfigType>();
        state_ = std::make_unique<RunStateType>();
        weight_ = std::make_unique<TransformerWeightsType>();
    }
};

// ----------------------------------------------------------------------------
// Transformer model

class Transformer {
private:
    std::unique_ptr<ConfigType> config_;
    std::unique_ptr<RunStateType> state_;
    std::unique_ptr<TransformerWeightsType> weight_;

public:
    Transformer() = default;
    ~Transformer() = default;
    
    void load_model(std::string_view ckpt_path);
    
    std::vector<float> forward(int token, int pos);
};


#endif // RUN_HPP_