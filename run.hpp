#ifndef RUN_HPP_
#define RUN_HPP_

#include "utility.hpp"

// ----------------------------------------------------------------------------
// struct definitions

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
    float *k;
    float *v;
    std::vector<float> att; // buffer for scores/attention values (n_heads, seq_len)
    std::vector<float> logits; // output logits
    // kv cache
    std::vector<float> key_cache;   // (layer, seq_len, dim)
    std::vector<float> value_cache; // (layer, seq_len, dim)
};

using RunStateType = RunState;
using TransformerWeightsType = TransformerWeights;

// transformer model
struct TransformerModel {
    std::unique_ptr<ConfigType> config;
    std::unique_ptr<RunStateType> state;
    std::unique_ptr<TransformerWeightsType> weight;
};
using TransformerModelType = TransformerModel;

// ----------------------------------------------------------------------------
// Transformer model

class Transformer {
private:
    bool shared_weights_;
    std::unique_ptr<ConfigType> config_;
    std::unique_ptr<RunStateType> state_;
    std::unique_ptr<TransformerWeightsType> weight_;
    std::unique_ptr<TransformerModelType> model_;
    void malloc_weights();
    void malloc_run_state();

public:
    explicit Transformer(bool shared_weights): shared_weights_(shared_weights) {
        config_ = std::make_unique<ConfigType>();
        state_ = std::make_unique<RunStateType>();
        weight_ = std::make_unique<TransformerWeightsType>();
        model_ = std::make_unique<TransformerModelType>();
    };

    Transformer(): shared_weights_(true) {
        config_ = std::make_unique<ConfigType>();
        state_ = std::make_unique<RunStateType>();
        weight_ = std::make_unique<TransformerWeightsType>();
        model_ = std::make_unique<TransformerModelType>();
    };
    ~Transformer() = default;
    
    void load_model(std::string_view ckpt_path);
    
    std::vector<float> forward(int token, int pos);

    int get_vocab_size() const { return model_->config->vocab_size; }

    int get_seq_len() const { return model_->config->seq_len; }
};



#endif // RUN_HPP_