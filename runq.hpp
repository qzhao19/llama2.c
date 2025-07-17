#ifndef RUNQ_HPP_
#define RUNQ_HPP_

#include "utility.hpp"

// Global var group size for quantization of the weights
extern int GS;

// forward definition
// struct QuantizedTensor;
struct RunState;
struct TransformerWeights;
struct TransformerModel;

using RunStateType = RunState;
using TransformerWeightsType = TransformerWeights;
using TransformerModelType = TransformerModel;

// Quantization functions
inline void quantize(QuantizedTensorType *xq, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        xq->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = static_cast<int8_t>(round(quant_value)); // round and clamp
            xq->q[group * GS + i] = quantized;
        }
    }
}

inline void dequantize(QuantizedTensorType *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

/* initialize `n` x quantized tensor (with `size_each` elements), starting from memory pointed at *ptr */
inline void init_quantized_tensors(std::ifstream &file, QuantizedTensorType *w, int n_layers, int size_each) {
    for(int i = 0; i < n_layers; i++) {
        w[i].q.resize(size_each);
        w[i].s.resize(size_each / GS);
        file.read(reinterpret_cast<char*>(w[i].q.data()), size_each * sizeof(int8_t));
        file.read(reinterpret_cast<char*>(w[i].s.data()), size_each / GS * sizeof(float));
    }
}

// ----------------------------------------------------------------------------
// struct definitions


// all weights params of model
struct TransformerWeights {
    // embedding layer
    std::vector<QuantizedTensorType> q_tokens; // (vocab_size, dim)
    std::vector<float> token_embedding_table;  // token embedding table
    
    // rmsnorms layer
    std::vector<float> rms_att_weight; // (layer, dim) rmsnorm weights
    std::vector<float> rms_ffn_weight; // (layer, dim)
    std::vector<float> rms_final_weight; // (dim,) for the last layer

    // attention block
    // weights for matmuls. note dim == n_heads * head_size
    std::vector<QuantizedTensorType> wq; // (layer, dim, n_heads * head_size)
    std::vector<QuantizedTensorType> wk; // (layer, dim, n_kv_heads * head_size)
    std::vector<QuantizedTensorType> wv; // (layer, dim, n_kv_heads * head_size)
    std::vector<QuantizedTensorType> wo; // (layer, n_heads * head_size, dim)
    
    // weights for ffn
    std::vector<QuantizedTensorType> w1; // (layer, hidden_dim, dim)
    std::vector<QuantizedTensorType> w2; // (layer, dim, hidden_dim)
    std::vector<QuantizedTensorType> w3; // (layer, hidden_dim, dim)

    // (optional) classifier weights for the logits, on the last layer
    std::vector<QuantizedTensorType> wcls;
};

// all activation buffers and intermediate states during forward propagation
struct RunState {
    // current wave of activations
    std::vector<float> x; // activation at current time stamp (dim,)
    std::vector<float> xb; // same, but inside a residual branch (dim,)
    std::vector<float> xb2; // an additional buffer just for convenience (dim,)
    std::vector<float> hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    std::vector<float> hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    
    std::vector<QuantizedTensorType> xq; // quantized x (dim,)
    std::vector<QuantizedTensorType> hq; // quantized hb (hidden_dim,)
    
    std::vector<float> q; // query (dim,)
    float *k;
    float *v;
    std::vector<float> att; // buffer for scores/attention values (n_heads, seq_len)
    std::vector<float> logits; // output logits
    // kv cache
    std::vector<float> key_cache;   // (layer, seq_len, kv_dim)
    std::vector<float> value_cache; // (layer, seq_len, kv_dim)
};

// transformer model
struct TransformerModel {
    std::unique_ptr<ConfigType> config;
    std::unique_ptr<RunStateType> state;
    std::unique_ptr<TransformerWeightsType> weight;
};

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


#endif // RUNQ_HPP_