#include "runq.hpp"

// ----------------------------------------------------------------------------
// Globals
int GS = 0; // group size global for quantization of the weights

// ----------------------------------------------------------------------------
// Transformer model

void Transformer::malloc_run_state() {
    int kv_dim = (config_->dim * config_->n_kv_heads) / config_->n_heads;
    // buffer for activation and hidden
    state_->x.resize(config_->dim);
    state_->xb.resize(config_->dim);
    state_->xb2.resize(config_->dim);
    state_->hb.resize(config_->hidden_dim);
    state_->hb2.resize(config_->hidden_dim);
    // buffer for attention
    state_->q.resize(config_->dim);    
    state_->k = nullptr;
    state_->v = nullptr;
    state_->att.resize(config_->n_heads * config_->seq_len);
    // buffer for output
    state_->logits.resize(config_->vocab_size);
    // kv cache
    state_->key_cache.resize(config_->n_layers * config_->seq_len * kv_dim);
    state_->value_cache.resize(config_->n_layers * config_->seq_len * kv_dim);
    // check if succeed alloc
    if (state_->x.empty() || state_->xb.empty() || state_->xb2.empty() || state_->hb.empty() || 
        state_->hb2.empty() || state_->q.empty() || state_->att.empty() || state_->logits.empty() ||
        state_->key_cache.empty() || state_->value_cache.empty()) {
            throw std::runtime_error("Malloc for run state failed.");
    } 

    // buffer for quantized
    state_->xq.resize(1);
    state_->hq.resize(1);
    if (state_->xq.empty() || state_->hq.empty()) {
        throw std::runtime_error("Malloc for run state xq or hq failed.");
    }

    state_->xq[0].q.resize(config_->dim);
    state_->xq[0].s.resize(config_->dim);
    if (state_->xq[0].q.empty() || state_->xq[0].s.empty()) {
        throw std::runtime_error("Malloc for run state xq[0] failed.");
    }

    state_->hq[0].q.resize(config_->hidden_dim);
    state_->hq[0].s.resize(config_->hidden_dim);
    if (state_->hq[0].q.empty() || state_->hq[0].s.empty()) {
        throw std::runtime_error("Malloc for run state hq[0] failed.");
    }

};

void Transformer::malloc_weights() {
    int head_size = config_->dim / config_->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = config_->n_layers;
    weight_->q_tokens.resize(1);
    weight_->token_embedding_table.resize(config_->vocab_size * config_->dim);
    // rmsnorms layer
    weight_->rms_att_weight.resize(n_layers * config_->dim);
    weight_->rms_ffn_weight.resize(n_layers * config_->dim);
    weight_->rms_final_weight.resize(config_->dim);
    // attention block
    weight_->wq.resize(n_layers * config_->dim * config_->n_heads * head_size);
    weight_->wk.resize(n_layers * config_->dim * config_->n_kv_heads * head_size);
    weight_->wv.resize(n_layers * config_->dim * config_->n_kv_heads * head_size);
    weight_->wo.resize(n_layers * config_->dim * config_->n_heads * head_size);
    // ffn
    weight_->w1.resize(n_layers * config_->dim * config_->hidden_dim);
    weight_->w2.resize(n_layers * config_->dim * config_->hidden_dim);
    weight_->w3.resize(n_layers * config_->dim * config_->hidden_dim);

    if (!shared_weights_) {
        weight_->wcls.resize(config_->vocab_size * config_->dim);
        if (weight_->wcls.empty()) {
            throw std::runtime_error("Malloc for wcls weights failed.");
        }
    }
    
    if (weight_->token_embedding_table.empty() || weight_->rms_att_weight.empty() || 
        weight_->rms_ffn_weight.empty() || weight_->rms_final_weight.empty() || 
        weight_->wq.empty() || weight_->wk.empty() || weight_->wv.empty() || 
        weight_->wo.empty() || weight_->w1.empty() || weight_->w2.empty() || 
        weight_->w3.empty()) {
            throw std::runtime_error("Malloc for weights failed.");
    }
};

void Transformer::load_model(std::string_view ckpt_path) {
    std::ifstream file(ckpt_path.data(), std::ios::binary);
    if (!file.is_open()) {
        std::ostringstream err_msg;
        err_msg << "Failed to open file: " << ckpt_path;
        throw std::runtime_error(err_msg.str());
    }  

    // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
    uint32_t magic_number;
    file.read(reinterpret_cast<char*>(&magic_number), sizeof(uint32_t));
    if (magic_number != 0x616b3432) {
        std::cerr << "Bad magic number\n";
        std::exit(EXIT_FAILURE);
    }
    // read in the version number (uint32), has to be 2
    int version;
    file.read(reinterpret_cast<char*>(&version), sizeof(int));
    if (version != 2) {
        std::cerr << "Bad version " << version << ", need version 2\n";
        std::exit(EXIT_FAILURE);
    }
    // read in the config header
    if (!file.read(reinterpret_cast<char*>(config_.get()), sizeof(ConfigType))) {
        throw std::runtime_error("Failed to read config data");
    }

    // read in flags
    uint8_t shared_classifier; // a byte to indicate if the classifier is shared
    file.read(reinterpret_cast<char*>(&shared_classifier), sizeof(uint8_t));

    int group_size;
    file.read(reinterpret_cast<char*>(&group_size), sizeof(int));
    GS = group_size; 

    shared_weights = static_cast<bool><shared_classifier>;

    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    // shared_weights_ = config_->vocab_size > 0 ? true : false;
    // config_->vocab_size = std::abs(config_->vocab_size);
    
    // init transformer weights
    malloc_weights();
    int head_size = config_->dim / config_->n_heads;
    unsigned long long n_layers = config_->n_layers;
    int header_size = 256;

    file.seekg(header_size, std::ios::beg);
    file.read(reinterpret_cast<char*>(weight_->rms_att_weight.data()), n_layers * config_->dim * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->rms_ffn_weight.data()), n_layers * config_->dim * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->rms_final_weight.data()), config_->dim * sizeof(float));

    init_quantized_tensors(file, weight_->q_tokens, 1, config_->vocab_size * config_->dim);
    dequantize(weight_->q_tokens, weight_->token_embedding_table, config_->vocab_size * config_->dim);

    init_quantized_tensors(file, weight_->wq, n_layers, config_->dim * config_->n_heads * head_size);
    







    // file.read(reinterpret_cast<char*>(weight_->token_embedding_table.data()), config_->vocab_size * config_->dim * sizeof(float));
    // file.read(reinterpret_cast<char*>(weight_->rms_att_weight.data()), n_layers * config_->dim * sizeof(float));
    // file.read(reinterpret_cast<char*>(weight_->wq.data()), n_layers * config_->dim * config_->n_heads * head_size * sizeof(float));
    // file.read(reinterpret_cast<char*>(weight_->wk.data()), n_layers * config_->dim * config_->n_kv_heads * head_size * sizeof(float));
    // file.read(reinterpret_cast<char*>(weight_->wv.data()), n_layers * config_->dim * config_->n_kv_heads * head_size * sizeof(float));
    // file.read(reinterpret_cast<char*>(weight_->wo.data()), n_layers * config_->dim * config_->n_heads * head_size * sizeof(float));
    // file.read(reinterpret_cast<char*>(weight_->rms_ffn_weight.data()), n_layers * config_->dim * sizeof(float));
    // file.read(reinterpret_cast<char*>(weight_->w1.data()), n_layers * config_->dim * config_->hidden_dim * sizeof(float));
    // file.read(reinterpret_cast<char*>(weight_->w2.data()), n_layers * config_->dim * config_->hidden_dim * sizeof(float));
    // file.read(reinterpret_cast<char*>(weight_->w3.data()), n_layers * config_->dim * config_->hidden_dim * sizeof(float));
    // file.read(reinterpret_cast<char*>(weight_->rms_final_weight.data()), config_->dim * sizeof(float));

    if (!shared_weights_) {
        // skip what used to be freq_cis_real (for RoPE), size is seq_len * head_size / 2
        // skip what used to be freq_cis_imag (for RoPE), size is seq_len * head_size / 2
        file.seekg((config_->seq_len * head_size) * sizeof(float), std::ios::cur);
        file.read(reinterpret_cast<char*>(weight_->wcls.data()), config_->vocab_size * config_->dim * sizeof(float));
    }
    else {
        weight_->wcls = weight_->token_embedding_table;
    }
    file.close();
    malloc_run_state();
    // 
    model_->config = std::move(config_);
    model_->state = std::move(state_);
    model_->weight = std::move(weight_);
};


