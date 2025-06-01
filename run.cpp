#include "run.hpp"
int GS = 0;

// ----------------------------------------------------------------------------
// Model loading

void ModelManager::malloc_run_state() {
    int kv_dim = (config_->dim * config_->n_kv_heads) / config_->n_heads;
    // buffer for activation and hidden
    state_->x.resize(config_->dim);
    state_->xb.resize(config_->dim);
    state_->xb2.resize(config_->dim);
    state_->hb.resize(config_->hidden_dim);
    state_->hb2.resize(config_->hidden_dim);
    // buffer for attention
    state_->q.resize(config_->dim);
    state_->k.resize(kv_dim);
    state_->v.resize(kv_dim);
    state_->att.resize(config_->n_heads * config_->seq_len);
    // buffer for output
    state_->logits.resize(config_->vocab_size);
    // kv cache
    state_->key_cache.resize(config_->n_layers * config_->seq_len * kv_dim);
    state_->value_cache.resize(config_->n_layers * config_->seq_len * kv_dim);
    // check if succeed alloc
    if (state_->x.empty() || state_->xb.empty() || state_->xb2.empty() || state_->hb.empty() || state_->hb2.empty() || 
        state_->q.empty() || state_->k.empty() || state_->v.empty() || state_->att.empty() || state_->logits.empty() || 
        state_->key_cache.empty() || state_->value_cache.empty()) {
            throw std::runtime_error("Malloc for run state failed.");
    }
    else {
        std::cout << "Succeed allocate for run state.\n";
    }
};

void ModelManager::malloc_weights() {
    int head_size = config_->dim / config_->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = config_->n_layers;
    weight_->token_embedding_table.resize(config_->vocab_size * config_->dim);
    weight_->rms_att_weight.resize(n_layers * config_->dim);
    weight_->rms_ffn_weight.resize(n_layers * config_->dim);
    weight_->rms_final_weight.resize(config_->dim);
    // 
    weight_->wq.resize(n_layers * config_->dim * config_->n_heads * head_size);
    weight_->wk.resize(n_layers * config_->dim * config_->n_kv_heads * head_size);
    weight_->wv.resize(n_layers * config_->dim * config_->n_kv_heads * head_size);
    weight_->wo.resize(n_layers * config_->dim * config_->n_heads * head_size);
    // 
    weight_->w1.resize(n_layers * config_->dim * config_->hidden_dim);
    weight_->w2.resize(n_layers * config_->dim * config_->hidden_dim);
    weight_->w3.resize(n_layers * config_->dim * config_->hidden_dim);
    weight_->wcls.resize(config_->vocab_size * config_->dim);
    
    if (weight_->token_embedding_table.empty() || weight_->rms_att_weight.empty() || 
        weight_->rms_ffn_weight.empty() || weight_->rms_final_weight.empty() || 
        weight_->wq.empty() || weight_->wk.empty() || weight_->wv.empty() || 
        weight_->wo.empty() || weight_->w1.empty() || weight_->w2.empty() || 
        weight_->w3.empty() || weight_->wcls.empty()) {
            throw std::runtime_error("Malloc for weights failed.");
    }
    else {
        std::cout << "Succeed allocate for transformer weights.\n";
    }
};

void ModelManager::load(std::string_view ckpt_path) {
    std::ifstream file(ckpt_path.data(), std::ios::binary);
    if (!file) {
        std::ostringstream err_msg;
        err_msg << "Failed to open file: " << ckpt_path;
        throw std::runtime_error(err_msg.str());
    }  

    // read in the config header
    if (!file.read(reinterpret_cast<char*>(config_.get()), sizeof(ConfigType))) {
        throw std::runtime_error("Failed to read config data");
    }

    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    shared_weights_ = config_->vocab_size > 0 ? true : false;
    config_->vocab_size = std::abs(config_->vocab_size);
    
    // init transformer weights
    malloc_weights();
    int head_size = config_->dim / config_->n_heads;
    unsigned long long n_layers = config_->n_layers;

    file.read(
        reinterpret_cast<char*>(weight_->token_embedding_table.data()), 
        config_->vocab_size * config_->dim * sizeof(float)
    );
    file.read(
        reinterpret_cast<char*>(weight_->rms_att_weight.data()), 
        n_layers * config_->dim * sizeof(float)
    );
    file.read(
        reinterpret_cast<char*>(weight_->rms_ffn_weight.data()), 
        n_layers * config_->dim * sizeof(float)
    );
    file.read(
        reinterpret_cast<char*>(weight_->rms_final_weight.data()), 
        config_->dim * sizeof(float)
    );
    file.read(
        reinterpret_cast<char*>(weight_->wq.data()), 
        n_layers * config_->dim * config_->n_heads * head_size * sizeof(float)
    );
    file.read(
        reinterpret_cast<char*>(weight_->wk.data()), 
        n_layers * config_->dim * config_->n_kv_heads * head_size * sizeof(float)
    );
    file.read(
        reinterpret_cast<char*>(weight_->wv.data()), 
        n_layers * config_->dim * config_->n_kv_heads * head_size * sizeof(float)
    );
    file.read(
        reinterpret_cast<char*>(weight_->wo.data()), 
        n_layers * config_->dim * config_->n_heads * head_size * sizeof(float)
    );
    file.read(
        reinterpret_cast<char*>(weight_->w1.data()), 
        n_layers * config_->dim * config_->hidden_dim * sizeof(float)
    );
    file.read(
        reinterpret_cast<char*>(weight_->w2.data()), 
        n_layers * config_->dim * config_->hidden_dim * sizeof(float)
    );
    file.read(
        reinterpret_cast<char*>(weight_->w3.data()), 
        n_layers * config_->dim * config_->hidden_dim * sizeof(float)
    );

    if (!shared_weights_) {
        // skip what used to be freq_cis_real (for RoPE), size is seq_len * head_size / 2
        // skip what used to be freq_cis_imag (for RoPE), size is seq_len * head_size / 2
        file.seekg((config_->seq_len * head_size) * sizeof(float), std::ios::cur);
        file.read(
            reinterpret_cast<char*>(weight_->wcls.data()), 
            config_->vocab_size * config_->dim * sizeof(float)
        );
    }
    else {
        weight_->wcls = weight_->token_embedding_table;
    }
    file.close();
    malloc_run_state();
};

// ----------------------------------------------------------------------------
// Transformer model

void Transformer::load_model(std::string_view ckpt_path) { 
    ModelManager& model_manager = ModelManager::get_instance();
    model_manager.load(ckpt_path); 

    // a few convenience variables 
    config_ = model_manager.config();
    state_ = model_manager.state();
    weight_ = model_manager.weight();
}

// token: index of the currently vocab
std::vector<float> Transformer::forward(int token, int pos) {
    int dim = config_->dim;
    int kv_dim = (config_->dim * config_->n_kv_heads) / config_->n_heads;
    int kv_mul = config_->n_heads / config_->n_kv_heads; // integer multiplier of the kv sharing in multi-query
    int hidden_dim = config_->hidden_dim;
    int head_size = dim / config_->n_heads;


    return {};

}