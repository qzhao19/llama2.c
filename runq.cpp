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

    // buffer for quantized x and h
    state_->xq.resize(1);
    state_->xq.data()->q.resize(config_->dim);
    state_->xq.data()->s.resize(config_->dim);
    if (!state_->xq.empty() || state_->xq->q.empty() || state_->xq->s.empty()) {
        throw std::runtime_error("Malloc for run state xq[0] failed.");
    }

    state_->hq.resize(1);
    state_->hq.data()->q.resize(config_->hidden_dim);
    state_->hq.data()->s.resize(config_->hidden_dim);
    if (state_->hq.empty() || state_->hq->q.empty() || state_->hq->s.empty()) {
        throw std::runtime_error("Malloc for run state hq[0] failed.");
    }

};

void Transformer::malloc_weights() {
    int head_size = config_->dim / config_->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = config_->n_layers;
    
    weight_->token_embedding_table.resize(config_->vocab_size * config_->dim);
    weight_->rms_att_weight.resize(n_layers * config_->dim);
    weight_->wq.resize(n_layers);
    weight_->wk.resize(n_layers);
    weight_->wv.resize(n_layers);
    weight_->wo.resize(n_layers);
    weight_->rms_ffn_weight.resize(n_layers * config_->dim);
    weight_->w1.resize(n_layers);
    weight_->w2.resize(n_layers);
    weight_->w3.resize(n_layers);
    weight_->rms_final_weight.resize(config_->dim);
    // weight_->q_tokens = new QuantizedTensorType();
    weight_->q_tokens.resize(1);

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

    shared_weights_ = static_cast<bool><shared_classifier>;

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

    init_quantized_tensors(file, weight_->q_tokens.data(), 1, config_->vocab_size * config_->dim);
    dequantize(weight_->q_tokens.data(), weight_->token_embedding_table.data(), config_->vocab_size * config_->dim);

    init_quantized_tensors(file, weight_->wq.data(), n_layers, config_->dim * config_->n_heads * head_size);
    init_quantized_tensors(file, weight_->wk.data(), n_layers, config_->dim * config_->n_kv_heads * head_size);
    init_quantized_tensors(file, weight_->wv.data(), n_layers, config_->dim * config_->n_kv_heads * head_size);
    init_quantized_tensors(file, weight_->wo.data(), n_layers, config_->dim * config_->n_heads * head_size);
    
    init_quantized_tensors(file, weight_->w1.data(), n_layers, config_->dim * config_->hidden_dim);
    init_quantized_tensors(file, weight_->w2.data(), n_layers, config_->dim * config_->hidden_dim);
    init_quantized_tensors(file, weight_->w3.data(), n_layers, config_->dim * config_->hidden_dim);

    if (!shared_weights_) {
        init_quantized_tensors(file, weight_->wcls.data(), 1, config_->vocab_size * config_->dim);
    }
    file.close();
    malloc_run_state();
    // 
    model_->config = std::move(config_);
    model_->state = std::move(state_);
    model_->weight = std::move(weight_);
};


std::vector<float> Transformer::forward(int token, int pos) {
    int dim = model_->config->dim;
    int kv_dim = (model_->config->dim * model_->config->n_kv_heads) / model_->config->n_heads;
    int kv_mul = model_->config->n_heads / model_->config->n_kv_heads; // integer multiplier of the kv sharing in multi-query
    int hidden_dim = model_->config->hidden_dim;
    int head_size = dim / model_->config->n_heads;

    // copy the token embedding into x
    std::memcpy(model_->state->x.data(), &model_->weight->token_embedding_table[token * dim], dim * sizeof(float));

    // forward all the layers
    for(int l = 0; l < model_->config->n_layers; l++) {

        // attention rmsnorm
        rmsnorm(model_->state->xb.data(), model_->state->x.data(), &model_->weight->rms_att_weight[l * dim], dim);

        // key and value point to the kv cache
        int loff = l * model_->config->seq_len * kv_dim; // kv cache layer offset for convenience
        model_->state->k = &model_->state->key_cache[loff + pos * kv_dim];
        model_->state->v = &model_->state->value_cache[loff + pos * kv_dim];

        // qkv matmuls for this position
        quantize(model_->state->xq.data(), model_->state->xb.data(), dim);
        matmul(model_->state->q.data(), model_->state->xq.data(), &model_->weight->wq[l], dim, dim);
        matmul(model_->state->k, model_->state->xq.data(), &model_->weight->wq[l], dim, dim);
        matmul(model_->state->v, model_->state->xq.data(), &model_->weight->wq[l], dim, dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                float *vec = v == 0 ? model_->state->q.data() : model_->state->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention. iterate over all heads
        int h;
        #pragma omp parallel for private(h)
        for (h = 0; h < model_->config->n_heads; h++) {
            // get the query vector for this head
            float* q = &model_->state->q[h * head_size];
            // attention scores for this head
            float* att = &model_->state->att[h * model_->config->seq_len];
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float *k = &model_->state->key_cache[loff + t * kv_dim + (h / kv_mul) * head_size]
                // calculate the attention score as the dot product of q and k
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                // save the score to the attention buffer
                att[t] = score;
            }
            // softmax the scores to get attention weights, from 0..pos inclusively
            softmax(att, pos + 1);

            float *xb = &model_->state->xb[h * head_size];
            std::memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                // get the value vector for this head and at this timestep
                float* v = &model_->state->value_cache[loff + t * kv_dim + (h / kv_mul) * head_size];
                // get the attention weight for this timestep
                float a = att[t];
                // accumulate the weighted value into xb
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        // final matmul to get the output of the attention
        quantize(model_->state->xq.data(), model_->state->xb, dim);
        matmul(model_->state->xb2.data(), model_->state->xq.data(), &model_->weight->wo[l], dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; ++i) {
            model_->state->x[i] += model_->state->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(model_->state->xb, model_->state->x, &model_->weight->rms_ffn_weight[l * dim], dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        quantize(model_->state->xq.data(), model_->state->xb.data(), dim);
        matmul(model_->state->hb.data(), model_->state->xq.data(), model_->weight->w1.data(), dim, hidden_dim);
        matmul(model_->state->hb2.data(), model_->state->xq.data(), model_->weight->w3.data(), dim, hidden_dim);

        // SwiGLU non-linearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = model_->state->hb[i];
            // silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
            val *= (1.0f / (1.0f + expf(-val)));
            // elementwise multiply with w3(x)
            val *= model_->state->hb2[i];
            model_->state->hb[i] = val;
        }

        // final matmul to get the output of the ffn
        quantize(model_->state->hq.data(), model_->state->hb.data(), hidden_dim);
        matmul(model_->state->xb.data(), model_->state->hq.data(), &model_->weight->w2[l], hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            model_->state->x[i] += model_->state->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(model_->state->x.data(), model_->state->x.data(), model_->weight->rms_final_weight.data(), dim);

    // classifier into logits
    quantize(model_->state->xq.data(), model_->state->x.data(), dim);
    if (!shared_weights_) {
        matmul(model_->state->logits.data(), model_->state->xq.data(), model_->weight->wcls.data(), dim, model_->config->vocab_size);
    }
    else {
        matmul(model_->state->logits.data(), model_->state->xq.data(), model_->weight->q_tokens.data(), dim, model_->config->vocab_size);

    }
    return model_->state->logits;
}