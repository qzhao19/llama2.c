#include "gemm.hpp"
#include "run.hpp"
int GS = 0;

// ----------------------------------------------------------------------------
// Tokenizer step

void Tokenizer::build_tokenizer(std::string_view tokenizer_path, int vocab_size) {
    tokenizer_data_->vocab_size = vocab_size ;
    tokenizer_data_->vocab.resize(vocab_size);
    tokenizer_data_->vocab_scores.resize(vocab_size);

    for (int i = 0; i < 256; i++) {
        tokenizer_data_->byte_pieces[i * 2] = static_cast<unsigned char>(i);
        tokenizer_data_->byte_pieces[i * 2 + 1] = '\0';
    }

    std::ifstream file(tokenizer_path.data(), std::ios::binary);
    if (!file.is_open()) {
        std::ostringstream err_msg;
        err_msg << "Failed to open file: " << tokenizer_path.data();
        throw std::runtime_error(err_msg.str());
    }

    file.read(reinterpret_cast<char*>(&tokenizer_data_->max_token_length), sizeof(int));
    int len = 0;
    for (int i = 0; i < vocab_size; i++) {
        file.read(reinterpret_cast<char*>(&tokenizer_data_->vocab_scores[i]), sizeof(float));
        file.read(reinterpret_cast<char*>(&len), sizeof(int));
        tokenizer_data_->vocab[i] = std::make_unique<char[]>(len + 1);
        file.read(tokenizer_data_->vocab[i].get(), len);
        tokenizer_data_->vocab[i][len] = '\0';
    }
    file.close();
};

void Tokenizer::encode(const std::string &text, const int8_t &bos, const int8_t &eos, 
                       std::vector<int> &tokens, int &num_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text.empty()) { 
        throw std::runtime_error("cannot encode NULL text."); 
    }
    
    // init sorted_vocab
    if (tokenizer_data_->sorted_vocab.empty()) {
        // lazily malloc and sort the vocabulary
        tokenizer_data_->sorted_vocab.resize(tokenizer_data_->vocab_size);
        for (int i = 0; i < tokenizer_data_->vocab_size; ++i) {
            tokenizer_data_->sorted_vocab[i].str = std::string(tokenizer_data_->vocab[i].get());
            tokenizer_data_->sorted_vocab[i].id = i;
        }
        std::sort(tokenizer_data_->sorted_vocab.begin(), tokenizer_data_->sorted_vocab.end(),
            [](const TokenIndexType& a, const TokenIndexType& b) -> bool {
                return a.str < b.str;
            });
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    std::string string_buffer;
    string_buffer.resize(tokenizer_data_->max_token_length*2 + 1 + 2);
    std::size_t str_len = 0;

    // start at 0 tokens
    num_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(num_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = string_lookup(" ", tokenizer_data_->vocab_size, tokenizer_data_->sorted_vocab);
        tokens[(num_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (const char *c = text.c_str(); *c != '\0'; c++) {
        // printf("text c = %s\n", c);
        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        string_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        string_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning string_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = string_lookup(string_buffer, tokenizer_data_->vocab_size, tokenizer_data_->sorted_vocab);
        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(num_tokens)++] = id;
        } 
        else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(num_tokens)++] = (unsigned char)string_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (num_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(string_buffer.data(), "%s%s", 
                    tokenizer_data_->vocab[tokens[i]].get(), 
                    tokenizer_data_->vocab[tokens[i+1]].get());
            int id = string_lookup(string_buffer, 
                tokenizer_data_->vocab_size, 
                tokenizer_data_->sorted_vocab);
            if (id != -1 && tokenizer_data_->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = tokenizer_data_->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (num_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        --num_tokens; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(num_tokens)++] = 2;

};

std::string Tokenizer::decode(int prev_token, int token) {
    char* piece = tokenizer_data_->vocab[token].get();

    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { 
        piece++; 
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)tokenizer_data_->byte_pieces + byte_val * 2;
    }
    return std::string(piece);
}

// ----------------------------------------------------------------------------
// Sampler

int Sampler::sample_argmax(const std::vector<float> &proba) {
    // return the index that has the highest probability
    auto iter = std::max_element(proba.begin(), proba.end());
    return std::distance(proba.begin(), iter);
}

int Sampler::sample_mult(const std::vector<float> &proba, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < vocab_size_; i++) {
        cdf += proba[i];
        if (coin < cdf) {
            return i;
        }
    }
    return vocab_size_ - 1; // in case of rounding errors
}

int Sampler::sample_topp(const std::vector<float> &proba, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp_) / (vocab_size_ - 1);
    for (int i = 0; i < vocab_size_; i++) {
        if (proba[i] >= cutoff) {
            proba_index_[n0].index = i;
            proba_index_[n0].proba = proba[i];
            n0++;
        }
    }
    std::sort(proba_index_.begin(), proba_index_.begin() + n0, 
            [](const ProbaIndexType& a, const ProbaIndexType& b) -> bool {
                return a.proba > b.proba;
            });
    
    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += proba_index_[i].proba;
        if (cumulative_prob > topp_) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += proba_index_[i].proba;
        if (r < cdf) {
            return proba_index_[i].index;
        }
    }
    return proba_index_[last_idx].index; // in case of rounding errors
}

int Sampler::sample(std::vector<float> &logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (temperature_ == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits);
    } 
    else {
        // apply the temperature to the logits
        for (int q=0; q < vocab_size_; q++) { 
            logits[q] /= temperature_; 
        }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits.data(), vocab_size_);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&rng_state_);
        // we sample from this distribution to get the next token
        if (topp_ <= 0 || topp_ >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, coin);
        } 
        else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, coin);
        }
    }
    return next;
}

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
};

void Transformer::malloc_weights() {
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
};

void Transformer::load_model(std::string_view ckpt_path) {
    std::ifstream file(ckpt_path.data(), std::ios::binary);
    if (!file.is_open()) {
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

    file.read(reinterpret_cast<char*>(weight_->token_embedding_table.data()), config_->vocab_size * config_->dim * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->rms_att_weight.data()), n_layers * config_->dim * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->wq.data()), n_layers * config_->dim * config_->n_heads * head_size * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->wk.data()), n_layers * config_->dim * config_->n_kv_heads * head_size * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->wv.data()), n_layers * config_->dim * config_->n_kv_heads * head_size * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->wo.data()), n_layers * config_->dim * config_->n_heads * head_size * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->rms_ffn_weight.data()), n_layers * config_->dim * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->w1.data()), n_layers * config_->dim * config_->hidden_dim * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->w2.data()), n_layers * config_->dim * config_->hidden_dim * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->w3.data()), n_layers * config_->dim * config_->hidden_dim * sizeof(float));
    file.read(reinterpret_cast<char*>(weight_->rms_final_weight.data()), config_->dim * sizeof(float));

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

// token: index of the currently vocab
std::vector<float> Transformer::forward(int token, int pos) {
    int dim = model_->config->dim;
    int kv_dim = (model_->config->dim * model_->config->n_kv_heads) / model_->config->n_heads;
    int kv_mul = model_->config->n_heads / model_->config->n_kv_heads; // integer multiplier of the kv sharing in multi-query
    int hidden_dim = model_->config->hidden_dim;
    int head_size = dim / model_->config->n_heads;

    // copy the token embedding into x
    std::memcpy(model_->state->x.data(), model_->weight->token_embedding_table.data() + token * dim, dim * sizeof(float));

    // forward all the layers
    for (unsigned long long l = 0; l < model_->config->n_layers; l++) {
        // attention rmsnorm
        rmsnorm(model_->state->xb.data(), model_->state->x.data(), model_->weight->rms_att_weight.data() + l * dim, dim);

        // qkv matmuls for this position
        matmul(model_->state->q.data(), model_->state->xb.data(), model_->weight->wq.data() + l * dim * dim, dim, dim);
        matmul(model_->state->k.data(), model_->state->xb.data(), model_->weight->wk.data() + l * dim * kv_dim, dim, kv_dim);
        matmul(model_->state->v.data(), model_->state->xb.data(), model_->weight->wv.data() + l * dim * kv_dim, dim, kv_dim);
        
        // RoPE relative positional encoding: complex-valued rotate q and k in each head
        for (int i = 0; i < dim; i+=2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / static_cast<float>(head_size));
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
            for (int v = 0; v < rotn; v++) {
                std::vector<float> &vec = v == 0 ? model_->state->q : model_->state->k; // the vector to rotate (query or key)
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        // save key,value at this time step (pos) to our kv cache
        int loff = l * model_->config->seq_len * kv_dim;
        std::memcpy(model_->state->key_cache.data() + loff + pos * kv_dim, model_->state->k.data(), kv_dim * sizeof(float));
        std::memcpy(model_->state->value_cache.data() + loff + pos * kv_dim, model_->state->v.data(), kv_dim * sizeof(float));

        // multihead attention. iterate over all heads
        int h;
        for (h = 0; h < model_->config->n_heads; h++) {
            // get the query vector for this head
            // float* q = model_->state->q.data() + h * head_size;
            float* q = &model_->state->q[h * head_size];
            // attention scores for this head
            float* att = &model_->state->att[h * model_->config->seq_len];
            // iterate over all timesteps, including the current one
            for (int t = 0; t <= pos; t++) {
                // get the key vector for this head and at this timestep
                float* k = &model_->state->key_cache[loff + t * kv_dim + (h / kv_mul) * head_size];
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

            // weighted sum of the values, store back into xb
            float* xb = &model_->state->xb[h * head_size];
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
        matmul(model_->state->xb2.data(), model_->state->xb.data(), model_->weight->wo.data() + l * dim * dim, dim, dim);

        // residual connection back into x
        for (int i = 0; i < dim; i++) {
            model_->state->x[i] += model_->state->xb2[i];
        }

        // ffn rmsnorm
        rmsnorm(model_->state->xb.data(), model_->state->x.data(), model_->weight->rms_ffn_weight.data() + l * dim, dim);
        
        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        matmul(model_->state->hb.data(), model_->state->xb.data(), model_->weight->w1.data() + l * dim * hidden_dim, dim, hidden_dim);
        matmul(model_->state->hb2.data(), model_->state->xb.data(), model_->weight->w3.data() + l * dim * hidden_dim, dim, hidden_dim);

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
        matmul(model_->state->xb.data(), model_->state->hb.data(), model_->weight->w2.data() + l * dim * hidden_dim, hidden_dim, dim);

        // residual connection
        for (int i = 0; i < dim; i++) {
            model_->state->x[i] += model_->state->xb[i];
        }
    }

    // final rmsnorm
    rmsnorm(model_->state->x.data(), model_->state->x.data(), model_->weight->rms_final_weight.data(), dim);

    // classifier into logits
    matmul(model_->state->logits.data(), model_->state->x.data(), model_->weight->wcls.data(), model_->config->dim, model_->config->vocab_size);
    return model_->state->logits;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer &transformer, Tokenizer &tokenizer, Sampler &sampler, std::string& prompt, int steps) {
    std::string empty_prompt(1, '\0');
    if (prompt.empty()) {
        prompt = empty_prompt;
    }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    std::vector<int> prompt_tokens(std::strlen(prompt.c_str()) + 3);
    tokenizer.encode(prompt, 1, 0, prompt_tokens, num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        std::ostringstream err_msg;
        err_msg << "something is wrong, expected at least 1 prompt token.";
        throw std::runtime_error(err_msg.str());

    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {
        // forward the transformer to get logits for the next token
        std::vector<float> logits = transformer.forward(token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sampler.sample(logits);
        }
        pos++;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (next == 1) { break; }

        // print the token as string, decode it with the Tokenizer object
        std::string piece = tokenizer.decode(token, next);
        safe_print(piece); // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }

    }
    std::cout << "\n";

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        std::cerr << "achieved tok/s: " << (pos-1) / static_cast<double>(end-start) * 1000 << std::endl;
    }
}

// ----------------------------------------------------------------------------
// chat loop

void chat(Transformer &transformer, Tokenizer &tokenizer, Sampler &sampler,
          std::string& cli_user_prompt, std::string& cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    std::string system_prompt;
    std::string user_prompt;
    std::string rendered_prompt;
    int num_prompt_tokens = 0;
    // std::unique_ptr<int[]> prompt_tokens = std::make_unique<int[]>(1152);
    std::vector<int> prompt_tokens(1152);
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    while (pos < steps) {
        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt.empty()) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    system_prompt = cli_system_prompt;
                }
            }
            // get the user prompt
            if (pos == 0 && !cli_user_prompt.empty()) {
                // user prompt for position 0 was passed in, use it
                user_prompt = cli_user_prompt;
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && !system_prompt.empty()) {
                std::string system_template = "[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n" + user_prompt + " [/INST]";
                rendered_prompt = system_template;
            } else {
                std::string user_template = "[INST] " + user_prompt + " [/INST]";
                rendered_prompt = user_template;
            }
            // encode the rendered prompt into tokens
            tokenizer.encode(rendered_prompt, 1, 0, prompt_tokens, num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            std::cout<<"Assistant: ";
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }
        // EOS (=2) token ends the Assistant turn
        if (token == 2) { user_turn = 1; }

        // forward the transformer to get logits for the next token
        std::vector<float> logits = transformer.forward(token, pos);
        next = sampler.sample(logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != 2) {
            // the Assistant is responding, so print its output
            std::string piece = tokenizer.decode(token, next);
            safe_print(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        if (next == 2) { std::cout<<"\n"; }
    }
    std::cout<<"\n";
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    std::cerr << R"(Usage:   run <checkpoint> [options]
Example: run model.bin -n 256 -i "Once upon a time"
Options:
  -t <float>  temperature in [0,inf], default 1.0
  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9
  -s <int>    random seed, default time(NULL)
  -n <int>    number of steps to run for, default 256. 0 = max_seq_len
  -i <string> input prompt
  -z <string> optional path to custom tokenizer
  -m <string> mode: generate|chat, default: generate
  -y <string> (optional) system prompt in chat mode
)";
    std::exit(EXIT_FAILURE);
}

int main(int argc, char* argv[]) {
    std::string ckpt_path;
    std::string tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    std::string prompt;
    unsigned long long rng_seed = 0; // seed rng with time by default
    std::string mode = "generate";
    std::string system_prompt;

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { ckpt_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }
    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    Transformer transformer;
    transformer.load_model(ckpt_path);
    int vocab_size = transformer.get_vocab_size();
    int seq_len = transformer.get_seq_len();

    if (steps == 0 || steps > seq_len) steps = seq_len; // ovrerride to ~max length
    Tokenizer tokenizer(tokenizer_path, vocab_size);
    Sampler sampler(vocab_size, temperature, topp, rng_seed);

    // run!
    if (mode == "generate") {
        generate(transformer, tokenizer, sampler, prompt, steps);
    } else if (mode == "chat") {
        chat(transformer, tokenizer, sampler, prompt, system_prompt, steps);
    } else {
        std::cerr << "unknown mode: " << mode << "\n" <<std::endl;
        error_usage();
    }

    return 0;
}
#endif


