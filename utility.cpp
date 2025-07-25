#include "utility.hpp"

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
