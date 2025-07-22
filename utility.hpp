#ifndef UTILITY_HPP_
#define UTILITY_HPP_

#include <algorithm>
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
#include <new>

#if defined _WIN32
    #include "win.h"
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

#if defined(__SSE__)
#include <smmintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h> 
#endif

#if defined(__SSE__)
inline __m128 add(__m128 x, __m128 y) { return _mm_add_ps(x, y); }
inline __m128 sub(__m128 x, __m128 y) { return _mm_sub_ps(x, y); }
inline __m128 mul(__m128 x, __m128 y) { return _mm_mul_ps(x, y); }
#endif  // __SSE4_2__

#if defined(__AVX2__)
inline __m256 add(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
inline __m256 sub(__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }
inline __m256 mul(__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }
#endif  // __AVX2__

/**
 * Computes a * b + c.
 * apply _mm256_fmadd_ps if enable fma
 */
template<typename T>
inline T madd(T a, T b, T c) {
    return add(mul(a, b), c);
}

#if defined(__SSE__) || defined(__AVX2__)
#if defined(__FMA__)
template<>
inline __m256 madd(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}
#endif
#endif

#if defined(__AVX2__)
template<>
inline __m256i madd(__m256i a, __m256i b, __m256i c) {
    // 1. plit 32 int8 values into two registers each containing 16 int8 values
    __m256i a_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 0));
    __m256i a_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
    __m256i b_lo = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 0));
    __m256i b_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));
    
    // 2. 16 int16 values are multiplied to get 16 int32 values
    __m256i prod_lo = _mm256_mullo_epi16(a_lo, b_lo);
    __m256i prod_hi = _mm256_mullo_epi16(a_hi, b_hi);
    
    // 3. accumulate 16 int32 values into 8 int32 values (by adding each pair of adjacent elements)
    __m256i sum_lo = _mm256_add_epi32(
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 0)),
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_lo, 1)));
    
    __m256i sum_hi = _mm256_add_epi32(
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 0)),
        _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_hi, 1)));
    
    // 4. combine the results and accumulate them into c
    __m256i sum = _mm256_hadd_epi32(sum_lo, sum_hi);
    return _mm256_add_epi32(c, sum);
}
#endif

#if defined(__SSE__)
inline float hsum(__m128 x) {
    __m128 t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_add_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = _mm_add_ss(x, t);
    return _mm_cvtss_f32(x);
}
#endif

#if defined(__AVX2__)
inline float hsum(__m256 x) {
    // split 256 bit vector into 128 bit, then use below hsum(__m128)
    __m128 hi = _mm256_extractf128_ps(x, 1);
    __m128 lo = _mm256_castps256_ps128(x);
    return hsum(_mm_add_ps(hi, lo));
}
inline int32_t hsum(__m256i x) {
    __m128i lo = _mm256_extracti128_si256(x, 0);
    __m128i hi = _mm256_extracti128_si256(x, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_extract_epi32(lo, 0);
}
#endif

// declaration of basic template function load
template <typename T, typename U> 
T load(const float *);

template <typename T, typename U> 
T load(const std::int8_t *);

#if defined(__SSE__)
template <> 
inline __m128 load<__m128, float>(const float *p) {
    return _mm_loadu_ps(p);
}

template <>
inline __m128i load<__m128i, std::int8_t>(const std::int8_t *p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}
#endif  // __SSE__

#if defined(__AVX2__)
template <> 
inline __m256 load<__m256, float>(const float *p) {
    return _mm256_loadu_ps(p);
}
template <>
inline __m256i load<__m256i, std::int8_t>(const std::int8_t *p) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
}
#endif // __AVX__

// declaration of basic template function setzeros
template <typename T>
inline T setzeros();

#if defined(__SSE__)
template <>
inline __m128 setzeros<__m128>() { return _mm_setzero_ps(); }
template <>
inline __m128i setzeros<__m128i>() { return _mm_setzero_si128(); }
#endif

#if defined(__AVX2__)
template <>
inline __m256 setzeros<__m256>() { return _mm256_setzero_ps(); }
template <>
inline __m256i setzeros<__m256i>() { return _mm256_setzero_si256(); }
#endif

// declaration set1 basic template function
template <typename T>
inline T set1(float x);

#if defined(__SSE__)
template <>
inline __m128 set1(float x) { return _mm_set1_ps(x); }
#endif

#if defined(__AVX2__)
template <>
inline __m256 set1(float x) { return _mm256_set1_ps(x); }
inline __m256i set1(short x) { return _mm256_set1_epi16(x); }
inline __m256i set1(int x) { return _mm256_set1_epi32(x); }
#endif

#if defined(__AVX2__)
inline void store(std::int8_t *a, __m256i b) {
    // _mm256_storeu_si256(a, b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(a), b);
}
#endif

#if defined(__SSE__)
inline void store(std::int8_t *a, __m128i b) {
    // _mm_storeu_si128(a, b);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a), b);
}
#endif

#define MEMORY_ALIGNMENT 32
#define UNROLLING_SIZE 16

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
// forward definition
struct Config;
struct TokenIndex;
struct TokenizerData;
struct ProbaIndex;

using ConfigType = Config;
using TokenIndexType = TokenIndex;
using TokenizerDataType = TokenizerData;
using ProbaIndexType = ProbaIndex;

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

struct TokenizerData {
    std::vector<std::unique_ptr<char[]>> vocab;
    std::vector<TokenIndexType> sorted_vocab;
    std::vector<float> vocab_scores;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
};

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

// ----------------------------------------------------------------------------

struct QuantizedTensor {
    std::vector<std::int8_t> q;   // quantized values
    std::vector<float> s;         // scaling factors
};
using QuantizedTensorType = QuantizedTensor;

template <typename T>
inline T* malloc_aligned(int m, int n, int size) {
    std::size_t bytes = m * n * size;
    if (bytes % MEMORY_ALIGNMENT != 0) {
        bytes += MEMORY_ALIGNMENT - (bytes % MEMORY_ALIGNMENT);
    }

    void *ptr = std::aligned_alloc(MEMORY_ALIGNMENT, bytes);
    if (!ptr) {
        throw std::bad_alloc();
    }
    std::memset(ptr, 0, m * n * size);

    return static_cast<T*>(ptr);
}

template <>
inline QuantizedTensorType* malloc_aligned(int m, int n, int size) {
    auto* ptr = static_cast<QuantizedTensorType*>(
        std::aligned_alloc(MEMORY_ALIGNMENT, size)
    );

    if (!ptr) {
        throw std::bad_alloc();
    }

    // use placement new to construct object
    new (ptr) QuantizedTensorType();

    std::vector<std::int8_t> q_vec(m * n, 0);
    const int GS = 32;
    std::vector<float> s_vec((m * n + GS - 1) / GS, 0.0f);

    ptr->q = std::move(q_vec);
    ptr->s = std::move(s_vec);

    return ptr;
}

inline void free_aligned(QuantizedTensorType* ptr) noexcept {
    if (!ptr) return;
    ptr->~QuantizedTensorType();
    std::free(ptr);
}

template <typename TA, typename TX, typename TY, 
          int RM = 4, int RN = 1, 
          int CM = 72, int CN = 256>
class GEMV {
private:
    const TA *const A_;
    const TX *const x_;
    TY *const y_;
    const int lda_;

public:
    GEMV(const TA *A, int lda, 
         const TX *x, TY *y) :
            A_(A), lda_(lda), 
            x_(x), y_(y) {};
    
    void multiply(int m, int n) {
        int ic, ib, jc, jb;
        
        #pragma omp parallel for
        for (ic = 0; ic < m; ic += CM) {
            ib = std::min(ic + CM, m);

            for (jc = 0; jc < n; jc += CN) {
                jb = std::min(jc + CN, n);

                for (int i = ic; i < ib; i += RM) {
                    const int nrows = std::min(ib - i, RM);
                    TY sum[RM] = {0.0f};
                    
                    // define RM AVX vector registers to accumulate results
                    __m256 y_j_ymm[RM];
                    for (int r = 0; r < nrows; ++r) {
                        y_j_ymm[r] = setzeros<__m256>();
                    }
                    // handle one row data
                    for (int j = jc; j + 7 < jb; j += 8) {
                        __m256 x_j_ymm = load<__m256, float>(&x_[j]);
                        for (int r = 0; r < nrows; ++r) {
                            __m256 a_rj_ymm = load<__m256, float>(&A_[(i + r) * lda_ + j]);
                            y_j_ymm[r] = madd<__m256>(a_rj_ymm, x_j_ymm, y_j_ymm[r]);
                        }
                    }
                    // compute horizontal sum of each row
                    for (int r = 0; r < nrows; ++r) {
                        sum[r] += hsum(y_j_ymm[r]);
                    }

                    // handle rest of elements
                    __m128 y_j_xmm[RM];
                    for (int r = 0; r < nrows; ++r) {
                        y_j_xmm[r] = setzeros<__m128>();
                    }
                    // find the remaining starting position after AVX2 processing
                    int j = jc;
                    while (j + 7 < jb) j += 8;
                    for (; j < jb; j += 4) {
                        __m128 x_j_xmm = load<__m128, float>(&x_[j]);
                        for (int r = 0; r < nrows; ++r) {
                            __m128 a_rj_xmm = load<__m128, float>(&A_[(i + r) * lda_ + j]);
                            y_j_xmm[r] = madd<__m128>(a_rj_xmm, x_j_xmm, y_j_xmm[r]);
                        }
                    }
                    // compute horizontal sum of each row
                    for (int r = 0; r < nrows; ++r) {
                        sum[r] += hsum(y_j_xmm[r]);
                    }

                    // handle last remaining
                    while (j + 7 < jb) j += 8;
                    while (j + 3 < jb) j += 4;
                    for (; j < jb; ++j) {
                        for (int r = 0; r < nrows; ++r) {
                            sum[r] += A_[(i + r) * lda_ + j] * x_[j];
                        }
                    }
                    for (int r = 0; r < nrows; ++r) {
                        y_[i + r] += sum[r];
                    }
                }
            }
        }
    }
};

inline void matmul_pseudo(float* xout, float* x, float* w, int n, int d) {
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
};

inline void matmul(float* xout, float* x, float* w, int n, int d) {
    int m = d;
    int lda = n;
    constexpr int RM = 4, RN = 1, CM = 72, CN = 256;
    float *C = malloc_aligned<float>(m, 1, sizeof(float));
    GEMV<float, float, float, RM, RN, CM, CN> gemv(
        w, lda, x, C
    );
    gemv.multiply(m, n);
    std::memcpy(xout, C, m * sizeof(float));
    free(C);
}

// ----------------------------------------------------------------------------

template <int GS, typename TA, typename TB, typename TC, int RM = 4, int RN = 1>
inline void AddDot_4x1_Q0(int k, const TA *a, int offset_A, const TB *b, int offset_B, TC *c, int ldc) {
    TC c_00_reg = 0.0, c_10_reg = 0.0, c_20_reg = 0.0, c_30_reg = 0.0;
    
    const int a_0p_ptr = offset_A + 0 * k;
    const int a_1p_ptr = offset_A + 1 * k;
    const int a_2p_ptr = offset_A + 2 * k;
    const int a_3p_ptr = offset_A + 3 * k;
    const auto& a_q = a->q;
    const auto& b_q = b->q;
    for (int group = 0; group < (k + GS - 1) / GS; ++group) {
        const int begin = group * GS;
        const int end = std::min(begin + GS, k);
        
        _mm_prefetch(reinterpret_cast<const char*>(&a->s[(a_0p_ptr + begin) / GS]), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(&b->s[(offset_B + begin) / GS]), _MM_HINT_T0);
        
        // 32-byte aligned accumulators
        alignas(32) int32_t acc_00_sum = 0;
        alignas(32) int32_t acc_10_sum = 0;
        alignas(32) int32_t acc_20_sum = 0;
        alignas(32) int32_t acc_30_sum = 0;
        
        int p = begin;
        if (end - begin >= 32) {
            __m256i acc_00_ymm = setzeros<__m256i>();
            __m256i acc_10_ymm = setzeros<__m256i>();
            __m256i acc_20_ymm = setzeros<__m256i>();
            __m256i acc_30_ymm = setzeros<__m256i>();
            
            for (; p + 31 < end; p += 32) {
                _mm_prefetch(reinterpret_cast<const char*>(&a_q[a_0p_ptr + p + 64]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&a_q[a_1p_ptr + p + 64]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&a_q[a_2p_ptr + p + 64]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&a_q[a_3p_ptr + p + 64]), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(&b_q[offset_B + p + 64]), _MM_HINT_T0);
                
                __m256i a_0p_ymm = load<__m256i, std::int8_t>(&a_q[a_0p_ptr + p]);
                __m256i a_1p_ymm = load<__m256i, std::int8_t>(&a_q[a_1p_ptr + p]);
                __m256i a_2p_ymm = load<__m256i, std::int8_t>(&a_q[a_2p_ptr + p]);
                __m256i a_3p_ymm = load<__m256i, std::int8_t>(&a_q[a_3p_ptr + p]);
                __m256i b_p0_ymm = load<__m256i, std::int8_t>(&b_q[offset_B + p]);
                
                acc_00_ymm = madd(a_0p_ymm, b_p0_ymm, acc_00_ymm);
                acc_10_ymm = madd(a_1p_ymm, b_p0_ymm, acc_10_ymm);
                acc_20_ymm = madd(a_2p_ymm, b_p0_ymm, acc_20_ymm);
                acc_30_ymm = madd(a_3p_ymm, b_p0_ymm, acc_30_ymm);
            }
            acc_00_sum = hsum(acc_00_ymm);
            acc_10_sum = hsum(acc_10_ymm);
            acc_20_sum = hsum(acc_20_ymm);
            acc_30_sum = hsum(acc_30_ymm);
        }
        
        if (end - p >= 4) {
            for (; p + 3 < end; p += 4) {
                for (int j = 0; j < 4; ++j) {
                    int index = p + j;
                    int8_t b_val = b_q[offset_B + index];
                    acc_00_sum += static_cast<int32_t>(a_q[a_0p_ptr + index]) * static_cast<int32_t>(b_val);
                    acc_10_sum += static_cast<int32_t>(a_q[a_1p_ptr + index]) * static_cast<int32_t>(b_val);
                    acc_20_sum += static_cast<int32_t>(a_q[a_2p_ptr + index]) * static_cast<int32_t>(b_val);
                    acc_30_sum += static_cast<int32_t>(a_q[a_3p_ptr + index]) * static_cast<int32_t>(b_val);
                }
            }
        }
        
        for (; p < end; ++p) {
            int8_t b_val = b_q[offset_B + p];
            acc_00_sum += static_cast<int32_t>(a_q[a_0p_ptr + p]) * static_cast<int32_t>(b_val);
            acc_10_sum += static_cast<int32_t>(a_q[a_1p_ptr + p]) * static_cast<int32_t>(b_val);
            acc_20_sum += static_cast<int32_t>(a_q[a_2p_ptr + p]) * static_cast<int32_t>(b_val);
            acc_30_sum += static_cast<int32_t>(a_q[a_3p_ptr + p]) * static_cast<int32_t>(b_val);
        }
        
        const float a_s0 = a->s[(a_0p_ptr + begin) / GS];
        const float a_s1 = a->s[(a_1p_ptr + begin) / GS];
        const float a_s2 = a->s[(a_2p_ptr + begin) / GS];
        const float a_s3 = a->s[(a_3p_ptr + begin) / GS];
        const float b_s = b->s[(offset_B + begin) / GS];
        
        const float combined_scale = b_s;
        c_00_reg += acc_00_sum * a_s0 * combined_scale;
        c_10_reg += acc_10_sum * a_s1 * combined_scale;
        c_20_reg += acc_20_sum * a_s2 * combined_scale;
        c_30_reg += acc_30_sum * a_s3 * combined_scale;
    }
    
    c[0 * ldc + 0] += c_00_reg;
    c[1 * ldc + 0] += c_10_reg;
    c[2 * ldc + 0] += c_20_reg;
    c[3 * ldc + 0] += c_30_reg;
}

template <int GS, typename TA, typename TB, typename TC, int RM, int RN>
using MicroKernelQ0Type = void (*)(int, const TA*, int, const TB*, int, TC*, int);

template <int GS, typename TA, typename TB, typename TC, 
          int RM = 4, int RN = 1, 
          int CM = 72, int CK = 256, int CN = 1020>
class GEMM_Q0 {
private:
    const TA *const A_;
    const TB *const B_;
    TC *const C_;
    const int lda_;
    const int ldb_;
    const int ldc_;
    MicroKernelQ0Type<GS, TA, TB, TC, RM, RN> micro_kernel_;
    
    void PackMatrixA(int m, int k, const TA *A, int row_offset, int col_offset, 
                TA *packA, int pack_offset) {
        const auto &src_q = A->q;
        const auto &src_s = A->s;
        
        using QType = typename decltype(A->q)::value_type;
        const QType *src_q_row[RM];
        
        for (int i = 0; i < RM; ++i) {
            if (i < m) {
                src_q_row[i] = &src_q[(row_offset + i) * lda_ + col_offset];
            } else {
                src_q_row[i] = &src_q[(row_offset + 0) * lda_ + col_offset];
            }
        }
        
        for (int i = 0; i < RM; ++i) {
            const QType *current_q_row = src_q_row[i];
            const int abs_row_index = row_offset + (i < m ? i : 0);
            int dst_q_base = pack_offset + i * k;
            int p = 0;
            for (; p + 31 < k; p += 32) {
                __m256i data = load<__m256i, std::int8_t>(current_q_row + p);
                store(&packA->q[dst_q_base + p], data);
                for (int j = p; j < p + 32 && j < k; j += GS) {
                    int dst_s_index = (dst_q_base + j) / GS;
                    int src_s_index = (abs_row_index * lda_ + col_offset + j) / GS;
                    packA->s[dst_s_index] = src_s[src_s_index];
                }
            }
            
            for (; p < k; ++p) {
                int dst_q_index = dst_q_base + p;
                int src_q_index = abs_row_index * lda_ + col_offset + p;
                packA->q[dst_q_index] = current_q_row[p];
                
                if (p % GS == 0) {
                    int dst_s_index = dst_q_index / GS;
                    int src_s_index = src_q_index / GS;
                    packA->s[dst_s_index] = src_s[src_s_index];
                }
            }
        }
    }

    void PackMatrixB(int k, int n, const TB *B, int row_offset, int col_offset, 
                TB *packB, int pack_offset) {
        const auto &src_q = B->q;
        const auto &src_s = B->s;
        
        int i = 0;
        for (; i + 31 < k; i += 32) {
            int src_base = (row_offset + i) * ldb_ + col_offset;
            int dst_base = pack_offset + i;
            __m256i data = load<__m256i, std::int8_t>(&src_q[src_base]);
            store(&packB->q[dst_base], data);

            for (int j = 0; j < 32 && i + j < k; j += GS) {
                int src_s_index = (src_base + j) / GS;
                int dst_s_index = (dst_base + j) / GS;
                packB->s[dst_s_index] = src_s[src_s_index];
            }
        }
        
        for (; i < k; ++i) {
            int src_q_index = (row_offset + i) * ldb_ + col_offset;
            int dst_q_index = pack_offset + i;
            packB->q[dst_q_index] = src_q[src_q_index];

            if (i % GS == 0) {
                int src_s_index = src_q_index / GS;
                int dst_s_index = dst_q_index / GS;
                packB->s[dst_s_index] = src_s[src_s_index];
            }
        }
    }

public:
    GEMM_Q0(const TA *A, int lda, 
            const TB *B, int ldb, 
            TC *C, int ldc, 
            MicroKernelQ0Type<GS, TA, TB, TC, RM, RN> micro_kernel) :
            A_(A), lda_(lda), 
            B_(B), ldb_(ldb), 
            C_(C), ldc_(ldc), 
            micro_kernel_(micro_kernel) {};
    void multiply(int m, int n, int k) {
        int i, j, p;
        int ic, ib, jc, jb, pc, pb;

        // iterate row of A
        #pragma omp parallel for private(ic, ib, pc, pb, i)
        for (ic = 0; ic < m; ic += CM) {
            ib = std::min(m - ic, CM);
            // col of A, row of B
            for (pc = 0; pc < k; pc += CK) {
                pb = std::min(k - pc, CK);
                TA *packed_A = malloc_aligned<TA>(ib * pb, 1, sizeof(TA));
                TB *packed_B = malloc_aligned<TB>(pb * n, 1, sizeof(TB));
                // n = 1, so do not need third loop of n
                // pack matrix A
                for (int i = 0; i < ib; i += RM) {
                    PackMatrixA(
                        std::min(ib - i, RM), pb, 
                        A_,
                        ic + i, pc, 
                        packed_A,
                        i * pb
                    );
                }
                // pack B
                jb = n;
                PackMatrixB(pb, jb, B_, pc, 0, packed_B, 0);
                // micro kernel
                for (int i = 0; i < ib; i += RM) {
                    micro_kernel_(
                        pb,
                        packed_A,                  // A block
                        i * pb,
                        packed_B,                  // B block (n=1)
                        0,
                        &C_[(ic + i) * ldc_],      // C block
                        ldc_
                    );
                }
                free_aligned(packed_A);
                free_aligned(packed_B);
            }
        }
    }
};

inline void matmul_pseudo(float* xout, const QuantizedTensorType *x, const QuantizedTensorType *w, int n, int d) {
    int i;
    int GS = 32;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        int32_t ival = 0;
        int row_index = i * n;
        int group;
        for (group = 0; group < (n + GS - 1) / GS; ++group) {
            int begin = group * GS;
            int end = std::min(begin + GS, n);
            ival = 0;
            for (int k = begin; k < end; ++k) {
                ival += static_cast<int32_t>(x->q[k]) * static_cast<int32_t>(w->q[row_index + k]);
            }
            val += ((float) ival) * w->s[(row_index + begin) / GS] * x->s[group];
            
        }
        xout[i] = val;
    }
}


// W (d,n) @ x (n,) -> xout (d,)
// by far the most amount of time is spent inside this little function
// inputs to this function are both quantized
inline void matmul(float* xout, const QuantizedTensorType *x, const QuantizedTensorType *w, int n, int d) {
    const int GS = 32;
    const int SMALL_MATRIX_THRESHOLD = 768 * 768;
    if (n * d <= SMALL_MATRIX_THRESHOLD) {
        matmul_pseudo(xout, x, w, n, d);
        return ;
    }

    int m = d;
    int k = n;
    int nn = 1;
    int lda = k;
    int ldb = nn;
    int ldc = nn;
    constexpr int RM = 4, RN = 1, CM = 32, Ck = 128, CN = 1020;
    float *C = malloc_aligned<float>(m, nn, sizeof(float));

    MicroKernelQ0Type<GS, QuantizedTensorType, QuantizedTensorType, float, RM, RN> micro_kernel;
    micro_kernel = &AddDot_4x1_Q0<GS, QuantizedTensorType, QuantizedTensorType, float, RM, RN>;
    GEMM_Q0<GS, QuantizedTensorType, QuantizedTensorType, float, RM, RN, CM, Ck, CN> gemm_q0(
        w, lda, x, ldb, C, ldc, 
        micro_kernel
    );
    gemm_q0.multiply(m, nn, k);
    std::memcpy(xout, C, m * nn * sizeof(float));
    free(C);
}

#endif // UTILITY_HPP_ 