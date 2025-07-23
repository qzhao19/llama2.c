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
    // Reduce two horizontal operations to one
    __m128i sum_lo = _mm256_extracti128_si256(x, 0);
    __m128i sum_hi = _mm256_extracti128_si256(x, 1);
    __m128i sum = _mm_add_epi32(sum_lo, sum_hi);
    
    // Single horizontal add with shuffling
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0x4E)); // 0x4E = 01001110
    sum = _mm_add_epi32(sum, _mm_shuffle_epi32(sum, 0xB1)); // 0xB1 = 10110001
    
    return _mm_cvtsi128_si32(sum);
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
    constexpr int RM = 4, RN = 1, CM = 72, CN = 256;
    std::memset(xout, 0, d * sizeof(float));
    GEMV<float, float, float, RM, RN, CM, CN> gemv(
        w, n, x, xout
    );
    gemv.multiply(d, n);
}

// ----------------------------------------------------------------------------

template <int GS, typename TA, typename TX, typename TY, 
          int RM = 4, int RN = 1, 
          int CM = 72, int CN = 1020>
class GEMV_Q0 {
private:
    const TA *const A_;
    const TX *const x_;
    TY *const y_;
    const int lda_;

public:
    GEMV_Q0(const TA *A, int lda, 
            const TX *x, TY *y) :
            A_(A), lda_(lda), 
            x_(x), y_(y) {};
    
    void multiply(int m, int n) {
        int ic, jc;
        // split by row block
        for (ic = 0; ic < m; ic += CM) {
            const int ib = std::min(m - ic, CM);
            const int ie = ic + ib;

            // split by column block
            for (jc = 0; jc < n; jc += CN) {
                const int jb = std::min(n - jc, CN);
                const int je = jc + jb;

                // handle current block with RM rows
                for (int i = ic; i < ie; i += RM) {
                    const int nrows = std::min(ie - i, RM);
                    TY sum[RM] = {0};
                    
                    for (int group = 0; group < (jb + GS - 1) / GS; ++group) {
                        const int begin = jc + group * GS;
                        const int end = std::min(begin + GS, je);
                        alignas(32) int32_t accum[RM] = {0}; 

                        // init register array: y_j_ymm 
                        __m256i y_j_ymm[RM];
                        for (int r = 0; r < nrows; ++r) {
                            y_j_ymm[r] = setzeros<__m256i>();
                        }

                        int j = begin;
                        // process 32 elements at a time using SIMD
                        for (; j + 31 < end; j += 32) {
                            // prefetch x_->q and A_->q
                            if (j + 128 < end) {
                                _mm_prefetch(reinterpret_cast<const char*>(&x_->q[j + 128]), _MM_HINT_T0);
                                for (int r = 0; r < nrows; ++r) {
                                    _mm_prefetch(reinterpret_cast<const char*>(&A_->q[(i + r) * lda_ + j + 128]), _MM_HINT_T0);
                                }
                            }
                            __m256i x_j_ymm0 = load<__m256i, std::int8_t>(&x_->q[j]);
                            for (int r = 0; r < nrows; ++r) {
                                __m256i a_rj_ymm0 = load<__m256i, std::int8_t>(&A_->q[(i + r) * lda_ + j]);
                                y_j_ymm[r] = madd(a_rj_ymm0, x_j_ymm0, y_j_ymm[r]);
                            }
                        }

                        // compute horizontal sum
                        for (int r = 0; r < nrows; ++r) {
                            accum[r] += hsum(y_j_ymm[r]);
                        }
                    
                        // handle remaining elements
                        for (; j < end; ++j) {
                            int8_t xval = x_->q[j];
                            for (int r = 0; r < nrows; ++r) {
                                accum[r] += static_cast<int32_t>(A_->q[(i + r) * lda_ + j]) * 
                                            static_cast<int32_t>(xval);
                            }
                        }
                        
                        // prefetch x_->s and A_->s
                        _mm_prefetch(reinterpret_cast<const char*>(&x_->s[group / GS]), _MM_HINT_T0);
                        for (int r = 0; r < nrows; r++) {
                            _mm_prefetch(reinterpret_cast<const char*>(&A_->s[((i + r) * lda_ + begin) / GS]), _MM_HINT_T0);
                        }
                        const float xs = x_->s[begin / GS];
                        for (int r = 0; r < nrows; ++r) {
                            const float as = A_->s[((i + r) * lda_ + begin) / GS];
                            sum[r] += static_cast<float>(accum[r]) * as * xs;
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
    constexpr int GS = 32;
    constexpr int RM = 4, RN = 1, CM = 72, CN = 256;
    std::memset(xout, 0, d * sizeof(float));
    
    GEMV_Q0<GS, QuantizedTensorType, QuantizedTensorType, float, RM, RN, CM, CN> gemv_q0(
        w, n, x, xout
    );
    gemv_q0.multiply(d, n);
}

#endif // UTILITY_HPP_ 