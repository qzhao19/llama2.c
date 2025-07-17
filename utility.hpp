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
// _mm256_maddubs_epi16：multiply 32 int8 -> 16 int16
// _mm256_set1_epi16: initialize int16 vector with value is 1
// _mm256_madd_epi16: 16 int16 add 2by2 -> 8 int32
// _mm256_add_epi32: accumulate
template<>
inline __m256i madd(__m256i a, __m256i b, __m256i c) {
    return _mm256_add_epi32(
        c, 
        _mm256_madd_epi16(
            _mm256_maddubs_epi16(a, b), 
            _mm256_set1_epi16(1)
        )
    );
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
    __m128i lo = _mm256_extracti128_si256(v, 0);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    return _mm_extract_epi32(lo, 0);
}
#endif

// declaration of basic template function load
template <typename T> 
T load(const float *);

#if defined(__SSE__)
template <> 
inline __m128 load<__m128>(const float *p) {
    return _mm_loadu_ps(p);
}
template <>
inline __m128i load<__m128i>(const std::int8_t *p) {
    return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}
#endif  // __SSE__

#if defined(__AVX2__)
template <> 
inline __m256 load<__m256>(const float *p) {
    return _mm256_loadu_ps(p);
}
template <>
inline __m256i load<__m256i>(const std::int8_t *p) {
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
inline __m128i setzeros<__m128i>() { return _mm_setzero_si(); }
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
inline __m128 set1<__m128>(float x) { return _mm_set1_ps(x); }
#endif

#if defined(__AVX2__)
template <>
inline __m256 set1<__m256>(float x) { return _mm256_set1_ps(x); }
template <>
inline __m256i set1<__m256i>(short x) { return _mm256_set1_epi16(x); }
template <>
inline __m256i set1<__m256i>(int x) { return _mm256_set1_epi32(x); }
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

template <typename TA, typename TB, typename TC, int RM = 4, int RN = 1>
inline void AddDot_4x1(int k, TA *a, TB *b, TC *c, int ldc) {
    TC c_00, c_10, c_20, c_30;
    c_00 = 0.0;
    c_10 = 0.0;
    c_20 = 0.0;
    c_30 = 0.0;

    // declare register vars for result matrix a
    __m256 c_00_ymm0 = setzeros<__m256>(); __m256 c_00_ymm1 = setzeros<__m256>();
    __m256 c_10_ymm0 = setzeros<__m256>(); __m256 c_10_ymm1 = setzeros<__m256>();
    __m256 c_20_ymm0 = setzeros<__m256>(); __m256 c_20_ymm1 = setzeros<__m256>();
    __m256 c_30_ymm0 = setzeros<__m256>(); __m256 c_30_ymm1 = setzeros<__m256>();

    // declare register vars for matrix a and vector b
    __m256 a_0p_ymm0 = setzeros<__m256>(); __m256 a_0p_ymm1 = setzeros<__m256>();
    __m256 a_1p_ymm0 = setzeros<__m256>(); __m256 a_1p_ymm1 = setzeros<__m256>();
    __m256 a_2p_ymm0 = setzeros<__m256>(); __m256 a_2p_ymm1 = setzeros<__m256>();
    __m256 a_3p_ymm0 = setzeros<__m256>(); __m256 a_3p_ymm1 = setzeros<__m256>();
    __m256 b_p0_ymm0 = setzeros<__m256>(); __m256 b_p0_ymm1 = setzeros<__m256>();

    // define pointers to a and b
    TA *a_0p_ptr = a + 0 * k;
    TA *a_1p_ptr = a + 1 * k;
    TA *a_2p_ptr = a + 2 * k;
    TA *a_3p_ptr = a + 3 * k;
    TB *b_p0_ptr = b;

    int p = 0;
    const int aligned_end = k & ~(UNROLLING_SIZE - 1);

    for (; p < aligned_end; p += UNROLLING_SIZE) {
        // pre-fetch data
        _mm_prefetch(reinterpret_cast<const char*>(a_0p_ptr + p + 16), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a_1p_ptr + p + 16), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a_2p_ptr + p + 16), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(a_3p_ptr + p + 16), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b_p0_ptr + p + 16), _MM_HINT_T0);

        // load vector b 
        b_p0_ymm0 = load<__m256>(b_p0_ptr + p + 0); 
        b_p0_ymm1 = load<__m256>(b_p0_ptr + p + 8);

        // load matrix a_0p and compute c00 = a_0p * b_p0
        a_0p_ymm0 = load<__m256>(a_0p_ptr + p + 0); 
        a_0p_ymm1 = load<__m256>(a_0p_ptr + p + 8);
        c_00_ymm0 = madd(a_0p_ymm0, b_p0_ymm0, c_00_ymm0);
        c_00_ymm1 = madd(a_0p_ymm1, b_p0_ymm1, c_00_ymm1);

        // load matrix a_1p and compute c10 = a_1p * b_p0
        a_1p_ymm0 = load<__m256>(a_1p_ptr + p + 0); 
        a_1p_ymm1 = load<__m256>(a_1p_ptr + p + 8);
        c_10_ymm0 = madd(a_1p_ymm0, b_p0_ymm0, c_10_ymm0); 
        c_10_ymm1 = madd(a_1p_ymm1, b_p0_ymm1, c_10_ymm1);

        // load matrix a_2p and compute c20 = a_2p * b_p0
        a_2p_ymm0 = load<__m256>(a_2p_ptr + p + 0); 
        a_2p_ymm1 = load<__m256>(a_2p_ptr + p + 8);
        c_20_ymm0 = madd(a_2p_ymm0, b_p0_ymm0, c_20_ymm0); 
        c_20_ymm1 = madd(a_2p_ymm1, b_p0_ymm1, c_20_ymm1);

        // load matrix a_3p and compute c30 = a_3p * b_p0
        a_3p_ymm0 = load<__m256>(a_3p_ptr + p + 0); 
        a_3p_ymm1 = load<__m256>(a_3p_ptr + p + 8);
        c_30_ymm0 = madd(a_3p_ymm0, b_p0_ymm0, c_30_ymm0); 
        c_30_ymm1 = madd(a_3p_ymm1, b_p0_ymm1, c_30_ymm1);
    }

    c_00 += hsum(add(c_00_ymm0, c_00_ymm1));
    c_10 += hsum(add(c_10_ymm0, c_10_ymm1)); 
    c_20 += hsum(add(c_20_ymm0, c_20_ymm1));
    c_30 += hsum(add(c_30_ymm0, c_30_ymm1));

    const int remainder = k - aligned_end;
    if (remainder > 0) {
        // load matrix a
        __m128 a_0p_xmm = setzeros<__m128>();
        __m128 a_1p_xmm = setzeros<__m128>();
        __m128 a_2p_xmm = setzeros<__m128>();
        __m128 a_3p_xmm = setzeros<__m128>();
        // load vector b
        __m128 b_p0_xmm = setzeros<__m128>(); 
        // load result matrix c
        __m128 c_00_xmm = setzeros<__m128>();
        __m128 c_10_xmm = setzeros<__m128>();
        __m128 c_20_xmm = setzeros<__m128>();
        __m128 c_30_xmm = setzeros<__m128>();

        int r = 0;
        for (; r + 3 < remainder; r += 4) {
            b_p0_xmm = _mm_loadu_ps(b_p0_ptr + p + r);
            // c_00_xmm = 
            a_0p_xmm = load<__m128>(a_0p_ptr + p + r);
            a_1p_xmm = load<__m128>(a_1p_ptr + p + r);
            a_2p_xmm = load<__m128>(a_2p_ptr + p + r);
            a_3p_xmm = load<__m128>(a_3p_ptr + p + r);

            // compute c += a * b
            c_00_xmm = add(c_00_xmm, mul(a_0p_xmm, b_p0_xmm));
            c_10_xmm = add(c_10_xmm, mul(a_1p_xmm, b_p0_xmm));
            c_20_xmm = add(c_20_xmm, mul(a_2p_xmm, b_p0_xmm));
            c_30_xmm = add(c_30_xmm, mul(a_3p_xmm, b_p0_xmm));
        }

        c_00 += hsum(c_00_xmm);
        c_10 += hsum(c_10_xmm);
        c_20 += hsum(c_20_xmm);
        c_30 += hsum(c_30_xmm);

        // the last elements which are less than 4
        for (; r < remainder; ++r) {
            TB b_p0 = b_p0_ptr[p + r];
            c_00 += a_0p_ptr[p + r] * b_p0;
            c_10 += a_1p_ptr[p + r] * b_p0;
            c_20 += a_2p_ptr[p + r] * b_p0;
            c_30 += a_3p_ptr[p + r] * b_p0;
        }   
    }

    c[0 * ldc + 0] += c_00;
    c[1 * ldc + 0] += c_10;
    c[2 * ldc + 0] += c_20;
    c[3 * ldc + 0] += c_30;
}

template <typename TA, typename TB, typename TC, int RM, int RN>
using MicroKernelType = void (*)(int, TA*, TB*, TC*, int);

// row-major order
// a(i, j) ==> a[(i) * lda + (j)]
// b(i, j) ==> b[(i) * lda + (j)]
// c(i, j) ==> c[(i) * lda + (j)]

template <typename TA, typename TB, typename TC, 
          int RM = 4, int RN = 1, 
          int CM = 72, int CK = 256, int CN = 1020>
class GEMM {
private:
    const TA *const A_;
    const TB *const B_;
    TC *const C_;

    // const int k_;
    const int lda_;
    const int ldb_;
    const int ldc_;

    MicroKernelType<TA, TB, TC, RM, RN> micro_kernel_;

    // packing matrix A following row-major order
    void PackMatrixA(int m, int k, const TA *sub_A, int offset, TA *packA) {
        int i, p;
        const TA *src[RM];

        for (i = 0; i < RM; ++i) {
            if (i < m) {
                src[i] = &sub_A[(offset + i) * lda_];
            }
            else {
                src[i] = &sub_A[(offset + 0) * lda_];
            }
        }

        // colnum-major order packing for (i = 0; i < RM; ++i)
        for (i = 0; i < RM; ++i) {
            for (p = 0; p < k; ++p) {
                *packA = src[i][p];
                packA++;
            }
        }
    }

    // packing vector A，just copy B to aligned packB
    void PackMatrixB(int k, int n, const TB *sub_B, int offset, TB *packB) {
        std::memcpy(packB, sub_B + offset, RN * k * sizeof(TB));
    }

public:
    GEMM(const TA *A, int lda, 
         const TB *B, int ldb, 
         TC *C, int ldc, 
         MicroKernelType<TA, TB, TC, RM, RN> micro_kernel) :
            A_(A), lda_(lda), 
            B_(B), ldb_(ldb), 
            C_(C), ldc_(ldc), 
            micro_kernel_(micro_kernel) {};
    
    void multiply(int m, int n, int k) {
        int i, j, p;
        int ic, ib, jc, jb, pc, pb;

        // iterate row of A
        for (ic = 0; ic < m; ic += CM) {
            ib = std::min(m - ic, CM);

            // col of A, row of B
            for (pc = 0; pc < k; pc += CK) {
                pb = std::min(k - pc, CK);

                TA *packed_A = malloc_aligned<TA>(ib * pb, 1, sizeof(TA));
                TB *packed_B = malloc_aligned<TB>(pb * n, 1, sizeof(TB));

                // n = 1, so do not need third loop of n
                // pack matrix A
                for (i = 0; i < ib; i += RM) {
                    PackMatrixA(
                        std::min(ib - i, RM), 
                        pb, 
                        &A_[(ic + i) * lda_ + pc], 
                        0, 
                        &packed_A[i * pb]
                    );
                }

                // pack B
                jb = n;
                PackMatrixB(pb, jb, &B_[pc], 0, packed_B);

                // micro kernel
                for (i = 0; i < ib; i += RM) {
                    micro_kernel_(
                        pb,
                        &packed_A[i * pb],         // A block
                        packed_B,                  // B block (n=1)
                        &C_[(ic + i) * ldc_],      // C block
                        ldc_
                    );
                }
                free(packed_A);
                free(packed_B);
            }
        }
    }
};

inline void matmul(float* xout, float* x, float* w, int n, int d) {
    int m = d;
    int k = n;
    int nn = 1;
    int lda = k;
    int ldb = nn;
    int ldc = nn;
    constexpr int RM = 4, RN = 1, CM = 72, Ck = 512, CN = 1020;
    float *C = malloc_aligned<float>(m, nn, sizeof(float));

    MicroKernelType<float, float, float, RM, RN> micro_kernel;
    micro_kernel = &AddDot_4x1;
    GEMM<float, float, float, RM, RN, CM, Ck, CN> gemm(
        w, lda, x, ldb, C, ldc, 
        micro_kernel
    );
    gemm.multiply(m, nn, k);
    std::memcpy(xout, C, m * nn * sizeof(float));
    free(C);
}

// ----------------------------------------------------------------------------

template <int GS, typename TA, typename TB, typename TC, int RM = 4, int RN = 1>
inline void AddDot_4x1_Q0(int k, const TA *a, int offset_A, const TB *b, int offset_B, TC *c, int ldc) {
    TC c_00_reg, c_10_reg, c_20_reg, c_30_reg;
    c_00_reg = 0.0;
    c_10_reg = 0.0;
    c_20_reg = 0.0;
    c_30_reg = 0.0;

    int group;
    for (group = 0; group < (k + GS - 1) / GS; ++group) {
        int begin = group * GS;
        int end = std::min(begin + GS, k);

        int32_t acc_00 = 0, acc_10 = 0, acc_20 = 0, acc_30 = 0;
        for (int p = begin; p <end; ++p) {
            int a_index_0p = offset_A + 0 * k + p;
            int a_index_1p = offset_A + 1 * k + p;
            int a_index_2p = offset_A + 2 * k + p;
            int a_index_3p = offset_A + 3 * k + p;
            int b_index_p0 = offset_B + p;

            acc_00 += static_cast<int32_t>(a->q[a_index_0p]) * static_cast<int32_t>(b->q[b_index_p0]);
            acc_10 += static_cast<int32_t>(a->q[a_index_1p]) * static_cast<int32_t>(b->q[b_index_p0]);
            acc_20 += static_cast<int32_t>(a->q[a_index_2p]) * static_cast<int32_t>(b->q[b_index_p0]);
            acc_30 += static_cast<int32_t>(a->q[a_index_3p]) * static_cast<int32_t>(b->q[b_index_p0]);
        }

        float a_s0 = a->s[(offset_A + 0 * k + begin) / GS];
        float a_s1 = a->s[(offset_A + 1 * k + begin) / GS];
        float a_s2 = a->s[(offset_A + 2 * k + begin) / GS];
        float a_s3 = a->s[(offset_A + 3 * k + begin) / GS];
        float b_s = b->s[(offset_B + begin) / GS];

        c_00_reg += static_cast<TC>(acc_00) * a_s0 * b_s;
        c_10_reg += static_cast<TC>(acc_10) * a_s1 * b_s;
        c_20_reg += static_cast<TC>(acc_20) * a_s2 * b_s;
        c_30_reg += static_cast<TC>(acc_30) * a_s3 * b_s;
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

    // packing matrix A following row-major order
    void PackMatrixA(int m, int k, const TA *A, int row_offset, int col_offset, TA *packA, int pack_offset) {
        const auto &src_q = A->q;
        const auto &src_s = A->s;
        
        using QType = typename decltype(A->q)::value_type;
        const QType *src_q_row[RM];
        
        int i, p;
        for (i = 0; i < RM; ++i) {
            if (i < m) {
                src_q_row[i] = &src_q[(row_offset + i) * lda_ + col_offset];
            }
            else {
                src_q_row[i] = &src_q[(row_offset + 0) * lda_ + col_offset];
            }
        }

        for (i = 0; i < RM; ++i) {
            const QType *current_q_row_index = src_q_row[i];
            const int abs_row_index = row_offset + (i < m ? i : 0);
            for (p = 0; p < k; ++p) {
                int dst_q_index = pack_offset + i * k + p;
                int dst_s_index = dst_q_index / GS;
                int src_q_index = abs_row_index * lda_ + col_offset + p;
                int src_s_index = src_q_index / GS;

                // packing q & s
                packA->q[dst_q_index] = current_q_row_index[p];
                packA->s[dst_s_index] = src_s[src_s_index];
            }
        }
    }

    void PackMatrixB(int k, int n, const TB *B, int row_offset, int col_offset, TB *packB, int pack_offset) {
        const auto &src_q = B->q;
        const auto &src_s = B->s;

        int i, j;
        for (j = 0; j < n; ++j) {
            for (i = 0; i < k; ++i) {
                int src_q_index = (row_offset + i) * ldb_ + (col_offset + j);
                int dst_q_index = pack_offset + j * k + i;
                int src_s_index = src_q_index / GS;
                int dst_s_index = dst_q_index / GS;
                // pack q & s
                packB->q[dst_q_index] = src_q[src_q_index];
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

// W (d,n) @ x (n,) -> xout (d,)
// by far the most amount of time is spent inside this little function
// inputs to this function are both quantized
inline void matmul(float* xout, const QuantizedTensorType *x, const QuantizedTensorType *w, int n, int d) {
    int m = d;
    int k = n;
    int nn = 1;
    int lda = k;
    int ldb = nn;
    int ldc = nn;
    const int GS = 32;
    constexpr int RM = 4, RN = 1, CM = 72, Ck = 256, CN = 1020;
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