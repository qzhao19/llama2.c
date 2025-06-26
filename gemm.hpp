#ifndef GEMM_HPP_
#define GEMM_HPP_

#include <cmath>
#include <cstring>

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#if defined(__SSE__)
#include <smmintrin.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h> 
#endif

// 
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
#endif

// declaration of basic template function load
template <typename T> 
T load(const float *);

#if defined(__SSE__)
template <> 
inline __m128 load<__m128>(const float *p) {
    return _mm_loadu_ps(p);
}
#endif  // __SSE__

#if defined(__AVX2__)
template <> 
inline __m256 load<__m256>(const float *p) {
    return _mm256_loadu_ps(p);
}
#endif // __AVX__

// declaration of basic template function setzeros
template <typename T>
inline T setzeros();

#if defined(__SSE__)
template <>
inline __m128 setzeros<__m128>() { return _mm_setzero_ps(); }
#endif

#if defined(__AVX2__)
template <>
inline __m256 setzeros<__m256>() { return _mm256_setzero_ps(); }
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
#endif

#define MEMORY_ALIGNMENT 32
#define UNROLLING_SIZE 16


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

inline void matmul_ref(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
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


// row-major order
// a(i, j) ==> a[(i) * lda + (j)]
// b(i, j) ==> b[(i) * lda + (j)]
// c(i, j) ==> c[(i) * lda + (j)]

template <typename T>
T* malloc_aligned(int m, int n, int size) {
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

template <int MR = 4, int NR = 1>
inline void AddDot_4x1(int k, float *a, float *b, float *c, int ldc) {
    float c_00, c_10, c_20, c_30;
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
    float *a_0p_ptr = a + 0 * k;
    float *a_1p_ptr = a + 1 * k;
    float *a_2p_ptr = a + 2 * k;
    float *a_3p_ptr = a + 3 * k;
    float *b_p0_ptr = b;

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
            float b_p0 = b_p0_ptr[p + r];
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

template <std::int64_t MR, std::int64_t NR>
using MicroKernelType = void (*)(std::int64_t, float*, float*, float*, std::int64_t);

template <int MR = 4, int NR = 1, int MC = 72, int KC = 256, int NC = 1020>
void PackMatrixA(int m, int k, float *A, int lda, int offset, float *packA) {
    int i, p;
    float *src[MR];

    for (i = 0; i < MR; ++i) {
        if (i < m) {
            src[i] = &A[(offset + i) * lda];
        }
        else {
            src[i] = &A[offset * lda];
        }
    }

    // colnum-major order packing for (i = 0; i < MR; ++i)
    for (i = 0; i < MR; ++i) {
        for (p = 0; p < k; ++p) {
            *packA = src[i][p];
            packA++;
        }
    }
}

template <int MR = 4, int NR = 1, int MC = 72, int KC = 256, int NC = 1020>
void PackMatrixB(int k, int n, float *B, int ldb, int offset, float *packB) {
    std::memcpy(packB, B + offset, NR * k * sizeof(float));
}

// MR, NR: register level block size
// MC, NC, KC: cache level block size
template <int MR = 4, int NR = 1, int MC = 72, int KC = 256, int NC = 1020>
inline void gemm(int m, int n, int k, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    // int i, j, p;
    int ic, jc, pc;
    int min_m, min_k, min_n;

    float *packA, *packB;
    packA = malloc_aligned<float>(KC, MC + 1, sizeof(float));
    packB = malloc_aligned<float>(KC, NC + 1, sizeof(float));

    // iterate row of A
    for (ic = 0; ic < m; ic += MC) {
        min_m = std::min(m - ic, MC);

        // col of A, row of B
        for (pc = 0; pc < k; pc += KC) {
            min_k = std::min(k - pc, KC);

            // n = 1, so do not need third loop of n
            // pack matrix A
            for (int i = 0; i < min_m; i += MR) {
                PackMatrixA<MR, NR, MC, KC, NC>(
                    std::min(min_m - i, MR), 
                    min_k, 
                    &A[(ic + i) * lda + pc], 
                    lda, 
                    0, 
                    &packA[i * min_k]
                );
            }

            // pack B
            min_n = n;
            PackMatrixB(min_k, min_n, &B[pc], ldb, 0, packB);

            // micro kernel
            for (int i = 0; i < min_m; i += MR) {
                AddDot_4x1<MR, NR>(
                    min_k,
                    &packA[i * min_k],         // A block
                    packB,                     // B block (n=1)
                    &C[(ic + i) * ldc],        // C block
                    ldc
                );
            }
        }
    }
    free(packA);
    free(packB);
}

// W (d,n) @ x (n,1) -> xout (d,)
// w: d x n, row-major
// x: n x 1, row-major
// xout: d x 1, row-major
template <int MR = 4, int NR = 1, int MC = 72, int KC = 512, int NC = 1020>
inline void matmul(float* xout, float* x, float* w, int n, int d) {
    int m = d;
    int k = n;
    int nn = 1;
    int lda = k;
    int ldb = nn;
    int ldc = nn;
    float *C = malloc_aligned<float>(m, nn, sizeof(float));
    gemm<MR, NR, MC, KC, NC>(m, nn, k, w, lda, x, ldb, C, ldc);
    std::memcpy(xout, C, m * nn * sizeof(float));
    free(C);
}


#endif // GEMM_HPP_