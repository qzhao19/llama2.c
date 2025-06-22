#ifndef GEMM_HPP_
#define GEMM_HPP_

#include <cmath>
#include <cstring>

#undef MIN
#undef MAX

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#if defined(USE_SSE) || defined(USE_AVX)
// --------------- if both defined ---------------
#if defined(USE_SSE) && defined(USE_AVX)
    #error "USE_SSE and USE_AVX cannot be defined simultaneously"
#endif

// --------------- compiler flag check ---------------
#if defined(USE_SSE) && !defined(__SSE4_2__)
    #error "USE_SSE defined but SSE4.2 not enabled. Compile with -msse4.2"
#endif

#if defined(USE_AVX) && !defined(__AVX2__)
    #error "USE_AVX defined but AVX2 not enabled. Compile with -mavx2"
#endif
#endif


#if defined(USE_SSE)
#include <smmintrin.h>
#endif

#if defined(USE_AVX)
#include <immintrin.h> 
#endif

// tell compiler do not inline function
#define NOINLINE __attribute__((__noinline__))

#define VECTOR_REGISTERS 16

// 
#if defined(USE_SSE)
inline __m128 add(__m128 x, __m128 y) { return _mm_add_ps(x, y); }
inline __m128 sub(__m128 x, __m128 y) { return _mm_sub_ps(x, y); }
inline __m128 mul(__m128 x, __m128 y) { return _mm_mul_ps(x, y); }
#endif  // __SSE4_2__

#if defined(USE_AVX)
inline __m256 add(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
inline __m256 sub(__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }
inline __m256 mul(__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }
#endif  // __AVX2__

/**
 * Computes a * b + c.
 * apply _mm256_fmadd_ps if enable fma
 */

#if defined(USE_SSE) || defined(USE_AVX)
#if defined(__FMA__)
template<>
inline __m256 madd(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}
#else
template<typename T>
inline T madd(T a, T b, T c) {
    return add(mul(a, b), c);
}
#endif
#endif

//
#if defined(USE_SSE) || defined(USE_AVX)
inline float hsum(__m128 x) {
#if defined(USE_AVX)
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
#else
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_add_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = _mm_add_ss(x, t);
#endif
    return _mm_cvtss_f32(x);
}
#endif

// 
#if defined(USE_AVX)
inline float hsum(__m256 x) {
    return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1),
                           _mm256_castps256_ps128(x)));
}
#endif // __AVX__

template <typename T, typename U> T load(const U *);

#if defined(USE_SSE) || defined(USE_AVX)
template <> 
inline __m128 load(const float *p) {
    return _mm_loadu_ps(p);
}
#endif  // __SSE__

#if defined(USE_AVX)
template <> 
inline __m256 load(const float *p) {
    return _mm256_loadu_ps(p);
}
#endif // __AVX__

// 添加set1函数定义
#if defined(USE_SSE)
inline __m128 set1(float x) { return _mm_set1_ps(x); }
#endif

#if defined(USE_AVX)
inline __m256 set1(float x) { return _mm256_set1_ps(x); }
#endif

#define MEMORY_ALIGNMENT 32

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
    // #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
};

// colmun-major order, lda = row_a, ldb = row_b, ldc = row_c
// a(i, j) ==> a[(j) * lda + (i)]
// b(i, j) ==> b[(j) * ldb + (i)]
// c(i, j) ==> c[(j) * ldc + (i)]

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
    float c_00_reg, c_10_reg, c_20_reg, c_30_reg;
    c_00_reg = 0.0;
    c_10_reg = 0.0;
    c_20_reg = 0.0;
    c_30_reg = 0.0;

    int p;
    for (p = 0; p < k; ++p) {
        float a_0p_reg = a[0 * k + p];
        float a_1p_reg = a[1 * k + p];
        float a_2p_reg = a[2 * k + p];
        float a_3p_reg = a[3 * k + p];
        float b_p0_reg = b[p * NR + 0];

        // C_ij += A_ip * B_pj
        c_00_reg += a_0p_reg * b_p0_reg;
        c_10_reg += a_1p_reg * b_p0_reg;
        c_20_reg += a_2p_reg * b_p0_reg;
        c_30_reg += a_3p_reg * b_p0_reg;
    }

    c[0 * ldc + 0] += c_00_reg;
    c[1 * ldc + 0] += c_10_reg;
    c[2 * ldc + 0] += c_20_reg;
    c[3 * ldc + 0] += c_30_reg;
}


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
template <int MR = 4, int NR = 1, int MC = 72, int KC = 256, int NC = 1020>
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