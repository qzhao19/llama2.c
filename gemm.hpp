#ifndef GEMM_HPP_
#define GEMM_HPP_

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

/**
 * 
 */
template<int KN, typename D, typename V, typename TA, typename TB, typename TC>
class tinyBLAS {
    public:
    tinyBLAS(int64_t k,
             const TA *A, int64_t lda,
             const TB *B, int64_t ldb,
             TC *C, int64_t ldc,
             int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

    private:
    NOINLINE void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        switch ((MIN(m - m0, 5) << 4) | MIN(n - n0, 5)) {
            case 0x55:
            case 0x54:
            case 0x53:
            case 0x45:
            case 0x44:
            case 0x43:
                mc = 4;
                nc = 3;
                gemm<4, 3>(m0, m, n0, n);
                break;
            case 0x35:
            case 0x34:
                mc = 3;
                nc = 4;
                gemm<3, 4>(m0, m, n0, n);
                break;
            case 0x52:
                mc = 5;
                nc = 2;
                gemm<5, 2>(m0, m, n0, n);
                break;
            case 0x33:
                mc = 3;
                nc = 3;
                gemm<3, 3>(m0, m, n0, n);
                break;
            case 0x25:
                mc = 2;
                nc = 5;
                gemm<2, 5>(m0, m, n0, n);
                break;
            case 0x42:
                mc = 4;
                nc = 2;
                gemm<4, 2>(m0, m, n0, n);
                break;
            case 0x24:
                mc = 2;
                nc = 4;
                gemm<2, 4>(m0, m, n0, n);
                break;
            case 0x32:
                mc = 3;
                nc = 2;
                gemm<3, 2>(m0, m, n0, n);
                break;
            case 0x23:
                mc = 2;
                nc = 3;
                gemm<2, 3>(m0, m, n0, n);
                break;
            case 0x51:
                mc = 5;
                nc = 1;
                gemm<5, 1>(m0, m, n0, n);
                break;
            case 0x41:
                mc = 4;
                nc = 1;
                gemm<4, 1>(m0, m, n0, n);
                break;
            case 0x22:
                mc = 2;
                nc = 2;
                gemm<2, 2>(m0, m, n0, n);
                break;
            case 0x15:
                mc = 1;
                nc = 5;
                gemm<1, 5>(m0, m, n0, n);
                break;
            case 0x14:
                mc = 1;
                nc = 4;
                gemm<1, 4>(m0, m, n0, n);
                break;
            case 0x31:
                mc = 3;
                nc = 1;
                gemm<3, 1>(m0, m, n0, n);
                break;
            case 0x13:
                mc = 1;
                nc = 3;
                gemm<1, 3>(m0, m, n0, n);
                break;
            case 0x21:
                mc = 2;
                nc = 1;
                gemm<2, 1>(m0, m, n0, n);
                break;
            case 0x12:
                mc = 1;
                nc = 2;
                gemm<1, 2>(m0, m, n0, n);
                break;
            case 0x11:
                mc = 1;
                nc = 1;
                gemm<1, 1>(m0, m, n0, n);
                break;
            default:
                return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN>
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        // std::cout << "RM = "<< RM << ", RN = "<< RN << std::endl;
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        // std::cout << "ytiles = "<< ytiles << ", xtiles = "<< xtiles << std::endl;
        // std::cout << "tiles = "<< tiles << std::endl;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;

        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            D Cv[RN][RM] = {};
            for (int64_t l = 0; l < k; l += KN)
                for (int64_t j = 0; j < RN; ++j)
                    for (int64_t i = 0; i < RM; ++i)
                        Cv[j][i] = madd(
                            load<V>(A + lda * (ii + i) + l),
                            load<V>(B + ldb * (jj + j) + l),
                            // load<V>(A + (ii + i) + l * lda),
                            // load<V>(B + l + (jj + j) * ldb),
                            Cv[j][i]
                        );
            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    // C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
                    C[(ii + i) + (jj + j) * ldc] = hsum(Cv[j][i]);
        }
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;

};


#endif // GEMM_HPP_