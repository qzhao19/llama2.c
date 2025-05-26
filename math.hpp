#ifndef MATH_HPP_
#define MATH_HPP_

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
template<int KTile, typename AccVecType, typename LoadVecType, typename MatAType, typename MatBType, typename MatCType>
class tinyBLAS {
    
}
















#endif // MATH_HPP_