#include <immintrin.h>
#include <math.h>
#include <string.h>

#define LARGE_VAL 1e20f

static float sum_m256(__m256 x) {
    float tot = 0;
    float* vals = (float*)&x;
    for (size_t i = 0; i < 8; ++i) tot += vals[i];
    return tot;
}

static float fastmse(float abort_threshold, size_t n, float *a, float *b) {
    __m256 total = _mm256_set1_ps(0);
    float a_aligned[8];
    float b_aligned[8];
    size_t i;
    for (i = 0; i + 8 <= n; i += 8) {
        memcpy(a_aligned, &a[i], sizeof(float) * 8);
        memcpy(b_aligned, &b[i], sizeof(float) * 8);
        __m256 res = _mm256_sub_ps(_mm256_loadu_ps(a_aligned), _mm256_loadu_ps(b_aligned));
        total = _mm256_add_ps(total, _mm256_mul_ps(res, res));
        if (i % 100 == 96) {
            // abort early
            if (sum_m256(total) / n >= abort_threshold) {
                return LARGE_VAL;
            }
        }
    }
    float sum = sum_m256(total);
    for (; i < n; ++i) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sum / n;
}


static size_t my_max(size_t a, size_t b) {
    return (a > b) ? a : b;
}

static size_t my_min(size_t a, size_t b) {
    return (a < b) ? a : b;
}

static size_t my_min3(size_t a, size_t b, size_t c) {
    return my_min(my_min(a, b), c);
}

typedef struct {
    int min_idx;
    float min_val;
} MinResult;

MinResult fast_find_alignment(size_t a_len, float *a,
                               size_t b_len, float *b,
                               size_t max_offset, size_t max_lookahead) {
    MinResult result = {0, LARGE_VAL};

    for (size_t i = 0; i < my_min(my_max(a_len, b_len), max_offset); ++i) {
        if (i < a_len) {
            float d1 = fastmse(result.min_val, my_min3(a_len - i, b_len, max_lookahead), &a[i], b);
            if (d1 < result.min_val) {
                result.min_val = d1;
                result.min_idx = i;
            }
        }
        if (i < b_len) {
            float d2 = fastmse(result.min_val, my_min3(a_len, b_len - i, max_lookahead), a, &b[i]);
            if (d2 < result.min_val) {
                result.min_val = d2;
                result.min_idx = -i;
            }
        }
    }

    return result;
}
