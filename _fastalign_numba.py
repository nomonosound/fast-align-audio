from cffi import FFI


ffibuilder = FFI()
ffibuilder.cdef("ssize_t fast_find_alignment(size_t, float *, size_t, float *, size_t, size_t);");
ffibuilder.set_source("_fastalign",
r"""
#include <immintrin.h>
#include <math.h>

#define LARGE_VAL 1e20

static float sum_m256(__m256 x) {
    float tot = 0;
    for (size_t i = 0; i < 8; ++i) tot += x[i];
    return tot;
}

static float fastmse(float abort_threshold, size_t n, float *a, float *b) {
    __m256 total = _mm256_set1_ps(0);
    float a_aligned[8] __attribute__ ((aligned (32)));
    float b_aligned[8] __attribute__ ((aligned (32)));
    size_t i;
    for (i = 0; i + 8 <= n; i += 8) {
        memcpy(a_aligned, &a[i], sizeof(float) * 8);
        memcpy(b_aligned, &b[i], sizeof(float) * 8);
        __m256 res = _mm256_sub_ps(_mm256_load_ps(a_aligned), _mm256_load_ps(b_aligned));
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
        sum += pow(a[i] - b[i], 2);
    }
    return sum / n;
}

static size_t min(size_t a, size_t b) {
    return (a < b) ? a : b;
}

static size_t min3(size_t a, size_t b, size_t c) {
    return min(min(a, b), c);
}

ssize_t fast_find_alignment(size_t a_len, float *a,
                            size_t b_len, float *b,
                            size_t max_offset, size_t max_lookahead) {
    float min_val = LARGE_VAL;
    ssize_t min_idx = 0;
    for (size_t i = 0; i < min3(a_len, b_len, max_offset); ++i) {
        if (i < a_len) {
            float d1 = fastmse(min_val, min3(a_len - i, b_len, max_lookahead), &a[i], b);
            if (d1 < min_val) {
                min_val = d1;
                min_idx = i;
            }
        }
        if (i < b_len) {
            float d2 = fastmse(min_val, min3(a_len, b_len - i, max_lookahead), a, &b[i]);
            if (d2 < min_val) {
                min_val = d2;
                min_idx = -i;
            }
        }
    }
    return min_idx;
}
""", extra_compile_args=["-mavx", "-Wall", "-Wextra", "-pedantic"])


if __name__ == "__main__":
    ffibuilder.compile(verbose=True)
