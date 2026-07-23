/* mock_air_compute.c - Mock FFI implementation of air_compute_api C functions */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void air_compute_rmsnorm(float* x, const float* weight, size_t size, float eps) {
    if (!x || size == 0) return;
    float sum_sq = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum_sq += x[i] * x[i];
    }
    float mean_sq = sum_sq / (float)size;
    float rms = 1.0f / sqrtf(mean_sq + eps);

    for (size_t i = 0; i < size; ++i) {
        float w = weight ? weight[i] : 1.0f;
        x[i] = x[i] * rms * w;
    }
}

void air_compute_rope(float* x, size_t pos, float theta, size_t dim, size_t seq_len) {
    if (!x || dim == 0 || seq_len == 0) return;
    for (size_t s = 0; s < seq_len; ++s) {
        size_t p = pos + s;
        for (size_t i = 0; i < dim; i += 2) {
            float freq = 1.0f / powf(theta, (float)i / (float)dim);
            float val = (float)p * freq;
            float cos_val = cosf(val);
            float sin_val = sinf(val);

            size_t idx0 = s * dim + i;
            size_t idx1 = s * dim + i + 1;
            float x0 = x[idx0];
            float x1 = x[idx1];

            x[idx0] = x0 * cos_val - x1 * sin_val;
            x[idx1] = x0 * sin_val + x1 * cos_val;
        }
    }
}

void air_compute_matmul(float* out, const float* lhs, const float* rhs, size_t m, size_t n, size_t k) {
    if (!out || !lhs || !rhs) return;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (size_t l = 0; l < k; ++l) {
                sum += lhs[i * k + l] * rhs[l * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}
