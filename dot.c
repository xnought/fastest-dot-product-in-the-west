#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define float_t float

float_t dot(float_t *a, float_t *b, int length) {
  float_t sum = 0.0;
  for (int i = 0; i < length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

float_t omp_dot(float_t *a, float_t *b, int length) {
  float_t sum = 0.0;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

float_t omp_simd_dot(float_t *a, float_t *b, int length) {
  float_t sum = 0.0;
#pragma omp simd reduction(+ : sum)
  for (int i = 0; i < length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

float_t asm_simd_dot(float_t *a, float_t *b, int length) {
  int simdSize = 4; // 4 floats at a time for SSE.
  float sum[] = {0.0, 0.0, 0.0, 0.0};

  // Outer for loop to process elements in SIMD chunks.
  for (int i = 0; i < length; i += simdSize) {
    // Use inline assembly to load and sum the elements in SIMD registers
    // (SSE).
    __asm__("movups (%1), %%xmm0\n" // Load 4 floats from inputArray into xmm0
            "movups (%2), %%xmm1\n" // Load 4 floats from inputArray into xmm0
            "mulps %%xmm1, %%xmm0\n"
            "addps %%xmm0, %0\n" // Store the sum in the sum variable.
            : "=x"(sum)
            : "r"(&a[i]), "r"(&b[i])
            : "%xmm0", "%xmm1");
  }
  // For any remaining elements (if the array size is not a multiple of
  float result = sum[0] + sum[1] + sum[2] + sum[3];
  for (int i = length - (length % simdSize); i < length; i++) {
    result += a[i] * b[i];
  }

  return result;
}

void randomData(float_t *out, int length) {
  srand((unsigned int)time(NULL));
  for (int i = 0; i < length; i++) {
    out[i] = ((float_t)rand() / (float_t)(RAND_MAX)) * 1.0;
  }
}

#define SIZE 10000000
int main() {
  {
    float_t start, end;
    float_t data[SIZE] = {};
    randomData(data, SIZE);

    start = omp_get_wtime();
    printf("%f\n", dot(data, data, SIZE));
    end = omp_get_wtime();
    printf("Time %f seconds\n", end - start);

    start = omp_get_wtime();
    printf("%f\n", omp_dot(data, data, SIZE));
    end = omp_get_wtime();
    printf("Time %f seconds\n", end - start);

    start = omp_get_wtime();
    printf("%f\n", omp_simd_dot(data, data, SIZE));
    end = omp_get_wtime();
    printf("Time %f seconds\n", end - start);

    start = omp_get_wtime();
    printf("%f\n", asm_simd_dot(data, data, SIZE));
    end = omp_get_wtime();
    printf("Time %f seconds\n", end - start);
  }

  return 0;
}
