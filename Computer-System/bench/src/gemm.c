/*
 * gemm - int8 matrix multiply, naive vs blocked.
 *
 * Same arithmetic, same instruction count (near enough), different traversal
 * order. The gap between mode 0 and mode 1 is pure data locality, which lets
 * lab 2 show that a software change can beat a hardware change.
 *
 * usage: gemm <n> <mode> <tile> [seed]      mode 0 = naive ijk, 1 = blocked
 */
#include "bench.h"

int main(int argc, char **argv) {
  int n = (argc > 1) ? atoi(argv[1]) : 128;
  int mode = (argc > 2) ? atoi(argv[2]) : 0;
  int tile = (argc > 3) ? atoi(argv[3]) : 32;
  uint32_t seed = bench_seed(argc, argv, 4);

  if (tile <= 0)
    tile = 32;

  int8_t *A = (int8_t *)malloc((size_t)n * n);
  int8_t *B = (int8_t *)malloc((size_t)n * n);
  int32_t *C = (int32_t *)calloc((size_t)n * n, sizeof(int32_t));
  if (!A || !B || !C) {
    printf("gemm: allocation failed\n");
    return 1;
  }

  rng_t r;
  rng_init(&r, seed);
  for (int i = 0; i < n * n; i++) {
    A[i] = rng_i8(&r);
    B[i] = rng_i8(&r);
  }

  ROI_BEGIN();
  if (mode == 0) {
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
        int32_t acc = 0;
        for (int k = 0; k < n; k++)
          acc += (int32_t)A[i * n + k] * (int32_t)B[k * n + j];
        C[i * n + j] = acc;
      }
  } else {
    for (int ii = 0; ii < n; ii += tile)
      for (int kk = 0; kk < n; kk += tile)
        for (int jj = 0; jj < n; jj += tile) {
          int imax = (ii + tile < n) ? ii + tile : n;
          int kmax = (kk + tile < n) ? kk + tile : n;
          int jmax = (jj + tile < n) ? jj + tile : n;
          for (int i = ii; i < imax; i++)
            for (int k = kk; k < kmax; k++) {
              int32_t a = A[i * n + k];
              for (int j = jj; j < jmax; j++)
                C[i * n + j] += a * (int32_t)B[k * n + j];
            }
        }
  }
  ROI_END();

  int64_t checksum = 0;
  for (int i = 0; i < n * n; i++)
    checksum += C[i];
  printf("gemm n=%d mode=%d tile=%d checksum=%lld\n", n, mode, tile,
         (long long)checksum);

  free(A);
  free(B);
  free(C);
  return 0;
}
