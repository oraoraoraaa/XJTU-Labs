/*
 * branchy - identical work, different branch predictability.
 *
 * The loop body is byte-for-byte the same in all three modes; only the *data*
 * changes, which changes how often the branch predictor is right. This isolates
 * misprediction cost from every other effect and is the core demo for lab 4.
 *
 *   mode 0  random   - ~50% taken, essentially unpredictable
 *   mode 1  sorted   - one single mispredict at the threshold crossing
 *   mode 2  periodic - repeating pattern, invisible to a bimodal predictor but
 *                      learnable by a history-based (tournament/TAGE) one
 *
 * usage: branchy <mode> <n> <iters> [seed]
 */
#include "bench.h"

static void sort_u32(uint32_t *a, size_t n) {
  for (size_t gap = n / 2; gap > 0; gap /= 2)
    for (size_t i = gap; i < n; i++) {
      uint32_t v = a[i];
      size_t j = i;
      while (j >= gap && a[j - gap] > v) {
        a[j] = a[j - gap];
        j -= gap;
      }
      a[j] = v;
    }
}

int main(int argc, char **argv) {
  int mode = (argc > 1) ? atoi(argv[1]) : 0;
  size_t n = (argc > 2) ? (size_t)atol(argv[2]) : 8192;
  int iters = (argc > 3) ? atoi(argv[3]) : 200;
  uint32_t seed = bench_seed(argc, argv, 4);

  uint32_t *a = (uint32_t *)malloc(n * sizeof(uint32_t));
  if (!a) {
    printf("branchy: allocation failed\n");
    return 1;
  }

  rng_t r;
  rng_init(&r, seed);
  for (size_t i = 0; i < n; i++)
    a[i] = rng_u32(&r) % 256;

  if (mode == 1)
    sort_u32(a, n);
  else if (mode == 2)
    for (size_t i = 0; i < n; i++)
      a[i] = (i % 7 < 3) ? 200 : 50; /* period-7 pattern */

  long taken = 0;
  int64_t sum = 0;

  ROI_BEGIN();
  for (int it = 0; it < iters; it++)
    for (size_t i = 0; i < n; i++) {
      if (a[i] >= 128) { /* the branch under study */
        sum += a[i];
        taken++;
      } else {
        sum -= a[i];
      }
    }
  ROI_END();

  printf("branchy mode=%d n=%zu iters=%d sum=%lld taken=%ld (%.1f%%)\n", mode, n,
         iters, (long long)sum, taken, 100.0 * taken / ((double)n * iters));
  free(a);
  return 0;
}
