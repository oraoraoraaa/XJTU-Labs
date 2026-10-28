/*
 * pchase - pointer chasing over a random cycle.
 *
 * Every load depends on the previous one, so there is no memory-level
 * parallelism and no stride for a prefetcher to lock onto. Sweeping the working
 * set size makes the cache hierarchy show up directly as steps in cycles/access.
 *
 * usage: pchase <kib> <accesses> [seed]
 */
#include "bench.h"

int main(int argc, char **argv) {
  int kib = (argc > 1) ? atoi(argv[1]) : 64;
  long accesses = (argc > 2) ? atol(argv[2]) : 1000000L;
  uint32_t seed = bench_seed(argc, argv, 3);

  size_t n = ((size_t)kib * 1024) / sizeof(uint32_t);
  if (n < 2)
    n = 2;

  uint32_t *idx = (uint32_t *)malloc(n * sizeof(uint32_t));
  if (!idx) {
    printf("pchase: allocation of %zu KiB failed\n", (size_t)kib);
    return 1;
  }

  for (size_t i = 0; i < n; i++)
    idx[i] = (uint32_t)i;

  /* Fisher-Yates on positions 1..n-1, then thread a single cycle through the
   * shuffled order so the chase visits every element exactly once. */
  rng_t r;
  rng_init(&r, seed);
  for (size_t i = n - 1; i > 1; i--) {
    size_t j = 1 + (rng_u32(&r) % (uint32_t)i);
    uint32_t t = idx[i];
    idx[i] = idx[j];
    idx[j] = t;
  }

  uint32_t *next = (uint32_t *)malloc(n * sizeof(uint32_t));
  if (!next) {
    printf("pchase: allocation failed\n");
    return 1;
  }
  for (size_t i = 0; i < n; i++)
    next[idx[i]] = idx[(i + 1) % n];

  free(idx);

  uint32_t p = 0;
  ROI_BEGIN();
  for (long k = 0; k < accesses; k++)
    p = next[p];
  ROI_END();

  printf("pchase kib=%d accesses=%ld final=%u\n", kib, accesses, p);
  free(next);
  return 0;
}
