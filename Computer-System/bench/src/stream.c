/*
 * stream - sequential triad, a[i] = b[i] + s * c[i].
 *
 * Pure streaming with unit stride: no reuse, perfectly prefetchable. This is the
 * bandwidth-bound corner of the roofline that lab 6 needs in order to show when
 * an accelerator stops being compute-bound.
 *
 * usage: stream <kib_per_array> <iters>
 */
#include "bench.h"

int main(int argc, char **argv) {
  int kib = (argc > 1) ? atoi(argv[1]) : 512;
  int iters = (argc > 2) ? atoi(argv[2]) : 20;

  size_t n = ((size_t)kib * 1024) / sizeof(int32_t);
  if (n < 16)
    n = 16;

  int32_t *a = (int32_t *)malloc(n * sizeof(int32_t));
  int32_t *b = (int32_t *)malloc(n * sizeof(int32_t));
  int32_t *c = (int32_t *)malloc(n * sizeof(int32_t));
  if (!a || !b || !c) {
    printf("stream: allocation failed\n");
    return 1;
  }

  for (size_t i = 0; i < n; i++) {
    b[i] = (int32_t)i;
    c[i] = (int32_t)(i * 3 + 1);
    a[i] = 0;
  }

  const int32_t s = 3;
  ROI_BEGIN();
  for (int it = 0; it < iters; it++)
    for (size_t i = 0; i < n; i++)
      a[i] = b[i] + s * c[i];
  ROI_END();

  int64_t checksum = 0;
  for (size_t i = 0; i < n; i += 64)
    checksum += a[i];
  printf("stream kib=%d iters=%d bytes_touched=%lld checksum=%lld\n", kib, iters,
         (long long)((int64_t)n * 12 * iters), (long long)checksum);

  free(a);
  free(b);
  free(c);
  return 0;
}
