#include <stdio.h>

extern long long fact (long long n, long long a);

int
main ()
{
  long long n = 5;
  long long a = 1;
  long long result = fact (n, a);
  printf ("fact(%lld, %lld) = %lld\n", n, a, result);
  return 0;
}