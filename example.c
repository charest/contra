#include <stdio.h>

void task( double * ptr, int n )
{
  for ( int i=0; i<n; ++i ) {
    printf("%f, ", ptr[i]);
  }
  printf("\n");
}
