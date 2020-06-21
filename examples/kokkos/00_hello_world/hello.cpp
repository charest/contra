#include <Kokkos_Core.hpp>

#include <cstdio>

void hello_world_task()
{ printf("Hello World!\n"); }
  

int main(int argc, char **argv)
{
  Kokkos::initialize( argc, argv );

  int num=1;
  Kokkos::parallel_for( num, KOKKOS_LAMBDA ( int i ) {
    hello_world_task();
  });
  
  Kokkos::finalize();
  return 0;
} 
