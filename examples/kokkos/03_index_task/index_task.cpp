#include <Kokkos_Core.hpp>

#include <cassert>
#include <cstdio>

void index_space_task(int i, int local_arg, int global_arg)
{
  printf("Hello world from task %i, with local arg %d, and global arg %d!\n",
		i, local_arg, global_arg);
}

int main(int argc, char **argv)
{

  Kokkos::initialize( argc, argv );

  int num_points = 10;
  printf("Running hello world redux for %d points...\n", num_points);

  Kokkos::parallel_for( Kokkos::RangePolicy<>(0,num_points), KOKKOS_LAMBDA ( int i ) {
    index_space_task(i, num_points, i+num_points);
  });

  Kokkos::finalize();
  return 0;
} 
