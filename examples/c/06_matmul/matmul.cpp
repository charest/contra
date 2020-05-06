#include <vector>

int main(int argc, char **argv)
{
  std::size_t num = 8192;
  std::size_t num_points = num*num;
  std::vector<double> matrix1(num_points, 1);
  std::vector<double> matrix2(num_points, 1);
  std::vector<double> matrix3(num_points, 0);

  for (int n=0; n<100; ++n) {
    for (std::size_t i=0; i<num_points; ++i) {
      matrix3[i] = matrix1[i] * matrix2[i];
    }
  }

  return 0;
} 
