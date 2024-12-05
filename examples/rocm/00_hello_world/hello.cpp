#include <hip/hip_runtime.h>
#include <iostream>

__global__ void add(int *a, int *b, int *c, int n)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    c[index] = a[index] + b[index];
		printf("Hello from thread %d\n", index);
  }
}

int main() {
  const int arraySize = 5;
  int a[arraySize] = {1, 2, 3, 4, 5};
  int b[arraySize] = {10, 20, 30, 40, 50};
  int c[arraySize] = {0};

  int *d_a, *d_b, *d_c;
  hipMalloc((void**)&d_a, arraySize * sizeof(int));
  hipMalloc((void**)&d_b, arraySize * sizeof(int));
  hipMalloc((void**)&d_c, arraySize * sizeof(int));

  hipMemcpy(d_a, a, arraySize * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(d_b, b, arraySize * sizeof(int), hipMemcpyHostToDevice);

  add<<<1, arraySize>>>(d_a, d_b, d_c, arraySize);

  hipMemcpy(c, d_c, arraySize * sizeof(int), hipMemcpyDeviceToHost);

  std::cout << "Result: ";
  for (int i = 0; i < arraySize; i++) {
      std::cout << c[i] << " ";
  }
  std::cout << std::endl;

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_c);

  return 0;
}

