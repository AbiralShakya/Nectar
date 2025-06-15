#include <cuda_runtime.h>
#include <iostream>

__global__ void addKernel(float* a, float* b, float* c, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int N = 256;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    // Allocate host memory
    a = new float[N];
    b = new float[N];
    c = new float[N];

    // Initialize
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Allocate device memory
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));

    // Copy to device
    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    addKernel<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c, N);

    // Copy result back
    cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
