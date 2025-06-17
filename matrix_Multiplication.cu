#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#define BLOCK_SIZE 16

using namespace std;

void print_matrices(float *M, const char *fname, int rows, int cols, int dim)
{
    ofstream f(fname);
    f << fixed << setprecision(2);
    for (int r = 0; r < rows; r++)
    {
        for (int c = 0; c < cols; c++)
        {
            f << M[r * dim + c] << " ";
        }
        f << "\n";
    }
}

__host__ void cpu_mult(float *A, float *B, float *C, int m)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float s = 0;
            for (int k = 0; k < m; k++)
            {
                s += A[i * m + k] * B[k * m + j];
            }
            C[i * m + j] = s;
        }
    }
}

__host__ int fill(float **A, float **B, int ax, int ay, int bx, int by)
{
    int mx = ax > bx ? ax : bx, my = ay > by ? ay : by;
    int m = mx > my ? mx : my;
    int blk = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
    m = blk * BLOCK_SIZE;
    size_t sz = m * m * sizeof(float);
    *A = (float *)malloc(sz);
    *B = (float *)malloc(sz);
    memset(*A, 0, sz);
    memset(*B, 0, sz);
    for (int i = 0; i < ax; i++)
    {
        for (int j = 0; j < ay; j++)
        {
            (*A)[i * m + j] = sinf(i * m + j);
        }
    }
    for (int i = 0; i < bx; i++)
    {
        for (int j = 0; j < by; j++)
        {
            (*B)[i * m + j] = cosf(i * m + j);
        }
    }
    return m;
}

__global__ void multKernel(float *A, float *B, float *C, int dim)
{
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    for (int t = 0; t < gridDim.x; t++)
    {
        int colA = t * BLOCK_SIZE + threadIdx.x;
        int rowB = t * BLOCK_SIZE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = A[row * dim + colA];
        Bs[threadIdx.y][threadIdx.x] = B[rowB * dim + col];
        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; k += 4)
        {
            sum += As[threadIdx.y][k + 0] * Bs[k + 0][threadIdx.x];
            sum += As[threadIdx.y][k + 1] * Bs[k + 1][threadIdx.x];
            sum += As[threadIdx.y][k + 2] * Bs[k + 2][threadIdx.x];
            sum += As[threadIdx.y][k + 3] * Bs[k + 3][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * dim + col] = sum;
}

int main()
{
    int ax, ay, bx, by;
    printf("Enter m n n k :\n");
    scanf("%d%d%d%d", &ax, &ay, &bx, &by);
    float *hA, *hB, *dA, *dB, *hC, *dC, *cC;
    int dim = fill(&hA, &hB, ax, ay, bx, by);
    print_matrices(hA, "A.txt", ax, ay, dim);
    print_matrices(hB, "B.txt", bx, by, dim);
    size_t sz = dim * dim * sizeof(float);
    hC = (float *)malloc(sz);
    cC = (float *)malloc(sz);
    cudaMalloc(&dA, sz);
    cudaMalloc(&dB, sz);
    cudaMalloc(&dC, sz);
    cudaMemcpy(dA, hA, sz, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sz, cudaMemcpyHostToDevice);
    dim3 bs(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gs(dim / BLOCK_SIZE, dim / BLOCK_SIZE);
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    multKernel<<<gs, bs>>>(dA, dB, dC, dim);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    cudaMemcpy(hC, dC, sz, cudaMemcpyDeviceToHost);
    clock_t t0 = clock();
    cpu_mult(hA, hB, cC, dim);
    double cpu_ms = 1000.0 * (clock() - t0) / CLOCKS_PER_SEC;
    printf("GPU time= %f ms\n", ms);
    printf("CPU time= %f ms\n", cpu_ms);
    print_matrices(hC, "C_gpu.txt", ax, by, dim);
    print_matrices(cC, "C_cpu.txt", ax, by, dim);
    bool eq = true;
    for (int i = 0; i < ax && eq; i++)
        for (int j = 0; j < by && eq; j++)
        {
            if (fabs(hC[i * dim + j] - cC[i * dim + j]) > 1e-3)
                eq = false;
        }
    if (eq)
        cout << "They are equal\n";
    else
        cout << "They are not equal\n";
    free(hA);
    free(hB);
    free(hC);
    free(cC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
