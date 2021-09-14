#ifndef REDUKCJA_REDUKCJA_H
#define REDUKCJA_REDUKCJA_H

#include <iostream>
#include <cstdio>

/* GPU parameters */

//#define BLOCK_WIDTH 128
int BLOCK_WIDTH;
#define SHARED_MEM_SIZE 1024

/* Data file consts*/

const char* DATA_FILE_NAME = "neuroblastoma_CNV.csv";
const int BUF_SIZE = 2000000;

const int ROWS = 145; // We know the dataset
const int COLUMNS = 39116; // We know the dataset

/* Normalization types */

enum Orientation {
    rows,
    columns
};

enum Operation {
    op_sum,
    op_sq_sum,
    op_add,
    op_mul,
    op_pow2
};


/* Utils */

#define NEXT_POW2(x) pow(2, ceil(log(x)/log(2)))

/* CPU functions*/

void ReadCSV(float* data_cpu);

void naive_normalization(float* data_cpu, float* result_cpu, Orientation orientation);

void reduction(float* data_cpu, float* result_cpu,
               void (*reduction_kernel)(const float*, float*, int));

void test_normalization(const float* result_cpu, Orientation orientation);


/* GPU kernels */


__global__ void naive_kernel_columns(float* data_gpu, int rows, int columns);
__global__ void naive_kernel_rows(float* data_gpu, int rows, int columns);
__global__ void modify_array(float* data_gpu, int columns, float arg, Operation operation);

__global__ void reduction_kernel_v1(const float* data_gpu, float* result_gpu, int columns);
__global__ void reduction_kernel_v2(const float* data_gpu, float* result_gpu, int columns);
__global__ void reduction_kernel_v3(const float* data_gpu, float* result_gpu, int columns);
__global__ void reduction_kernel_v4(const float* data_gpu, float* result_gpu, int columns);
__global__ void reduction_kernel_v5(const float* data_gpu, float* result_gpu, int columns);
template <unsigned blockSize>
__global__ void reduction_kernel_v6(const float* data_gpu, float* result_gpu, int columns);

#endif //REDUKCJA_REDUKCJA_H
