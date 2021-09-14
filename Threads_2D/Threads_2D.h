#ifndef THREADS_2D_THREADS_2D_H
#define THREADS_2D_THREADS_2D_H


#include <iostream>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <cmath>

const int TEST_RUNS = 50;

#define STEP_LENGTH 8
#define MAX_BLOCK_SIZE 64
#define ROWS 145
#define COLUMNS 39116

/* GPU parameters */

const unsigned NORMALIZATION_BLOCK_WIDTH = 64;
const unsigned BLOCK_SIZES[] = {4, 8, 16, 24, 32, 64};
const unsigned MIN_BLOCK = 32;
#define MAX_BLOCK 1024

/* Data file consts*/
const char* DATA_FILE_NAME = "neuroblastoma_CNV.csv";
const int BUF_SIZE = 2000000;


/* CPU functions*/

void ReadCSV(float* data_cpu);

void normalization(float* data_cpu, float* result_cpu);

void calculate_scalars(float* data_gpu, float* result_gpu,
                          float* result_cpu, int kernel_version);


void test_normalization(const float* result_cpu);
void test_scalars(const float* result_cpu);

/* GPU kernels */

__global__ void naive_kernel_rows(float* data_gpu, int rows, int columns);
__global__ void compute_scalar_kernel_v1(const float* data_gpu, float* result_gpu);
__global__ void compute_scalar_kernel_v2(const float* data_gpu, float* result_gpu);

#endif //THREADS_2D_THREADS_2D_H
