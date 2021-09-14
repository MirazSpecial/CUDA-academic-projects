#ifndef STRUMIENIE_I_2D_MAIN_H
#define STRUMIENIE_I_2D_MAIN_H

#include <iostream>
#include <cstdio>
#include <chrono>
#include <ctime>
#include <cmath>

#define MAX_BLOCK_SIZE 1024
#define ROWS 145
#define COLUMNS 39116

#define TESTS 3

/* GPU parameters */

const unsigned NORMALIZATION_BLOCK_WIDTH = 64;
const unsigned BLOCK_SIZES[] = {32, 64, 128, 256, 512, 1024};
const unsigned STREAMS[] = {1, 2, 4, 8, 13, 14, 15, 16, 24, 28, 32, 64, 128};

/* Data file consts*/

const char* DATA_FILE_NAME = "neuroblastoma_CNV.csv";
const int BUF_SIZE = 2000000;


/* CPU functions*/

void ReadCSV(float* data_cpu);

void normalization(float* data_cpu, float* result_cpu);

double baseline_scalars(float* data_gpu, float* result_gpu,
                      unsigned threads_per_block);
double streams_scalars(float* data_gpu, float* result_gpu,
                     unsigned threads_per_block, unsigned streams_number);
double two_dim_scalars(float* data_gpu, float* result_gpu,
                     unsigned threads_per_block);

void test_normalization(const float* result_cpu);
void test_scalars(const float* result_cpu);

/* GPU kernels */

__global__ void normalization_kernel(float* data_gpu);
__global__ void baseline_scalars_kernel(const float* vec_1, const float* vec_2,
                                        float* result_gpu, unsigned result_index);
__global__ void two_dim_scalars_kernel(const float* data_gpu, float* result_gpu);


#endif //STRUMIENIE_I_2D_MAIN_H
