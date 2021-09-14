#ifndef THREADS_2D_THREADS_2D_H
#define THREADS_2D_THREADS_2D_H


#include <iostream>
#include <cstdio>
#include <chrono>
#include <ctime>

/* GPU parameters */

const unsigned NORMALIZATION_BLOCK_WIDTH = 64;
const unsigned BLOCK_SIZES[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
const unsigned MIN_BLOCK = 32;
const unsigned MAX_BLOCK = 1024;

/* Result parameters */

const int RESULT_SIZE = 8096;

/* Data file consts*/

const char* DATA_FILE_NAME = "neuroblastoma_CNV.csv";
const int BUF_SIZE = 2000000;

const int ROWS = 145; // We know the dataset
const int COLUMNS = 39116; // We know the dataset


/* CPU functions*/

void ReadCSV(float* data_cpu);

void normalization(float* data_cpu, float* result_cpu);
void count_scalars(const float* result_cpu);

void test_normalization(const float* result_cpu);
void test_scalars(const float* result_cpu);

/* GPU kernels */

__global__ void naive_kernel_columns(float* data_gpu, int rows, int columns);
__global__ void compute_scalar_kernel(const float* data_gpu, float* result_gpu,
                                      unsigned result_size, int rows, int columns);

#endif //THREADS_2D_THREADS_2D_H
