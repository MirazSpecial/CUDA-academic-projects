#ifndef REDUKCJA_SHUFFLE_REDUKCJA_SHUFFLE_H
#define REDUKCJA_SHUFFLE_REDUKCJA_SHUFFLE_H

#include <chrono>
#include <ctime>

/* Kernel execution constants */

const int RUN_NUM = 10;
const int BLOCK_SIZE = 1024;
#define SHARED_MEM_SIZE 1024
#define CLEAN_MEM 16

/* Utils */

enum Operation {
    op_sub_avg,
    op_dev_len,
};

enum SumType {
    plain,
    pow2,
};

/* CPU functions */

void ReadCSV(float* data_cpu);
void reduction(float* data_cpu, float* result_cpu, int version);
void test_normalization(const float* result_cpu);


/* Data file consts*/

const char* DATA_FILE_NAME = "neuroblastoma_CNV.csv";
const int BUF_SIZE = 2000000;

#define ROWS 145 // We know the dataset
#define COLUMNS 39116 // We know the dataset

/* Kernels */

__global__ void modify_array(float* data_gpu, const float* args, Operation operation);

__global__ void mixed_reduction_kernel(const float* data_gpu, float* result_gpu,
                                       SumType sum_type);
__global__ void clean_reduction_kernel(const float* data_gpu, float* result_gpu,
                                       SumType sum_type);

#endif //REDUKCJA_SHUFFLE_REDUKCJA_SHUFFLE_H
