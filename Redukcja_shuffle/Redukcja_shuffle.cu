#include <iostream>

#include "Redukcja_shuffle.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "One argument required - reduction version" << std::endl;
        std::cout << "1 - mixed version" << std::endl;
        std::cout << "2 - clean version" << std::endl;
        exit(1);
    }
    int version = (int)strtol(argv[1], nullptr, 10);

    auto *data_cpu = (float*)malloc(ROWS * COLUMNS * sizeof (float)),
            *result_cpu = (float*)malloc(ROWS * COLUMNS * sizeof (float));

    ReadCSV(data_cpu);

    reduction(data_cpu, result_cpu, version);

//    test_normalization(result_cpu);
    return 0;
}

void reduction(float* data_cpu, float* result_cpu, int version) {
    float *data_gpu, *result_gpu;
    double min_time = MAXFLOAT;

    if (cudaMalloc((void **) &data_gpu, ROWS * COLUMNS * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void **) &result_gpu, COLUMNS * sizeof(float)) != cudaSuccess) {
        std::cout << "cudaMalloc error!" << std::endl;
        exit(1);
    }
    if (cudaMemcpy(data_gpu, data_cpu, ROWS * COLUMNS * sizeof (float), cudaMemcpyHostToDevice) !=
        cudaSuccess){
        std::cout << "cudaMemcpy error!" << std::endl;
        exit(1);
    }

    for (int test_run = 0; test_run < RUN_NUM; ++test_run) {

        auto reduction_start = std::chrono::steady_clock::now();

        if (version == 1) {
            mixed_reduction_kernel<<<COLUMNS, BLOCK_SIZE>>>(data_gpu, result_gpu, plain);
            modify_array<<<COLUMNS, BLOCK_SIZE>>>(data_gpu, result_gpu, op_sub_avg);
            mixed_reduction_kernel<<<COLUMNS, BLOCK_SIZE>>>(data_gpu, result_gpu, pow2);
            modify_array<<<COLUMNS, BLOCK_SIZE>>>(data_gpu, result_gpu, op_dev_len);
        }
        else if (version == 2) {
            clean_reduction_kernel<<<COLUMNS, BLOCK_SIZE>>>(data_gpu, result_gpu, plain);
            modify_array<<<COLUMNS, BLOCK_SIZE>>>(data_gpu, result_gpu, op_sub_avg);
            clean_reduction_kernel<<<COLUMNS, BLOCK_SIZE>>>(data_gpu, result_gpu, pow2);
            modify_array<<<COLUMNS, BLOCK_SIZE>>>(data_gpu, result_gpu, op_dev_len);
        }

        cudaDeviceSynchronize();

        auto reduction_end = std::chrono::steady_clock::now();
        double test_run_time = std::chrono::duration <double, std::milli> (reduction_end - reduction_start).count();

        std::cout << "Run time in ms: " << test_run_time << std::endl;

        min_time = std::min(min_time, test_run_time);
    }

    std::cout << "Minimal time in ms: " << min_time << std::endl;

    if (cudaMemcpy(result_cpu, data_gpu, ROWS * COLUMNS * sizeof (float), cudaMemcpyDeviceToHost) != cudaSuccess){
        std::cout << "cudaMemcpy error!" << std::endl;
        exit(1);
    }

}

__global__ void modify_array(float* data_gpu, const float* args, Operation operation) {
    if (threadIdx.x < ROWS) {
        float *target = &data_gpu[threadIdx.x * COLUMNS + blockIdx.x];
        switch (operation) {
            case op_sub_avg:
                *target -= args[blockIdx.x] / ROWS;
                break;
            case op_dev_len:
                *target /= sqrt(args[blockIdx.x]);
                break;
        }
    }
}

__global__ void mixed_reduction_kernel(const float* data_gpu, float* result_gpu,
                                       SumType sum_type) {
    __shared__ float sums[SHARED_MEM_SIZE];
    float sum;

    if (threadIdx.x < ROWS) {
        sums[threadIdx.x] = data_gpu[threadIdx.x * COLUMNS + blockIdx.x];
        if (sum_type == pow2)
            sums[threadIdx.x] *= data_gpu[threadIdx.x * COLUMNS + blockIdx.x];
    }
    __syncthreads();

    for (unsigned jump = blockDim.x / 2; jump >= 32; jump >>= 1) {
        if (threadIdx.x < jump && threadIdx.x + jump < COLUMNS)
            sums[threadIdx.x] += sums[threadIdx.x + jump];
        __syncthreads();
    }

    sum = sums[threadIdx.x];
    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    if (threadIdx.x == 0)
        result_gpu[blockIdx.x] = sum;
}

__global__ void clean_reduction_kernel(const float* data_gpu, float* result_gpu,
                                       SumType sum_type) {
    __shared__ float sums[CLEAN_MEM];
    float sum = 0;

    if (threadIdx.x < ROWS) {
        sum = data_gpu[threadIdx.x * COLUMNS + blockIdx.x];
        if (sum_type == pow2)
            sum *= data_gpu[threadIdx.x * COLUMNS + blockIdx.x];
    }

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);

    if (threadIdx.x % 32 == 0)
        sums[threadIdx.x / 32] = sum;

    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < ROWS / 32 + 1; ++i) {
            result_gpu[blockIdx.x] += sums[i];
        }
    }
}



void ReadCSV(float* data_cpu) {
    FILE* CNVfile;
    float* row;
    char buffer[BUF_SIZE];
    int col_num;


    row = (float*)malloc(COLUMNS * sizeof(float));
    CNVfile = fopen(DATA_FILE_NAME,"r");
    if (CNVfile == nullptr) {
        std::cout << "Opening data file failed!" << std::endl;
        exit(1);
    }

    int row_count = -1;
    while (fgets(buffer, BUF_SIZE - 1, CNVfile) != nullptr){
        if (row_count < 0) {
            row_count++;
            continue; // We skip header line
        }

        char *col = strtok(buffer, ",");
        for (col_num = -1; col; col_num++) {
            if (col_num >= 0)
                row[col_num] = (float)strtod(col, nullptr);
            col = strtok(nullptr, ",");
        }

        for (int i=0;i<=COLUMNS;i++) {
            data_cpu[row_count * COLUMNS + i] = row[i];
        }

        row_count++;
    }

    fclose(CNVfile);
}

void test_normalization(const float* cpu_result) {
    float eps = 1e-4, sum, len;

    for (int column = 0; column < COLUMNS; ++column) {
        sum = 0;
        len = 0;
        for (int row = 0; row < ROWS; ++row) {
            sum += cpu_result[row * COLUMNS + column];
            len += cpu_result[row * COLUMNS + column] * cpu_result[row * COLUMNS + column];
        }
        if (abs(sum / ROWS) > eps) {
            std::cout << "Average in column " << column << " is non-zero, it is "
                      << sum / ROWS << std::endl;
            return;
        }
        else if (abs(std::sqrt(len) - 1) > eps) {
            std::cout << "Length in column " << column << " is non-one, it is "
                      << std::sqrt(len) << std::endl;
            return;
        }
    }
}