#include <iostream>
#include <vector>
#include <algorithm>

#include "main.h"

int main(int argc, char* argv[]) {

    if (argc != 2) {
        std::cout << "One argument required - code version number" << std::endl;
        std::cout << "1 - baseline implementation (1 block per kernel)" << std::endl;
        std::cout << "2 - implementation with streams" << std::endl;
        std::cout << "3 - implementation with block numbering" << std::endl;
        exit(1);
    }
    int version = (int)strtol(argv[1], nullptr, 10);

    /* Initialize cpu memory */
    auto *data_cpu = (float*)malloc(ROWS * COLUMNS * sizeof (float)),
         *data_cpu_normalized = (float*)malloc(ROWS * COLUMNS * sizeof (float)),
         *result_cpu = (float*) malloc(ROWS * ROWS * sizeof (float));

    /* Read data and normalize data*/
    ReadCSV(data_cpu);
    normalization(data_cpu, data_cpu_normalized);

    test_normalization(data_cpu_normalized);

    /* Initialize gpu memory */
    float *data_gpu, *result_gpu;
    if (cudaMalloc ((void**)&data_gpu, ROWS * COLUMNS * sizeof (float)) != cudaSuccess ||
        cudaMalloc ((void**)&result_gpu, ROWS * ROWS * sizeof (float)) != cudaSuccess) {
        std::cout << "cudaMalloc error!" << std::endl;
        exit(1);
    }
    cudaMemcpy(data_gpu, data_cpu_normalized, ROWS * COLUMNS * sizeof(float), cudaMemcpyHostToDevice);

    /* Compute scalar on different number of threads */

    std::vector<double> times;

    switch (version) {
        case 1:
            for (unsigned int i : BLOCK_SIZES) {

                times.clear();
                for (int test = 0; test < TESTS; ++test) {
                    times.push_back(baseline_scalars(data_gpu, result_gpu, i));
                    cudaMemcpy(result_cpu, result_gpu, ROWS * ROWS * sizeof(float), cudaMemcpyDeviceToHost);
                    test_scalars(result_cpu);
                }
                sort(times.begin(), times.end());
                std::cout << "Baseline version for " << i << " block size, time: "
                          << times[TESTS / 2] << " (ms)" << std::endl;
            }
            break;
        case 2:
            for (unsigned int i : STREAMS) {
                for (unsigned int j : BLOCK_SIZES) {

                    times.clear();
                    for (int test = 0; test < TESTS; ++test) {
                        times.push_back(streams_scalars(data_gpu, result_gpu, j, i));
                        cudaMemcpy(result_cpu, result_gpu, ROWS * ROWS * sizeof(float), cudaMemcpyDeviceToHost);
                        test_scalars(result_cpu);
                    }
                    sort(times.begin(), times.end());
                    std::cout << "Streams version for " << j << " block size and " << i
                              << " streams, time: " << times[TESTS / 2] << " (ms)" << std::endl;
                }
            }
            break;
        case 3:
            for (unsigned int i : BLOCK_SIZES) {

                times.clear();
                for (int test = 0; test < TESTS; ++test) {
                    times.push_back(two_dim_scalars(data_gpu, result_gpu, i));
                    cudaMemcpy(result_cpu, result_gpu, ROWS * ROWS * sizeof(float), cudaMemcpyDeviceToHost);
                    test_scalars(result_cpu);
                }
                sort(times.begin(), times.end());
                std::cout << "2D version for " << i << " block size, time: "
                          << times[TESTS / 2] << " (ms)" << std::endl;
            }
            break;
        default:
            std::cout << "Version not implemented" << std::endl;
            return 1;
    }


    return 0;
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

void normalization(float* data_cpu, float* result_cpu) {
    float dt_ms;
    float *data_gpu;

    if (cudaMalloc ((void**)&data_gpu , ROWS * COLUMNS * sizeof (float)) != cudaSuccess){
        std::cout << "cudaMalloc error!" << std::endl;
        exit(1);
    }

    if (cudaMemcpy(data_gpu, data_cpu, ROWS * COLUMNS * sizeof (float), cudaMemcpyHostToDevice) !=
        cudaSuccess){
        std::cout << "cudaMemcpy error!" << std::endl;
        exit(1);
    }

    cudaEvent_t event1, event2;
    cudaEventCreate(&event1);
    cudaEventCreate(&event2);

    cudaEventRecord(event1, nullptr);

    dim3 threadsPerBlock(NORMALIZATION_BLOCK_WIDTH, 1,1);
    dim3 numBlocks(ROWS / NORMALIZATION_BLOCK_WIDTH + 1, 1, 1);
    normalization_kernel<<<numBlocks, threadsPerBlock>>>(data_gpu);

    cudaEventRecord(event2, nullptr);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);

    cudaEventElapsedTime(&dt_ms, event1, event2);
    std::cout << "Normalization time: " << dt_ms << " ms" << std::endl;

    if (cudaMemcpy(result_cpu, data_gpu, ROWS * COLUMNS * sizeof (float), cudaMemcpyDeviceToHost) != cudaSuccess){
        std::cout << "cudaMemcpy error!" << std::endl;
        exit(1);
    }
    cudaDeviceSynchronize();
}

double baseline_scalars(float* data_gpu, float* result_gpu,
                      unsigned threads_per_block) {
    /*
     * Kernel launches slow down (from 0.06 ms to 2ms) after about 1020 launches.
     * I makes this implementation extremely slow. This issue is explained here:
     * https://stackoverflow.com/questions/53970187/cuda-stream-is-blocked-when-launching-many-kernels-1000
     */
    const unsigned num_blocks = 1;
    auto start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < ROWS; ++j) {
            baseline_scalars_kernel<<<num_blocks,
                             threads_per_block
                             >>>(data_gpu + (COLUMNS * i),
                                 data_gpu + (COLUMNS * j),
                                 result_gpu,
                                 i * ROWS + j);
        }
    }

    cudaDeviceSynchronize();
    auto end_time = std::chrono::steady_clock::now();
    double whole_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return whole_time;
}

double streams_scalars(float* data_gpu, float* result_gpu,
                     unsigned threads_per_block, const unsigned streams_number) {
    /*
     * Kernel launches slow down (from 0.06 ms to 2ms) after about 1020 launches.
     * I makes this implementation extremely slow. This issue is explained here:
     * https://stackoverflow.com/questions/53970187/cuda-stream-is-blocked-when-launching-many-kernels-1000
     */
    cudaStream_t stream[streams_number];
    for (int i = 0; i < streams_number; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    const unsigned num_blocks = 1;
    auto start_time = std::chrono::steady_clock::now();

    for (int i = 0; i < ROWS; ++i) {
        for (int j = 0; j < ROWS; ++j) {
            baseline_scalars_kernel<<<num_blocks,
                                      threads_per_block,
                                      0,
                                      stream[(i * ROWS + j) % streams_number]
                                      >>>(data_gpu + (COLUMNS * i),
                                          data_gpu + (COLUMNS * j),
                                          result_gpu,
                                          i * ROWS + j);
        }
    }
    cudaDeviceSynchronize();
    auto end_time = std::chrono::steady_clock::now();
    double whole_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    for (int i = 0; i < streams_number; ++i) {
        cudaStreamDestroy(stream[i]);
    }
    return whole_time;
}

double two_dim_scalars(float* data_gpu, float* result_gpu,
                     unsigned threads_per_block) {
    dim3 num_blocks(ROWS, ROWS, 1);

    auto start_time = std::chrono::steady_clock::now();
    two_dim_scalars_kernel<<<num_blocks, threads_per_block>>>(data_gpu, result_gpu);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::steady_clock::now();
    double whole_time = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    return whole_time;
}

__global__ void normalization_kernel(float* data_gpu) {
    unsigned vector_num = threadIdx.x + blockIdx.x * blockDim.x;
    if (vector_num < ROWS) {
        // Find sum
        float avg = 0;
        for (int i = 0; i < COLUMNS; ++i) {
            avg += data_gpu[vector_num * COLUMNS + i];
        }
        avg /= COLUMNS;

        // Decrease values by avg
        for (int i = 0; i < COLUMNS; ++i) {
            data_gpu[vector_num * COLUMNS + i] -= avg;
        }

        // Find length
        float len = 0;
        for (int i = 0; i < COLUMNS; ++i) {
            len += data_gpu[vector_num * COLUMNS + i] * data_gpu[vector_num * COLUMNS + i];
        }
        len = 1 / sqrt(len);

        // Normalize vector
        for (int i = 0; i < COLUMNS; ++i) {
            data_gpu[vector_num * COLUMNS + i] *= len;
        }
    }
}

__global__ void baseline_scalars_kernel(const float* vec_1, const float* vec_2,
                                        float* result_gpu, unsigned result_index) {
    __shared__ float partial_sums[MAX_BLOCK_SIZE];

    partial_sums[threadIdx.x] = 0;
    for (unsigned i = threadIdx.x; i < COLUMNS; i += blockDim.x) {
        partial_sums[threadIdx.x] += vec_1[i] * vec_2[i];
    }

    __syncthreads();

    /* Sum results of every thread (reduction) */

    for (unsigned jump = blockDim.x / 2; jump > 0; jump >>= 1) {
        if (threadIdx.x < jump)
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + jump];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        result_gpu[result_index] = partial_sums[0];

}

__global__ void two_dim_scalars_kernel(const float* data_gpu, float* result_gpu) {
    unsigned vec_1 = blockIdx.x, vec_2 = blockIdx.y;
    __shared__ float partial_sums[MAX_BLOCK_SIZE];

    unsigned first_index = COLUMNS * threadIdx.x / blockDim.x,
             end_index = COLUMNS * (threadIdx.x + 1) / blockDim.x;

    partial_sums[threadIdx.x] = 0;
    for (unsigned i = first_index; i < end_index; ++i) {
        partial_sums[threadIdx.x] += data_gpu[vec_1 * COLUMNS + i] * data_gpu[vec_2 * COLUMNS + i];
    }

    __syncthreads();

    /* Sum results of every thread (reduction) */

    for (unsigned jump = blockDim.x / 2; jump > 0; jump >>= 1) {
        if (threadIdx.x < jump)
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + jump];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        result_gpu[vec_1 * ROWS + vec_2] = partial_sums[0];
}


void test_normalization(const float* cpu_result) {
    float eps = 1e-4, sum, len;

    for (int row = 0; row < ROWS; ++row) {
        sum = 0;
        len = 0;
        for (int column = 0; column < COLUMNS; ++column) {
            sum += cpu_result[row * COLUMNS + column];
            len += cpu_result[row * COLUMNS + column] * cpu_result[row * COLUMNS + column];
        }
        if (abs(sum / ROWS) > eps) {
            std::cout << "Average in row " << row << " is non-zero" << std::endl;
            return;
        }
        else if (abs(std::sqrt(len) - 1) > eps) {
            std::cout << "Length in row " << row << " is non-one" << std::endl;
            return;
        }
    }

    std::cout << "Normalization tested" << std::endl;
}

void test_scalars(const float* result_cpu) {
    float eps = 1e-4, val;

    for (int i = 0; i < ROWS; ++i) {
        val = result_cpu[i * ROWS + i];
        if (abs(val - 1) > eps) {
            std::cout << "Scalar product of vector " << i << " with itself is non-one" << std::endl;
            std::cout << "It's " << val << std::endl;
            exit(1);
        }

    }
}

 & 310.086 & 308.331 & 308.816 & 310.414 & 312.6 & 340.708

