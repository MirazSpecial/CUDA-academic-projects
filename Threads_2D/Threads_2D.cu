#include "Threads_2D.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "One argument required - code version number" << std::endl;
        std::cout << "1 - baseline implementation (Task 4)" << std::endl;
        std::cout << "2 - implementation with shared memory (Task 5)" << std::endl;
        std::cout << "3 - implementation with resolved bank conflict (Task 6)" << std::endl;
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
    if (cudaMalloc ((void**)&data_gpu , ROWS * COLUMNS * sizeof (float)) != cudaSuccess ||
        cudaMalloc ((void**)&result_gpu , ROWS * ROWS * sizeof (float)) != cudaSuccess){
        std::cout << "cudaMalloc error!" << std::endl;
        exit(1);
    }
    cudaMemcpy(data_gpu, data_cpu_normalized, ROWS * COLUMNS * sizeof(float), cudaMemcpyHostToDevice);

    /* Compute scalar on different number of threads */

    calculate_scalars(data_gpu, result_gpu, result_cpu, version);

    return 0;
}

void calculate_scalars(float* data_gpu, float* result_gpu,
                       float* result_cpu, int kernel_version) {
    for (auto dim_x : BLOCK_SIZES) {
        for (auto dim_y : BLOCK_SIZES) {
            unsigned threads_in_block = dim_x * dim_y;
            if (threads_in_block < MIN_BLOCK || threads_in_block > MAX_BLOCK)
                continue;

            unsigned blocks_x = ROWS / dim_x + 1;
            unsigned blocks_y = ROWS / dim_y + 1;
            dim3 threadsPerBlock(dim_x, dim_y,1);
            dim3 numBlocks(blocks_x, blocks_y, 1);

            double time_sum = 0;
            for (int i = 0; i < TEST_RUNS; ++i) {
                auto kernel_start = std::chrono::steady_clock::now();

                switch (kernel_version) {
                    case 1:
                        compute_scalar_kernel_v1<<<numBlocks,
                                                   threadsPerBlock
                                                   >>>(data_gpu, result_gpu);
                        break;
                    case 2:
                        compute_scalar_kernel_v2<<<numBlocks,
                                                   threadsPerBlock
                                                   >>>(data_gpu, result_gpu);
                        break;
                    default:
                        std::cout << "Version not implemented!" << std::endl;
                        exit(1);
                }

                cudaDeviceSynchronize();

                auto kernel_end = std::chrono::steady_clock::now();
                double kernel_time = std::chrono::duration<double, std::milli>(kernel_end - kernel_start).count();
                time_sum += kernel_time;
            }

            std::cout << "For block " << dim_x << " " << dim_y << " kernels took "
                  << time_sum / TEST_RUNS << " ms" << std::endl;

            cudaMemcpy(result_cpu, result_gpu,
                       ROWS * ROWS * sizeof(float), cudaMemcpyDeviceToHost);

            test_scalars(result_cpu);
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
    naive_kernel_rows<<<numBlocks, threadsPerBlock>>>(data_gpu, ROWS, COLUMNS);

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

__global__ void naive_kernel_rows(float* data_gpu, int rows, int columns) {
    unsigned vector_num = threadIdx.x + blockIdx.x * blockDim.x;
    if (vector_num < rows) {
        // Find sum
        float avg = 0;
        for (int i = 0; i < columns; ++i) {
            avg += data_gpu[vector_num * COLUMNS + i];
        }
        avg /= COLUMNS;

        // Decrease values by avg
        for (int i = 0; i < columns; ++i) {
            data_gpu[vector_num * COLUMNS + i] -= avg;
        }

        // Find length
        float len = 0;
        for (int i = 0; i < columns; ++i) {
            len += data_gpu[vector_num * COLUMNS + i] * data_gpu[vector_num * COLUMNS + i];
        }
        len = 1 / sqrt(len);

        // Normalize vector
        for (int i = 0; i < columns; ++i) {
            data_gpu[vector_num * COLUMNS + i] *= len;
        }
    }
}


__global__ void compute_scalar_kernel_v1(const float* data_gpu, float* result_gpu) {
    unsigned x_pos = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y_pos = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_pos < ROWS && y_pos < ROWS && x_pos >= y_pos) {
        float sum = 0;
        for (unsigned i = 0; i < COLUMNS; ++i) {
            sum += data_gpu[x_pos * COLUMNS + i] * data_gpu[y_pos * COLUMNS + i];
        }

        result_gpu[y_pos * ROWS + x_pos] = sum;
        result_gpu[x_pos * ROWS + y_pos] = sum;
    }
}

__global__ void compute_scalar_kernel_v2(const float* data_gpu, float* result_gpu) {
    __shared__ float values_1[MAX_BLOCK_SIZE][STEP_LENGTH];
    __shared__ float values_2[MAX_BLOCK_SIZE][STEP_LENGTH];
    __shared__ float results[MAX_BLOCK];

    unsigned x_pos = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y_pos = blockDim.y * blockIdx.y + threadIdx.y;

    results[threadIdx.y * blockDim.x + threadIdx.x] = 0;

    for (int vec_pos = 0; vec_pos < COLUMNS; vec_pos += STEP_LENGTH) {

        /* Load data to shared memory and synchronize */

        if (x_pos < ROWS && y_pos < ROWS) {
            for (unsigned i = threadIdx.x; i < STEP_LENGTH; i += blockDim.x) {
                values_1[threadIdx.y][i] =
                        data_gpu[y_pos * COLUMNS + vec_pos * STEP_LENGTH + i];
            }
            for (unsigned i = threadIdx.y; i < STEP_LENGTH; i += blockDim.y) {
                values_2[threadIdx.x][i] =
                        data_gpu[x_pos * COLUMNS + vec_pos * STEP_LENGTH + i];
            }
        }

        __syncthreads();

        /* Multiply partial */

        if (x_pos < ROWS && y_pos < ROWS) {
            for (int i = 0; i < STEP_LENGTH && vec_pos * STEP_LENGTH + i < COLUMNS; ++i) {
                results[threadIdx.y * blockDim.x + threadIdx.x] +=
                        values_1[threadIdx.y][i] * values_2[threadIdx.x][i];
            }
        }
    }

    if (x_pos < ROWS && y_pos < ROWS)
        result_gpu[y_pos * ROWS + x_pos] = results[threadIdx.y * blockDim.x + threadIdx.x];

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
            exit(1);
        }
        else if (abs(std::sqrt(len) - 1) > eps) {
            std::cout << "Length in row " << row << " is non-one" << std::endl;
            exit(1);
        }
    }

    std::cout << "Normalization tests passed." << std::endl;
}

void test_scalars(const float* result_cpu) {
    float eps = 1e-4, val;

    for (int i = 0; i < ROWS; ++i) {
        val = result_cpu[i * ROWS + i];
        if (abs(val - 1) > eps) {
            std::cout << "Scalar product of vector " << i << " with itself is non-one, it's "
                      << val << std::endl;
            exit(1);
        }
    }
}
