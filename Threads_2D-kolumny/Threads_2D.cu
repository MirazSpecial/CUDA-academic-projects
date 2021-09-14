#include <cmath>

#include "Threads_2D.h"

int main() {
    /* Initialize cpu memory */
    auto *data_cpu = (float*)malloc(ROWS * COLUMNS * sizeof (float)),
         *data_cpu_normalized = (float*)malloc(ROWS * COLUMNS * sizeof (float)),
         *result_cpu = (float*) malloc(RESULT_SIZE * RESULT_SIZE * sizeof (float));

    /* Read data */
    ReadCSV(data_cpu);

    /* Normalize vectors */
    normalization(data_cpu, data_cpu_normalized);

    /* Initialize gpu memory */

    float *data_gpu, *result_gpu;
    if (cudaMalloc ((void**)&data_gpu , ROWS * COLUMNS * sizeof (float)) != cudaSuccess ||
        cudaMalloc ((void**)&result_gpu , RESULT_SIZE * RESULT_SIZE * sizeof (float)) != cudaSuccess){
        std::cout << "cudaMalloc error!" << std::endl;
        exit(1);
    }
    cudaMemcpy(data_gpu, data_cpu_normalized, ROWS * COLUMNS * sizeof(float), cudaMemcpyHostToDevice);

    /* Compute scalar on different number of threads */

    for (auto dim_x : BLOCK_SIZES) {
        for (auto dim_y : BLOCK_SIZES) {
            unsigned threads_in_block = dim_x * dim_y;
            if (threads_in_block < MIN_BLOCK || threads_in_block > MAX_BLOCK)
                continue;

            unsigned blocks_x = RESULT_SIZE / dim_x + 1;
            unsigned blocks_y = RESULT_SIZE / dim_y + 1;
            dim3 threadsPerBlock(dim_x, dim_y,1);
            dim3 numBlocks(blocks_x, blocks_y, 1);

            auto kernel_start = std::chrono::steady_clock::now();

            compute_scalar_kernel<<<numBlocks, threadsPerBlock>>>(data_gpu, result_gpu,
                                                                  RESULT_SIZE, ROWS, COLUMNS);
            cudaDeviceSynchronize();

            auto kernel_end = std::chrono::steady_clock::now();
            double kernel_time = std::chrono::duration <double, std::milli> (kernel_end - kernel_start).count();
            std::cout << "For block " << dim_x << " " << dim_y << " kernels took "
                      << kernel_time << " ms" << std::endl;

            cudaMemcpy(result_cpu, result_gpu,
                       RESULT_SIZE * RESULT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            test_scalars(result_cpu);
            count_scalars(result_cpu);
        }
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
    dim3 numBlocks(COLUMNS / NORMALIZATION_BLOCK_WIDTH + 1, 1, 1);
    naive_kernel_columns<<<numBlocks, threadsPerBlock>>>(data_gpu, ROWS, COLUMNS);

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

__global__ void naive_kernel_columns(float* data_gpu, int rows, int columns) {
    unsigned vector_num = threadIdx.x + blockIdx.x * blockDim.x;

    // Omit first column
    if (0 < vector_num && vector_num < columns) {
        // Find sum
        float avg = 0;
        for (int i = 0; i < rows; ++i) {
            avg += data_gpu[i * COLUMNS + vector_num];
        }
        avg /= ROWS;

        // Decrease values by avg
        for (int i = 0; i < rows; ++i) {
            data_gpu[i * COLUMNS + vector_num] -= avg;
        }

        // Find length
        float len = 0;
        for (int i = 0; i < rows; ++i) {
            len += data_gpu[i * COLUMNS + vector_num] * data_gpu[i * COLUMNS + vector_num];
        }
        len = 1 / sqrt(len);

        // Normalize vector
        for (int i = 0; i < rows; ++i) {
            data_gpu[i * COLUMNS + vector_num] *= len;
        }
    }
}


__global__ void compute_scalar_kernel(const float* data_gpu, float* result_gpu,
                                      unsigned result_size, int rows, int columns) {
    unsigned x_pos = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned y_pos = blockDim.y * blockIdx.y + threadIdx.y;

    if (x_pos < result_size && y_pos < result_size && x_pos >= y_pos) {
        float sum = 0;
        for (unsigned i = 0; i < rows; ++i) {
            sum += data_gpu[i * columns + x_pos + 1] * data_gpu[i * columns + y_pos + 1];
            // +1 in index because we omit first column
        }
        result_gpu[y_pos * result_size + x_pos] = sum;
        result_gpu[x_pos * result_size + y_pos] = sum;
    }
}

void count_scalars(const float* result_cpu) {
    float eps = 1e-4;
    int scalars_counters[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    for (int i = 0; i < RESULT_SIZE * RESULT_SIZE; ++i) {
        if (abs(result_cpu[i]) <= eps)
            scalars_counters[0]++;
        if (result_cpu[i] > 0.01)
            scalars_counters[1]++;
        if (result_cpu[i] > 0.02)
            scalars_counters[2]++;
        if (result_cpu[i] > 0.05)
            scalars_counters[3]++;
        if (result_cpu[i] > 0.1)
            scalars_counters[4]++;
        if (result_cpu[i] > 0.2)
            scalars_counters[5]++;
        if (result_cpu[i] > 0.5)
            scalars_counters[6]++;
        if (abs(result_cpu[i] - 1) < eps)
            scalars_counters[7]++;
    }

    std::cout << "Scalars counted: " << scalars_counters[0] << " "
                                     << scalars_counters[1] << " "
                                     << scalars_counters[2] << " "
                                     << scalars_counters[3] << " "
                                     << scalars_counters[4] << " "
                                     << scalars_counters[5] << " "
                                     << scalars_counters[6] << " "
                                     << scalars_counters[7] << std::endl;
}

void test_normalization(const float* cpu_result) {
    float eps = 1e-4, sum, len;

    // Omit first column in normalization
    for (int column = 1; column < COLUMNS; ++column) {
        sum = 0;
        len = 0;
        for (int row = 0; row < ROWS; ++row) {
            sum += cpu_result[row * COLUMNS + column];
            len += cpu_result[row * COLUMNS + column] * cpu_result[row * COLUMNS + column];
        }
        if (abs(sum / ROWS) > eps) {
            std::cout << "Average in column " << column << " is non-zero" << std::endl;
            return;
        }
        else if (abs(std::sqrt(len) - 1) > eps) {
            std::cout << "Length in column " << column << " is non-one" << std::endl;
            return;
        }
    }

}

void test_scalars(const float* result_cpu) {
    float eps = 1e-4, val;

    for (int i = 0; i < RESULT_SIZE; ++i) {
        val = result_cpu[i * RESULT_SIZE + i];
        if (abs(val - 1) > eps) {
            std::cout << "Scalar product of vector " << i << " with itself is non-one" << std::endl;
            return;
        }
    }
}
