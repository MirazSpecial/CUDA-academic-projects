#include <cmath>

#include "Redukcja.h"

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Arguments are: block_width, reduction_version" << std::endl;
        exit(1);
    }

    BLOCK_WIDTH = (int)strtol(argv[1], nullptr, 10);
    int version = (int)strtol(argv[2], nullptr, 10);

    auto *data_cpu = (float*)malloc(ROWS * COLUMNS * sizeof (float)),
         *result_cpu = (float*)malloc(ROWS * COLUMNS * sizeof (float));
    ReadCSV(data_cpu);

    switch (version) {
        case 0:
            naive_normalization(data_cpu, result_cpu, rows);
//            naive_normalization(data_cpu, result_cpu, columns);
            break;
        case 1:
            reduction(data_cpu, result_cpu, reduction_kernel_v1);
            break;
        case 2:
            reduction(data_cpu, result_cpu, reduction_kernel_v2);
            break;
        case 3:
            reduction(data_cpu, result_cpu, reduction_kernel_v3);
            break;
        case 4:
            reduction(data_cpu, result_cpu, reduction_kernel_v4);
            break;
        case 5:
            reduction(data_cpu, result_cpu, reduction_kernel_v5);
            break;
        case 6: {
//            reduction(data_cpu, result_cpu, reduction_kernel_v6);
            break;
        }
        default:
            std::cout << "Version not implemented!" << std::endl;
            return 0;
    }

    test_normalization(result_cpu, rows);
    return 0;
}



void naive_normalization(float* data_cpu, float* result_cpu, Orientation orientation) {
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

    if (orientation == columns) {
        dim3 threadsPerBlock(BLOCK_WIDTH, 1,1);
        dim3 numBlocks(COLUMNS / BLOCK_WIDTH + 1, 1, 1);
        naive_kernel_columns<<<numBlocks, threadsPerBlock>>>(data_gpu, ROWS, COLUMNS);
    }
    else if (orientation == rows) {
        dim3 threadsPerBlock(BLOCK_WIDTH, 1,1);
        dim3 numBlocks(ROWS / BLOCK_WIDTH + 1, 1, 1);
        naive_kernel_rows<<<numBlocks, threadsPerBlock>>>(data_gpu, ROWS, COLUMNS);
    }

    cudaEventRecord(event2, nullptr);

    cudaEventSynchronize(event1);
    cudaEventSynchronize(event2);

    cudaEventElapsedTime(&dt_ms, event1, event2);
    std::cout << "Execution time: " << dt_ms << " ms" << std::endl;

    if (cudaMemcpy(result_cpu, data_gpu, ROWS * COLUMNS * sizeof (float), cudaMemcpyDeviceToHost) != cudaSuccess){
        std::cout << "cudaMemcpy error!" << std::endl;
        exit(1);
    }
    cudaDeviceSynchronize();
}


void reduction(float* data_cpu, float* result_cpu,
               void (*reduction_kernel)(const float*, float*, int)) {
    float dt_ms = 0, dt_ms_part, avg, len;
    float *data_gpu, *result_gpu, *help_gpu;
    int blocks;

    for (int row = 0; row < ROWS; ++ row) {

        if (cudaMalloc((void **) &data_gpu, COLUMNS * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void **) &result_gpu, COLUMNS * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void **) &help_gpu, COLUMNS * sizeof(float)) != cudaSuccess) {
            std::cout << "cudaMalloc error!" << std::endl;
            exit(1);
        }

        if (cudaMemcpy(data_gpu, data_cpu+(COLUMNS * row), COLUMNS * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cout << "cudaMemcpy error!" << std::endl;
            exit(1);
        }

        cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);
        cudaEventRecord(event1, nullptr);

        blocks = COLUMNS / BLOCK_WIDTH + 1;
        /* Calculating average */

        if (reduction_kernel == &reduction_kernel_v4)
            (*reduction_kernel)<<<blocks / 2, BLOCK_WIDTH>>>(data_gpu, result_gpu, COLUMNS);
        else
            (*reduction_kernel)<<<blocks, BLOCK_WIDTH>>>(data_gpu, result_gpu, COLUMNS);
        (*reduction_kernel)<<<1, NEXT_POW2(blocks)>>>(result_gpu, result_gpu, blocks);

        if (cudaMemcpy(&avg, result_gpu, sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cout << "cudaMemcpy error!" << std::endl;
            exit(1);
        }
        avg /= COLUMNS;
        modify_array<<<blocks, BLOCK_WIDTH>>>(data_gpu, COLUMNS, -avg, op_add);

        /* Calculating product */

        if (cudaMemcpy(help_gpu, data_gpu, COLUMNS * sizeof(float),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) {
            std::cout << "cudaMemcpy error!" << std::endl;
            exit(1);
        }
        modify_array<<<blocks, BLOCK_WIDTH>>>(help_gpu, COLUMNS, 0, op_pow2);

        if (reduction_kernel == &reduction_kernel_v4)
            (*reduction_kernel)<<<blocks / 2, BLOCK_WIDTH>>>(help_gpu, result_gpu, COLUMNS);
        else
            (*reduction_kernel)<<<blocks, BLOCK_WIDTH>>>(help_gpu, result_gpu, COLUMNS);
        (*reduction_kernel)<<<1, NEXT_POW2(blocks)>>>(result_gpu, result_gpu, blocks);

        if (cudaMemcpy(&len, result_gpu, sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cout << "cudaMemcpy error!" << std::endl;
            exit(1);
        }
        len = std::sqrt(len);
        modify_array<<<blocks, BLOCK_WIDTH>>>(data_gpu, COLUMNS, 1 / len, op_mul);

        cudaEventRecord(event2, nullptr);
        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);

        cudaEventElapsedTime(&dt_ms_part, event1, event2);
        dt_ms += dt_ms_part;

        if (cudaMemcpy(result_cpu + COLUMNS * row, data_gpu, COLUMNS * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cout << "cudaMemcpy error!" << std::endl;
            exit(1);
        }
        cudaDeviceSynchronize();
    }
    std::cout << "Execution time (without copying memory between CPU and GPU) in ms: " << dt_ms << std::endl;
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



__global__ void naive_kernel_columns(float* data_gpu, int rows, int columns) {
    unsigned vector_num = threadIdx.x + blockIdx.x * blockDim.x;
    if (vector_num < columns) {
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

__global__ void modify_array(float* data_gpu, int columns, float arg, Operation operation) {
    unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < columns) {
        switch (operation) {
            case op_add:
                data_gpu[tid] += arg;
                break;
            case op_mul:
                data_gpu[tid] *= arg;
                break;
            case op_pow2:
                data_gpu[tid] *= data_gpu[tid];
                break;
            default:
                return;
        }
    }
}

__global__ void reduction_kernel_v1(const float* data_gpu, float* result_gpu, int columns) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x, tix = threadIdx.x;
    __shared__ float sums[SHARED_MEM_SIZE];

    if (tid < columns)
        sums[tix] = data_gpu[tid];
    __syncthreads();

    for (unsigned jump = 1; jump < blockDim.x; jump *= 2) {
        if (tix % (2 * jump) == 0 && tix + jump < blockDim.x && tid + jump < columns)
            sums[tix] += sums[tix + jump];
        __syncthreads();
    }

    if (tix == 0)
        result_gpu[blockIdx.x] = sums[0];
}

__global__ void reduction_kernel_v2(const float* data_gpu, float* result_gpu, int columns) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x, tix = threadIdx.x;
    __shared__ float sums[SHARED_MEM_SIZE];

    if (tid < columns)
        sums[tix] = data_gpu[tid];
    __syncthreads();

    for (unsigned jump = 1; jump < blockDim.x; jump *= 2) {
        unsigned index = 2 * jump * tix;
        if (index + jump < blockDim.x && tid - tix + index + jump < columns)
            sums[index] += sums[index + jump];
        __syncthreads();
    }

    if (tix == 0)
        result_gpu[blockIdx.x] = sums[0];
}

__global__ void reduction_kernel_v3(const float* data_gpu, float* result_gpu, int columns) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x, tix = threadIdx.x;
    __shared__ float sums[SHARED_MEM_SIZE];

    if (tid < columns)
        sums[tix] = data_gpu[tid];
    __syncthreads();

    for (unsigned jump = blockDim.x / 2; jump > 0; jump >>= 1) {
        if (tix < jump && tid + jump < columns)
            sums[tix] += sums[tix + jump];
        __syncthreads();
    }

    if (tix == 0)
        result_gpu[blockIdx.x] = sums[0];
}

__global__ void reduction_kernel_v4(const float* data_gpu, float* result_gpu, int columns) {
    unsigned tid = blockIdx.x * (blockDim.x * 2) + threadIdx.x, tix = threadIdx.x;
    __shared__ float sums[SHARED_MEM_SIZE];

    if (tid < columns)
        sums[tix] = data_gpu[tid];
    if (tid + blockDim.x < columns)
        sums[tix] += data_gpu[tid + blockDim.x];
    __syncthreads();

    for (unsigned jump = blockDim.x / 2; jump > 0; jump >>= 1) {
        if (tix < jump && tid + jump < columns)
            sums[tix] += sums[tix + jump];
        __syncthreads();
    }

    if (tix == 0)
        result_gpu[blockIdx.x] = sums[0];
}

__global__ void reduction_kernel_v5(const float* data_gpu, float* result_gpu, int columns) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x, tix = threadIdx.x;
    __shared__ float sums[SHARED_MEM_SIZE];

    if (tid < columns)
        sums[tix] = data_gpu[tid];
    __syncthreads();

    for (unsigned jump = blockDim.x / 2; jump > 32; jump >>= 1) {
        if (tix < jump && tid + jump < columns)
            sums[tix] += sums[tix + jump];
        __syncthreads();
    }
    if (tix < 32) {
        if (tix + 32 < columns) sums[tix] += sums[tix + 32];
        if (tix + 16 < columns) sums[tix] += sums[tix + 16];
        if (tix + 8 < columns) sums[tix] += sums[tix + 8];
        if (tix + 4 < columns) sums[tix] += sums[tix + 4];
        if (tix + 2 < columns) sums[tix] += sums[tix + 2];
        if (tix + 1 < columns) sums[tix] += sums[tix + 1];
    }

    if (tix == 0)
        result_gpu[blockIdx.x] = sums[0];
}

template <unsigned blockSize>
__global__ void reduction_kernel_v6(const float* data_gpu, float* result_gpu, int columns) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x, tix = threadIdx.x;
    __shared__ float sums[SHARED_MEM_SIZE];

    if (tid < columns)
        sums[tix] = data_gpu[tid];
    __syncthreads();

    for (unsigned jump = blockDim.x / 2; jump > 32; jump >>= 1) {
        if (tix < jump && tid + jump < columns)
            sums[tix] += sums[tix + jump];
        __syncthreads();
    }

    if (blockSize >= 512) {
        if (tix < 256)
            sums[tix] += sums[tix + 256];
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tix < 128)
            sums[tix] += sums[tix + 128];
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tix < 64)
            sums[tix] += sums[tix + 64];
        __syncthreads();
    }

    if (tix < 32) {
        if (tix + 32 < columns) sums[tix] += sums[tix + 32];
        if (tix + 16 < columns) sums[tix] += sums[tix + 16];
        if (tix + 8 < columns) sums[tix] += sums[tix + 8];
        if (tix + 4 < columns) sums[tix] += sums[tix + 4];
        if (tix + 2 < columns) sums[tix] += sums[tix + 2];
        if (tix + 1 < columns) sums[tix] += sums[tix + 1];
    }

    if (tix == 0)
        result_gpu[blockIdx.x] = sums[0];
}

void test_normalization(const float* cpu_result, Orientation orientation) {
    float eps = 1e-4, sum, len;

    if (orientation == columns) {
        for (int column = 0; column < COLUMNS; ++column) {
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
    else if (orientation == rows) {
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
    }

}