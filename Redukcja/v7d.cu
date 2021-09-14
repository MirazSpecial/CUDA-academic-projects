#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define blockSize 512
#define real double


template <int bSize> __global__ void redukcja (int N, real* v, real* out)
{
 size_t s = threadIdx.x + blockIdx.x * blockDim.x*2;
 int sID = threadIdx.x;
 size_t i;

 __shared__ real pom[blockSize];
 
 pom[sID] = 0;
 if (s<N/2)
	 pom[sID] = v[s] + v[s + blockDim.x];
 __syncthreads();

 if (bSize >=512) {
 	if (sID<256) pom[sID] += pom[sID + 256]; __syncthreads();
 }
 if (bSize >=256) {
	if (sID<128) pom[sID] += pom[sID + 128]; __syncthreads();
 }
 if (bSize >= 128){
 	if (sID<64) pom[sID] += pom[sID + 64]; __syncthreads();
 }	
 if (sID < 32){
	if (bSize >= 64) pom[sID] += pom[sID + 32];
__syncthreads();	
	if (bSize >= 32) pom[sID] += pom[sID + 16];
__syncthreads();	
	if (bSize >= 16) pom[sID] += pom[sID +  8];
__syncthreads();	
	if (bSize >= 8) pom[sID] += pom[sID +  4];
__syncthreads();	
	if (bSize >= 4) pom[sID] += pom[sID +  2];
__syncthreads();	
	if (bSize >= 2) pom[sID] += pom[sID +  1];
__syncthreads();	
 }	
 if (sID==0) out[blockIdx.x] = pom[0];
}

__global__ void redukcja2 (int N, real* v, real* out)
{
 size_t s = threadIdx.x + blockIdx.x * blockDim.x;
 int sID = threadIdx.x;
 size_t i;

 __shared__ real pom[blockSize];

 pom[sID] = 0;
 if (s<N)
         pom[sID] = v[s];
 __syncthreads();

 for (i=blockDim.x/2; i>0; i>>=1){
        if (sID<i){
                pom[sID] += pom[sID + i];
        }
        __syncthreads();
 }
 if (sID==0) out[blockIdx.x] = pom[0];
}

__global__ void wypelnij (int N, real* v)
{
 size_t s = threadIdx.x + blockIdx.x * blockDim.x;

 if (s<N) {
 	v[s] = sin(s * 2. * M_PI / 10.);
 }	
}

int main ()
{
 size_t N = blockSize * blockSize * blockSize;
 int blocks = (N + blockSize-1) / blockSize;

 float dt_ms;

 cudaEvent_t event1, event2;
 cudaEventCreate(&event1);
 cudaEventCreate(&event2);

 real* v;
 cudaMalloc( (void**) &v, N * sizeof(real) );
 real* outV;
 cudaMalloc( (void**) &outV, blockSize * blockSize * sizeof(real) );
 real* outVV;
 cudaMalloc( (void**) &outVV, blockSize * sizeof(real) );

 real out;
 int i;
 int M = 10;

 wypelnij <<<blocks, blockSize>>> (N, v);


 cudaEventRecord(event1, 0);
 	for (i=0; i<M; i++){
 		redukcja<blockSize><<<blocks/2, blockSize>>> (N, v, outV);
		redukcja<blockSize><<<blockSize/2, blockSize>>> (blockSize*blockSize, outV, outVV);
		redukcja2<<<1, blockSize>>> (blockSize, outVV, v);
	}
 cudaEventRecord(event2, 0);

 cudaEventSynchronize(event1);
 cudaEventSynchronize(event2);

 cudaEventElapsedTime(&dt_ms, event1, event2);
 cudaMemcpy (&out, v, 1 * sizeof(real), cudaMemcpyDeviceToHost);
  
 printf ("Czas redukcji: %f ms   wynik; %f\n", dt_ms * 1./M, out);

 return 0;
} 
