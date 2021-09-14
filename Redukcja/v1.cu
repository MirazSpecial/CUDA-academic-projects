#include <cuda.h>
#include <stdio.h>
#include <math.h>

#define blockSize 512
#define real float


__global__ void redukcja (int N, real* v)
{
 size_t s = threadIdx.x + blockIdx.x * blockDim.x;
 size_t i;

 real p = 0;
 if (s==0){
//	*out = 0;
	for (i=0; i<N; i++)
		p += v[i];
 	v[0] = p;		
 }		
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
 size_t N = blockSize*blockSize*blockSize;
 int blocks = (N + blockSize-1) / blockSize;
 float dt_ms;

 cudaEvent_t event1, event2;
 cudaEventCreate(&event1);
 cudaEventCreate(&event2);

 real* v;
 cudaMalloc( (void**) &v, N * sizeof(real) );

 real out;
 int i;
 int M = 10;

 wypelnij <<<blocks, blockSize>>> (N, v);


 cudaEventRecord(event1, 0);
 	for (i=0; i<M; i++){
 		redukcja<<<blocks, blockSize>>> (N, v);
	
	}
 cudaEventRecord(event2, 0);

 cudaEventSynchronize(event1);
 cudaEventSynchronize(event2);

 cudaEventElapsedTime(&dt_ms, event1, event2);
 cudaMemcpy (&out, v, 1 * sizeof(real), cudaMemcpyDeviceToHost);
  
 printf ("Czas redukcji: %f ms   wynik; %f\n", dt_ms * 1./M, out);

 return 0;
} 
