#define NO_FREETYPE

#include "Mandelbrot_cuda.h"
#include <cmath>
#include <chrono>
#include <iostream>
#include <pngwriter.h>
using namespace std;



int main(int argc, char **argv) {
    if (argc != 12 ) {
        printf("Wywołanie %s LD_Re, LD_Im, PG_Re, PG_Im, Poziom, Pion, Iteracje, Compare  Picture \n",argv[0]);
        printf("Flagi:  Compare: 0/1 - porównaj rezultat z CPU lub nie\n");
        printf("Flagi:  Picture: 0/1 - generuj obrazki lub nie \n");
        exit(1);
    }

    real X0=atof(argv[1]);
    real Y0=atof(argv[2]);

    real X1=atof(argv[3]);
    real Y1=atof(argv[4]);

    int POZ=atoi(argv[5]);
    int PION=atoi(argv[6]);

    int ITER=atoi(argv[7]);

    int CMP=atoi(argv[8]);
    int PIC=atoi(argv[9]);

    int NUM_THREADS_X = atoi(argv[10]);
    int NUM_THREADS_Y = atoi(argv[11]);

    cudaError_t status;

    int* Mandel_host;
    int* Mandel_device;
    int *Iters;
    Iters= (int*) malloc(sizeof(int)*POZ*PION);


    status = cudaMalloc ((void**)&Mandel_device , POZ*PION * sizeof (int ));
    if (status != cudaSuccess){
        cout << cudaGetErrorString(status) << endl;
    };

    status = cudaMallocHost( (void**) & Mandel_host , POZ*PION * sizeof ( int ));
    if (status != cudaSuccess){
        cout << cudaGetErrorString(status) << endl;
    };

    time_t start, end;

    printf("Corners - (%lf , %lf) and ",X0,Y0);
    printf("(%lf , %lf)\n",X1,Y1);

    // settings for cudaMandelbrot
//    dim3 threadsPerBlock(NUM_THREADS,1,1);
//    dim3 numBlocks(PION*POZ/threadsPerBlock.x+1,1,1);

    // settings for cudaMandelbrot2D and cudaMandelbrot_steps
    int block_width=NUM_THREADS_X;
    int block_height=NUM_THREADS_Y;
    dim3 threadsPerBlock(block_width,block_height,1);
    dim3 numBlocks(POZ/block_width+1,PION/block_height+1,1);

    start=clock();
    auto start2 = chrono::steady_clock::now();

    vector<double> execution_time_results, entire_time_results;

    for (int i = 0; i < RUNS; ++i) {
        cout << "Run number " << i << " ";
        auto run_start = chrono::steady_clock::now();

//        cudaMandelbrot<<<numBlocks,threadsPerBlock,0>>>(X0,Y0,X1,Y1,POZ,PION,ITER,Mandel_device);
//        cudaMandelbrot2D<<<numBlocks,threadsPerBlock,0>>>(X0,Y0,X1,Y1,POZ,PION,ITER,Mandel_device);
        cudaMandelbrot_steps<<<numBlocks,threadsPerBlock,0>>>(X0,Y0,X1,Y1,POZ,PION,ITER,Mandel_device);

        auto run_kernel_end = chrono::steady_clock::now();

        status = cudaMemcpy(Mandel_host , Mandel_device , POZ*PION * sizeof ( int ), cudaMemcpyDeviceToHost);
        if (status != cudaSuccess){
            cout << cudaGetErrorString(status) << endl;
            exit(1);
        };

        auto run_end = chrono::steady_clock::now();

        double execution_time = chrono::duration <double, milli> (run_kernel_end - run_start).count(),
               entire_time = chrono::duration <double, milli> (run_end - run_start).count();

        cout << "execution_time " << execution_time << " entire_time " << entire_time << endl;

        execution_time_results.push_back(execution_time);
        entire_time_results.push_back(entire_time);
    }

    sort(execution_time_results.begin(), execution_time_results.end());
    sort(entire_time_results.begin(), entire_time_results.end());

    cout << "Typical execution time (ms): " << execution_time_results[RUNS / 2] << endl;
    cout << "Minimal execution time (ms): " << execution_time_results[0] << endl;

    // We want to find out CPU time on smaller input
    int CPU_POZ = POZ / 10, CPU_PION = PION / 10;

    auto cpu_run_start = chrono::steady_clock::now();
    computeMandelbrot(X0,Y0,X1,Y1,CPU_POZ,CPU_PION,ITER,Iters);
    auto cpu_run_end = chrono::steady_clock::now();

    double cpu_time = chrono::duration <double, milli> (cpu_run_end - cpu_run_start).count();

    cout << "CPU time (ms): " << cpu_time * 100 << endl;
    cout << "Acceleration over CPU time (%): " << entire_time_results[RUNS / 2] / cpu_time << endl;


//    auto stop = chrono::steady_clock::now();
//    end=clock();
//    auto diff = stop - start2;
//
//    cout << "Kernel " << chrono::duration <real, milli> (diff).count() << " ms" << endl;
//    cout << "Kernel " <<chrono::duration <real, micro> (diff).count() << " us" << endl;
//    cout << "Kernel " <<chrono::duration <real, nano> (diff).count() << " ns" << endl;
//
//    //printf("\nComputations took %f  clocks\n",(float) (end-start));
//    printf("Start %f End %f clock ticks\n",(float) start, (float) end);
//    printf("Computations and transfer %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);

//    if (PIC==1) {
//        start=clock();
//        makePicturePNG(Mandel_host, POZ, PION, ITER);
//        makePicture(Mandel_host, POZ, PION, ITER);
//        end=clock();
//        printf("Saving took %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);
//    }

    status = cudaFree(Mandel_device);
    if (status != cudaSuccess){
        cout << cudaGetErrorString(status) << endl;
    };

    if (CMP==1) {
        printf("Computing reference\n");
        start=clock();
        int SUM = computeMandelbrot(X0,Y0,X1,Y1,POZ,PION,ITER,Iters);
        end=clock();
        printf("Time %lf s\n\n",1.0*(end-start)/CLOCKS_PER_SEC);
        int ident = Compare(Mandel_host,Iters,PION*POZ);
        printf("%d out of %d pixels are identical (%8.2lf) %% \n",ident,PION*POZ,100.0*ident/PION/POZ);
    }

    status = cudaFreeHost (Mandel_host);
    if (status != cudaSuccess){
        cout << cudaGetErrorString(status) << endl;
    };


}


__global__ void cudaMandelbrot(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER,int *Mandel) {
    real dX = (X1 - X0) / (POZ - 1);
    real dY = (Y1 - Y0) / (PION - 1);
    int i;
    unsigned pion, poz;
    real x, y, Zx, Zy, tZx;
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < PION * POZ) {
        pion  = idx / POZ;
        poz = idx - pion * POZ;
        x = X0 + poz * dX;
        y = Y0 + pion * dY;
        Zx = x;
        Zy = y;
        i = 0;
        //printf("%d %d %lf %lf\n",pion,poz,y,x);
        while ( (i<ITER) && ((Zx*Zx+Zy*Zy)<4) )
        {
            tZx = Zx*Zx-Zy*Zy+x;
            Zy = 2*Zx*Zy+y;
            Zx = tZx;

            i++;
        }
        Mandel[pion*POZ+poz] = i;
    }
}

__global__ void cudaMandelbrot2D(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER,int *Mandel){
    real    dX=(X1-X0)/(POZ-1);
    real    dY=(Y1-Y0)/(PION-1);
    int i;
    real x,y,Zx,Zy,tZx;

    unsigned poz = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned pion = (blockIdx.y * blockDim.y) + threadIdx.y;

    if ( (poz<POZ) &&  (pion<PION) ) {

        x = X0 + poz * dX;
        y = Y0 + pion * dY;
        Zx = x;
        Zy = y;
        i = 0;
        //printf("%d %d %lf %lf\n",pion,poz,y,x);
        while ( (i<ITER) && ((Zx*Zx+Zy*Zy)<4) )
        {
            tZx = Zx*Zx-Zy*Zy+x;
            Zy = 2*Zx*Zy+y;
            Zx = tZx;

            i++;
        }
        Mandel[pion*POZ+poz] = i;

    }
}


__global__ void cudaMandelbrot_steps(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER,int *Mandel){
    real    dX=(X1-X0)/(POZ-1);
    real    dY=(Y1-Y0)/(PION-1);
    int i;
    real x,y,Zx,Zy,tZx;
    unsigned poz = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned pion = (blockIdx.y * blockDim.y) + threadIdx.y;

    const int STEP_CONST = 8;

    if ( (poz<POZ) &&  (pion<PION) ) {

        /*
         * I understand that this version means finding Mandelbrot in lower resolution by
         * checking continuing condition not in every iteration
         */

        x = X0 + poz * dX;
        y = Y0 + pion * dY;
        Zx = x;
        Zy = y;
        i = 0;
        //printf("%d %d %lf %lf\n",pion,poz,y,x);
        while ( (i<ITER) && ((Zx*Zx+Zy*Zy)<4) )
        {
            for (int j = 0; j < STEP_CONST; j++) {
                tZx = Zx*Zx-Zy*Zy+x;
                Zy = 2*Zx*Zy+y;
                Zx = tZx;

                i++;
            }
        }

        Mandel[pion*POZ+poz] = (i < ITER) ? i : (ITER - 1);

    }

}

//void makePicture(const int *Mandel,int width, int height, int MAX){
//
//    int red_value, green_value, blue_value;
//    float scale = 256.0f / (float)MAX;
//    int MyPalette[41][3]={
//            {255,255,255}, {255,255,255}, {255,255,255}, {255,255,255}, //3
//            {255,255,255}, {255,180,255}, {255,180,255}, {255,180,255}, //7
//            {248,128,240}, {248,128,240}, {240,64,224},  {240,64,224}, //11
//            {232,32,208},  {224,16,192},  {216,8,176},   {208,4,160}, //15
//            {200,2,144},   {192,1,128},   {184,0,112},   {176,0,96}, //19
//            {168,0,80},    {160,0,64},    {152,0,48},    {144,0,32}, //23
//            {136,0,16},    {128,0,0},     {120,16,0},    {112,32,0}, //27
//            {104,48,0},    {96,64,0},     {88,80,0},     {80,96,0}, //31
//            {72,112,0},    {64,128,0},    {56,144,0},    {48,160,0}, //35
//            {40,176,0},    {32,192,0},    {16,224,0},    {8,240,0}, //40
//            {0,0,0}
//    };
//
//    FILE *f = fopen("Mandel.ppm", "wb");
//    fprintf(f, "P6\n%i %i 255\n", width, height);
//    for (int j=height-1; j>=0; j--) {
//        for (int i=0; i<width; i++) {
//            int indx= (int) floor(5.0*scale*log2f(1.0f*(float)Mandel[j*width+i]+1));
//            red_value=MyPalette[indx][0];
//            green_value=MyPalette[indx][2];
//            blue_value=MyPalette[indx][1];
//
//            fputc(red_value, f);   // 0 .. 255
//            fputc(green_value, f); // 0 .. 255
//            fputc(blue_value, f);  // 0 .. 255
//        }
//    }
//    fclose(f);
//
//}
//
//void makePictureInt(int *Mandel,int width, int height, int MAX){
//
//    real scale = 256.0/MAX;
//
//    int red_value, green_value, blue_value;
//
//    int MyPalette[33][3]={
//            {255,255,255}, {255,0,255}, {248,0,240}, {240,0,224},
//            {232,0,208}, {224,0,192}, {216,0,176}, {208,0,160},
//            {200,0,144}, {192,0,128}, {184,0,112}, {176,0,96},
//            {168,0,80},  {160,0,64},  {152,0,48},  {144,0,32},
//            {136,0,16},  {128,0,0},   {120,16,0},  {112,32,0},
//            {104,48,0},  {96,64,0},   {88,80,0},   {80,96,0},
//            {72,112,0},  {64,128,0},  {56,144,0},  {48,160,0},
//            {40,176,0},  {32,192,0},  {16,224,0},  {8,240,0}, {0,0,0}
//    };
//
//    FILE *f = fopen("Mandel.ppm", "wb");
//
//    fprintf(f, "P3\n%i %i 255\n", width, height);
//    printf("MAX = %d, scale %lf\n",MAX,scale);
//    for (int j=(height-1); j>=0; j--) {
//        for (int i=0; i<width; i++)
//        {
//            //if ( ((i%4)==0) && ((j%4)==0) ) printf("%d ",Mandel[j*width+i]);
//            //red_value = (int) round(scale*(Mandel[j*width+i])/16);
//            //green_value = (int) round(scale*(Mandel[j*width+i])/16);
//            //blue_value = (int) round(scale*(Mandel[j*width+i])/16);
//            int indx= (int) round(4.0*scale*log2f(1.0f*Mandel[j*width+i]+1));
//            red_value=MyPalette[indx][0];
//            green_value=MyPalette[indx][2];
//            blue_value=MyPalette[indx][1];
//
//            fprintf(f,"%d ",red_value);   // 0 .. 255
//            fprintf(f,"%d ",green_value); // 0 .. 255
//            fprintf(f,"%d ",blue_value);  // 0 .. 255
//        }
//        fprintf(f,"\n");
//        //if ( (j%4)==0)  printf("\n");
//
//    }
//    fclose(f);
//
//}
//
//void makePicturePNG(int *Mandel,int width, int height, int MAX){
//    real red_value, green_value, blue_value;
//    float scale = 256.0/MAX;
//    real MyPalette[41][3]={
//            {1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0},// 0, 1, 2, 3,
//            {1.0,1.0,1.0},{1.0,0.7,1.0},{1.0,0.7,1.0},{1.0,0.7,1.0},// 4, 5, 6, 7,
//            {0.97,0.5,0.94},{0.97,0.5,0.94},{0.94,0.25,0.88},{0.94,0.25,0.88},//8, 9, 10, 11,
//            {0.91,0.12,0.81},{0.88,0.06,0.75},{0.85,0.03,0.69},{0.82,0.015,0.63},//12, 13, 14, 15,
//            {0.78,0.008,0.56},{0.75,0.004,0.50},{0.72,0.0,0.44},{0.69,0.0,0.37},//16, 17, 18, 19,
//            {0.66,0.0,0.31},{0.63,0.0,0.25},{0.60,0.0,0.19},{0.56,0.0,0.13},//20, 21, 22, 23,
//            {0.53,0.0,0.06},{0.5,0.0,0.0},{0.47,0.06,0.0},{0.44,0.12,0},//24, 25, 26, 27,
//            {0.41,0.18,0.0},{0.38,0.25,0.0},{0.35,0.31,0.0},{0.31,0.38,0.0},//28, 29, 30, 31,
//            {0.28,0.44,0.0},{0.25,0.50,0.0},{0.22,0.56,0.0},{0.19,0.63,0.0},//32, 33, 34, 35,
//            {0.16,0.69,0.0},{0.13,0.75,0.0},{0.06,0.88,0.0},{0.03,0.94,0.0},//36, 37, 38, 39,
//            {0.0,0.0,0.0}//40
//    };
//
//    //FILE *f = fopen("Mandel.txt","w");
//    pngwriter png(width,height,1.0,"Mandelbrot.png");
//    for (int j=height-1; j>=0; j--) {
//        for (int i=0; i<width; i++) {
//            // compute index to the palette
//            int indx= (int) floor(5.0*scale*log2f(1.0f*Mandel[j*width+i]+1));
//            //fprintf(f,"%3d, ",Mandel[j*width+i]);
//            red_value=MyPalette[indx][0];
//            green_value=MyPalette[indx][2];
//            blue_value=MyPalette[indx][1];
//            png.plot(i,j, red_value, green_value, blue_value);
//        }
//        //fprintf(f,"\n");
//    }
//    // fclose(f);
//    png.close();
//}

int computeMandelbrot(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER,int *Mandel ){
    real    dX=(X1-X0)/(POZ-1);
    real    dY=(Y1-Y0)/(PION-1);
    real x,y,Zx,Zy,tZx;
    int SUM=0;
    int i;

    printf("Computations for rectangle { (%lf %lf), (%lf %lf) }\n",X0,Y0,X1,Y1);

    for (int pion=0; pion<PION; pion++) {
        for (int poz=0;poz<POZ; poz++) {
            x=X0+poz*dX;
            y=Y0+pion*dY;
            Zx=x;
            Zy=y;
            i=0;
            //printf("%d %d %lf %lf\n",pion,poz,y,x);
            while ( (i<ITER) && ((Zx*Zx+Zy*Zy)<4) )
            {
                tZx = Zx*Zx-Zy*Zy+x;
                Zy = 2*Zx*Zy+y;
                Zx = tZx;

                i++;
            }
            Mandel[pion*POZ+poz] = i;
            SUM+=i;
        }
        //printf("Line %d, sum=%d\n",pion,SUM);
    }
    return SUM;
}


int computeMandelbrot2(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER,int *Mandel ){
    real    dX=(X1-X0)/(POZ-1);
    real    dY=(Y1-Y0)/(PION-1);
    real x,y,Zx,Zy,tZx;
    int SUM=0;
    int i;
    int SIZE=POZ*PION;
    int pion, poz;

    for (int indx=0;indx<SIZE;indx++) {
        pion=indx / POZ;
        poz=indx % POZ;
        //printf("%d %d %d \n",indx,pion,poz);
        x=X0+poz*dX;
        y=Y0+pion*dY;
        //printf("%lf %lf\n",x,y);
        Zx=x;
        Zy=y;
        i=0;

        while ( (i<ITER) && ((Zx*Zx+Zy*Zy)<4) ){
            tZx = Zx*Zx-Zy*Zy+x;
            Zy = 2*Zx*Zy+y;
            Zx = tZx;

            i++;
        }
        Mandel[indx] = i;
        SUM+=i;
    }
    return SUM;
}

int Compare(int *mand1, int *mand2, int LEN){
    int sum =0;
    int in1, in2;

    //for (int i=0;i<LEN;i++) sum+= (int) (mand1[i]==mand2[i]);
    for (int i=0;i<LEN;i++) {
        in1 = (mand1[i]>255) ? 1:0;
        in2 = (mand2[i]>255) ? 1:0;
        sum += (int) in1==in2;
    }
    return sum;
}
