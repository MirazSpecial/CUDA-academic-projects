#ifndef ____Mandelbrot__
#define ____Mandelbrot__

#define real double

#include <cstdio>
#include <cstring>
#include <ctime>
#include <iostream>
#include <vector>
#include <algorithm>

#endif /* defined(____Mandelbrot__) */

const int RUNS = 20;

int computeMandelbrot(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER, int *Mandel);
void makePicture(const int *Mandel, int width, int height, int MAX);
void makePictureInt(int *Mandel,int width, int height, int MAX);
void makePicturePNG(int *Mandel,int width, int height, int MAX);
int Compare(int *mand1, int *mand2, int LEN);

__global__ void cudaMandelbrot(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER,int *Mandel);

__global__ void cudaMandelbrot2D(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER,int *Mandel);

__global__ void cudaMandelbrot_steps(real X0, real Y0, real X1, real Y1, int POZ, int PION, int ITER,int *Mandel);

