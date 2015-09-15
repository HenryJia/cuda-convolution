#ifndef CONV_H
#define CONV_H

#include <cuda.h>

#include "cudadefs.h"
#include <iostream>

using namespace std;

#define BLOCK_DIM1 64
#define BLOCK_DIM2 16

#define FILTER_DIM 5

/*
 * An 1D convolution kernel.
 */

#define O_TILE_DIM1 (BLOCK_DIM1 - FILTER_DIM + 1)

void convValidGPU(const float* in, const float* filter, float* out, int M, int filterM);
__global__ void kernelConvValid(const float* in, const float* __restrict__ filter, float* out, int M, int filterM, int outM);

/*
 * A 2D convolution kernel.
 */

#define O_TILE_DIM2 (BLOCK_DIM2 - FILTER_DIM + 1)

void conv2ValidGPU(const float* in, const float* filter, float* out, int M, int N, int filterM, int filterN);
__global__ void kernelConv2Valid(const float* in, const float* __restrict__ filter, float* out, int M, int N, int filterM, int filterN,
                                 int outM, int outN);

#endif // CONV_H