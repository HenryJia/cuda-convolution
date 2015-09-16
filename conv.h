#ifndef CONV_H
#define CONV_H

#include <cuda.h>

#include "cudadefs.h"
#include <iostream>

using namespace std;

/*
 * An 1D convolution kernel.
 */

void convValidGPU(const float* in, const float* filter, float* out, int M, int filterM);
__global__ void kernelConvValid(const float* in, const float* __restrict__ filter, float* out, int M, int filterM, int outM, int oTile);

/*
 * A 2D convolution kernel.
 */

void conv2ValidGPU(const float* in, const float* filter, float* out, int M, int N, int filterM, int filterN);
__global__ void kernelConv2Valid(const float* in, const float* __restrict__ filter, float* out, int M, int N, int filterM, int filterN,
                                 int outM, int outN, int oTileM, int oTileN);

#endif // CONV_H