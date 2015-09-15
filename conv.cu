#include "conv.h"

/*
 * A 1D convolution kernel.
 */

void convValidGPU(const float* in, const float* filter, float* out, int M, int filterM)
{
	int outM = M - filterM + 1;
	kernelConvValid<<<((outM - 1) / O_TILE_DIM1 + 1), BLOCK_DIM1>>>(in, filter, out, M, filterM, outM);
}

__global__ void kernelConvValid(const float* in, const float* __restrict__ filter, float* out, int M, int filterM, int outM)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int row = bx * O_TILE_DIM1 + tx;
	float CValue = 0;

	__shared__ float ds_in[BLOCK_DIM1];

	if(row < M)
		ds_in[tx] = in[row];
	else
		ds_in[tx] = 0.0;

	__syncthreads();

	if(tx < O_TILE_DIM1 && row < outM)
	{
		for(int i = 0; i < filterM; i++)
			CValue += ds_in[tx + i] * filter[i];
		out[row] = CValue;
	}
}

/*
 * A 2D convolution kernel.
 */

void conv2ValidGPU(const float* in, const float* filter, float* out, int M, int N, int filterM, int filterN)
{
	int outM = M - filterM + 1;
	int outN = N - filterN + 1;
	cout << "Total threads" << ((outM - 1) / O_TILE_DIM2 + 1) * BLOCK_DIM2 << endl;
	dim3 gridDim(((outM - 1) / O_TILE_DIM2 + 1), ((outN - 1) / O_TILE_DIM2 + 1));
	dim3 blockDim(BLOCK_DIM2, BLOCK_DIM2);
	kernelConv2Valid<<<gridDim, blockDim>>>(in, filter, out, M, N, filterM, filterN, outM, outN);
}

__global__ void kernelConv2Valid(const float* in, const float* __restrict__ filter, float* out, int M, int N, int filterM, int filterN,
                                int outM, int outN)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by * O_TILE_DIM2 + ty;
	int col = bx * O_TILE_DIM2 + tx;
	float CValue = 0;

	__shared__ float ds_in[BLOCK_DIM2][BLOCK_DIM2];

	if(row < M && col < N)
		//ds_in[ty][tx] = in[row * M + col];
		ds_in[ty][tx] = in[IDX2C(row, col, M)];
	else
		ds_in[ty][tx] = 0.0;

	__syncthreads();

	if(tx < O_TILE_DIM2 && ty < O_TILE_DIM2 && row < outM && col < outN)
	{
		for(int i = 0; i < filterM; i++)
			for(int j = 0; j < filterN; j++)
				CValue += ds_in[ty + i][tx + j] * filter[IDX2C(i, j, filterM)];
		//out[col * M + row] = CValue;
		out[IDX2C(row, col, outM)] = CValue;
	}
}