#include "conv.h"

/*
 * A 1D Valid convolution kernel.
 */

void convValidGPU(const float* in, const float* filter, float* out, int M, int filterM)
{
	int outM = M - filterM + 1;
	int oTile = O_TILE_DIM(BLOCK_DIM1, filterM);
	kernelConvValid<<<((outM - 1) / oTile + 1), BLOCK_DIM1>>>(in, filter, out, M, filterM, outM, oTile);
}

__global__ void kernelConvValid(const float* in, const float* __restrict__ filter, float* out, int M, int filterM, int outM, int oTile)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int row = bx * oTile + tx;
	float CValue = 0;

	__shared__ float ds_in[BLOCK_DIM1];

	if(row < M)
		ds_in[tx] = in[row];
	else
		ds_in[tx] = 0.0;

	__syncthreads();

	if(tx < oTile && row < outM)
	{
		for(int i = 0; i < filterM; i++)
			CValue += ds_in[tx + i] * filter[i];
		out[row] = CValue;
	}
}

/*
 * A 1D Full convolution kernel.
 */

void convFullGPU(const float* in, const float* filter, float* out, int M, int filterM)
{
	// We can do full convolution by simply padding valid convolution.
	// So we shift indices and add more threads to deal with extra elements
	int outM = M + filterM - 1;
	int oTile = O_TILE_DIM(BLOCK_DIM1, filterM);
	kernelConvFull<<<((outM - 1) / oTile + 1), BLOCK_DIM1>>>(in, filter, out, M, filterM, outM, oTile);
}

__global__ void kernelConvFull(const float* in, const float* __restrict__ filter, float* out, int M, int filterM, int outM, int oTile)
{
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int row = bx * oTile + tx;
	int row_i = row - filterM + 1;
	float CValue = 0;

	__shared__ float ds_in[BLOCK_DIM1];

	if(row_i < M && row_i >= 0)
		ds_in[tx] = in[row_i];
	else
		ds_in[tx] = 0.0;

	__syncthreads();

	if(tx < oTile && row < outM)
	{
		for(int i = 0; i < filterM; i++)
			CValue += ds_in[tx + i] * filter[i];
		out[row] = CValue;
	}
}

/*
 * A 2D Valid convolution kernel.
 */

void conv2ValidGPU(const float* in, const float* filter, float* out, int M, int N, int filterM, int filterN)
{
	int outM = M - filterM + 1;
	int outN = N - filterN + 1;
	int oTileM = O_TILE_DIM(BLOCK_DIM2, filterM);
	int oTileN = O_TILE_DIM(BLOCK_DIM2, filterN);
	dim3 gridDim(((outN - 1) / oTileN + 1), ((outM - 1) / oTileM + 1));
	dim3 blockDim(BLOCK_DIM2, BLOCK_DIM2);
	kernelConv2Valid<<<gridDim, blockDim>>>(in, filter, out, M, N, filterM, filterN, outM, outN, oTileM, oTileN);
}

__global__ void kernelConv2Valid(const float* in, const float* __restrict__ filter, float* out, int M, int N, int filterM, int filterN,
                                int outM, int outN, int oTileM, int oTileN)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by * oTileM + ty;
	int col = bx * oTileN + tx;
	float CValue = 0;

	__shared__ float ds_in[BLOCK_DIM2][BLOCK_DIM2];

	if(row < M && col < N)
		ds_in[ty][tx] = in[IDX2C(row, col, M)];
	else
		ds_in[ty][tx] = 0.0;

	__syncthreads();

	if(ty < oTileM && tx < oTileN && row < outM && col < outN)
	{
		for(int i = 0; i < filterM; i++)
			for(int j = 0; j < filterN; j++)
				CValue += ds_in[ty + i][tx + j] * filter[IDX2C(i, j, filterM)];
		out[IDX2C(row, col, outM)] = CValue;
	}
}