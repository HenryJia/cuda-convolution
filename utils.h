#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <chrono>

#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#include "cudadefs.h"

using namespace std;

vector<vector<float>> readCSV(string fileName, bool header)
{
	vector<vector<float>> result;
	ifstream in(fileName);
	string lineStr;
	string delimiter = ",";

	if(!in.is_open())
		cerr << "failed to open file\n";
	if(header == true)
		std::getline(in, lineStr);

	while(std::getline(in, lineStr))
	{
		vector<float> lineVec;
		size_t pos = 0;
		while((pos = lineStr.find(delimiter)) != std::string::npos)
		{
			lineVec.push_back(stold(lineStr.substr(0, pos)));
			lineStr.erase(0, pos + delimiter.length());
		}
		lineVec.push_back(stold(lineStr));
		result.push_back(lineVec);
	}

	return result;
}

void writeCSV(vector<vector<float>> data, string fileName)
{
	ofstream out(fileName);

	for(int i = 0; i < (data.size() - 1); i++)
	{
		for(int j = 0; j < (data[i].size() - 1); j++)
			out << data[i][j] << ',';
		out << data[i][data[i].size() - 1] << endl;
	}
	//out << data[data.size() - 1][data[data.size() - 1].size() - 1];
	for(int j = 0; j < (data[data.size() - 1].size() - 1); j++)
		out << data[data.size() - 1][j] << ',';
	out << data[data.size() - 1][data[data.size() - 1].size() - 1] << endl;
}

float* vector2dToMat(vector<vector<float>> data)
{
	float* result;
	int a = data.size();
	int b = data[0].size();

	result = (float*)malloc(a * b * sizeof(*result));
	if(!result)
	{
		cout << "Malloc Failed" << endl;
		return nullptr;
	}
	for(int i = 0; i < a; i++)
		for(int j = 0; j < b; j++)
			result[IDX2C(i, j, a)] = data[i][j];

	return result;
}

vector<vector<float>> matToVector2d(float* data, int a, int b)
{
	vector<vector<float>> result;

	for(int i = 0; i < a; i++)
	{
		vector<float> lineVec;
		for(int j = 0; j < b; j++)
			lineVec.push_back(data[IDX2C(i, j, a)]);
		result.push_back(lineVec);
	}

	return result;
}

float* copyToGPU(float* data, int a, int b)
{
	float* dataGPU;
	cudaError_t cudaStat = cudaMalloc((void**)&dataGPU, a * b * sizeof(*data));
	if (cudaStat != cudaSuccess)
	{
		cout << "cudaMalloc Failed" << endl;
		return nullptr;
	}

	cudaStat = cudaMemcpy(dataGPU, data, a * b * sizeof(*data), cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess)
	{
		cout << "cudaMemcpy Failed" << endl;
		return nullptr;
	}
	return dataGPU;
}

float* copyFromGPU(float* dataGPU, int a, int b)
{
	float* data = (float*)malloc(a * b * sizeof(*data));
	if(data == nullptr)
	{
		cout << "malloc Failed" << endl;
		return nullptr;
	}

	cudaError_t cudaStat = cudaMemcpy(data, dataGPU, a * b * sizeof(*data), cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess)
	{
		cout << "cudaMemcpy Failed" << endl;
		return nullptr;
	}

	return data;
}

#endif // UTILS_H