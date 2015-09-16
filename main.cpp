#include "utils.h"
#include "conv.h"
#include <vector>

int main(int argc, char **argv)
{
	vector<vector<float>> test1Vec = readCSV("../Test1.csv", false);
	vector<vector<float>> test2Vec = readCSV("../Test2.csv", false);
	vector<vector<float>> testFilter1Vec = readCSV("../TestFilter1.csv", false);
	vector<vector<float>> testFilter2Vec = readCSV("../TestFilter2.csv", false);
	float* test1Host = vector2dToMat(test1Vec);
	float* test2Host = vector2dToMat(test2Vec);
	float* testFilter1Host = vector2dToMat(testFilter1Vec);
	float* testFilter2Host = vector2dToMat(testFilter2Vec);

	int m1 = test1Vec.size();
	int m2 = test2Vec.size();
	int n2 = test2Vec[0].size();

	cudaEvent_t custart, custop;
	cudaEventCreate(&custart);
	cudaEventCreate(&custop);

	/*
	 * 1D Valid Convolution
	 */

	float* result1;
	cudaMalloc((void**)&result1, (m1 - 5 + 1) * sizeof(float));
	float* test1 = copyToGPU(test1Host, m1, 1);
	float* testFilter1 = copyToGPU(testFilter1Host, 5, 1);

	auto start = chrono::steady_clock::now();
	convValidGPU(test1, testFilter1, result1, m1, 5);
	auto end = chrono::steady_clock::now();
	auto elapsed = end - start;
	cout << "1D Convolution time: " << chrono::duration <float, nano> (elapsed).count() << " ns" << endl;

	float* result1Host = copyFromGPU(result1, (m1 - 5 + 1), 1);

	for(int i = 0; i < 5; i++)
		cout << result1Host[i] << endl;

	vector<vector<float>> result1Vec = matToVector2d(result1Host, (m1 - 5 + 1), 1);
	writeCSV(result1Vec, "../result1.csv");

	/*
	 * 2D Valid Convolution
	 */

	float* result2;
	cudaMalloc((void**)&result2, (m2 - 5 + 1) * (n2 - 5 + 1) * sizeof(float));
	float* test2 = copyToGPU(test2Host, m2, n2);
	float* testFilter2 = copyToGPU(testFilter2Host, 5, 5);

	auto start2 = chrono::steady_clock::now();
	conv2ValidGPU(test2, testFilter2, result2, m2, n2, 5, 5);
	auto end2 = chrono::steady_clock::now();
	auto elapsed2 = end2 - start2;
	cout << "2D Convolution time (warm up): " << chrono::duration <float, nano> (elapsed2).count() << " ns" << endl;

	cudaEventRecord(custart);
	conv2ValidGPU(test2, testFilter2, result2, m2, n2, 5, 5);
	cudaEventRecord(custop);

	float* result2Host = copyFromGPU(result2, (m2 - 5 + 1), (n2 - 5 + 1));

	cudaEventSynchronize(custop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, custart, custop);
	cout << "2D Convolution time: " << milliseconds << " ms" << endl;

	for(int i = 0; i < 5; i++)
	{
		for(int j = 0; j < 5; j++)
			cout << result2Host[IDX2C(i, j, (m2 - 5 + 1))] << '\t';
		cout << endl;
	}

	vector<vector<float>> result2Vec = matToVector2d(result2Host, (m2 - 5 + 1), (n2 - 5 + 1));
	writeCSV(result2Vec, "../result2.csv");

	free(test1Host);
	free(testFilter1Host);
	free(result1Host);

	cudaFree(result1);
	cudaFree(test1);
	cudaFree(testFilter1);

	free(test2Host);
	free(testFilter2Host);
	free(result2Host);

	cudaFree(result2);
	cudaFree(test2);
	cudaFree(testFilter2);

	getchar();
}