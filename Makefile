OutDir = ./Debug/
CXX = g++
NVCC = nvcc
DEBUG = -g
NVCCFLAGS = -O3 -arch=sm_52 -Xptxas -dlcm=ca 
CXXFLAGS = $(DEBUG) -O -O0 -O1 -O2 -O3 -std=c++11 -Wall
IncludePath = -I. -I/opt/cuda/include/
LibPaths = -L. -L/opt/cuda/lib64/
Libs = -lcuda -lcudart -lcurand -lcublas 
LFLAGS = $(LibPaths) $(Libs)

CUBLAS-ML: Directories conv.cu.o main.cpp.o
	$(CXX) -o $(OutDir)cuda-convolution $(OutDir)main.cpp.o $(OutDir)conv.cu.o $(LFLAGS)

Directories:
	mkdir -p $(OutDir)

conv.cu.o: #conv.cu
	$(NVCC) -c $(NVCCFLAGS) conv.cu -o $(OutDir)conv.cu.o

main.cpp.o: main.cpp
	$(CXX) -c main.cpp $(CXXFLAGS) -o $(OutDir)main.cpp.o $(IncludePath)

##
## Clean
##
clean:
	$(RM) -r $(OutDir)