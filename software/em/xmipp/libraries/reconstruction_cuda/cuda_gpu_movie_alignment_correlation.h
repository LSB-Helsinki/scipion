
#include "utils.h"
#include <vector>
#include "reconstruction_cuda/cuda_xmipp_utils.h"
#include "data/multidim_array.h"


float* loadToGPU(float* data, size_t items);

void release(float* data);

size_t getFreeMem(int device);

void getBestSize(int imgsToProcess, int origXSize, int origYSize, int &batchSize, int &xSize, int &ySize,
		int extraMem = 0);
size_t getFreeMem(int device);

std::complex<float>* performFFTAndScale(float* h_imgs, int noOfImgs,
		int inSizeX, int inSizeY, int inBatch,
		int outSizeX, int outSizeY,  float* d_filter);

void processInput(GpuMultidimArrayAtGpu<float>& imagesGPU,
		GpuMultidimArrayAtGpu<std::complex<float> >& resultingFFT,
		mycufftHandle& handle,
		int inSizeX, int inSizeY, int inBatch,
		int outSizeX, int outSizeY, float* d_filter,
		std::complex<float>* result);

void computeCorrelations(double maxShift, size_t noOfImgs, std::complex<float>* h_FFTs,
		int fftSizeX, int imgSizeX, int imgSizeY, size_t maxFFTsInBuffer,
		int fftBatchSize, float*& result);
