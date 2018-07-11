#include "utils.h"
#include <vector>
#include "reconstruction_cuda/cuda_xmipp_utils.h"
#include "data/multidim_array.h"
#include <type_traits>

template<typename T>
T* loadToGPU(T* data, size_t items);

template<typename T>
void release(T* data);

size_t getFreeMem(int device);

void getBestSize(int imgsToProcess, int origXSize, int origYSize, int &batchSize, int &xSize, int &ySize,
		int extraMem = 0);

size_t getFreeMem(int device);

template<typename T>
std::complex<T>* performFFTAndScale(T* h_imgs, int noOfImgs,
		int inSizeX, int inSizeY, int inBatch,
		int outSizeX, int outSizeY,  T* d_filter);

template<typename T>
void processInput(GpuMultidimArrayAtGpu<T>& imagesGPU,
		GpuMultidimArrayAtGpu<std::complex<T> >& resultingFFT,
		mycufftHandle& handle,
		int inSizeX, int inSizeY, int inBatch,
		int outSizeX, int outSizeY, T* d_filter,
		std::complex<T>* result);

template<typename T>
void computeCorrelations(T maxShift, size_t noOfImgs, std::complex<T>* h_FFTs,
		int fftSizeX, int imgSizeX, int imgSizeY, size_t maxFFTsInBuffer,
		int fftBatchSize, T*& result);
