
#include "utils.h"
#include <vector>
#include "reconstruction_cuda/cuda_xmipp_utils.h"
#include "data/multidim_array.h"
#include "data/transformations.h"

template<typename T>
void applyGeometry_2D_S3_wrap(int SplineDegree,
                   MultidimArray<T>& __restrict__ V2,
                   const MultidimArray<T>& __restrict__ V1,
                   const Matrix2D<T> &A, bool inv,
                   bool wrap, T outside = 0, MultidimArray<T> *BcoeffsPtr=NULL);


float* loadToGPU(float* data, size_t items);

void release(float* data);

size_t getFreeMem(int device);

void getBestSize(int imgsToProcess, int origXSize, int origYSize, int &batchSize, int &xSize, int &ySize,
		int extraMem = 0);
size_t getFreeMem(int device);

void kernel1(float* imgs, size_t oldX, size_t oldY, int noOfImages, size_t newX, size_t newY,
		float* filter,
		std::complex<float>*& result);
//		float*& result);

void kernel3(float maxShift, size_t noOfImgs,
		const std::complex<float>* imgs, size_t fftXdim, size_t fftYdim,
		float*& result, std::complex<float>*& result2);

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
