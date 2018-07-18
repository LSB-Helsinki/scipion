
#include <cuda_runtime_api.h>
#include "reconstruction_cuda/cuda_utils.h"
#include "advisor.h"
#include "cudaUtils.h"
#include "cuda_gpu_movie_alignment_correlation.h"
#include "reconstruction_cuda/cuda_basic_math.h"

#define BLOCK_DIM_X 32
#define TILE 8

template<typename T, typename U, bool applyFilter, bool normalize, bool center>
__global__
void scaleFFTKernel(const T* __restrict__ src, T* __restrict__ dest, int noOfImages, size_t oldX, size_t oldY, size_t newX, size_t newY,
		const U* __restrict__ filter, U normFactor) {
	// assign pixel to thread
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;

	if (idx >= newX || idy >= newY ) return;
	size_t fIndex = idy*newX + idx; // index within single image
	U filterCoef = filter[fIndex];
	U centerCoef = 1-2*((idx+idy)&1); // center FFT, input must have be even
	int yhalf = newY/2;

	size_t origY = (idy <= yhalf) ? idy : (oldY - (newY-idy)); // take top N/2+1 and bottom N/2 lines
	for (int n = 0; n < noOfImages; n++) {
		size_t iIndex = n*oldX*oldY + origY*oldX + idx; // index within consecutive images
		size_t oIndex = n*newX*newY + fIndex; // index within consecutive images
		dest[oIndex] = src[iIndex];
		if (applyFilter) {
			dest[oIndex] *= filterCoef;
		}
		if (normalize) {
			dest[oIndex] *= normFactor;
		}
		if (center) {
			dest[oIndex] *= centerCoef;
		}
	}
}

void scaleFFT2D(dim3& dimGrid, dim3& dimBlock, const std::complex<float>* d_inFFT, std::complex<float>* d_outFFT, int noOfFFT, size_t inFFTX, size_t inFFTY, size_t outFFTX, size_t outFFTY,
		float* d_filter, float normFactor, bool center) {
	if (NULL == d_filter) {
		if ((float)1 == normFactor) {
			if (center) {
				scaleFFTKernel<float2, float, false, false, true>
					<<<dimGrid, dimBlock>>>((float2*)d_inFFT, (float2*)d_outFFT, noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, 1.f);
			} else {
				scaleFFTKernel<float2, float, false, false, false>
					<<<dimGrid, dimBlock>>>((float2*)d_inFFT, (float2*)d_outFFT, noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, 1.f);
			}
		} else { // normalize
			if (center) {
				scaleFFTKernel<float2, float, false, true, true>
					<<<dimGrid, dimBlock>>>((float2*)d_inFFT, (float2*)d_outFFT, noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, normFactor);
			} else {
				scaleFFTKernel<float2, float, false, true, false>
					<<<dimGrid, dimBlock>>>((float2*)d_inFFT, (float2*)d_outFFT, noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, NULL, normFactor);
			}
		}
	} else { // apply filter (on output)
		if ((float)1 == normFactor) {
			if (center) {
				scaleFFTKernel<float2, float, true, false, true>
					<<<dimGrid, dimBlock>>>((float2*)d_inFFT, (float2*)d_outFFT, noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, 1.f);
			} else {
				scaleFFTKernel<float2, float, true, false, false>
					<<<dimGrid, dimBlock>>>((float2*)d_inFFT, (float2*)d_outFFT, noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, 1.f);
			}
		} else { // normalize
			if (center) {
				scaleFFTKernel<float2, float, true, true, true>
					<<<dimGrid, dimBlock>>>((float2*)d_inFFT, (float2*)d_outFFT, noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, normFactor);
			} else {
				scaleFFTKernel<float2, float, true, true, false>
					<<<dimGrid, dimBlock>>>((float2*)d_inFFT, (float2*)d_outFFT, noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY, d_filter, normFactor);
			}
		}
	}
	gpuErrchk(cudaPeekAtLastError());
}

template void applyFilterAndCrop<float>(const std::complex<float>* src, std::complex<float>* dest, int noOfImages, size_t oldX, size_t oldY, size_t newX, size_t newY,
		float* filter);

template<typename T>
void applyFilterAndCrop(const std::complex<T>* h_inFFT, std::complex<T>* h_outFFT, int noOfFFT, size_t inFFTX, size_t inFFTY, size_t outFFTX, size_t outFFTY,
		T* h_filter) {
	std::complex<T>* d_inFFT = loadToGPU(h_inFFT, inFFTX * inFFTY);
	std::complex<T>* d_outFFT = loadToGPU(h_outFFT, outFFTX * outFFTY);
	T* d_filter = h_filter ? loadToGPU(h_filter, outFFTX * outFFTY) : NULL;
	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
	dim3 dimGrid(ceil(outFFTX/(float)dimBlock.x), ceil(outFFTY/(float)dimBlock.y));
	scaleFFT2D(dimGrid, dimBlock, d_inFFT, d_outFFT, noOfFFT, inFFTX, inFFTY, outFFTX, outFFTY,
			d_filter, (T)1, false);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaMemcpy((void*)h_outFFT, (void*)d_outFFT, sizeof(std::complex<T>) * outFFTX * outFFTY, cudaMemcpyDeviceToHost));
	release(d_inFFT);
	release(d_outFFT);
	release(d_filter);
}

template std::complex<float>* performFFTAndScale<float>(float* inOutData, int noOfImgs,
		int inSizeX, int inSizeY, int inBatch,
		int outSizeX, int outSizeY,  float* d_filter);
template<typename T>
std::complex<T>* performFFTAndScale(T* inOutData, int noOfImgs,
		int inSizeX, int inSizeY, int inBatch,
		int outSizeX, int outSizeY,  T* d_filter) {
	mycufftHandle handle;
	int counter = 0;
	std::complex<T>* h_result = (std::complex<T>*)inOutData;
	GpuMultidimArrayAtGpu<T> imagesGPU(inSizeX, inSizeY, 1, inBatch);
	GpuMultidimArrayAtGpu<std::complex<T> > resultingFFT;

	while (counter < noOfImgs) {
		int imgToProcess = std::min(inBatch, noOfImgs - counter);
		T* h_imgLoad = inOutData + counter * inSizeX * inSizeY;
		size_t bytes = imgToProcess * inSizeX * inSizeY * sizeof(T);
		gpuErrchk(cudaMemcpy(imagesGPU.d_data, h_imgLoad, bytes, cudaMemcpyHostToDevice));
		std::complex<T>* h_imgStore = h_result + counter * outSizeX * outSizeY;
		processInput(imagesGPU, resultingFFT, handle, inSizeX, inSizeY, imgToProcess, outSizeX, outSizeY, d_filter, h_imgStore);
		counter += inBatch;
	}
	handle.clear();

	return h_result;
}

size_t getFreeMem(int device) {
	return cuFFTAdvisor::toMB(cuFFTAdvisor::getFreeMemory(device));
}

void getBestSize(int imgsToProcess, int origXSize, int origYSize, int &batchSize, int &xSize, int &ySize, int reserveMem,
		bool verbose) {
	int device = 0; // FIXME detect device or add to cmd param

	size_t freeMem = getFreeMem(device);
	std::vector<cuFFTAdvisor::BenchmarkResult const *> *results =
			cuFFTAdvisor::Advisor::find(10, device,
					origXSize, origYSize, 1, imgsToProcess,
					cuFFTAdvisor::Tristate::TRUE,
					cuFFTAdvisor:: Tristate::TRUE,
					cuFFTAdvisor::Tristate::TRUE,
					cuFFTAdvisor::Tristate::FALSE,
					cuFFTAdvisor::Tristate::TRUE, INT_MAX,
					freeMem - reserveMem, false, true);

	batchSize = results->at(0)->transform->N;
	xSize = results->at(0)->transform->X;
	ySize = results->at(0)->transform->Y;
	if (verbose) {
		results->at(0)->print(stdout);
		printf("\n");
	}
}

template float* loadToGPU<float>(const float* data, size_t items);
template<typename T>
T* loadToGPU(const T* data, size_t items) {
	T* d_data;
	size_t bytes = items * sizeof(T);
	gpuMalloc((void**) &d_data,bytes);
	gpuErrchk(cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice));
	return d_data;
}

template void release<float>(float* data);
template<typename T>
void release(T* data) {
	gpuErrchk(cudaFree(data));
}


template void processInput<float>(GpuMultidimArrayAtGpu<float>& imagesGPU,
		GpuMultidimArrayAtGpu<std::complex<float> >& resultingFFT,
		mycufftHandle& handle,
		int inSizeX, int inSizeY, int inBatch,
		int outSizeX, int outSizeY, float* d_filter, std::complex<float>* result);
template<typename T>
void processInput(GpuMultidimArrayAtGpu<T>& imagesGPU,
		GpuMultidimArrayAtGpu<std::complex<T> >& resultingFFT,
		mycufftHandle& handle,
		int inSizeX, int inSizeY, int inBatch,
		int outSizeX, int outSizeY, T* d_filter, std::complex<T>* result) {
	imagesGPU.fft(resultingFFT, handle);

	// crop FFT, reuse already allocated space
	size_t noOfCroppedItems = inBatch * outSizeX * outSizeY ;
	size_t bytes =  noOfCroppedItems * sizeof(T) * 2; // complex
	cudaMemset(imagesGPU.d_data, 0.f, bytes);

	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
	dim3 dimGrid(ceil(outSizeX/(float)dimBlock.x), ceil(outSizeY/(float)dimBlock.y));
	scaleFFT2D(dimGrid, dimBlock, (std::complex<T>*)resultingFFT.d_data, (std::complex<T>*)imagesGPU.d_data, inBatch, resultingFFT.Xdim, resultingFFT.Ydim, outSizeX, outSizeY, d_filter, 1.f/imagesGPU.yxdim, false);
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk(cudaMemcpy((void*)result, (void*)imagesGPU.d_data, bytes, cudaMemcpyDeviceToHost));
}

template<typename T>
__global__
void correlate(const T* __restrict__ in1, const T* __restrict__ in2, T* correlations, int xDim, int yDim, int noOfImgs,
		bool isWithin, int iStart, int iStop, int jStart, int jStop, size_t jSize, size_t offset1, size_t offset2) {
	// assign pixel to thread
#if TILE > 1
	int id = threadIdx.y * blockDim.x + threadIdx.x;
	int tidX = threadIdx.x % TILE + (id / (blockDim.y * TILE)) * TILE;
	int tidY = (id / TILE) % blockDim.y;
	int idx = blockIdx.x*blockDim.x + tidX;
	int idy = blockIdx.y*blockDim.y + tidY;
#else
	volatile int idx = blockIdx.x*blockDim.x + threadIdx.x;
	volatile int idy = blockIdx.y*blockDim.y + threadIdx.y;
#endif
	int a = 1-2*((idx+idy)&1); // center FFT, input must have be even

	if (idx >= xDim || idy >= yDim ) return;
	size_t pixelIndex = idy*xDim + idx; // index within single image

	bool compute = false;
	int counter = 0;
	for (int i = iStart; i <= iStop; i++) {
		int tmpOffset = i * xDim * yDim;
		T tmp = in1[tmpOffset + pixelIndex];
		for (int j = isWithin ? i + 1 : 0; j < jSize; j++) {
			if (!compute) {
				compute = true;
				j = jStart;
				continue; // skip first iteration
			}
			if (compute) {
				int tmp2Offset = j * xDim * yDim;
				T tmp2 = in2[tmp2Offset + pixelIndex];
				T res;
				res.x = (tmp.x*tmp2.x) + (tmp.y*tmp2.y);
				res.y = (tmp.y*tmp2.x) - (tmp.x*tmp2.y);
				correlations[counter*xDim*yDim + pixelIndex] = res * a;
				counter++;
			}
			if ((iStop == i) && (jStop == j)) {
				return;
			}
		}
	}
}

template<typename T>
void copyInRightOrder(T* imgs, T* result, int xDim, int yDim, int noOfImgs,
		bool isWithin, int iStart, int iStop, int jStart, int jStop, size_t jSize, size_t offset1, size_t offset2, size_t maxImgs) {
	size_t pixelsPerImage =  xDim * yDim;
	size_t counter = 0;
	bool ready = false;
	for (int i = iStart; i <= iStop; i++) {
		for (int j = isWithin ? i + 1 : 0; j < jSize; j++) {
			if (!ready) {
				ready = true;
				j = jStart;
				continue; // skip first iteration
			}
			if (ready) {
				size_t actualI = offset1 + i;
				size_t actualJ = offset2 + j;
				size_t toCopy = jSize - j;
				// imagine correlation in layers, correlation of 0th img with other is first layer, 1st with other is second etc
				// compute sum of images in complete layers
				size_t imgsInPreviousLayers = (((maxImgs - 1) + (maxImgs - actualI)) * (actualI)) / 2;
				size_t imgsInCurrentLayer = actualJ - actualI - 1;
//				size_t imgs = imgsInLayers + actualJ;
				gpuErrchk(cudaMemcpy(result + (pixelsPerImage * (imgsInPreviousLayers + imgsInCurrentLayer)),
					imgs + (counter * pixelsPerImage),
					toCopy * pixelsPerImage * sizeof(T),
					cudaMemcpyDeviceToHost));
				counter += toCopy;
				break; // skip to next outer iteration
			}
			if ((iStop == i) && (jStop == j)) {
				return;
			}
		}
	}
}

template<typename T>
__global__
void cropCenter(const T* __restrict__ in, T* out, int xDim, int yDim, int noOfImgs,
		int outDim) {
	// assign pixel to thread
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;

	if (idx >= outDim || idy >= outDim ) return;

	int inputImgSize = xDim * yDim;
	int outputImgSize = outDim * outDim;

	int inCenterX = (int)((T) (xDim) / 2.f);
	int inCenterY = (int)((T) (yDim) / 2.f);

	int outCenter = (int)((T) (outDim) / 2.f);

	for (int n = 0; n < noOfImgs; n++) {
		int iX = idx - outCenter + inCenterX;
		int iY = idy - outCenter + inCenterY;
		int inputPixelIdx = (n * inputImgSize) + (iY * xDim) + iX;
		int outputPixelIdx = (n * outputImgSize) + (idy * outDim) + idx;
		out[outputPixelIdx] = in[inputPixelIdx];
	}
}

template<typename T>
void computeCorrelations(int N, T maxShift, void* d_in1, size_t in1Size, void* d_in2, size_t in2Size,
		int fftSizeX, int imgSizeX, int imgSizeY, int fftBatchSize, size_t fixmeOffset1, size_t fixmeOffset2,
		GpuMultidimArrayAtGpu<std::complex<T> >& ffts,
			GpuMultidimArrayAtGpu<T>& imgs, mycufftHandle& handler,
			T*& result) {
	bool isWithin = d_in1 == d_in2; // correlation is done within the same buffer

	int cropSize = maxShift * 2 + 1;

	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
	dim3 dimGridCorr(ceil(fftSizeX/(float)dimBlock.x), ceil(imgSizeY/(float)dimBlock.y));
	dim3 dimGridCrop(ceil(cropSize/(float)dimBlock.x), ceil(cropSize/(float)dimBlock.y));

	size_t batchCounter = 0;
	size_t counter = 0;
	int origI = 0;
	int origJ = isWithin ? 0 : -1; // kernel must skip first iteration
	for (int i = 0; i < in1Size; i++) {
		for (int j = isWithin ? i + 1 : 0; j < in2Size; j++) {
			counter++;
			bool isLastIIter = isWithin ? (i == in1Size - 2) : (i == in1Size -1);
			if (counter == fftBatchSize || (isLastIIter && (j == in2Size -1)) ) {
				// kernel must perform last iteration
				// compute correlation from input buffers. Result are FFT images
				if (std::is_same<T, float>::value) {
				correlate<<<dimGridCorr, dimBlock>>>((float2*)d_in1, (float2*)d_in2,(float2*)ffts.d_data, fftSizeX, imgSizeY, counter,
						isWithin, origI, i, origJ, j, in2Size, fixmeOffset1, fixmeOffset2);
//				} else if (std::is_same<T, double>::value) {
//					correlate<<<dimGridCorr, dimBlock>>>((double2*)d_in1, (double2*)d_in2,(double2*)ffts.d_data, fftSizeX, imgSizeY, counter,
//											isWithin, origI, i, origJ, j, in2Size, fixmeOffset1, fixmeOffset2);
				} else {
					throw std::logic_error("unsupported type");
				}
				// convert FFTs to space domain
				ffts.ifft(imgs, handler);
				// crop images in space domain, use memory for FFT to avoid realocation
				cropCenter<<<dimGridCrop, dimBlock>>>((T*)imgs.d_data, (T*)ffts.d_data, imgSizeX, imgSizeY,
						counter, cropSize);

				copyInRightOrder((T*)ffts.d_data, result,
						cropSize, cropSize, counter,
						isWithin, origI, i, origJ, j, in2Size, fixmeOffset1, fixmeOffset2,N);

				origI = i;
				origJ = j;
				counter = 0;
				batchCounter++;
			}
		}
	}
}

template void computeCorrelations<float>(float maxShift, size_t noOfImgs, std::complex<float>* h_FFTs,
		int fftSizeX, int imgSizeX, int imgSizeY, size_t maxFFTsInBuffer,
		int fftBatchSize, float*& result);
template<typename T>
void computeCorrelations(T maxShift, size_t noOfImgs, std::complex<T>* h_FFTs,
		int fftSizeX, int imgSizeX, int imgSizeY, size_t maxFFTsInBuffer,
		int fftBatchSize, T*& result) {

	GpuMultidimArrayAtGpu<std::complex<T> > ffts(fftSizeX, imgSizeY, 1, fftBatchSize);
	GpuMultidimArrayAtGpu<T> imgs(imgSizeX, imgSizeY, 1, fftBatchSize);
	mycufftHandle myhandle;

	size_t resSize = 2*maxShift + 1;
	size_t singleImgPixels = resSize * resSize;
	size_t noOfCorrelations = (noOfImgs * (noOfImgs-1)) / 2;

	size_t singleFFTPixels = fftSizeX * imgSizeY;
	size_t singleFFTBytes = singleFFTPixels * sizeof(T) * 2;

	result = new T[noOfCorrelations * singleImgPixels];

	size_t buffer1Size = std::min(maxFFTsInBuffer, noOfImgs);
	void* d_fftBuffer1;
	gpuMalloc((void**) &d_fftBuffer1, buffer1Size * singleFFTBytes);

	void* d_fftBuffer2;
	size_t buffer2Size = std::max((size_t)0, std::min(maxFFTsInBuffer, noOfImgs - buffer1Size));
	gpuMalloc((void**) &d_fftBuffer2, buffer2Size * singleFFTBytes);

	size_t buffer1Offset = 0;
	do {
		size_t buffer1ToCopy = std::min(buffer1Size, noOfImgs - buffer1Offset);
		size_t inputOffsetBuffer1 = buffer1Offset * singleFFTPixels;
		gpuErrchk(cudaMemcpy(d_fftBuffer1, h_FFTs + inputOffsetBuffer1, buffer1ToCopy * singleFFTBytes, cudaMemcpyHostToDevice));

		// compute inter-buffer correlations
		computeCorrelations(noOfImgs, maxShift, d_fftBuffer1, buffer1ToCopy, d_fftBuffer1, buffer1ToCopy, fftSizeX, imgSizeX, imgSizeY, fftBatchSize, buffer1Offset, buffer1Offset, ffts, imgs, myhandle, result);
		size_t buffer2Offset = buffer1Offset + buffer1ToCopy;
		while (buffer2Offset < noOfImgs) {
			// copy other buffer
			size_t buffer2ToCopy = std::min(buffer2Size, noOfImgs - buffer2Offset);
			size_t inputOffsetBuffer2 = buffer2Offset * singleFFTPixels;
			gpuErrchk(cudaMemcpy(d_fftBuffer2, h_FFTs + inputOffsetBuffer2, buffer2ToCopy * singleFFTBytes, cudaMemcpyHostToDevice));

			computeCorrelations(noOfImgs, maxShift,d_fftBuffer1, buffer1ToCopy, d_fftBuffer2, buffer2ToCopy, fftSizeX, imgSizeX, imgSizeY, fftBatchSize, buffer1Offset, buffer2Offset, ffts, imgs, myhandle, result);

			buffer2Offset += buffer2ToCopy;
		}

		buffer1Offset += buffer1ToCopy;

	} while (buffer1Offset < noOfImgs);

	cudaFree(d_fftBuffer1);
	cudaFree(d_fftBuffer2);

	gpuErrchk( cudaPeekAtLastError() );
}
