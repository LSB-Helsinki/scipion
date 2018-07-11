
#include <cuda_runtime_api.h>
#include "reconstruction_cuda/cuda_utils.h"
#include "advisor.h"
#include "cudaUtils.h"
#include "cuda_gpu_movie_alignment_correlation.h"
#include "reconstruction_cuda/cuda_basic_math.h"

#define BLOCK_DIM_X 32
#define TILE 8

__global__
void applyFilterAndCrop(const float2* __restrict__ src, float2* dest, int noOfImages, size_t oldX, size_t oldY, size_t newX, size_t newY,
		const float* __restrict__ filter) {
	// assign pixel to thread
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;

	if (idx >= newX || idy >= newY ) return;
	size_t fIndex = idy*newX + idx; // index within single image
	float lpfw = filter[fIndex];
	int yhalf = (newY+1)/2;

	size_t origY = (idy <= yhalf) ? idy : (oldY - (newY-idy)); // take top N/2+1 and bottom N/2 lines
	for (int n = 0; n < noOfImages; n++) {
		size_t iIndex = n*oldX*oldY + origY*oldX + idx; // index within consecutive images
		size_t oIndex = n*newX*newY + fIndex; // index within consecutive images
		dest[oIndex] = src[iIndex] * lpfw;
	}
}

std::complex<float>* performFFTAndScale(float* inOutData, int noOfImgs,
		int inSizeX, int inSizeY, int inBatch,
		int outSizeX, int outSizeY,  float* d_filter) {
	mycufftHandle handle;
	int counter = 0;
	std::complex<float>* h_result = (std::complex<float>*)inOutData;
	GpuMultidimArrayAtGpu<float> imagesGPU(inSizeX, inSizeY, 1, inBatch);
	GpuMultidimArrayAtGpu<std::complex<float> > resultingFFT;

	while (counter < noOfImgs) {
		int imgToProcess = std::min(inBatch, noOfImgs - counter);
		float* h_imgLoad = inOutData + counter * inSizeX * inSizeY;
		size_t bytes = imgToProcess * inSizeX * inSizeY * sizeof(float);
		gpuErrchk(cudaMemcpy(imagesGPU.d_data, h_imgLoad, bytes, cudaMemcpyHostToDevice));
		std::complex<float>* h_imgStore = h_result + counter * outSizeX * outSizeY;
		processInput(imagesGPU, resultingFFT, handle, inSizeX, inSizeY, imgToProcess, outSizeX, outSizeY, d_filter, h_imgStore);
		counter += inBatch;
	}
	handle.clear();

	return h_result;
}

size_t getFreeMem(int device) {
	return cuFFTAdvisor::toMB(cuFFTAdvisor::getFreeMemory(device));
}

void getBestSize(int imgsToProcess, int origXSize, int origYSize, int &batchSize, int &xSize, int &ySize, int extraMem) {
	int device = 0; // FIXME detect device or add to cmd param

	size_t freeMem = getFreeMem(device);
	std::vector<cuFFTAdvisor::BenchmarkResult const *> *results =
			cuFFTAdvisor::Advisor::find(50, device,
					origXSize, origYSize, 1, imgsToProcess,
					cuFFTAdvisor::Tristate::TRUE,
					cuFFTAdvisor:: Tristate::TRUE,
					cuFFTAdvisor::Tristate::TRUE,
					cuFFTAdvisor::Tristate::FALSE,
					cuFFTAdvisor::Tristate::TRUE, INT_MAX,
						  freeMem - extraMem);

	batchSize = results->at(0)->transform->N;
	xSize = results->at(0)->transform->X;
	ySize = results->at(0)->transform->Y;
	results->at(0)->print(stdout);
}

float* loadToGPU(float* data, size_t items) {
	float* d_data;
	size_t bytes = items * sizeof(float);
	gpuMalloc((void**) &d_data,bytes);
	gpuErrchk(cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice));
	return d_data;
}

void release(float* data) {
	gpuErrchk(cudaFree(data));
}

void processInput(GpuMultidimArrayAtGpu<float>& imagesGPU,
		GpuMultidimArrayAtGpu<std::complex<float> >& resultingFFT,
		mycufftHandle& handle,
		int inSizeX, int inSizeY, int inBatch,
		int outSizeX, int outSizeY, float* d_filter, std::complex<float>* result) {
	imagesGPU.fft(resultingFFT, handle);

	// crop FFT, reuse already allocated space
	size_t noOfCroppedFloats = inBatch * outSizeX * outSizeY ; // complex
	cudaMemset(imagesGPU.d_data, 0.f, noOfCroppedFloats*sizeof(float2));

	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
	dim3 dimGrid(ceil(outSizeX/(float)dimBlock.x), ceil(outSizeY/(float)dimBlock.y));
	applyFilterAndCrop<<<dimGrid, dimBlock>>>((float2*)resultingFFT.d_data, (float2*)imagesGPU.d_data, inBatch, resultingFFT.Xdim, resultingFFT.Ydim, outSizeX, outSizeY, d_filter);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

	gpuErrchk(cudaMemcpy((void*)result, (void*)imagesGPU.d_data, noOfCroppedFloats*sizeof(float2), cudaMemcpyDeviceToHost));
}

__global__
void correlate(const float2* __restrict__ in1, const float2* __restrict__ in2, float2* correlations, int xDim, int yDim, int noOfImgs,
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
	float a = 1-2*((idx+idy)&1); // center FFT

	if (idx >= xDim || idy >= yDim ) return;
	size_t pixelIndex = idy*xDim + idx; // index within single image

	bool compute = false;
	int counter = 0;
	for (int i = iStart; i <= iStop; i++) {
		int tmpOffset = i * xDim * yDim;
		float2 tmp = in1[tmpOffset + pixelIndex];
		for (int j = isWithin ? i + 1 : 0; j < jSize; j++) {
			if (!compute) {
				compute = true;
				j = jStart;
				continue; // skip first iteration
			}
			if (compute) {
				int tmp2Offset = j * xDim * yDim;
				float2 tmp2 = in2[tmp2Offset + pixelIndex];
				float2 res;
				res.x = ((tmp.x*tmp2.x) + (tmp.y*tmp2.y))*(yDim*yDim);
				res.y = ((tmp.y*tmp2.x) - (tmp.x*tmp2.y))*(yDim*yDim);
				correlations[counter*xDim*yDim + pixelIndex] = res*a;
				counter++;
			}
			if ((iStop == i) && (jStop == j)) {
				return;
			}
		}
	}
}


void copyInRightOrder(float* imgs, float* result, int xDim, int yDim, int noOfImgs,
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
					toCopy * pixelsPerImage * sizeof(float),
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

__global__
void cropCenter(const float* __restrict__ in, float* out, int xDim, int yDim, int noOfImgs,
		int outDim) {
	// assign pixel to thread
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;

	if (idx >= outDim || idy >= outDim ) return;

	int inputImgSize = xDim * yDim;
	int outputImgSize = outDim * outDim;

	int inCenterX = (int)((float) (xDim) / 2.f);
	int inCenterY = (int)((float) (yDim) / 2.f);

	int outCenter = (int)((float) (outDim) / 2.f);

	for (int n = 0; n < noOfImgs; n++) {
		int iX = idx - outCenter + inCenterX;
		int iY = idy - outCenter + inCenterY;
		int inputPixelIdx = (n * inputImgSize) + (iY * xDim) + iX;
		int outputPixelIdx = (n * outputImgSize) + (idy * outDim) + idx;
		out[outputPixelIdx] = in[inputPixelIdx];
	}
}

void computeCorrelations(int N, double maxShift, void* d_in1, size_t in1Size, void* d_in2, size_t in2Size,
		int fftSizeX, int imgSizeX, int imgSizeY, int fftBatchSize, size_t fixmeOffset1, size_t fixmeOffset2,
		GpuMultidimArrayAtGpu<std::complex<float> >& ffts,
			GpuMultidimArrayAtGpu<float>& imgs, mycufftHandle& handler,
			float*& result) {
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
				correlate<<<dimGridCorr, dimBlock>>>((float2*)d_in1, (float2*)d_in2,(float2*)ffts.d_data, fftSizeX, imgSizeY, counter,
						isWithin, origI, i, origJ, j, in2Size, fixmeOffset1, fixmeOffset2);
				// convert FFTs to space domain
				ffts.ifft(imgs, handler);
				// crop images in space domain, use memory for FFT to avoid realocation
				cropCenter<<<dimGridCrop, dimBlock>>>((float*)imgs.d_data, (float*)ffts.d_data, imgSizeX, imgSizeY,
						counter, cropSize);

				copyInRightOrder((float*)ffts.d_data, result,
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

void computeCorrelations(double maxShift, size_t noOfImgs, std::complex<float>* h_FFTs,
		int fftSizeX, int imgSizeX, int imgSizeY, size_t maxFFTsInBuffer,
		int fftBatchSize, float*& result) {

	GpuMultidimArrayAtGpu<std::complex<float> > ffts(fftSizeX, imgSizeY, 1, fftBatchSize);
	GpuMultidimArrayAtGpu<float> imgs(imgSizeX, imgSizeY, 1, fftBatchSize);
	mycufftHandle myhandle;

	size_t resSize = 2*maxShift + 1;
	size_t singleImgPixels = resSize * resSize;
	size_t noOfCorrelations = (noOfImgs * (noOfImgs-1)) / 2;

	size_t singleFFTPixels = fftSizeX * imgSizeY;
	size_t singleFFTBytes = singleFFTPixels * sizeof(float2);

	result = new float[noOfCorrelations * singleImgPixels];

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
