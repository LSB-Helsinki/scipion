
#include <cuda_runtime_api.h>
#include "reconstruction_cuda/cuda_utils.h" // cannot be in header as it includes cuda headers
#include "advisor.h"
#include "cudaUtils.h"
#include "cuda_gpu_reconstruct_fourier.h"
#include "cuda_gpu_movie_alignment_correlation.h"
#include "reconstruction_cuda/cuda_basic_math.h"

#define BLOCK_DIM_X 32

// run per each pixel of the dest array
__global__
void kernel2(const float2* __restrict__ src, float2* dest, int noOfImages, size_t oldX, size_t oldY, size_t newX, size_t newY,
		const float* __restrict__ filter) {
	// assign pixel to thread
	volatile int idx = blockIdx.x*blockDim.x + threadIdx.x;
	volatile int idy = blockIdx.y*blockDim.y + threadIdx.y;

//	FIXME add tiling. currently limited by memory transfers (even without filtering)

//	if (idx == 0 && idy ==0) {
//		printf("kernle2 called %p %p %d old:%lu %lu new:%lu %lu filter %p\n", src, dest, noOfImages, oldX, oldY, newX, newY, filter);
//	}
	if (idx >= newX || idy >= newY ) return;
	size_t fIndex = idy*newX + idx; // index within single image
	float lpfw = filter[fIndex];
	int yhalf = (newY+1)/2;
	float a = 1.f;//1-2*((idx+idy)&1);

	size_t origY = (idy <= yhalf) ? idy : (oldY - (newY-idy)); // take top N/2+1 and bottom N/2 lines
	for (int n = 0; n < noOfImages; n++) {
		size_t iIndex = n*oldX*oldY + origY*oldX + idx; // index within consecutive images
		size_t oIndex = n*newX*newY + fIndex; // index within consecutive images
//		if (iIndex >= 16785408l || iIndex < 0 || oIndex >= 5177900l || oIndex < 0) {
//			printf("problem: %p %p old:%lu %lu new:%lu %lu : i:%lu o:%lu\nyhalf: %d origY %lu thread %d %d \n", src, dest, oldX, oldY, newX, newY, iIndex, oIndex,
//					yhalf, origY, idx, idy);
//		}
//		if (fIndex >= 2588950) {
//			printf("problem: %p %p old:%lu %lu new:%lu %lu : i:%lu o:%lu f:%lu \nyhalf: %d origY %lu thread %d %d \n", src, dest, oldX, oldY, newX, newY, iIndex, oIndex, fIndex,
//								yhalf, origY, idx, idy);
//		}
		dest[oIndex] = src[iIndex] * lpfw*a;
	}

//	int halfY = iSizeY / 2;
//	float normFactor = iSizeY*iSizeY;
//	int oSizeX = oBuffer->fftSizeX;
//
//	// input is an image in Fourier space (not normalized)
//	// with low frequencies in the inner corners
//	for (int n = 0; n < iLength; n++) {
//		float2 freq;
//		if ((idy < iSizeY) // for all input lines
//				&& (idx < oSizeX)) { // for all output pixels in the line
//			// process line only if it can hold sufficiently high frequency, i.e. process only
//			// first and last N lines
//			if (idy < oSizeX || idy >= (iSizeY - oSizeX)) {
//				// check the frequency
//				freq.x = FFT_IDX2DIGFREQ(idx, iSizeY);
//				freq.y = FFT_IDX2DIGFREQ(idy, iSizeY);
//				if ((freq.x * freq.x + freq.y * freq.y) > maxResolutionSqr) {
//					continue;
//				}
//				// do the shift (lower line will move up, upper down)
//				int newY = (idy < halfY) ? (idy + oSizeX) : (idy - iSizeY + oSizeX);
//				int oIndex = newY*oSizeX + idx;
//
//				int iIndex = n*iSizeY*iSizeX + idy*iSizeX + idx;
//				float* iValue = (float*)&(iFouriers[iIndex]);
//
//				// copy data and perform normalization
//				oBuffer->getNthItem(oBuffer->FFTs, n)[2*oIndex] = iValue[0] / normFactor;
//				oBuffer->getNthItem(oBuffer->FFTs, n)[2*oIndex + 1] = iValue[1] / normFactor;
//			}
//		}
//	}
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
	printf("free: %lu extra %d result %d\n", freeMem, extraMem, freeMem - extraMem);
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


//	std::cout << "about to do FFT" << std::endl;
	imagesGPU.fft(resultingFFT, handle);

	// crop FFT, reuse already allocated space
	size_t noOfCroppedFloats = inBatch * outSizeX * outSizeY ; // complex
	cudaMemset(imagesGPU.d_data, 0.f, noOfCroppedFloats*sizeof(float2));

	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
	dim3 dimGrid(ceil(outSizeX/(float)dimBlock.x), ceil(outSizeY/(float)dimBlock.y));
//	printf("about to run kernel\n");
	kernel2<<<dimGrid, dimBlock>>>((float2*)resultingFFT.d_data, (float2*)imagesGPU.d_data, inBatch, resultingFFT.Xdim, resultingFFT.Ydim, outSizeX, outSizeY, d_filter);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );

// copy out results
//	std::cout << "about to copy to host" << std::endl;
//	result = new std::complex<float>[noOfImages*newFFTX*newY]();
//	printf("result: %p\nFFTs: %p\n", result, resultingFFT.d_data );
//	resultingFFT.copyToCpu(result);
//	printf ("about to copy to host: %p %p %d\n", result, d_cropped, noOfCroppedFloats*sizeof(float));
	gpuErrchk(cudaMemcpy((void*)result, (void*)imagesGPU.d_data, noOfCroppedFloats*sizeof(float2), cudaMemcpyDeviceToHost));
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
//	std::cout << "copy to host done" << std::endl;
//	result = (std::complex<float>*) d_cropped;
//	resultingFFT.d_data = NULL;
//	std::cout << "No of elems: " << resultingFFT.nzyxdim  << " X:" << resultingFFT.Xdim << " Y:" << resultingFFT.Ydim<< std::endl;

//	cudaMemGetInfo(&free, &total);
//	printf("Mem kernel1 end: %lu %lu\n", free/1024/1024, total);
//	printf("---------------- kernel1 end\n");
//	fflush(stdout);
}

void kernel1(float* imgs, size_t oldX, size_t oldY, int noOfImages, size_t newX, size_t newY,
		float* filter,
		std::complex<float>*& result) {
//		float*& result) {

//	printf("----------------\n");

//	size_t free, total;
//	cudaMemGetInfo(&free, &total);
//	printf("Mem kernel1: %lu %lu\n", free/1024/1024, total);

//	FIXME Assert newX <= oldX. same with Y


	size_t noOfFloats = noOfImages * std::max(oldX*oldY, (oldX/2+1) * oldY * 2);
	float* d_imgs;
	gpuMalloc((void**) &d_imgs,noOfFloats*sizeof(float)); // no dealoc here, destructor will take care of it
//	printf("allocated %p of size %lu\n", d_imgs, noOfFloats*sizeof(float)/1048576);
	gpuErrchk(cudaMemcpy(d_imgs, imgs, noOfFloats*sizeof(float), cudaMemcpyHostToDevice));
	// store to proper structure
	GpuMultidimArrayAtGpu<float> imagesGPU(oldX, oldY, 1, noOfImages, d_imgs);
//	imagesGPU.copyToGpu(imgs);

//	************
//	IN-OF-PLACE
//	************
	GpuMultidimArrayAtGpu<std::complex<float> > resultingFFT(imagesGPU.Xdim / 2 + 1,
			imagesGPU.Ydim,
			imagesGPU.Zdim,
			imagesGPU.Ndim,
			(std::complex<float>*)imagesGPU.d_data);

//	************
//	OUT-OF-PLACE
//	************
//	GpuMultidimArrayAtGpu<std::complex<float> > resultingFFT;

// perform FFT
	mycufftHandle myhandle;
//	std::cout << "about to do FFT" << std::endl;
	imagesGPU.fft(resultingFFT, myhandle);
//	myhandle.clear(); // release unnecessary l || oIndex < 0) {
	//			printf("problem: %p %p old:%lu %lu new:%lu %lu : i:%lu o:%lu\nyhalf: %d origY %lu thread %d %d \n", src, dest, oldX, oldY, newX, newY, iIndex, oIndex,
	//					yhalf, origY, idx, idy);
	//		}memory
//	std::cout << "FFT done" << std::endl;
	myhandle.clear();

//	gpuErrchk( cudaPeekAtLastError() );
//	gpuErrchk( cudaDeviceSynchronize() );


	// crop FFT
	float2* d_cropped;
	size_t newFFTX = newX / 2 + 1;
	size_t noOfCroppedFloats = noOfImages * newFFTX * newY ; // complex

	float* d_filter; // FIXME

	gpuMalloc((void**) &d_cropped,noOfCroppedFloats*sizeof(float2));
	cudaMemset(d_cropped, 0.f, noOfCroppedFloats*sizeof(float2));
	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
	dim3 dimGrid(ceil(newFFTX/(float)dimBlock.x), ceil(newY/(float)dimBlock.y));
	printf("byl jsem zde\n");
	kernel2<<<dimGrid, dimBlock>>>((float2*)resultingFFT.d_data, d_cropped, noOfImages, resultingFFT.Xdim, resultingFFT.Ydim, newFFTX, newY, d_filter);
	cudaFree(d_filter);
	resultingFFT.clear();
	imagesGPU.d_data = NULL; // pointed to resultingFFT.d_data, which was cleared above
//	gpuErrchk( cudaPeekAtLastError() );
//	gpuErrchk( cudaDeviceSynchronize() );
//	gpuErrchk( cudaPeekAtLastError() );

// copy out results
//	std::cout << "about to copy to host" << std::endl;
//	result = new std::complex<float>[noOfImages*newFFTX*newY]();
//	printf("result: %p\nFFTs: %p\n", result, resultingFFT.d_data );
//	resultingFFT.copyToCpu(result);
//	printf ("about to copy to host: %p %p %d\n", result, d_cropped, noOfCroppedFloats*sizeof(float));
//	gpuErrchk(cudaMemcpy((void*)result, (void*)d_cropped, noOfCroppedFloats*sizeof(float), cudaMemcpyDeviceToHost));
//	cudaFree(d_cropped);
//	std::cout << "copy to host done" << std::endl;
	result = (std::complex<float>*) d_cropped;
//	resultingFFT.d_data = NULL;
//	std::cout << "No of elems: " << resultingFFT.nzyxdim  << " X:" << resultingFFT.Xdim << " Y:" << resultingFFT.Ydim<< std::endl;

//	cudaMemGetInfo(&free, &total);
//	printf("Mem kernel1 end: %lu %lu\n", free/1024/1024, total);
//	printf("---------------- kernel1 end\n");
//	fflush(stdout);
}

#define IDX2R(i,j,N) (((i)*(N))+(j))

__global__ void fftshift_2D(double2 *data, int N1, int N2)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockDim.y * blockIdx.y;

    if (i < N1 && j < N2) {
    	double a = 1-2*((i+j)&1);

       data[j*blockDim.x*gridDim.x+i].x *= a;
       data[j*blockDim.x*gridDim.x+i].y *= a;

    }
}

__global__
void kernel4(const float2* __restrict__ imgs, float2* correlations, int xDim, int yDim, int noOfImgs) {
	// assign pixel to thread
		volatile int idx = blockIdx.x*blockDim.x + threadIdx.x;
		volatile int idy = blockIdx.y*blockDim.y + threadIdx.y;
		float a = 1-2*((idx+idy)&1); // center FFT

//		if (idx == 0 && idy ==0) {
//			printf("kernel4 called %p %p %d %d %d\n", imgs, correlations, xDim, yDim, noOfImgs);
//		}
		if (idx >= xDim || idy >= yDim ) return;
		size_t pixelIndex = idy*xDim + idx; // index within single image

		int counter = 0;
		for (int i = 0; i < (noOfImgs - 1); i++) {
			int tmpOffset = i * xDim * yDim;
			float2 tmp = imgs[tmpOffset + pixelIndex];
			for (int j = i+1; j < noOfImgs; j++) {
				int tmp2Offset = j * xDim * yDim;
				float2 tmp2 = imgs[tmp2Offset + pixelIndex];
				float2 res;
				// FIXME why conjugate and multiply?
				res.x = ((tmp.x*tmp2.x) + (tmp.y*tmp2.y))*(yDim*yDim);
				res.y = ((tmp.y*tmp2.x) - (tmp.x*tmp2.y))*(yDim*yDim);
				correlations[counter*xDim*yDim + pixelIndex] = res*a;
				counter++;
			}
		}
}

#pragma GCC optimize("O0") // FIXME
void test(bool isWithin, int iStart, int iStop, int jStart, int jStop, size_t jSize, size_t offset1, size_t offset2) {
//	int i = iStart;
//	int j = jStart;
//	while (i != iStop || j != jStop) {
//		printf("correlation %03d - %03d\n", i+offset1, j+offset2);
//		if (j == jSize - 1) {
//			i++;
//			j = isWithin ? i + 1 : 0;
//		} else {
//			j++;
//		}
//	}

	bool compute = false;
	for (int i = iStart; i <= iStop; i++) {
		for (int j = isWithin ? i + 1 : 0; j < jSize; j++) {
			if (!compute) {// && (iStart == i) && (jStart == j)) {
				compute = true;
				j = jStart;
				continue; // skip first iteration
			}
			if (compute) {
				printf("correlation %03d - %03d\n", i+offset1, j+offset2);
			}
			if ((iStop == i) && (jStop == j)) {
				return;
			}
		}
	}
//
//
//	int counter = 0;
//	for (int i = iStart; i < iStop; i++) {
//		for (int j = jStart; j < jStop; j++) {
//			printf("correlation %03d - %03d\n", i+offset1, j+offset2);
//		}
//	}
}

void computeCorrelations(void* d_in1, size_t in1Size, void* d_in2, size_t in2Size,
		int fftSizeX, int imgSizeX, int imgSizeY, int fftBatchSize, size_t fixmeOffset1, size_t fixmeOffset2) {
	bool isWithin = d_in1 == d_in2; // correlation is done within the same buffer

	size_t counter = 0;
	int origI = 0;
	int origJ = isWithin ? 0 : -1; // kernel must skip first iteration
	for (int i = 0; i < in1Size; i++) {
		for (int j = isWithin ? i + 1 : 0; j < in2Size; j++) {
			counter++;
			bool isLastIIter = isWithin ? (i == in1Size - 2) : (i == in1Size -1);
			if (counter == fftBatchSize || (isLastIIter && (j == in2Size -1)) ) {
				// kernel must perform last iteration
				printf("volej kernel, i: %d-%d j: %d-%d, len = %lu\n", origI, i, origJ, j, counter);
				test(isWithin, origI, i, origJ, j, in2Size, fixmeOffset1, fixmeOffset2);
				origI = i;
				origJ = j;
				counter = 0;
			}
			// tohle bude kernel . musi vracet i a j, podle toho kde skoncil
		}
	}
}

void computeCorrelations(double maxShift, size_t noOfImgs, std::complex<float>* h_FFTs,
		int fftSizeX, int imgSizeX, int imgSizeY, size_t maxFFTsInBuffer,
		int fftBatchSize, std::complex<float>*& result) {

	size_t singleFFTPixels = fftSizeX * imgSizeY;
	size_t singleFFTBytes = singleFFTPixels * sizeof(float2);

	size_t buffer1Size = std::min(maxFFTsInBuffer, noOfImgs);
	void* d_fftBuffer1;
	gpuMalloc((void**) &d_fftBuffer1, buffer1Size * singleFFTBytes);

	void* d_fftBuffer2;
	size_t buffer2Size = std::max((size_t)0, std::min(maxFFTsInBuffer, noOfImgs - buffer1Size));
	gpuMalloc((void**) &d_fftBuffer2, buffer2Size * singleFFTBytes);

	size_t buffer1Offset = 0;
	size_t buffer2Offset = buffer1Size;
	do {
		printf("copying data 1 ... \n");
		size_t buffer1ToCopy = std::min(buffer1Size, noOfImgs - buffer1Offset);
		printf("buffer 1 starts at %d, length %d\n", buffer1Offset, buffer1ToCopy);
		size_t inputOffsetBuffer1 = buffer1Offset * singleFFTPixels;
		gpuErrchk(cudaMemset(d_fftBuffer1, 0.f, buffer1ToCopy * singleFFTBytes));
		gpuErrchk(cudaMemcpy(d_fftBuffer1, h_FFTs + inputOffsetBuffer1, buffer1ToCopy * singleFFTBytes, cudaMemcpyHostToDevice));

		// compute inter-buffer correlations
		printf("computing inter-buffer correlations... \n");
		computeCorrelations(d_fftBuffer1, buffer1ToCopy, d_fftBuffer1, buffer1ToCopy, fftSizeX, imgSizeX, imgSizeY, fftBatchSize, buffer1Offset, buffer1Offset);
		while (buffer2Offset < noOfImgs) {
			// copy other buffer
			printf("copying data 2 ... \n");
			size_t buffer2ToCopy = std::min(buffer2Size, noOfImgs - buffer2Offset);
			printf("buffer 2 starts at %d, length %d\n", buffer2Offset, buffer2ToCopy);
			size_t inputOffsetBuffer2 = buffer2Offset * singleFFTPixels;
			gpuErrchk(cudaMemset(d_fftBuffer2, 0.f, buffer2ToCopy * singleFFTBytes));
			gpuErrchk(cudaMemcpy(d_fftBuffer2, h_FFTs + inputOffsetBuffer2, buffer2ToCopy * singleFFTBytes, cudaMemcpyHostToDevice));

			printf("computing extra-buffer correlations... \n");
			computeCorrelations(d_fftBuffer1, buffer1ToCopy, d_fftBuffer2, buffer2ToCopy, fftSizeX, imgSizeX, imgSizeY, fftBatchSize, buffer1Offset, buffer2Offset);

			buffer2Offset += buffer2ToCopy;
		}

		buffer1Offset += buffer1ToCopy;

	} while (buffer1Offset < noOfImgs);

}


void kernel3(float maxShift, size_t noOfImgs, const std::complex<float>* imgs, size_t fftXdim, size_t fftYdim, float*& result,
		std::complex<float>*& result2) {
//	printf("---------------- kernel 3 start %lu %lu \n",  fftXdim, fftYdim);
	size_t noOfCorellations = noOfImgs * (noOfImgs - 1) / 2;
//	float2* d_b;
//
//	gpuMalloc((void**) &d_b,noOfCorellations*sizeof(float)*2);
//	cudaMemset(d_b, 0.f, noOfCorellations*sizeof(float)*2);
//	size_t free, total;
//	cudaMemGetInfo(&free, &total);
//	printf("Mem before plan: %lu %lu\n", free/1024/1024, total);

	size_t noOfPixels = noOfImgs * fftXdim * fftYdim;
	float2* d_imgs;
	d_imgs = (float2*) imgs;
//	gpuMalloc((void**) &d_imgs, noOfPixels*sizeof(float2));
//	cudaMemcpy((void*)d_imgs, (void*)imgs, noOfPixels*sizeof(float2), cudaMemcpyHostToDevice);

//	cudaMemGetInfo(&free, &total);
//	printf("Mem: %lu %lu\n", free/1024/1024, total);

	size_t noOfCorrPixels = noOfCorellations * fftXdim * fftYdim;
	float2* d_corrs;
	gpuMalloc((void**) &d_corrs, std::max(noOfCorrPixels*sizeof(float2), noOfCorellations*fftYdim*fftYdim*sizeof(float)));
	cudaMemset(d_corrs, 0.f, std::max(noOfCorrPixels*sizeof(float2), noOfCorellations*fftYdim*fftYdim*sizeof(float)));

//	cudaMemGetInfo(&free, &total);
//	printf("Mem: %lu %lu\n", free/1024/1024, total);

	dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
	dim3 dimGrid(ceil(fftXdim/(float)dimBlock.x), ceil(fftYdim/(float)dimBlock.y));
	kernel4<<<dimGrid, dimBlock>>>((float2*)d_imgs,(float2*) d_corrs, fftXdim, fftYdim, noOfImgs);

//	cudaMemGetInfo(&free, &total);
//	printf("Mem: %lu %lu\n", free/1024/1024, total);

//	gpuErrchk( cudaDeviceSynchronize() );
//	gpuErrchk( cudaPeekAtLastError() );

	cudaFree(d_imgs);

//	cudaMemGetInfo(&free, &total);
//	printf("Mem: %lu %lu\n", free/1024/1024, total);


//	output correlations in FFT
//	result2 = new std::complex<float>[noOfCorrPixels]();
//	cudaMemcpy((void*)result2, (void*)d_corrs, noOfCorrPixels*sizeof(float2), cudaMemcpyDeviceToHost);


// perform IFFT
	GpuMultidimArrayAtGpu<std::complex<float> > tmp(fftXdim, fftYdim, 1, noOfCorellations-1, (std::complex<float>*)d_corrs);
	GpuMultidimArrayAtGpu<float> tmp1(fftYdim, fftYdim, 1, noOfCorellations, (float*)d_corrs);
	mycufftHandle myhandle;
//	std::cout << "about to do IFFT" << std::endl;

//	cudaMemGetInfo(&free, &total);
//	printf("Mem: %lu %lu\n", free/1024/1024, total);

	tmp.ifft(tmp1, myhandle);
//	myhandle.clear(); // release unnecessary l || oIndex < 0) {
	//			printf("problem: %p %p old:%lu %lu new:%lu %lu : i:%lu o:%lu\nyhalf: %d origY %lu thread %d %d \n", src, dest, oldX, oldY, newX, newY, iIndex, oIndex,
	//					yhalf, origY, idx, idy);
	//		}memory
//	std::cout << "IFFT done" << std::endl;
	tmp1.d_data = NULL; // unbind
//	gpuErrchk( cudaPeekAtLastError() );
//	gpuErrchk( cudaDeviceSynchronize() );



	result = new float[fftYdim*fftYdim*noOfCorellations]();
	cudaMemcpy((void*)result, (void*)d_corrs, fftYdim*fftYdim*noOfCorellations*sizeof(float), cudaMemcpyDeviceToHost);

//	gpuErrchk( cudaDeviceSynchronize() );
//	gpuErrchk( cudaPeekAtLastError() );
//
//	printf("---------------- kernel 3 done \n");
//	fflush(stdout);

}
