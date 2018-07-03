
#include <cuda_runtime_api.h>
#include "reconstruction_cuda/cuda_utils.h" // cannot be in header as it includes cuda headers
#include "advisor.h"
#include "cudaUtils.h"
#include "cuda_gpu_reconstruct_fourier.h"
#include "cuda_gpu_movie_alignment_correlation.h"
#include "reconstruction_cuda/cuda_basic_math.h"

#define BLOCK_DIM_X 32
#define TILE 8

__device__
float bspline03(float x){
	float Argument = fabsf(x);
	if (Argument < 1.f)
		return Argument * Argument * (Argument - 2.f) * 0.5f + 2.f / 3.f;
	else if (Argument < 2.f)
	{
		Argument -= 2.f;
		return Argument * Argument * Argument * (-1.f / 6.f);
	}
	else
		return 0.f;
}

__device__
double bspline03(double x){
	double Argument = fabs(x);
	if (Argument < 1.0)
		return Argument * Argument * (Argument - 2.0) * 0.5 + 2.0 / 3.0;
	else if (Argument < 2.0)
	{
		Argument -= 2.0;
		return Argument * Argument * Argument * (-1.0 / 6.0);
	}
	else
		return 0.0;
}

template<typename T>
__device__
T interpolatedElementBSpline2D_Degree3(T x, T y, int xinit, int yinit, int xdim, int ydim, T* data)
{
	bool	firstTime=true;			// Inner loop first time execution flag.
	T	*ref;

	// Logical to physical
	y -= yinit;
	x -= xinit;

	int l1 = (int)ceil(x - 2);
	int l2 = l1 + 3;
	int m1 = (int)ceil(y - 2);
	int m2 = m1 + 3;

	T columns = 0.0;
	T aux;

	int		equivalent_l_Array[LOOKUP_TABLE_LEN]; // = new int [l2 - l1 + 1];
	T 	aux_Array[LOOKUP_TABLE_LEN];// = new double [l2 - l1 + 1];

	for (int m = m1; m <= m2; m++)
	{
		int equivalent_m=m;
		if      (m<0)
			equivalent_m=-m-1;
		else if (m>=ydim)
			equivalent_m=2*ydim-m-1;
		T rows = 0.0;
		int	index=0;
//		ref = &DIRECT_A2D_ELEM(*this, equivalent_m,0);
		ref = data + (equivalent_m*xdim);
		for (int l = l1; l <= l2; l++)
		{
			int equivalent_l;
			// Check if it is first time executing inner loop.
			if (firstTime)
			{
				T xminusl = x - (T) l;
				equivalent_l=l;
				if (l<0)
				{
					equivalent_l=-l-1;
				}
				else if (l>=xdim)
				{
					equivalent_l=2*xdim-l-1;
				}

				equivalent_l_Array[index] = equivalent_l;
				aux = bspline03(xminusl);
				aux_Array[index] = aux;
				index++;
			}
			else
			{
				equivalent_l = equivalent_l_Array[index];
				aux = aux_Array[index];
				index++;
			}

			//double Coeff = DIRECT_A2D_ELEM(*this, equivalent_m,equivalent_l);
			T Coeff = ref[equivalent_l];
			rows += Coeff * aux;
		}

		// Set first time inner flag is executed to false.
		firstTime = false;

		T yminusm = y - (T) m;
		aux = bspline03(yminusm);
		columns += rows * aux;
	}

	return columns;
}




// explicit instatiation to avoid linking issues
//template void applyGeometryGPU<float>(int, MultidimArray<double>&, MultidimArray<double> const&, Matrix2D<double> const&, bool, bool, double, MultidimArray<double>*);
template<typename T>
__global__
void applyGeometryKernel(T x, T y,
		bool wrap, T minxpp, T maxxpp, T minypp, T maxypp,
		T minxp, T maxxp, T minyp, T maxyp, T xShift,
		T yShift, T* data, int xdim, int ydim, int xinit, int yinit, T* coefs,
		int coefsXDim, int coefsYDim) {

	// assign pixel to thread
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
//	if (idx == 0 && idy == 0) {
//		printf("x:%f y:%f\n", xShift, yShift);
//	}
	if (j >= xdim || i >= ydim) return;


//	for (size_t i = 0; i < YSIZE(V2); i++) {
		// Calculate this position in the input image according to the
		// geometrical transformation
		// they are related by
		// coords_output(=x,y) = A * coords_input (=xp,yp)
		T xp = j + xShift + x;
		T yp = i + yShift + y;

//		printf("init idx: %d idy: %d xp: %f yp: %f\n", idx, idy, xp, yp);

		// Inner loop boundaries.
//		int globalMin = 0, globalMax = XSIZE(V2);

		// Loop over j is splitted according to wrap (wrap==true is not
		// vectorizable) and also according to SplineDegree value
		// (I have not fully analyzed vector dependences for
		// SplineDegree==3 and else branch)
		if (wrap) {
			// This is original implementation
//			for (int j = globalMin; j < globalMax; j++) {
				bool x_isOut = XMIPP_RANGE_OUTSIDE_FAST(xp, minxpp, maxxpp);
				bool y_isOut = XMIPP_RANGE_OUTSIDE_FAST(yp, minypp, maxypp);
//				printf("init i: %d j: %d xp: %f yp: %f, xisOut %d yisOut %d\n", i, j, xp, yp, x_isOut, y_isOut);

				if (x_isOut) {
					xp = realWRAP(xp, minxp - 0.5, maxxp + 0.5);
				}

				if (y_isOut) {
					yp = realWRAP(yp, minyp - 0.5, maxyp + 0.5);
				}

				// B-spline interpolation
//				if (xp >=-1 && xp <= 0 && yp >= -2 && yp <= -1) {
//				}
					T res = interpolatedElementBSpline2D_Degree3(xp, yp, xinit, yinit, coefsXDim, coefsYDim, coefs);
					size_t index  = i*xdim+j;
				data[index] = res;
//					printf("interpolate %lu : %f (i: %d j: %d xp: %f yp: %f\n", index, res, i, j, xp, yp);

				// Compute new point inside input image
//			}
		} /* wrap == true */
//		y++;
//	}
}


template void applyGeometry_2D_S3_wrap(int SplineDegree,
        MultidimArray<float>& __restrict__ result,
        const MultidimArray<float>& __restrict__ V1,
        const Matrix2D<float> &A, bool inv,
        bool wrap, float outside, MultidimArray<float> *BcoeffsPtr);
template<typename T>
void applyGeometry_2D_S3_wrap(int SplineDegree,
                   MultidimArray<T>& __restrict__ result,
                   const MultidimArray<T>& __restrict__ V1,
                   const Matrix2D<T> &A, bool inv,
                   bool wrap, T outside, MultidimArray<T> *BcoeffsPtr)
{
	clock_t begin = clock();
	MultidimArray<T> Bcoeffs;
    static MultidimArray<T> *BcoeffsToUse=NULL; // FIXME remove static
    Matrix2D<T> Ainv;
    const Matrix2D<T> * Aptr=&A;
    if (!inv)
    {
        Ainv = A.inv();
        Aptr=&Ainv;
    }
    const Matrix2D<T> &Aref=*Aptr;

    // For scalings the output matrix is resized outside to the final
    // size instead of being resized inside the routine with the
    // same size as the input matrix
    if (XSIZE(result) == 0)
    	result.resizeNoCopy(V1);

    printf("zacatek %f\n", ((float)clock()-begin)/CLOCKS_PER_SEC);

    begin = clock();
    if (outside != 0.)
    {
        // Initialize output matrix with value=outside
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(result)
        DIRECT_MULTIDIM_ELEM(result, n) = outside;
    }
    else
    	result.initZeros();

	printf("initValues %f\n", ((float)clock()-begin)/CLOCKS_PER_SEC);

        // 2D transformation
        T xShift=MAT_ELEM(Aref,0,2);
        T yShift=MAT_ELEM(Aref,1,2);

        // Find center and limits of image
        T cen_y  = (int)(YSIZE(result) / 2);
        T cen_x  = (int)(XSIZE(result) / 2);
        T cen_yp = (int)(YSIZE(V1) / 2);
        T cen_xp = (int)(XSIZE(V1) / 2);
        T minxp  = -cen_xp;
        T minyp  = -cen_yp;
        T minxpp = minxp-XMIPP_EQUAL_ACCURACY;
        T minypp = minyp-XMIPP_EQUAL_ACCURACY;
        T maxxp  = XSIZE(V1) - cen_xp - 1;
        T maxyp  = YSIZE(V1) - cen_yp - 1;
        T maxxpp = maxxp+XMIPP_EQUAL_ACCURACY;
        T maxypp = maxyp+XMIPP_EQUAL_ACCURACY;
        size_t Xdim   = XSIZE(V1);
        size_t Ydim   = YSIZE(V1);

        if (SplineDegree > 1)
        {
            // Build the B-spline coefficients
        	if (BcoeffsPtr!=NULL)
        		BcoeffsToUse=BcoeffsPtr;
        	else {
        		begin = clock();
        		MultidimArray<double> tmp;
        		produceSplineCoefficients(SplineDegree, tmp, V1); //Bcoeffs is a single image
        		typeCast(tmp, Bcoeffs);
        		printf("produceSplineCoefs%f\n", ((float)clock()-begin)/CLOCKS_PER_SEC);
        		BcoeffsToUse = &Bcoeffs;
        	}
            STARTINGX(*BcoeffsToUse) = (int) minxp;
            STARTINGY(*BcoeffsToUse) = (int) minyp;
        }

        // Now we go from the output image to the input image, ie, for any pixel
        // in the output image we calculate which are the corresponding ones in
        // the original image, make an interpolation with them and put this value
        // at the output pixel

#ifdef DEBUG_APPLYGEO
        std::cout << "A\n" << Aref << std::endl
        << "(cen_x ,cen_y )=(" << cen_x  << "," << cen_y  << ")\n"
        << "(cen_xp,cen_yp)=(" << cen_xp << "," << cen_yp << ")\n"
        << "(min_xp,min_yp)=(" << minxp  << "," << minyp  << ")\n"
        << "(max_xp,max_yp)=(" << maxxp  << "," << maxyp  << ")\n";
#endif
        // Calculate position of the beginning of the row in the output image
        T x = -cen_x;
        T y = -cen_y;

        dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
		dim3 dimGrid(ceil(XSIZE(result)/(T)dimBlock.x), ceil(YSIZE(result)/(T)dimBlock.y));

		static T* d_data = NULL; // FIXME
		size_t bytes = YXSIZE(result) * sizeof(T);
		if (NULL == d_data) gpuMalloc((void**) &d_data,bytes);
//		gpuErrchk(cudaMemcpy(d_data, result.data, bytes, cudaMemcpyHostToDevice));
//
		static T* d_coefs = NULL; // FIXME
		size_t coefs_bytes = YXSIZE(*BcoeffsToUse) * sizeof(T);
		if (NULL == d_coefs) gpuMalloc((void**) &d_coefs, coefs_bytes);
		gpuErrchk(cudaMemcpy(d_coefs, BcoeffsToUse->data, coefs_bytes, cudaMemcpyHostToDevice));

		printf("about to run kernel\n");

		applyGeometryKernel<<<dimGrid, dimBlock>>>(x, y, wrap, minxpp, maxxpp, minypp, maxypp,
				minxp, maxxp, minyp, maxyp, xShift, yShift, d_data, XSIZE(result), YSIZE(result), STARTINGX(*BcoeffsToUse), STARTINGY(*BcoeffsToUse), d_coefs, XSIZE(*BcoeffsToUse), YSIZE(*BcoeffsToUse));
		gpuErrchk(cudaMemcpy(result.data, d_data, bytes, cudaMemcpyDeviceToHost));
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
		gpuErrchk( cudaPeekAtLastError() );


    }





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
//	gpuErrchk( cudaPeekAtLastError() );
//	gpuErrchk( cudaDeviceSynchronize() );
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

//		if (idx == 0 && idy ==0) {
//			printf("kernel4 called %p %p %d %d %d\n", imgs, correlations, xDim, yDim, noOfImgs);
//		}
	if (idx >= xDim || idy >= yDim ) return;
	size_t pixelIndex = idy*xDim + idx; // index within single image

	bool compute = false;
	int counter = 0;
	for (int i = iStart; i <= iStop; i++) {
		int tmpOffset = i * xDim * yDim;
		float2 tmp = in1[tmpOffset + pixelIndex];
		for (int j = isWithin ? i + 1 : 0; j < jSize; j++) {
			if (!compute) {// && (iStart == i) && (jStart == j)) {
				compute = true;
				j = jStart;
				continue; // skip first iteration
			}
			if (compute) {
				int tmp2Offset = j * xDim * yDim;
				float2 tmp2 = in2[tmp2Offset + pixelIndex];
				float2 res;
				// FIXME why conjugate and multiply?
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
//				printf("imgsInPreviousLayers: %lu, imgsInCurrentLayer = %lu\n", imgsInPreviousLayers, imgsInCurrentLayer);
//				printf("copy at %p, offset %d, i:%d j:%d (%d)\n", result, imgsInPreviousLayers + imgsInCurrentLayer, actualI, actualJ, toCopy);
				break; // skip to next outer iteration

//				float2 tmp = in1[tmpOffset + pixelIndex];
//				int tmp2Offset = j * xDim * yDim;
//				float2 tmp2 = in2[tmp2Offset + pixelIndex];
//				float2 res;
//				// FIXME why conjugate and multiply?
//				res.x = ((tmp.x*tmp2.x) + (tmp.y*tmp2.y))*(yDim*yDim);
//				res.y = ((tmp.y*tmp2.x) - (tmp.x*tmp2.y))*(yDim*yDim);
//				correlations[counter*xDim*yDim + pixelIndex] = res*a;
//				counter++;
			}
			if ((iStop == i) && (jStop == j)) {
				return;
			}
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
//		if (n == 0) {
//			printf("thread [%d,%d] in: [%d,%d](%d), out: (%d)\n", idx, idy, iX, iY, inputPixelIdx, outputPixelIdx);
//		}
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
//		kernel4<<<dimGrid, dimBlock>>>((float2*)d_imgs,(float2*) d_corrs, fftXdim, fftYdim, noOfImgs);

	size_t singleImgPixels = cropSize * cropSize;

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
//				printf("volej kernel, i: %d-%d j: %d-%d, len = %lu\n", origI, i, origJ, j, counter);
				// compute correlation from input buffers. Result are FFT images
				correlate<<<dimGridCorr, dimBlock>>>((float2*)d_in1, (float2*)d_in2,(float2*)ffts.d_data, fftSizeX, imgSizeY, counter,
						isWithin, origI, i, origJ, j, in2Size, fixmeOffset1, fixmeOffset2);
//				test();
				// convert FFTs to space domain
				ffts.ifft(imgs, handler);
				// crop images in space domain, use memory for FFT to avoid realocation
//				gpuErrchk(cudaMemset(ffts.d_data, 0, counter * singleImgPixels * sizeof(float)));
				cropCenter<<<dimGridCrop, dimBlock>>>((float*)imgs.d_data, (float*)ffts.d_data, imgSizeX, imgSizeY,
						counter, cropSize);

				copyInRightOrder((float*)ffts.d_data, result,
						cropSize, cropSize, counter,
						isWithin, origI, i, origJ, j, in2Size, fixmeOffset1, fixmeOffset2,N);
				// copy cropped images to host
//				gpuErrchk(cudaMemcpy(result + (singleImgPixels * batchCounter * fftBatchSize),
//						ffts.d_data,
//						counter * singleImgPixels * sizeof(float),
//						cudaMemcpyDeviceToHost));

				origI = i;
				origJ = j;
				counter = 0;
				batchCounter++;
			}
			// tohle bude kernel . musi vracet i a j, podle toho kde skoncil
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
//		printf("copying data 1 ... \n");
		size_t buffer1ToCopy = std::min(buffer1Size, noOfImgs - buffer1Offset);
//		printf("buffer 1 starts at %d, length %d\n", buffer1Offset, buffer1ToCopy);
		size_t inputOffsetBuffer1 = buffer1Offset * singleFFTPixels;
//		gpuErrchk(cudaMemset(d_fftBuffer1, 0.f, buffer1ToCopy * singleFFTBytes));
		gpuErrchk(cudaMemcpy(d_fftBuffer1, h_FFTs + inputOffsetBuffer1, buffer1ToCopy * singleFFTBytes, cudaMemcpyHostToDevice));

		// compute inter-buffer correlations
//		printf("computing inter-buffer correlations... \n");
		computeCorrelations(noOfImgs, maxShift, d_fftBuffer1, buffer1ToCopy, d_fftBuffer1, buffer1ToCopy, fftSizeX, imgSizeX, imgSizeY, fftBatchSize, buffer1Offset, buffer1Offset, ffts, imgs, myhandle, result);
		size_t buffer2Offset = buffer1Offset + buffer1ToCopy;
		while (buffer2Offset < noOfImgs) {
			// copy other buffer
//			printf("copying data 2 ... \n");
			size_t buffer2ToCopy = std::min(buffer2Size, noOfImgs - buffer2Offset);
//			printf("buffer 2 starts at %d, length %d\n", buffer2Offset, buffer2ToCopy);
			size_t inputOffsetBuffer2 = buffer2Offset * singleFFTPixels;
//			gpuErrchk(cudaMemset(d_fftBuffer2, 0.f, buffer2ToCopy * singleFFTBytes));
			gpuErrchk(cudaMemcpy(d_fftBuffer2, h_FFTs + inputOffsetBuffer2, buffer2ToCopy * singleFFTBytes, cudaMemcpyHostToDevice));

//			printf("computing extra-buffer correlations... \n");
			computeCorrelations(noOfImgs, maxShift,d_fftBuffer1, buffer1ToCopy, d_fftBuffer2, buffer2ToCopy, fftSizeX, imgSizeX, imgSizeY, fftBatchSize, buffer1Offset, buffer2Offset, ffts, imgs, myhandle, result);

			buffer2Offset += buffer2ToCopy;
		}

		buffer1Offset += buffer1ToCopy;

	} while (buffer1Offset < noOfImgs);



	cudaFree(d_fftBuffer1);
	cudaFree(d_fftBuffer2);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );
	gpuErrchk( cudaPeekAtLastError() );
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
