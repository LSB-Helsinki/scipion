/***************************************************************************
 *
 * Authors:    David Strelak (davidstrelak@gmail.com)
 *
 * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
 * 02111-1307  USA
 *
 *  All comments concerning this program package may be sent to the
 *  e-mail address 'xmipp@cnb.csic.es'
 ***************************************************************************/

#include "reconstruction_adapt_cuda/movie_alignment_correlation_gpu.h"


// FIXME: REMOVE
#include <sstream>
#include <set>
#include "data/filters.h"
#include "data/xmipp_fftw.h"
#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

const int primes[4] = {2, 3, 5, 7};

int intpow(int x, int p) {
  if (p == 0) return 1;
  if (p == 1) return x;
  return x * intpow(x, p-1);
}

int findNext2357Multiple(int num) {
	printf("Computing padding for size %i\n", num);

	int N = num * 2;

	int length = (int) (log2(N) + 0.5) + 1;
	int primepowers[4][length];

	for (int i = 0; i < 4; i++)
		for (int j = 1; j < length; j++) {
			int power = intpow(primes[i], j);
			if (power < N)
				primepowers[i][j] = power;
			else
				primepowers[i][j] = 1;
		}

	std::set<int> goodnumbers;
	for (int a = 0; a < length; a++)
		for (int b = 0; b < length; b++)
			for (int c = 0; c < length; c++)
				for (int d = 0; d < length; d++)
					/* mask < 2: only 2^a,
					 mask < 4: 2^a * 3^b
					 mask < 8: 2^a * 3^b * 5^c
					 mask < 16: 2^a * 3^b * 5^c * 7^d */
					for (int mask = 1; mask < 16; mask++) {
						int mul = ((mask & 1) ? primepowers[0][a] : 1)
								* ((mask & 2) ? primepowers[1][b] : 1)
								* ((mask & 4) ? primepowers[2][c] : 1)
								* ((mask & 8) ? primepowers[3][d] : 1);
	                        if (mul <= N && mul > 0) /* overflow protection */
	                            goodnumbers.insert(mul);
//						if (mul >= num)
//							return mul;
					}
	for (std::set<int>::iterator i = goodnumbers.begin(); i != goodnumbers.end(); i++)
	        if (*i >= num) return *i;


	return 0;
}

template<typename T>
void __attribute__((optimize("O0"))) ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
		const MetaData& movie, const Image<T>& dark,
		const Image<T>& gain, Image<T>& initialMic,
		size_t& Ninitial, Image<T>& averageMicrograph, size_t& N) {
	// Apply shifts and compute average
	Image<T> frame, croppedFrame, reducedFrame, shiftedFrame;
	Matrix1D<T> shift(2);
	FileName fnFrame;
	int j = 0;
	int n = 0;
	Ninitial = N = 0;
	GeoTransformer<T> transformer;
	FOR_ALL_OBJECTS_IN_METADATA(movie)
	{
		if (n >= this->nfirstSum && n <= this->nlastSum) {
			movie.getValue(MDL_IMAGE, fnFrame, __iter.objId);
			movie.getValue(MDL_SHIFT_X, XX(shift), __iter.objId);
			movie.getValue(MDL_SHIFT_Y, YY(shift), __iter.objId);

			std::cout << fnFrame << " shiftX=" << XX(shift) << " shiftY="
					<< YY(shift) << std::endl;
			clock_t begin = clock();
			frame.read(fnFrame);
			if (XSIZE(dark()) > 0)
				frame() -= dark();
			if (XSIZE(gain()) > 0)
				frame() *= gain();
			if (this->yDRcorner != -1)
				frame().window(croppedFrame(), this->yLTcorner, this->xLTcorner, this->yDRcorner,
						this->xDRcorner);
			else
				croppedFrame() = frame();
			if (this->bin > 0) {
				// FIXME add templates to respective functions/classes to avoid type casting
				Image<double>croppedFrameDouble;
				Image<double>reducedFrameDouble;
				typeCast(croppedFrame(), croppedFrameDouble());

				scaleToSizeFourier(1, floor(YSIZE(croppedFrame()) / this->bin),
						floor(XSIZE(croppedFrame()) / this->bin), croppedFrameDouble(),
						reducedFrameDouble());

				typeCast(reducedFrameDouble(), reducedFrame());

				shift /= this->bin;
				croppedFrame() = reducedFrame(); // FIXME what is this supposed to do?
			}

			if (this->fnInitialAvg != "") {
				if (j == 0)
					initialMic() = croppedFrame();
				else
					initialMic() += croppedFrame();
				Ninitial++;
			}

			if (this->fnAligned != "" || this->fnAvg != "") {
				if (this->outsideMode == OUTSIDE_WRAP) {
	//					translate(BsplineOrder, shiftedFrame(), croppedFrame(),
	//							shift, WRAP);
					Matrix2D<float> tmp;
//
//					XX(shift) = 0.1;
//					YY(shift) = 0.2;
//
//					shiftedFrame = Image<double>();
//					croppedFrame = Image<double>(5, 10);
//					for (int i = 0; i < croppedFrame.data.yxdim; i++) {
//						croppedFrame.data.data[i] = i+20;
//					}
//					translation2DMatrix(shift, tmp, true);
//					applyGeometryGPU(BsplineOrder, shiftedFrame(), croppedFrame(), tmp, IS_INV, WRAP, 0.);
//

					translation2DMatrix(shift, tmp, true);
					printf("pred applyGeomtery %f\n", ((float)clock()-begin)/CLOCKS_PER_SEC);
					begin = clock();
					transformer.initLazy(croppedFrame().xdim, croppedFrame().ydim);
					transformer.applyGeometry(this->BsplineOrder, shiftedFrame(), croppedFrame(), tmp, IS_INV, WRAP);
					printf("applyGeomtery %f\n", ((float)clock()-begin)/CLOCKS_PER_SEC);
					begin = clock();
				}
				else if (this->outsideMode == OUTSIDE_VALUE)
					translate(this->BsplineOrder, shiftedFrame(), croppedFrame(),
							shift, DONT_WRAP, this->outsideValue);
				else
					translate(this->BsplineOrder, shiftedFrame(), croppedFrame(),
							shift, DONT_WRAP, (T)croppedFrame().computeAvg());
				if (this->fnAligned != "")
					shiftedFrame.write(this->fnAligned, j + 1, true, WRITE_REPLACE);
				if (this->fnAvg != "") {
					if (j == 0)
						averageMicrograph() = shiftedFrame();
					else
						averageMicrograph() += shiftedFrame();
					N++;
				}
				printf("po averageMicrograph %f\n", ((float)clock()-begin)/CLOCKS_PER_SEC);
			}
		}
		j++;
	}
	n++;
}


// FIXME move to parent, use in store function?
template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::loadFrame(const MetaData& movie, size_t objId, bool crop, Image<T>& out) {
	FileName fnFrame;
	movie.getValue(MDL_IMAGE, fnFrame, objId);
	if (crop) {
		Image<double>tmp;
		tmp.read(fnFrame);
		tmp().window(out(), this->yLTcorner, this->xLTcorner, this->yDRcorner, this->xDRcorner);
	} else {
		out.read(fnFrame);
	}
}

int getMaxFilterSize(Image<float>& frame) { // FIXME put to header
	size_t maxXPow2 = std::ceil(log(frame.data.xdim) / log(2));
	size_t maxX = std::pow(2, maxXPow2);
	size_t maxFFTX = maxX / 2 + 1;
	size_t maxYPow2 = std::ceil(log(frame.data.ydim) / log(2));
	size_t maxY = std::pow(2, maxYPow2);
	size_t bytes = maxFFTX * maxY * sizeof(float);
	return bytes / (1024*1024);
}

template<typename T>
float* ProgMovieAlignmentCorrelationGPU<T>::loadToRAM(const MetaData& movie, int noOfImgs,
		const Image<T>& dark, const Image<T>& gain, bool cropInput) {
	float* imgs = new float[noOfImgs * inputOptSizeX * inputOptSizeY]();
	Image<float> frame, gainF, darkF;
	// copy image correction data, convert to float
	gainF.data.resize(gain(), true);
	darkF.data.resize(dark(), true);

	int movieImgIndex = -1;
	FOR_ALL_OBJECTS_IN_METADATA(movie) {
		// update variables
		movieImgIndex++;
		if (movieImgIndex < this->nfirst ) continue;
		if (movieImgIndex > this->nlast) break;

		// load image
		loadFrame(movie, __iter.objId, cropInput, frame);
		if (XSIZE(darkF()) > 0)
			frame() -= darkF();
		if (XSIZE(gainF()) > 0)
			frame() *= gainF();

		// copy line by line, adding offset at the end of each line
		// result is the same image, padded in the X and Y dimensions
		float* dest = imgs + ((movieImgIndex-this->nfirst) * inputOptSizeX * inputOptSizeY); // points to first float in the image
		for (size_t i = 0; i < frame.data.ydim; ++i) {
			memcpy(dest + (inputOptSizeX * i),
					frame.data.data + i*frame.data.xdim,
					frame.data.xdim * sizeof(float));
		}
	}

	//	Image<float> aaaa(inputOptSizeX, inputOptSizeY, 1, noOfImgs);
	//	aaaa.data.data = imgs;
	//	aaaa.write("images.vol");

	return imgs;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::setSizes(Image<T> frame,
		int noOfImgs) {
	// get best sizes
	int maxFilterSize = getMaxFilterSize(frame);
	if (this->verbose)
		std::cerr << "Benchmarking cuFFT ..." << std::endl;


	size_t availableMemMB = getFreeMem(0); // FIXME pass device
	size_t noOfCorrelations = (noOfImgs * (noOfImgs-1)) / 2;
	correlationBufferSizeMB = availableMemMB / 3; // divide available memory to 3 parts (2 buffers + 1 FFT)

	// we also need enough memory for filter
//	getBestSize(noOfImgs, frame.data.xdim, frame.data.ydim, inputOptBatchSize,
//			inputOptSizeX, inputOptSizeY, maxFilterSize); // FIXME uncomment
	inputOptBatchSize = 7;
	inputOptSizeX = 4096;
	inputOptSizeY = 4096;

	inputOptSizeFFTX = inputOptSizeX / 2 + 1;
	printf("best FFT for input is %d images of %d x %d (%d)\n",
			inputOptBatchSize, inputOptSizeX, inputOptSizeY, inputOptSizeFFTX);


	printf("benchmarking for %d imgs of %d x %d, %d of Memory (out of %d)\n",
			noOfCorrelations, this->newXdim, this->newYdim, correlationBufferSizeMB * 2, availableMemMB);
//	getBestSize(noOfCorrelations, newXdim, newYdim, croppedOptBatchSize,
//			croppedOptSizeX, croppedOptSizeY, correlationBufferSizeMB * 2); // FIXME uncomment
	croppedOptBatchSize = 15;
	croppedOptSizeX =  2304;
	croppedOptSizeY = 2304;

	croppedOptSizeFFTX = croppedOptSizeX / 2 + 1;
	printf("best FFT for cropped imgs is %d images of %d x %d (%d)\n",
			croppedOptBatchSize, croppedOptSizeX, croppedOptSizeY,
			croppedOptSizeFFTX);

	float corrSizeMB = ((size_t)croppedOptSizeFFTX * croppedOptSizeY * sizeof(std::complex<float>)) / 1048576.f;
	correlationBufferImgs = std::ceil(correlationBufferSizeMB / corrSizeMB);

	printf("one correlation %f MB, buffer (%d) will hold %d imgs\n", corrSizeMB, correlationBufferSizeMB, correlationBufferImgs);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::loadData(const MetaData& movie,
		const Image<T>& dark, const Image<T>& gain,
		T targetOccupancy, const MultidimArray<T>& lpf) {
	// allocate space for data on CPU
	bool cropInput = (this->yDRcorner != -1);
	int noOfImgs = this->nlast - this->nfirst + 1;

	// get frame info
	Image<float> frame;
	loadFrame(movie, movie.firstObject(), cropInput, frame);
	setSizes(frame, noOfImgs);
	// prepare filter
	MultidimArray<float> filter;
	filter.initZeros(croppedOptSizeY, croppedOptSizeFFTX);
	this->scaleLPF(lpf, croppedOptSizeX, croppedOptSizeY, targetOccupancy, filter);
	float* d_filter = loadToGPU(filter.data, croppedOptSizeFFTX * croppedOptSizeY);

	// load all frames to RAM
	float* imgs = loadToRAM(movie, noOfImgs, dark, gain, cropInput);
	tmpResult = performFFTAndScale(imgs, noOfImgs, inputOptSizeX, inputOptSizeY, inputOptBatchSize, croppedOptSizeFFTX, croppedOptSizeY, d_filter);


//	float* imgsToProcess = imgs;
//	float* imgsEnd = imgs + noOfImgs * inputOptSizeX * inputOptSizeY;
//	std::complex<float>* result = scaledFFTs;
//	while (imgsToProcess != imgsEnd) {
//		processInput(imgsToProcess, inputOptSizeX, inputOptSizeY, inputOptBatchSize, croppedOptSizeX, croppedOptSizeY, d_filter, result);
//		result += croppedOptSizeFFTX * croppedOptSizeY * inputOptBatchSize;
//		imgsToProcess = std::min(imgsEnd, imgsToProcess + inputOptSizeX * inputOptSizeY * inputOptBatchSize);
//	}
//	delete[] imgs;
	release(d_filter);
//
//	printf("hotovo\n");
//	fflush(stdout);
//	Image<double> bbb(croppedOptSizeFFTX, croppedOptSizeY, 1, noOfImgs);
//	for (size_t i = 0; i < ((size_t)croppedOptSizeFFTX * croppedOptSizeY * noOfImgs); i++) {
//		double d = tmpResult[i].real() / (frame.data.xdim*frame.data.ydim);
//		if (d < 3) bbb.data[i] = d;
//	}
//	bbb.write("fftFromGPU_nove.vol");
//	printf("juchuuu\n");
//	fflush(stdout);


//
//	return;


//	float* result;
//	size_t newFFTXDim = newXdim/2+1;
//	kernel1(imgs, frame.data.xdim, frame.data.ydim, noOfImgs, newXdim, newYdim, filter.data, tmpResult);
// 	******************
//	FIXME normalization has to be done using original img size, i.e frame.data.xdim*frame.data.ydim
//	******************

//	MultidimArray<std::complex<double> > V(1, 1, newYdim, newFFTXDim);
//	for (size_t i = 0; i < (newFFTXDim*newYdim); i++) {
//		V.data[i].real() = tmpResult[i].real() / (frame.data.xdim*frame.data.ydim);
//		V.data[i].imag() = tmpResult[i].imag() / (frame.data.xdim*frame.data.ydim);
//	}
//	Image<double> aaa(newFFTXDim, newYdim, 1, noOfImgs);
//	for (size_t i = 0; i < (newFFTXDim*newYdim*noOfImgs); i++) {
//		double d = tmpResult[i].real() / (frame.data.xdim*frame.data.ydim);
//		if (d < 3) aaa.data[i] = d;
//	}
//	aaa.write("fftFromGPU.vol");
//	std::cout << "normalization done" << std::endl;
//	Image<double> yyy (newXdim, newYdim, 1, 1);
//	FourierTransformer transformer;
//	std::cout << "about to do IFFT" << std::endl;
//	transformer.inverseFourierTransform(V, yyy.data);
//	std::cout << "IFFT done" << std::endl;
//	yyy.write("filteredCroppedInputGPU0.vol");


	// 16785408 X:2049 Y:4096
//	Image<float> tmp(newFFTXDim, newYdim, 1, noOfImgs);
//	for (size_t i = 0; i < (newFFTXDim*newYdim*2); i++) {
////	for (size_t i = 0; i < 8388608L; i++) {
//		float val = result[i].real() / (newYdim*newYdim);
//		if (val < 3) tmp.data[i] = val;
//		else std::cout << "skipping " << val << " at position " << i << std::endl;
//
//	}
//	tmp.write("fftFromGPU" + SSTR(counter) + ".vol");

}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::computeShifts(size_t N,
		const Matrix1D<T>& bX, const Matrix1D<T>& bY,
		const Matrix2D<T>& A) {

	float* correlations;
	computeCorrelations(this->maxShift, N, tmpResult,
			croppedOptSizeFFTX, croppedOptSizeX, croppedOptSizeY, correlationBufferImgs,
			croppedOptBatchSize, correlations);

	int resultSize = this->maxShift*2+1;

	int noOfCorrelations = (N * (N-1))/2;
	Image<float> imgs(resultSize, resultSize, 1, noOfCorrelations);
	imgs.data.data = correlations;
	imgs.write("correlationIFFTGPU_nove.vol");



//return;

//	float* result1;
//	std::complex<float>* result2;
//	kernel3(maxShift, N, tmpResult, newXdim/2+1, newYdim, result1, result2);
//	std::cout << "kernel3 done" << std::endl;
//	size_t framexdim = 4096;
//	size_t frameydim = 4096; // FIXME

//	size_t newFFTXDim = newXdim/2+1;
//	int noOfCorrelations = (N * (N-1)/2);
//	Image<float> ffts(newFFTXDim, newYdim, 1, noOfCorrelations);
//	for (size_t i = 0; i < newFFTXDim*newYdim*noOfCorrelations; i++) {
//		double d = result2[i].real() / (framexdim*frameydim*newFFTXDim*newYdim);
//		if (std::abs(d) < 3) ffts.data[i] = d;
//	}
//	ffts.write("correlationFFTGPU.vol");

	int idx = 0;
	MultidimArray<float> Mcorr (resultSize, resultSize);
	for (size_t i = 0; i < N - 1; ++i) {
		for (size_t j = i + 1; j < N; ++j) {
			size_t offset = idx * resultSize * resultSize;
//			for (size_t t = 0; t < croppedOptSizeX * croppedOptSizeY; t++) {
//				Mcorr.data[t] = correlations[offset + t] / (croppedOptSizeX * croppedOptSizeY);
//			}
			Mcorr.data = correlations + offset;
//			CenterFFT(Mcorr, true);
			Mcorr.setXmippOrigin();
			bestShift(Mcorr, bX(idx), bY(idx), NULL, this->maxShift);
			if (this->verbose)
				std::cerr << "Frame " << i + this->nfirst << " to Frame "
						<< j + this->nfirst << " -> (" << bX(idx) << "," << bY(idx)
						<< ")\n";
			for (int ij = i; ij < j; ij++)
				A(idx, ij) = 1;

			idx++;
		}
	}
	Mcorr.data = NULL;

//	for (int img = 0; img < (N * (N-1)/2); img++) {
//		MultidimArray<std::complex<double> > V(1, 1, newYdim, newFFTXDim);
//		for (size_t i = 0; i < (newFFTXDim*newYdim); i++) {
//			V.data[i].real() = result[i + img*newYdim*newFFTXDim].real() / (framexdim*frameydim);
//			V.data[i].imag() = result[i + img*newYdim*newFFTXDim].imag() / (framexdim*frameydim);
//		}
//		std::cout << "V done" << std::endl;
//		Image<double> aaa(newFFTXDim, newYdim, 1, 1);
//		for (size_t i = 0; i < (newFFTXDim*newYdim); i++) {
//			double d = result[i + img*newYdim*newFFTXDim].real() / (framexdim*frameydim);
//			if (d < 3) aaa.data[i] = d;
//		}
//		aaa.write("correlationGPU" + SSTR(img) + ".vol");
//		std::cout << "correlation done" << std::endl;
//		Image<double> yyy (newXdim, newYdim, 1, 1);
//		FourierTransformer transformer;
//		std::cout << "about to do IFFT" << std::endl;
//		transformer.inverseFourierTransform(V, yyy.data);
//		std::cout << "IFFT done" << std::endl;
//		CenterFFT(yyy.data, true);
//		yyy.write("correlationIFFTGPU" + SSTR(img) + ".vol");
//		Image<float>tmp(newXdim, newYdim, 1, noOfCorrelations);
//		tmp.data.data = result1;
////		CenterFFT(tmp.data, true);
//		tmp.write("correlationIFFTGPU.vol");
//	}


	return;
	// FIXME refactor

//	int idx = 0;
//	MultidimArray<double> Mcorr;
//	Mcorr.resizeNoCopy(newYdim, newXdim);
//	Mcorr.setXmippOrigin();
//	CorrelationAux aux;
//	for (size_t i = 0; i < N - 1; ++i) {
//		for (size_t j = i + 1; j < N; ++j) {
//			bestShift(*frameFourier[i], *frameFourier[j], Mcorr, bX(idx),
//					bY(idx), aux, NULL, maxShift);
//			if (verbose)
//				std::cerr << "Frame " << i + nfirst << " to Frame "
//						<< j + nfirst << " -> (" << bX(idx) << "," << bY(idx)
//						<< ")\n";
//			for (int ij = i; ij < j; ij++)
//				A(idx, ij) = 1;
//
//			idx++;
//		}
//		delete frameFourier[i];
//	}
}

template class ProgMovieAlignmentCorrelationGPU<float>;
