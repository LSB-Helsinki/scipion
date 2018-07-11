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

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::applyShiftsComputeAverage(
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
				croppedFrame() = reducedFrame();
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
					Matrix2D<T> tmp;
					translation2DMatrix(shift, tmp, true);
					transformer.initLazy(croppedFrame().xdim, croppedFrame().ydim);
					transformer.applyGeometry(this->BsplineOrder, shiftedFrame(), croppedFrame(), tmp, IS_INV, WRAP);
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
			}
		}
		j++;
	}
	n++;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::loadFrame(const MetaData& movie, size_t objId, bool crop, Image<T>& out) {
	FileName fnFrame;
	movie.getValue(MDL_IMAGE, fnFrame, objId);
	if (crop) {
		Image<T>tmp;
		tmp.read(fnFrame);
		tmp().window(out(), this->yLTcorner, this->xLTcorner, this->yDRcorner, this->xDRcorner);
	} else {
		out.read(fnFrame);
	}
}

template<typename T>
int ProgMovieAlignmentCorrelationGPU<T>::getMaxFilterSize(Image<T> &frame) {
	size_t maxXPow2 = std::ceil(log(frame.data.xdim) / log(2));
	size_t maxX = std::pow(2, maxXPow2);
	size_t maxFFTX = maxX / 2 + 1;
	size_t maxYPow2 = std::ceil(log(frame.data.ydim) / log(2));
	size_t maxY = std::pow(2, maxYPow2);
	size_t bytes = maxFFTX * maxY * sizeof(T);
	return bytes / (1024*1024);
}

template<typename T>
T* ProgMovieAlignmentCorrelationGPU<T>::loadToRAM(const MetaData& movie, int noOfImgs,
		const Image<T>& dark, const Image<T>& gain, bool cropInput) {
	T* imgs = new T[noOfImgs * inputOptSizeX * inputOptSizeY]();
	Image<T> frame, gainF, darkF;
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
		T* dest = imgs + ((movieImgIndex-this->nfirst) * inputOptSizeX * inputOptSizeY); // points to first float in the image
		for (size_t i = 0; i < frame.data.ydim; ++i) {
			memcpy(dest + (inputOptSizeX * i),
					frame.data.data + i*frame.data.xdim,
					frame.data.xdim * sizeof(T));
		}
	}
	return imgs;
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::setSizes(Image<T> frame,
		int noOfImgs) {
	// get best sizes
	int maxFilterSize = getMaxFilterSize(frame);
	if (this->verbose)
		std::cerr << "Benchmarking cuFFT ..." << std::endl; // FIXME add support for storing user data on drive

	size_t availableMemMB = getFreeMem(0); // FIXME pass device
	size_t noOfCorrelations = (noOfImgs * (noOfImgs-1)) / 2;
	correlationBufferSizeMB = availableMemMB / 3; // divide available memory to 3 parts (2 buffers + 1 FFT)

	// we also need enough memory for filter
	getBestSize(noOfImgs, frame.data.xdim, frame.data.ydim, inputOptBatchSize,
			inputOptSizeX, inputOptSizeY, maxFilterSize);

	inputOptSizeFFTX = inputOptSizeX / 2 + 1;
	if (this->verbose)
		printf("best FFT for input is %d images of %d x %d (%d)\n",
			inputOptBatchSize, inputOptSizeX, inputOptSizeY, inputOptSizeFFTX);


	if (this->verbose)
		printf("benchmarking for %lu imgs of %d x %d, %d of Memory (out of %lu)\n",
			noOfCorrelations, this->newXdim, this->newYdim, correlationBufferSizeMB * 2, availableMemMB);
	getBestSize(noOfCorrelations, this->newXdim, this->newYdim, croppedOptBatchSize,
			croppedOptSizeX, croppedOptSizeY, correlationBufferSizeMB * 2);

	croppedOptSizeFFTX = croppedOptSizeX / 2 + 1;
	if (this->verbose)
		printf("best FFT for cropped imgs is %d images of %d x %d (%d)\n",
			croppedOptBatchSize, croppedOptSizeX, croppedOptSizeY,
			croppedOptSizeFFTX);

	T corrSizeMB = ((size_t)croppedOptSizeFFTX * croppedOptSizeY * sizeof(std::complex<T>)) / (1024 * 1024.);
	correlationBufferImgs = std::ceil(correlationBufferSizeMB / corrSizeMB);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::loadData(const MetaData& movie,
		const Image<T>& dark, const Image<T>& gain,
		T targetOccupancy, const MultidimArray<T>& lpf) {
	// allocate space for data on CPU
	bool cropInput = (this->yDRcorner != -1);
	int noOfImgs = this->nlast - this->nfirst + 1;

	// get frame info
	Image<T> frame;
	loadFrame(movie, movie.firstObject(), cropInput, frame);
	setSizes(frame, noOfImgs);
	// prepare filter
	MultidimArray<T> filter;
	filter.initZeros(croppedOptSizeY, croppedOptSizeFFTX);
	this->scaleLPF(lpf, croppedOptSizeX, croppedOptSizeY, targetOccupancy, filter);
	T* d_filter = loadToGPU(filter.data, croppedOptSizeFFTX * croppedOptSizeY);

	// load all frames to RAM
	T* imgs = loadToRAM(movie, noOfImgs, dark, gain, cropInput);
	data = performFFTAndScale(imgs, noOfImgs, inputOptSizeX, inputOptSizeY, inputOptBatchSize, croppedOptSizeFFTX, croppedOptSizeY, d_filter);

	release(d_filter);
}

template<typename T>
void ProgMovieAlignmentCorrelationGPU<T>::computeShifts(size_t N,
		const Matrix1D<T>& bX, const Matrix1D<T>& bY,
		const Matrix2D<T>& A) {

	T* correlations;
	computeCorrelations(this->maxShift, N, data,
			croppedOptSizeFFTX, croppedOptSizeX, croppedOptSizeY, correlationBufferImgs,
			croppedOptBatchSize, correlations);

	int idx = 0;
	int resultSize = this->maxShift*2+1;
	MultidimArray<T> Mcorr (resultSize, resultSize);
	for (size_t i = 0; i < N - 1; ++i) {
		for (size_t j = i + 1; j < N; ++j) {
			size_t offset = idx * resultSize * resultSize;
			Mcorr.data = correlations + offset;
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
}

// explicit specialization
template class ProgMovieAlignmentCorrelationGPU<float>;
