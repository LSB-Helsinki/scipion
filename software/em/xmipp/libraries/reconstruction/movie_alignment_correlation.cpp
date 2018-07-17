/***************************************************************************
 *
 * Authors:    Carlos Oscar Sanchez Sorzano coss@cnb.csic.es
 *             David Strelak (davidstrelak@gmail.com)
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

#include "reconstruction/movie_alignment_correlation.h"

#include <sstream>

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

template<typename T>
void ProgMovieAlignmentCorrelation<T>::loadData(const MetaData& movie,
		const Image<T>& dark, const Image<T>& gain,
		T targetOccupancy, const MultidimArray<T>& lpf) {
	MultidimArray<T> filter;
	FourierTransformer transformer;
	bool firstImage = true;
	int n = 0;
	FileName fnFrame;
	Image<T> frame, croppedFrame, reducedFrame;

	if (this->verbose)
	{
		std::cout << "Computing Fourier transform of frames ..." << std::endl;
		init_progress_bar(movie.size());
	}

	FOR_ALL_OBJECTS_IN_METADATA(movie)
	{
		if (n >= this->nfirst && n <= this->nlast) {
			movie.getValue(MDL_IMAGE, fnFrame, __iter.objId);
			if (this->yDRcorner == -1)
				croppedFrame.read(fnFrame);
			else {
				frame.read(fnFrame);
				frame().window(croppedFrame(), this->yLTcorner, this->xLTcorner, this->yDRcorner,
						this->xDRcorner);
			}
			if (XSIZE(dark()) > 0)
				croppedFrame() -= dark();
			if (XSIZE(gain()) > 0)
				croppedFrame() *= gain();
			// Reduce the size of the input frame
			scaleToSizeFourier(1, this->newYdim, this->newXdim, croppedFrame(),
					reducedFrame());

			// Now do the Fourier transform and filter
			MultidimArray<std::complex<T> > *reducedFrameFourier =
					new MultidimArray<std::complex<T> >;
			transformer.FourierTransform(reducedFrame(), *reducedFrameFourier,
					true);
			if (firstImage) {
				firstImage = false;
				filter.initZeros(*reducedFrameFourier);
				this->scaleLPF(lpf, this->newXdim, this->newYdim, targetOccupancy, filter);
			}
			for (size_t nn = 0; nn < filter.nzyxdim; ++nn) {
				T wlpf = DIRECT_MULTIDIM_ELEM(filter, nn);
					DIRECT_MULTIDIM_ELEM(*reducedFrameFourier,nn) *= wlpf;
			}
			frameFourier.push_back(reducedFrameFourier);
//			Image<double> Vout(newXdim, newYdim);
//			transformer.inverseFourierTransform(*reducedFrameFourier, Vout.data);
////			Vout.data = transformer.getReal();
//			Vout.write("filteredCroppedInput" + SSTR(n) + ".vol");
		}
		++n;
		if (this->verbose)
			progress_bar(n);



		Image<double> bbb(frameFourier.at(0)->xdim, frameFourier.at(0)->ydim, 1, frameFourier.size());
		size_t imgSize = frameFourier.at(0)->yxdim;
		for (size_t img = 0; img < frameFourier.size();img++ ) {
			for (size_t i = 0; i < ((size_t)frameFourier.at(0)->yxdim); i++) {
				double d = frameFourier.at(img)->data[i].real();
				if (d < 3) bbb.data[img*imgSize + i] = d;
			}
		}
		bbb.write("fftFromCPU.vol");

		Image<T> tmp2(filter.xdim, filter.ydim);
		tmp2.data = filter;
		tmp2.write("filterCPU.vol");

	}
	if (this->verbose)
		progress_bar(movie.size());
}

template<typename T>
void ProgMovieAlignmentCorrelation<T>::computeShifts(size_t N,
		const Matrix1D<T>& bX, const Matrix1D<T>& bY,
		const Matrix2D<T>& A) {
	int idx = 0;
	MultidimArray<T> Mcorr;
	Mcorr.resizeNoCopy(this->newYdim, this->newXdim);
	Mcorr.setXmippOrigin();
	CorrelationAux aux;
	for (size_t i = 0; i < N - 1; ++i) {
		for (size_t j = i + 1; j < N; ++j) {
			bestShift(*frameFourier[i], *frameFourier[j], Mcorr, bX(idx),
					bY(idx), aux, NULL, this->maxShift);
			if (this->verbose)
				std::cerr << "Frame " << i + this->nfirst << " to Frame "
						<< j + this->nfirst << " -> (" << bX(idx) / this->sizeFactor
						<< "," << bY(idx) / this->sizeFactor
						<< ")\n";
			for (int ij = i; ij < j; ij++)
				A(idx, ij) = 1;

			idx++;
		}
		delete frameFourier[i];
	}
}

template<typename T>
void ProgMovieAlignmentCorrelation<T>::applyShiftsComputeAverage(
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
				scaleToSizeFourier(1, floor(YSIZE(croppedFrame()) / this->bin),
						floor(XSIZE(croppedFrame()) / this->bin), croppedFrame(),
						reducedFrame());
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
				if (this->outsideMode == OUTSIDE_WRAP)
					translate(this->BsplineOrder, shiftedFrame(), croppedFrame(),
							shift, WRAP);
				else if (this->outsideMode == OUTSIDE_VALUE)
					translate(this->BsplineOrder, shiftedFrame(), croppedFrame(),
							shift, DONT_WRAP, this->outsideValue);
				else
					translate(this->BsplineOrder, shiftedFrame(), croppedFrame(),
							shift, DONT_WRAP, croppedFrame().computeAvg());
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
			j++;
		}
		n++;
	}
}

template class ProgMovieAlignmentCorrelation<double>;
