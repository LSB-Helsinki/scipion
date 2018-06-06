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

void ProgMovieAlignmentCorrelation::loadData(const MetaData& movie,
		const Image<double>& dark, const Image<double>& gain,
		double targetOccupancy, const MultidimArray<double>& lpf) {
	MultidimArray<double> filter;
	FourierTransformer transformer;
	bool firstImage = true;
	int n = 0;
	FileName fnFrame;
	Image<double> frame, croppedFrame, reducedFrame;

	if (verbose)
	{
		std::cout << "Computing Fourier transform of frames ..." << std::endl;
		init_progress_bar(movie.size());
	}

	FOR_ALL_OBJECTS_IN_METADATA(movie)
	{
		if (n >= nfirst && n <= nlast) {
			movie.getValue(MDL_IMAGE, fnFrame, __iter.objId);
			if (yDRcorner == -1)
				croppedFrame.read(fnFrame);
			else {
				frame.read(fnFrame);
				frame().window(croppedFrame(), yLTcorner, xLTcorner, yDRcorner,
						xDRcorner);
			}
			if (XSIZE(dark()) > 0)
				croppedFrame() -= dark();
			if (XSIZE(gain()) > 0)
				croppedFrame() *= gain();
			// Reduce the size of the input frame
			scaleToSizeFourier(1, newYdim, newXdim, croppedFrame(),
					reducedFrame());

			// Now do the Fourier transform and filter
			MultidimArray<std::complex<double> > *reducedFrameFourier =
					new MultidimArray<std::complex<double> >;
			transformer.FourierTransform(reducedFrame(), *reducedFrameFourier,
					true);
			if (firstImage) {
				firstImage = false;
				filter.initZeros(*reducedFrameFourier);
				scaleLPF(lpf, newXdim, newYdim, targetOccupancy, filter);
			}
			for (size_t nn = 0; nn < filter.nzyxdim; ++nn) {
				double wlpf = DIRECT_MULTIDIM_ELEM(filter, nn);
					DIRECT_MULTIDIM_ELEM(*reducedFrameFourier,nn) *= wlpf;
			}
			frameFourier.push_back(reducedFrameFourier);
//			Image<double> Vout(newXdim, newYdim);
//			transformer.inverseFourierTransform(*reducedFrameFourier, Vout.data);
////			Vout.data = transformer.getReal();
//			Vout.write("filteredCroppedInput" + SSTR(n) + ".vol");
		}
		++n;
		if (verbose)
			progress_bar(n);



//		Image<double> bbb(frameFourier.at(0)->xdim, frameFourier.at(0)->ydim, 1, frameFourier.size());
//		size_t imgSize = frameFourier.at(0)->yxdim;
//		for (size_t img = 0; img < frameFourier.size();img++ ) {
//			for (size_t i = 0; i < ((size_t)frameFourier.at(0)->yxdim); i++) {
//				double d = frameFourier.at(img)->data[i].real();
//				if (d < 3) bbb.data[img*imgSize + i] = d;
//			}
//		}
//		bbb.write("fftFromCPU.vol");


	}
	if (verbose)
		progress_bar(movie.size());
}

void ProgMovieAlignmentCorrelation::computeShifts(size_t N,
		const Matrix1D<double>& bX, const Matrix1D<double>& bY,
		const Matrix2D<double>& A) {
	int idx = 0;
	MultidimArray<double> Mcorr;
	Mcorr.resizeNoCopy(newYdim, newXdim);
	Mcorr.setXmippOrigin();
	CorrelationAux aux;
	for (size_t i = 0; i < N - 1; ++i) {
		for (size_t j = i + 1; j < N; ++j) {
			bestShift(*frameFourier[i], *frameFourier[j], Mcorr, bX(idx),
					bY(idx), aux, NULL, maxShift);
			if (verbose)
				std::cerr << "Frame " << i + nfirst << " to Frame "
						<< j + nfirst << " -> (" << bX(idx) << "," << bY(idx)
						<< ")\n";
			for (int ij = i; ij < j; ij++)
				A(idx, ij) = 1;

			idx++;
		}
		delete frameFourier[i];
	}
}
