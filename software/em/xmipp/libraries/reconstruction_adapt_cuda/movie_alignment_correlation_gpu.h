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

#ifndef MOVIE_ALIGNMENT_CORRELATION_GPU
#define MOVIE_ALIGNMENT_CORRELATION_GPU

#include "reconstruction/movie_alignment_correlation_base.h"
#include "reconstruction_cuda/cuda_gpu_movie_alignment_correlation.h"
#include "reconstruction_cuda/cuda_gpu_geo_transformer.h"
#include "data/filters.h"

template<typename T>
class ProgMovieAlignmentCorrelationGPU: public AProgMovieAlignmentCorrelation<T>
{
private:
	void loadData(const MetaData& movie, const Image<T>& dark,
			const Image<T>& gain,
			T targetOccupancy,
			const MultidimArray<T>& lpf);

	void computeShifts(size_t N, const Matrix1D<T>& bX,
			const Matrix1D<T>& bY, const Matrix2D<T>& A);

	T* loadToRAM(const MetaData& movie, int noOfImgs,
			const Image<T>& dark, const Image<T>& gain, bool cropInput);

	void applyShiftsComputeAverage(
				const MetaData& movie, const Image<T>& dark,
				const Image<T>& gain, Image<T>& initialMic,
				size_t& Ninitial, Image<T>& averageMicrograph, size_t& N);
	void loadFrame(const MetaData& movie, size_t objId, bool crop, Image<T>& out);
	void setSizes(Image<T> frame, int noOfImgs);

	int getMaxFilterSize(Image<T> &frame);

private:
	// downscaled Fourier transforms of the input images
	std::complex<T>* data;

	int inputOptSizeX;
	int inputOptSizeY;
	int inputOptSizeFFTX;
	int inputOptBatchSize;

	int croppedOptSizeX;
	int croppedOptSizeY;
	int croppedOptSizeFFTX;
	int croppedOptBatchSize;

	int correlationBufferSizeMB;
	int correlationBufferImgs;
};


#endif /* MOVIE_ALIGNMENT_CORRELATION_GPU */
