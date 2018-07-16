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

#ifndef _PROG_MOVIE_ALIGNMENT_CORRELATION_BASE
#define _PROG_MOVIE_ALIGNMENT_CORRELATION_BASE

#include "data/xmipp_program.h"
#include "data/metadata_extension.h"
#include "data/xmipp_fftw.h"

/**@defgroup MovieAlignmentCorrelation Movie alignment by correlation
   @ingroup ReconsLibrary */
//@{

/** Movie alignment correlation Parameters. */
template<typename T>
class AProgMovieAlignmentCorrelation: public XmippProgram
{

#define OUTSIDE_WRAP 0
#define OUTSIDE_AVG 1
#define OUTSIDE_VALUE 2

public:
    /// Read argument from command line
    void readParams();

    /// Show
    void show();

    /// Define parameters
    void defineParams();

    /// Run
    void run();

protected:
    void scaleLPF(const MultidimArray<T>& lpf, int xSize, int ySize, T targetOccupancy, MultidimArray<T>& result);

private:
    virtual void loadData(const MetaData& movie, const Image<T>& dark,
    			const Image<T>& gain,
    			T targetOccupancy,
    			const MultidimArray<T>& lpf) = 0;

    virtual void computeShifts(size_t N, const Matrix1D<T>& bX,
    			const Matrix1D<T>& bY, const Matrix2D<T>& A) = 0;
	virtual void applyShiftsComputeAverage(const MetaData& movie,
			const Image<T>& dark, const Image<T>& gain,
			Image<T>& initialMic, size_t& Ninitial,
			Image<T>& averageMicrograph, size_t& N) = 0;

private:
	int findReferenceImage(size_t N, const Matrix1D<T>& shiftX,
			const Matrix1D<T>& shiftY);

	void solveEquationSystem(Matrix1D<T>& bX, Matrix1D<T>& bY,
			Matrix2D<T>& A, Matrix1D<T>& shiftX,
			Matrix1D<T>& shiftY);
	void loadDarkCorrection(Image<T>& dark);
	void loadGainCorrection(Image<T>& gain);

	void constructLPF(T targetOccupancy, const MultidimArray<T>& lpf);
	void setNewDimensions(T& targetOccupancy, const MetaData& movie,
			T& sizeFactor);
	void readMovie(MetaData& movie);
	void storeRelativeShifts(int bestIref, const Matrix1D<T>& shiftX,
			const Matrix1D<T>& shiftY, T sizeFactor, MetaData& movie);
	void setZeroShift(MetaData& movie);
	int findShiftsAndStore(MetaData& movie, Image<T>& dark,
			Image<T>& gain);
	void storeResults(Image<T>& initialMic, size_t Ninitial,
			Image<T>& averageMicrograph, size_t N, const MetaData& movie,
			int bestIref);
	void correctLoopIndices(const MetaData& movie);
	void computeTotalShift(int iref, int j, const Matrix1D<T> &shiftX, const Matrix1D<T> &shiftY,
	                       T &totalShiftX, T &totalShiftY);

protected:
	// Target size of the frames
	int newXdim, newYdim;
	/** First and last frame (inclusive)*/
	int nfirst, nlast;
	/** Max shift */
	T maxShift;
	/*****************************/
	/** crop corner **/
	/*****************************/
	/** x left top corner **/
	int xLTcorner;
	/** y left top corner **/
	int yLTcorner;
	/** x right down corner **/
	int xDRcorner;
	/** y right down corner **/
	int yDRcorner;
	/** Aligned movie */
	FileName fnAligned;
	/** Aligned micrograph */
	FileName fnAvg;
	/** First and last frame*/
	int nfirstSum, nlastSum;
	/** Aligned micrograph */
	FileName fnInitialAvg;
	/** Binning factor */
	T bin;
	/** Bspline order */
	int BsplineOrder;
	/** Outside mode */
	int outsideMode;
	/** Outside value */
	T outsideValue;
	T sizeFactor;

private:
	// Target sampling rate
	T newTs;
	 /** Filename of movie metadata */
	FileName fnMovie;
	/** Correction images */
	FileName fnDark, fnGain;
	/** Sampling rate */
	T Ts;
	/** Max freq. */
	T maxFreq;
	/** Solver iterations */
	int solverIterations;
	/** Metadata with shifts */
	FileName fnOut;
	/** Do not calculate and use the input shifts */
	bool useInputShifts;

};
//@}
#endif
