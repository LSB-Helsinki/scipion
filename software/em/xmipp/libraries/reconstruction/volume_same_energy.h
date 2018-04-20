/***************************************************************************
 *
 * Authors:    Jose Luis Vilas, 					  jlvilas@cnb.csic.es
 * 			   Carlos Oscar S. Sorzano            coss@cnb.csic.es (2016)
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

#ifndef _PROG_SAME_ENERGY
#define _PROG_SAME_ENERGY

#include <iostream>
#include <data/xmipp_program.h>
#include <data/xmipp_image.h>
#include <data/metadata.h>
#include <data/xmipp_fft.h>
#include <data/xmipp_fftw.h>
#include <math.h>
#include <limits>
#include <complex>
#include "fourier_filter.h"
#include <data/filters.h>
#include <string>
#include "symmetrize.h"

/**@defgroup same Energy
   @ingroup ReconsLibrary */
//@{
/** SSNR parameters. */

class ProgSameEnergy : public XmippProgram
{
public:
	 /** Filenames */
	FileName fnOut, fnVol, fnRes;

	/** sampling rate, minimum resolution, and maximum resolution */
	double sampling, maxRes, minRes, maxFreq, minFreq;
	int Nthread;

public:

    void defineParams();
    void readParams();
    void produceSideInfo();

    /* Mogonogenid amplitud of a volume, given an input volume,
     * the monogenic amplitud is calculated and low pass filtered at frequency w1*/
    void lowPassFilterFunction(const MultidimArray< std::complex<double> > &myfftV,
    		double w, double wL, MultidimArray<double> &filteredVol, int count);

    void bandPassFilterFunction(const MultidimArray< std::complex<double> > &myfftV,
    		double w, double wL, MultidimArray<double> &filteredVol, int count);

      void maxMinResolution(MultidimArray<double> &resVol,
			double &maxRes, double &minRes);

    void localfiltering(MultidimArray< std::complex<double> > &myfftV,
    										MultidimArray<double> &localfilteredVol,
    										double &minFreq, double &maxFreq, double &step);

    void amplitudeMonogenicSignalBP(MultidimArray< std::complex<double> > &myfftV,
    		double w, double step, int count, MultidimArray<double> &bpVol);

    void sameEnergy(MultidimArray<double> Vorig,
			double &minFreq, double &maxFreq, double &step, MultidimArray<double> &bpVol);

    void run();

public:
    CDF cdfS;
    std::vector<int> idxList;
    MultidimArray<int> idxVol, mask;
    MultidimArray<double> Vorig;//, VsoftMask;
    MultidimArray<double> resVol;
    MultidimArray<double> iu, sharpenedMap; // Inverse of the frequency
	MultidimArray< std::complex<double> > fftV, fftVfilter; // Fourier transform of the input volume
	FourierTransformer transformer_inv, transformer;
	FourierFilter FilterBand;
	Image<double> Vfiltered, VresolutionFiltered;
};
//@}
#endif
