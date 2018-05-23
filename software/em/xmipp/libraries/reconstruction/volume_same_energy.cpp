/***************************************************************************
 *
 * Authors:    Erney Ramirez , 				     	  eramirez@cnb.csic.es
 * 			   Jose Luis Vilas                        jlvilas@cnb.csic.es
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

#include "volume_same_energy.h"
//#define DEBUG
//#define DEBUG_MASK
//#define DEBUG_FILTER
void ProgSameEnergy::readParams()
{
	fnVol = getParam("--vol");
	fnRes = getParam("--resolution_map");
	sampling = getDoubleParam("--sampling");
    Nthread = getIntParam("-n");
	fnOut = getParam("-o");
}

void ProgSameEnergy::defineParams()
{
	addUsageLine("This function performs same contrast");
	addParamsLine("  --vol <vol_file=\"\">   : Input volume");
	addParamsLine("  --resolution_map <vol_file=\"\">: Resolution map");
	addParamsLine("  -o <output=\"Sharpening.vol\">: sharpening volume");
	addParamsLine("  --sampling <s=1>: sampling");
    addParamsLine("  [-n <Nthread=1>]: threads number");
}

void ProgSameEnergy::produceSideInfo()
{
	std::cout << "Starting..." << std::endl;
	Image<double> V;
    V.read(fnVol);
    V().setXmippOrigin();

    if (Nthread>1)
    {
       std::cout << "used procesors = " << Nthread << std::endl;
       transformer_inv.setThreadsNumber(Nthread);
       transformer.setThreadsNumber(Nthread);
    }

	MultidimArray<double> &inputVol = V();

	Vorig = inputVol;

	transformer.FourierTransform(inputVol, fftV);

	iu.initZeros(fftV);
	double uz, uy, ux, uz2, u2, uz2y2;
	long n=0;
	for(size_t k=0; k<ZSIZE(fftV); ++k)
	{
		FFT_IDX2DIGFREQ(k,ZSIZE(inputVol),uz);
		uz2=uz*uz;

		for(size_t i=0; i<YSIZE(fftV); ++i)
		{
			FFT_IDX2DIGFREQ(i,YSIZE(inputVol),uy);
			uz2y2=uz2+uy*uy;

			for(size_t j=0; j<XSIZE(fftV); ++j)
			{
				FFT_IDX2DIGFREQ(j,XSIZE(inputVol),ux);
				u2=uz2y2+ux*ux;
				if ((k != 0) || (i != 0) || (j != 0))
					DIRECT_MULTIDIM_ELEM(iu,n) =1/sqrt(u2);
				else
					DIRECT_MULTIDIM_ELEM(iu,n) = 1e38;
				++n;
			}
		}
	}

    Image<double> resolutionVolume;
    resolutionVolume.read(fnRes);
    resVol = resolutionVolume();

    mask.initZeros(resVol);

	resVol.setXmippOrigin();

	maxMinResolution(resVol, maxRes, minRes);
	std::cout << "minRes = " << minRes << "  maxRes = " << maxRes << std::endl;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resVol)
	{

		if ((DIRECT_MULTIDIM_ELEM(resVol, n) < 2*sampling) && (DIRECT_MULTIDIM_ELEM(resVol, n)>0))
		{
			DIRECT_MULTIDIM_ELEM(resVol, n) = 2*sampling;
		}
	}

	resVol.setXmippOrigin();

//	FourierFilter Filter;
//	Filter.FilterShape=REALGAUSSIAN;
//	Filter.FilterBand=LOWPASS;
//	Filter.w1=1;
//	Filter.apply(resVol);

}

void ProgSameEnergy::maxMinResolution(MultidimArray<double> &resVol,
										double &maxRes, double &minRes)
{
	// Count number of voxels with resolution
	size_t n=0;
	double lastMinRes=1e38, lastMaxRes=1e-38, value;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resVol)
	{
		value = DIRECT_MULTIDIM_ELEM(resVol, n);

		if (value>lastMaxRes)
			lastMaxRes = value;
		if (value<lastMinRes && value>0)
			lastMinRes = value;
	}

	maxRes = lastMaxRes;
	minRes = lastMinRes;
}


void ProgSameEnergy::sameEnergy(MultidimArray<double> Vorig,
									double &minFreq, double &maxFreq, double &step,
									MultidimArray<double> &finalbpVol)
{
	finalbpVol.initZeros(Vorig);
	MultidimArray<double> bpVol;
	bpVol.initZeros(Vorig);

	FourierTransformer transformer;
	MultidimArray<double> auxVol;
	auxVol = Vorig;
	transformer.FourierTransform(auxVol, fftV);


	double lastfreq=1e38;
	int idx=0, lastidx = -1;


	for (double freq = minFreq; freq<=maxFreq; freq+=step)
	{
		double res = sampling/freq;

        DIGFREQ2FFT_IDX(freq+step, ZSIZE(fftV), idx);

        if (idx == lastidx)
        {
            std::cout << "idx--------------------------- = " << idx << std::endl;
                continue;
        }
//        if ((fabs(lastfreq-freq))<(step))
//        {
//        	std::cout << "res = " << res << "  sampling/step = " << sampling/step << std::endl;
//           	std::cout << "lastfreq = " << lastfreq << "  freq = " << freq << "  step = " << step << std::endl;
//                continue;
//        }

		std::cout << "Resol= " << res << " freq = " << freq <<  " freq+step = " << freq+step << " idx " << idx << std::endl;

		amplitudeMonogenicSignalBP(fftV, freq, step, idx, bpVol);
		std::cout << " ====== " << std::endl;

		finalbpVol += bpVol;

		lastfreq = freq;
        lastidx = idx;
	}
}

void ProgSameEnergy::amplitudeMonogenicSignalBP(MultidimArray< std::complex<double> > &myfftV,
		double w, double step, int count, MultidimArray<double> &bpVol)
{
	double u;
	Matrix1D<double> freq_fourier;
	int size_fourier = ZSIZE(myfftV);
	freq_fourier.initZeros(size_fourier);

	int size = ZSIZE(Vorig);

	VEC_ELEM(freq_fourier,0) = 1e-38;
	for(size_t k=0; k<size_fourier; ++k)
	{
		FFT_IDX2DIGFREQ(k,size, u);
		VEC_ELEM(freq_fourier,k) = u;
	}

	MultidimArray< std::complex<double> > fftVRiesz, fftVRiesz_aux;
	MultidimArray<double> VRiesz, amplitude;
	MultidimArray<int> test;
	test.initZeros(myfftV);
	fftVRiesz.initZeros(myfftV);
	VRiesz.resizeNoCopy(Vorig);
	amplitude.resizeNoCopy(Vorig);
	fftVRiesz_aux.initZeros(myfftV);
	std::complex<double> J(0,1);

	// Filter the input volume and add it to amplitude
	long n=0;
	double w_inf = w;
	double w_sup = w+step;

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(myfftV)
	{
		double un=1/DIRECT_MULTIDIM_ELEM(iu,n);
		if (un<w_sup && un>=w_inf)
		{
			DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
			DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = -J;
			DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(fftVRiesz, n);
			DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) *= DIRECT_MULTIDIM_ELEM(iu,n);
			DIRECT_MULTIDIM_ELEM(test, n) = 1;
		}
	}
//	Image<int> filteredvolume;
//	filteredvolume() = test;
//	filteredvolume.write(formatString("Volumen_filtrado_%i.vol", count));

	transformer_inv.inverseFourierTransform(fftVRiesz, bpVol);

	#ifdef DEBUG
	Image<double> filteredvolume;
	filteredvolume = VRiesz;
	filteredvolume.write(formatString("Volumen_filtrado_%i.vol", count));
	#endif

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n)=DIRECT_MULTIDIM_ELEM(bpVol,n)*DIRECT_MULTIDIM_ELEM(bpVol,n);

	// Calculate first component of Riesz vector
	fftVRiesz.initZeros(myfftV);
	double uz, uy, ux;
	n=0;
//	std::cout << "2" << std::endl;
	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				ux = VEC_ELEM(freq_fourier,j);
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = ux*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}

	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);

	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	// Calculate second and third component of Riesz vector
	fftVRiesz.initZeros(myfftV);
	n=0;

	for(size_t k=0; k<ZSIZE(myfftV); ++k)
	{
		uz = VEC_ELEM(freq_fourier,k);
		for(size_t i=0; i<YSIZE(myfftV); ++i)
		{
			uy = VEC_ELEM(freq_fourier,i);
			for(size_t j=0; j<XSIZE(myfftV); ++j)
			{
				DIRECT_MULTIDIM_ELEM(fftVRiesz, n) = uy*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n) = uz*DIRECT_MULTIDIM_ELEM(fftVRiesz_aux, n);
				++n;
			}
		}
	}
	transformer_inv.inverseFourierTransform(fftVRiesz, VRiesz);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);

	transformer_inv.inverseFourierTransform(fftVRiesz_aux, VRiesz);
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		DIRECT_MULTIDIM_ELEM(amplitude,n)+=DIRECT_MULTIDIM_ELEM(VRiesz,n)*DIRECT_MULTIDIM_ELEM(VRiesz,n);
		DIRECT_MULTIDIM_ELEM(amplitude,n)=sqrt(DIRECT_MULTIDIM_ELEM(amplitude,n));
	}

	fftVRiesz_aux.clear();
	fftVRiesz.clear();

//	FileName iternumber;
//	Image<double> saveImg;
//	saveImg = amplitude;
//	iternumber = formatString("_Amplitude_%i.vol", count);
//	saveImg.write(iternumber);
//	saveImg.clear();
//
//	// Low pass filter the monogenic amplitude
//	FourierFilter lowPassFilter;
//	lowPassFilter.w1 = w;
//	amplitude.setXmippOrigin();
//	lowPassFilter.applyMaskSpace(amplitude);

//	std::cout << " 4" << std::endl;
	double meanAmplitude = 0;
	long NA = 0;
	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(amplitude)
	{
		//if (DIRECT_MULTIDIM_ELEM(mask,n) > 0)
		if (DIRECT_MULTIDIM_ELEM(resVol, n) > 2*sampling)
		{
			double freqmap=sampling/DIRECT_MULTIDIM_ELEM(resVol, n);
			if (freqmap>w)
			 {
			    meanAmplitude += DIRECT_MULTIDIM_ELEM(amplitude,n);
			    ++NA;
			 }
		}
	}
//	meanAmplitude = meanAmplitude/((double) NA);
	meanAmplitude = meanAmplitude/NA;

	//if (w<sampling/10)
	//{
		std::cout << "-------change contrast-------- " << std::endl;
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(bpVol)
		{

			if (DIRECT_MULTIDIM_ELEM(resVol, n) > 2*sampling)
			{
				double freqmap=sampling/DIRECT_MULTIDIM_ELEM(resVol, n);
				if (freqmap>w)
				    DIRECT_MULTIDIM_ELEM(bpVol,n) *= meanAmplitude/DIRECT_MULTIDIM_ELEM(amplitude,n);
			}
		}
	//}



	#ifdef DEBUG
	saveImg2 = amplitude;
	FileName fnSaveImg2;
	if (fnDebug.c_str() != "")
	{
		iternumber = formatString("_Filtered_Amplitude_%i.vol", count);
		saveImg2.write(fnDebug+iternumber);
	}
	saveImg2.clear();
	#endif // DEBUG

}


void ProgSameEnergy::run()
{
	produceSideInfo();


	MultidimArray<double>  bpVol;

    double extMinFreq = 0.0001;
    double extMaxFreq = 0.5;

	//std::cout << "Resolutions between " << minRes << " and " << maxRes << std::endl;

	FourierTransformer transformer;

	///////////////////
	//Same Energy
	double step = 0.01;
	sameEnergy(Vorig, extMinFreq, extMaxFreq, step, bpVol);

	Image<double> save;
	save() = bpVol;
	save.write(fnOut);
	///////////////////


}
