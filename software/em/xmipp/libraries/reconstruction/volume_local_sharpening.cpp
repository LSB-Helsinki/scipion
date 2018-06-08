/***************************************************************************
 *
 * Authors:    Erney Ramirez-Aportela,                                    eramirez@cnb.csic.es
 *             Jose Luis Vilas,                                           jlvilas@cnb.csic.es
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

#include "volume_local_sharpening.h"
//#define DEBUG
//#define DEBUG_MASK
#define DEBUG_FILTER
void ProgLocSharpening::readParams()
{
        fnVol = getParam("--vol");
        fnRes = getParam("--resolution_map");
        sampling = getDoubleParam("--sampling");
        lambda = getDoubleParam("-l");
        Niter = getIntParam("-i");
        Nthread = getIntParam("-n");
        fnOut = getParam("-o");
}

void ProgLocSharpening::defineParams()
{
        addUsageLine("This function performs local sharpening");
        addParamsLine("  --vol <vol_file=\"\">   : Input volume");
        addParamsLine("  --resolution_map <vol_file=\"\">: Resolution map");
        addParamsLine("  -o <output=\"Sharpening.vol\">: sharpening volume");
        addParamsLine("  --sampling <s=1>: sampling");
        addParamsLine("  [-l <lambda=1>]: regularization param");
        addParamsLine("  -i <Niter=5>: iteration");
        addParamsLine("  [-n <Nthread=1>]: threads number");
}

void ProgLocSharpening::produceSideInfo()
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
                                        DIRECT_MULTIDIM_ELEM(iu,n) =sqrt(u2);
                                else
                                        DIRECT_MULTIDIM_ELEM(iu,n) = 1e-38;
                                ++n;
                        }
                }
        }

        inputVol.clear();

    Image<double> resolutionVolume;
    resolutionVolume.read(fnRes);

    resVol = resolutionVolume();
    resolutionVolume().clear();

        maxMinResolution(resVol, maxRes, minRes);
        std::cout << "maxRes = " << maxRes << "  minRes = " << minRes << std::endl;

    	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resVol)
    	{

    		if ((DIRECT_MULTIDIM_ELEM(resVol, n) < 2*sampling) && (DIRECT_MULTIDIM_ELEM(resVol, n)>0))
    		{
    			DIRECT_MULTIDIM_ELEM(resVol, n) = 2*sampling;
    		}
//    		else if (DIRECT_MULTIDIM_ELEM(resVol, n) ==0)
//    			DIRECT_MULTIDIM_ELEM(resVol, n) = maxRes+5;
    	}

        resVol.setXmippOrigin();

        computeAvgStdev_within_binary_mask(resVol, Vorig, desvOutside_Vorig, true);
        //std::cout << "desvOutside_Vorig = " << desvOutside_Vorig << std::endl;

}

void ProgLocSharpening::maxMinResolution(MultidimArray<double> &resVol,
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
                //if (value<lastMinRes)
                if (value<lastMinRes && value>0)
                        lastMinRes = value;
        }

        maxRes = lastMaxRes;
        minRes = lastMinRes;
}

void ProgLocSharpening::computeAvgStdev_within_binary_mask(const MultidimArray< double >&resVol,
										const MultidimArray< double >&vol, double &stddev, bool outside )
{
    SPEED_UP_tempsInt;
    double sum1 = 0;
    double sum2 = 0;
    int N = 0;
    double avg=0;

    FOR_ALL_ELEMENTS_IN_COMMON_IN_ARRAY3D(resVol, vol)
    {
        if ((not outside && A3D_ELEM(resVol, k, i, j) > 2*sampling) ||
        		(outside && A3D_ELEM(resVol, k, i, j) < 2*sampling))
        {
            ++N;
            double aux=A3D_ELEM(vol, k, i, j);
            sum1 += aux;
            sum2 += aux*aux;
        }
    }

    // average and standard deviation
      avg  = sum1 / (double) N;
    if (N > 1)
      stddev = sqrt(fabs(sum2 / N - avg * avg) * N / (N - 1));
    else
        stddev = 0;
}

void ProgLocSharpening::bandPassFilterFunction(const MultidimArray< std::complex<double> > &myfftV,
                double w, double wL, MultidimArray<double> &filteredVol, int count)
{
        fftVfilter.initZeros(myfftV);
        size_t xdim, ydim, zdim, ndim;
        Vorig.getDimensions(xdim, ydim, zdim, ndim);
        filteredVol.resizeNoCopy(Vorig);

        double delta = wL-w;
        double w_inf = w-delta;
        // Filter the input volume and add it to amplitude
        long n=0;
        double ideltal=PI/(delta);
        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(myfftV)
        {
                double un=DIRECT_MULTIDIM_ELEM(iu,n);
                if (un>=w && un<=wL)
                {
                        DIRECT_MULTIDIM_ELEM(fftVfilter, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
                        DIRECT_MULTIDIM_ELEM(fftVfilter, n) *= 0.5*(1+cos((un-w)*ideltal));//H;
                } else if (un<=w && un>=w_inf)
                {
                        DIRECT_MULTIDIM_ELEM(fftVfilter, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
                        DIRECT_MULTIDIM_ELEM(fftVfilter, n) *= 0.5*(1+cos((un-w)*ideltal));//H;
                }
        }

        transformer_inv.inverseFourierTransform(fftVfilter, filteredVol);

//        #ifdef DEBUG_FILTER
//        Image<double> filteredvolume;
//        filteredvolume() = filteredVol;
//        filteredvolume.write(formatString("Volumen_filtrado_%i.vol", count));
//        #endif
}


void ProgLocSharpening::localfiltering(MultidimArray< std::complex<double> > &myfftV,
                                       MultidimArray<double> &localfilteredVol,
                                       double &minRes, double &maxRes, double &step)
{
        MultidimArray<double> filteredVol, lastweight, weight;
        localfilteredVol.initZeros(Vorig);
        weight.initZeros(Vorig);
        lastweight.initZeros(Vorig);

        double freq, lastResolution=1e38;
        int idx, lastidx = -1;

        for (double res = minRes; res<maxRes; res+=step)
        {
                freq = sampling/res;

                DIGFREQ2FFT_IDX(freq, ZSIZE(myfftV), idx);

                if (idx == lastidx)
                {
                        continue;
                }

                double wL = sampling/(res - step);

                bandPassFilterFunction(myfftV, freq, wL, filteredVol, idx);

                FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(filteredVol)
                {

					   if (DIRECT_MULTIDIM_ELEM(resVol, n) < 2*sampling)
							{
						   	   DIRECT_MULTIDIM_ELEM(filteredVol, n)=0;
							}
					   else
						    {
						   	   double res_map = DIRECT_MULTIDIM_ELEM(resVol, n);//+1e-38;
						   	   DIRECT_MULTIDIM_ELEM(weight, n) = (exp(-0.025*(res-res_map)*(res-res_map)));
						   	   DIRECT_MULTIDIM_ELEM(filteredVol, n) *= DIRECT_MULTIDIM_ELEM(weight, n);
						    }

                }

                localfilteredVol += filteredVol;
                lastweight += weight;
                lastResolution = res;
                lastidx = idx;
        }
//		double sigmaBefore=0;
//        computeAvgStdev_within_binary_mask(resVol, localfilteredVol, sigmaBefore);

        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(localfilteredVol)
        {
        	if (DIRECT_MULTIDIM_ELEM(lastweight, n)>0)
                DIRECT_MULTIDIM_ELEM(localfilteredVol, n) /=DIRECT_MULTIDIM_ELEM(lastweight, n);
        }
//		double sigmaAfter=0;
//        computeAvgStdev_within_binary_mask(resVol, localfilteredVol, sigmaAfter);
//        std::cout << "sigmaBefore div=" << sigmaBefore << " sigmaAfter div=" << sigmaAfter << std::endl;
//        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(localfilteredVol)
//        {
//        	if (DIRECT_MULTIDIM_ELEM(lastweight, n)>0)
//                DIRECT_MULTIDIM_ELEM(localfilteredVol, n) *=0.01*sigmaBefore/sigmaAfter;
//        }
}




void ProgLocSharpening::run()
{
        produceSideInfo();

        MultidimArray<double> auxVol;
        MultidimArray<double> operatedfiltered, Vk, filteredVol;
        double lastnorm=0, lastporc=1;
        double freq;
        double step = 0.2;
        double lastResolution=1e38;
        int  idx, bool1=1, bool2=1;
        int lastidx = -1;

        minRes = 2*sampling;
        //maxRes = sampling/0.0001;//Esto solo para este caso
        maxRes=maxRes+2;
        //maxRes=13;
        //maxRes = maxRes+1;

        std::cout << "Resolutions between " << minRes << " and " << maxRes << std::endl;

        filteredVol = Vorig;

        sharpenedMap.resizeNoCopy(Vorig);
		double normOrig=0;

    for (size_t i = 1; i<=Niter; ++i)
        {
        std::cout << "----------------Iteration " << i << "----------------" << std::endl;

        auxVol = filteredVol;
        transformer.FourierTransform(auxVol, fftV);

        localfiltering(fftV, operatedfiltered, minRes, maxRes, step);

                filteredVol = Vorig;
        		filteredVol -= operatedfiltered;

        		//calculate norm for Vorig
        		if (i==1)
        		{
            		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(Vorig)
            		{
                       //if (DIRECT_MULTIDIM_ELEM(resVol, n) <= maxRes-5) //la mascara
            			   normOrig +=(DIRECT_MULTIDIM_ELEM(Vorig,n)*DIRECT_MULTIDIM_ELEM(Vorig,n));
            		}
            		normOrig=sqrt(normOrig);
                    std::cout << "norma del original  " << normOrig << std::endl;
        		}


        		//calculate norm for operatedfiltered
        		double norm=0;
        		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(operatedfiltered)
        		{
                    //if (DIRECT_MULTIDIM_ELEM(resVol, n) <= maxRes-5) //la mascara
        			    norm +=(DIRECT_MULTIDIM_ELEM(operatedfiltered,n)*DIRECT_MULTIDIM_ELEM(operatedfiltered,n));
        		}
        		norm=sqrt(norm);


                double porc=lastnorm*100/norm;
                std::cout << "norma " << norm << " porciento " << porc << std::endl;

                double subst=porc-lastporc;

                if ((subst<1)&&(bool1==1)&&(i>2))
                {
                	bool1=2;
                    std::cout << "-------------la resta es menor que 1 para iter  ----- " << i << std::endl;
                }
//                if ((subst<0.5)&&(bool2==1)&&(i>2))
//                {
//                	bool2=2;
//                    std::cout << "-------------la resta es menor que 0.5 para iter  ----- " << i << std::endl;
//                }

                lastnorm=norm;
                lastporc=porc;

                if (i==1 && lambda==1)
                {
                	lambda=(normOrig/norm)/12;
                }
               	std::cout << "iteration "<< i << "  lambda  " << lambda << std::endl;



                ////Second operator
        transformer.FourierTransform(filteredVol, fftV);
        localfiltering(fftV, filteredVol, minRes, maxRes, step);

                if (i == 1)
                        Vk = Vorig;
                else
                        Vk = sharpenedMap;

                //sharpenedMap=Vk+lambda*(filteredVol);
        		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(sharpenedMap)
                {
        			DIRECT_MULTIDIM_ELEM(sharpenedMap,n)=DIRECT_MULTIDIM_ELEM(Vk,n)+
        			                     lambda*DIRECT_MULTIDIM_ELEM(filteredVol,n);
        			                     //-0.01*DIRECT_MULTIDIM_ELEM(Vk,n)*SGN(DIRECT_MULTIDIM_ELEM(Vk,n));
        			if (DIRECT_MULTIDIM_ELEM(sharpenedMap,n)<-4*desvOutside_Vorig)
        				DIRECT_MULTIDIM_ELEM(sharpenedMap,n)=-4*desvOutside_Vorig;
                }

//                Image<double> filteredvolume;
//                filteredvolume = sharpenedMap;
//                filteredvolume.write(formatString("sharpenedMapE_%i.vol", i));
//                filteredvolume.clear();

//        		double desv_sharp=0;
//                computeAvgStdev_within_binary_mask(resVol, sharpenedMap, desv_sharp);
//                std::cout << "desv_sharp = " << desv_sharp << std::endl;

                filteredVol = sharpenedMap;

                if (bool1==2)
                {
					Image<double> filteredvolume;
					filteredvolume() = sharpenedMap;
					filteredvolume.write(fnOut);
					break;
                }
        }
        Image<double> filteredvolume;
        filteredvolume() = sharpenedMap;
        filteredvolume.write(fnOut);

}








































///***************************************************************************
// *
// * Authors:    Erney Ramirez , 				     	  eramirez@cnb.csic.es
// * 			   Jose Luis Vilas                        jlvilas@cnb.csic.es
// *
// * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
// *
// * This program is free software; you can redistribute it and/or modify
// * it under the terms of the GNU General Public License as published by
// * the Free Software Foundation; either version 2 of the License, or
// * (at your option) any later version.
// *
// * This program is distributed in the hope that it will be useful,
// * but WITHOUT ANY WARRANTY; without even the implied warranty of
// * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// * GNU General Public License for more details.
// *
// * You should have received a copy of the GNU General Public License
// * along with this program; if not, write to the Free Software
// * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
// * 02111-1307  USA
// *
// *  All comments concerning this program package may be sent to the
// *  e-mail address 'xmipp@cnb.csic.es'
// ***************************************************************************/
//
//#include "volume_local_sharpening.h"
////#define DEBUG
////#define DEBUG_MASK
////#define DEBUG_FILTER
//void ProgLocSharpening::readParams()
//{
//	fnVol = getParam("--vol");
//	fnRes = getParam("--resolution_map");
//	sampling = getDoubleParam("--sampling");
//	lambda = getDoubleParam("-l");
//	Niter = getIntParam("-n");
//	fnOut = getParam("-o");
//}
//
//void ProgLocSharpening::defineParams()
//{
//	addUsageLine("This function performs local sharpening");
//	addParamsLine("  --vol <vol_file=\"\">   : Input volume");
//	addParamsLine("  --resolution_map <vol_file=\"\">: Resolution map");
//	addParamsLine("  -o <output=\"Sharpening.vol\">: sharpening volume");
//	addParamsLine("  --sampling <s=1>: sampling");
//	addParamsLine("  -l <s=1>: regularization param");
//	addParamsLine("  -n <s=5>: iteration");
//}
//
//void ProgLocSharpening::produceSideInfo()
//{
//	std::cout << "Starting..." << std::endl;
//	Image<double> V;
//    V.read(fnVol);
//    V().setXmippOrigin();
//
//	FourierTransformer transformer;
//	MultidimArray<double> &inputVol = V();
//
//	Vorig = inputVol;
//
//	transformer.FourierTransform(inputVol, fftV);
//
//	iu.initZeros(fftV);
//	double uz, uy, ux, uz2, u2, uz2y2;
//	long n=0;
//	for(size_t k=0; k<ZSIZE(fftV); ++k)
//	{
//		FFT_IDX2DIGFREQ(k,ZSIZE(inputVol),uz);
//		uz2=uz*uz;
//
//		for(size_t i=0; i<YSIZE(fftV); ++i)
//		{
//			FFT_IDX2DIGFREQ(i,YSIZE(inputVol),uy);
//			uz2y2=uz2+uy*uy;
//
//			for(size_t j=0; j<XSIZE(fftV); ++j)
//			{
//				FFT_IDX2DIGFREQ(j,XSIZE(inputVol),ux);
//				u2=uz2y2+ux*ux;
//				if ((k != 0) || (i != 0) || (j != 0))
//					DIRECT_MULTIDIM_ELEM(iu,n) =sqrt(u2);
//				else
//					DIRECT_MULTIDIM_ELEM(iu,n) = 1e-38;
//				++n;
//			}
//		}
//	}
//
//    Image<double> resolutionVolume;
//    resolutionVolume.read(fnRes);
//    resVol = resolutionVolume();
//
//	resVol.setXmippOrigin();
//
//	maxMinResolution(resVol, maxRes, minRes);
//	std::cout << "minRes = " << minRes << "  maxRes = " << maxRes << std::endl;
//
////	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resVol)
////	{
////		if (DIRECT_MULTIDIM_ELEM(resVol, n) < 2*sampling)
////			DIRECT_MULTIDIM_ELEM(resVol, n) = 2*sampling;
////	}
//
////	resVol.setXmippOrigin();
//
//	FourierFilter Filter;
//	Filter.FilterShape=REALGAUSSIAN;
//	Filter.FilterBand=LOWPASS;
//	Filter.w1=1;
////	Filter.apply(resVol);
//
//}
//
//void ProgLocSharpening::maxMinResolution(MultidimArray<double> &resVol,
//                                         double &maxRes, double &minRes)
//{
//        // Count number of voxels with resolution
//        size_t n=0;
//        double lastMinRes=1e38, lastMaxRes=1e-38, value;
//        FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(resVol)
//        {
//                value = DIRECT_MULTIDIM_ELEM(resVol, n);
//                if (value>lastMaxRes)
//                        lastMaxRes = value;
//                //if (value<lastMinRes)
//                if (value<lastMinRes && value>0)
//                        lastMinRes = value;
//        }
//
//        maxRes = lastMaxRes;
//        minRes = lastMinRes;
//}
//
//void ProgLocSharpening::bandPassFilterFunction(const MultidimArray< std::complex<double> > &myfftV,
//		double w, double step, MultidimArray<double> &filteredVol, int count)
//{
//	fftVfilter.initZeros(myfftV);
//	size_t xdim, ydim, zdim, ndim;
//	Vorig.getDimensions(xdim, ydim, zdim, ndim);
//	filteredVol.resizeNoCopy(Vorig);
//
//	double w_inf = w;
//	double w_sup = w+step;
//	// Filter the input volume
//	long n=0;
//	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(myfftV)
//	{
//		double un=DIRECT_MULTIDIM_ELEM(iu,n);
//
//		if (un<w_sup && un>=w_inf)
//		{
//			DIRECT_MULTIDIM_ELEM(fftVfilter, n) = DIRECT_MULTIDIM_ELEM(myfftV, n);
//		}
//	}
//
//	transformer_inv.inverseFourierTransform(fftVfilter, filteredVol);
//
//	#ifdef DEBUG_FILTER
//	Image<double> filteredvolume;
//	filteredvolume() = filteredVol;
//	filteredvolume.write(formatString("Volumen_filtrado_%i.vol", count));
//	#endif
//}
//
//
//void ProgLocSharpening::localfiltering(MultidimArray< std::complex<double> > &myfftV,
//										MultidimArray<double> &localfilteredVol,
//										double &minFreq, double &maxFreq, double &step)
//{
//	MultidimArray<double>  filteredVol;
//	localfilteredVol.initZeros(Vorig);
//
//	double lastResolution=1e38;
//	int idx=1, lastidx = -1;
//
//	//First band
//	double freq0=0.0;
//	double step0 = minFreq;
//	bandPassFilterFunction(myfftV, freq0, step0, filteredVol, idx);
//
//	localfilteredVol = filteredVol;
//
//	for (double freq = minFreq; freq<=maxFreq; freq+=step)
//	{
//
//			DIGFREQ2FFT_IDX(freq, ZSIZE(myfftV), idx);
//
//			if (idx == lastidx)
//			{
//			   std::cout << "idx = " << idx << std::endl;
//					continue;
//			}
//
//
//		double res = sampling/freq;
//		double freqstep=sampling/(res-0.2);
//               step=freqstep-freq;
//
//		//std::cout << " freqmin = " << freq <<  " freqmax = " << freq+step << "  resol= " << res << std::endl;
//
//		bandPassFilterFunction(myfftV, freq, step, filteredVol, idx);
//
//		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(filteredVol)
//		{
//			double resOk = DIRECT_MULTIDIM_ELEM(resVol, n);//+1e-38;
//
//			if (res<resOk)
//				DIRECT_MULTIDIM_ELEM(filteredVol, n)=0;
//
////	        if (DIRECT_MULTIDIM_ELEM(filteredVol, n)<0)
////				DIRECT_MULTIDIM_ELEM(filteredVol, n)=0;
//
//		}
////		Image<double> save;
////		save() = filteredVol;
////		save.write(formatString("filtro_%f.vol", freq));
//
//		localfilteredVol += filteredVol;
//
//		//++idx;
//	    lastidx = idx;
//	}
//
//}
//
//
//
//void ProgLocSharpening::run()
//{
//	produceSideInfo();
//
//	MultidimArray<double> auxVol;
//
//	double step;
//
//	MultidimArray<double> operatedfiltered, Vk, filteredVol;
//
//	//minRes = 2*sampling;
//    maxFreq=0.5;
//    minFreq=sampling/maxRes;
//
//	std::cout << "Resolutions between " << minRes << " and " << maxRes << std::endl;
//
//	FourierTransformer transformer;
//
//	filteredVol = Vorig;
//	sharpenedMap.resizeNoCopy(Vorig);
//
//
//    for (size_t i = 0; i<Niter; ++i)
//	{
//    	std::cout << "----------------Iteration " << i << "----------------" << std::endl;
//    	auxVol = filteredVol;
//    	transformer.FourierTransform(auxVol, fftV);
//
//
//    	   localfiltering(fftV, operatedfiltered, minFreq, maxFreq, step);
//
////    	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(operatedfiltered)
////		{
////    		if (DIRECT_MULTIDIM_ELEM(operatedfiltered, n) <0)
////    			DIRECT_MULTIDIM_ELEM(operatedfiltered, n) = 0;
////		}
//
//
//		Image<double> save;
//		save() = operatedfiltered;
//		save.write(formatString("sharpenedMapA_%i.vol", i));
//
//		filteredVol = Vorig;
//		filteredVol -= operatedfiltered;
//
//      //Image<double> save;
//		save() = filteredVol;
//		save.write(formatString("sharpenedMapB_%i.vol", i));
//
//		////Second operator
//    	transformer.FourierTransform(filteredVol, fftV);
//    	localfiltering(fftV, filteredVol, minFreq, maxFreq, step);
//
////    	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(filteredVol)
////		{
////    		if (DIRECT_MULTIDIM_ELEM(filteredVol, n) <0)
////    			DIRECT_MULTIDIM_ELEM(filteredVol, n) = 0;
////		}
//
//
//   	//Image<double> save;
//		save() = filteredVol;
//		save.write(formatString("sharpenedMapC_%i.vol", i));
//
//		if (i == 0)
//			Vk = Vorig;
//		else
//			Vk = sharpenedMap;
//
//		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(sharpenedMap)
//			DIRECT_MULTIDIM_ELEM(sharpenedMap,n)=DIRECT_MULTIDIM_ELEM(Vk,n)+
//			   lambda*DIRECT_MULTIDIM_ELEM(filteredVol,n);
//
////    	FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(sharpenedMap)
////		{
////    		if (DIRECT_MULTIDIM_ELEM(sharpenedMap, n) <0)
////    			DIRECT_MULTIDIM_ELEM(sharpenedMap, n) = 0;
////		}
//
//		Image<double> filteredvolume0;
//		filteredvolume0() = sharpenedMap;
//		filteredvolume0.write(formatString("sharpenedMapD_%i.vol", i));
//
//		filteredVol = sharpenedMap;
//	}
//	Image<double> filteredvolume;
//	filteredvolume() = sharpenedMap;
//	filteredvolume.write(fnOut);
//
//}
