
#include "cuda_gpu_geo_transformer.h"
#include "cuda_utils.h"
#include <cuda_runtime_api.h>
#include "cuda_gpu_geo_transformer_cu.cpp"

template class GeoTransformer<float>;

template<typename T>
void GeoTransformer<T>::release() {
	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_trInv);
	d_in = d_out = d_trInv = NULL;
	isReady = false;
}

template<typename T>
void GeoTransformer<T>::init(size_t x, size_t y, size_t z) {
	release();

	this->X = x;
	this->Y = y;
	this->Z = z;

	size_t matSize = (0 == z) ? 9 : 16;
	gpuErrchk(cudaMalloc((void**) &d_trInv, matSize * sizeof(T)));
	gpuErrchk(cudaMalloc((void**) &d_in, x * y * z * sizeof(T)));
	gpuErrchk(cudaMalloc((void**) &d_out, x * y * z * sizeof(T)));

	isReady = true;
}

template<typename T>
void GeoTransformer<T>::initLazy(size_t x, size_t y, size_t z) {
	if (!isReady) {
		init(x, y, z);
	}
}

template void GeoTransformer<float>::applyGeometry<float, float>(int splineDegree,
        MultidimArray<float> &output,
        const MultidimArray<float> &input,
        const Matrix2D<float> &transform, bool isInv,
        bool wrap, float outside);

template<typename T>
template<typename T_IN, typename T_MAT>
void GeoTransformer<T>::applyGeometry(int splineDegree,
	                   MultidimArray<T> &output,
	                   const MultidimArray<T_IN> &input,
	                   const Matrix2D<T_MAT> &transform, bool isInv,
	                   bool wrap, T outside) {
	applyGeometry<T_IN, T_MAT, T_IN>(splineDegree, output, input, transform, isInv, wrap, outside, NULL);
}

template<typename T>
void GeoTransformer<T>::test() {
	Matrix1D<T> shift(2);
	shift.vdata[0] = 0.45;
	shift.vdata[1] = 0.62;
	Matrix2D<T> transform;
	translation2DMatrix(shift, transform, true);

	test(transform);
}

template<typename T>
void GeoTransformer<T>::test(const Matrix2D<T> &transform) {
	MultidimArray<T> resGpu, resCpu;
	MultidimArray<T> input(100, 100);
	for(int i=0; i < input.ydim; ++i) {
		for(int j=0; j < input.xdim; ++j) {
			input.data[i * input.xdim + j] = i * 10 + j;
		}
	}

	this->init(input.xdim, input.ydim, input.zdim);
	this->applyGeometry(3, resGpu, input, transform, false, true);
	fprintf(stderr, "\n\nAAA gpu VVV cpu\n\n");
	::applyGeometry(3, resCpu, input, transform, false, true);

	bool failed = false;
	for(int i=0; i < input.ydim; ++i) {
		for(int j=0; j < input.xdim; ++j) {
			int index = i * input.xdim + j;
			T gpu = resGpu[index];
			T cpu = resCpu[index];
			if (std::abs(cpu - gpu) > 0.001) {
				failed = true;
				fprintf(stderr, "error[%d]: GPU %.4f CPU %.4f\n", index, gpu, cpu);
			}
		}
	}

	fprintf(stderr, "test result: %s\n", failed ? "FAIL" : "OK");
}


template<typename T>
template<typename T_IN, typename T_MAT, typename T_COEFFS>
void GeoTransformer<T>::applyGeometry(int splineDegree,
	                   MultidimArray<T> &output,
	                   const MultidimArray<T_IN> &input,
	                   const Matrix2D<T_MAT> &transform, bool isInv,
	                   bool wrap, T outside, const MultidimArray<T_COEFFS> *bCoeffsPtr) {
	checkRestrictions(splineDegree, output, input, transform);
	if (transform.isIdentity()) {
		typeCast(input, output);
		return;
	}

	prepareAndLoadTransform(transform, isInv);
	prepareAndLoadOutput(output, outside);

	if (splineDegree > 1) {
		if (NULL != bCoeffsPtr) {
			loadInput(*bCoeffsPtr);
		} else {
			prepareAndLoadCoeffs(splineDegree, input);
		}
	} else {
		loadInput(input);
	}

	if (input.getDim() == 2) {
		if (wrap) {
			applyGeometry_2D_wrap(splineDegree);
		} else {
			throw std::logic_error("Not implemented yet");
		}
	} else {
		throw std::logic_error("Not implemented yet");
	}

	gpuErrchk(cudaMemcpy(output.data, d_out, output.zyxdim * sizeof(T), cudaMemcpyDeviceToHost));

}

template<typename T>
template<typename T_MAT>
void GeoTransformer<T>::prepareAndLoadTransform(const Matrix2D<T_MAT> &transform, bool isInv) {
	Matrix2D<T_MAT> trInv = isInv ? transform : transform.inv();
	Matrix2D<T>tmp;
	typeCast(trInv, tmp);
	gpuErrchk(cudaMemcpy(d_trInv, tmp.mdata, tmp.mdim * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
template<typename T_IN>
void GeoTransformer<T>::prepareAndLoadCoeffs(int splineDegree, const MultidimArray<T_IN> &input) {
	MultidimArray<double> tmp; // FIXME this should be T, once produceSplineCoefficients can handle it
	produceSplineCoefficients(splineDegree, tmp, input);
	loadInput(tmp);
}

template<typename T>
void GeoTransformer<T>::applyGeometry_2D_wrap(int splineDegree) {
	T cen_yp = (int)(Y / 2);
	T cen_xp = (int)(X / 2);
	T minxp  = 0;
	T minyp  = 0;
	T minxpp = minxp-XMIPP_EQUAL_ACCURACY;
	T minypp = minyp-XMIPP_EQUAL_ACCURACY;
	T maxxp  = X - 1;
	T maxyp  = Y - 1;
	T maxxpp = maxxp+XMIPP_EQUAL_ACCURACY;
	T maxypp = maxyp+XMIPP_EQUAL_ACCURACY;

    dim3 dimBlock(BLOCK_DIM_X, BLOCK_DIM_X);
	dim3 dimGrid(ceil(X/(T)dimBlock.x), ceil(Y/(T)dimBlock.y));

	switch(splineDegree) {
	case 3:
		applyGeometryKernel_2D_wrap<T, 3,true><<<dimGrid, dimBlock>>>(d_trInv,
				minxpp, maxxpp, minypp, maxypp,
				minxp, maxxp, minyp, maxyp,
				d_out, (int)X, (int)Y, d_in, (int)X, (int)Y);
		gpuErrchk( cudaPeekAtLastError() );
		break;
	default:
		throw std::logic_error("not implemented");
	}
}

template<typename T>
template<typename T_IN>
void GeoTransformer<T>::loadInput(const MultidimArray<T_IN> &input) {
	MultidimArray<T> tmp;
	typeCast(input, tmp);
	gpuErrchk(cudaMemcpy(d_in, tmp.data, tmp.zyxdim * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
void GeoTransformer<T>::prepareAndLoadOutput(MultidimArray<T> &output, T outside) {
	if (output.xdim == 0) {
		output.resize(Z, Y, X);
	}
	if (outside != (T)0) {
		// Initialize output matrix with value=outside
		FOR_ALL_DIRECT_ELEMENTS_IN_MULTIDIMARRAY(output) {
			DIRECT_MULTIDIM_ELEM(output, n) = outside;
		}
	} else {
		output.initZeros();
	}
	gpuErrchk(cudaMemcpy(d_out, output.data, output.zyxdim * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
template<typename T_IN, typename T_MAT>
void GeoTransformer<T>::checkRestrictions(int splineDegree, MultidimArray<T> &output,
        const MultidimArray<T_IN> &input,
        const Matrix2D<T_MAT> &transform) {
	if (!isReady)
		throw std::logic_error("Transformer is not ready yet.");
	if (!input.xdim)
		throw std::invalid_argument("Input is empty");
	if ((X != input.xdim) || (Y != input.ydim) || (Z != input.zdim))
		throw std::logic_error("Transformer has been initialized for a different size of the input");
	if (&input == (MultidimArray<T_IN>*)&output)
		throw std::invalid_argument("The input array cannot be the same as the output array");
	if ((input.getDim() == 2)
			&& ((transform.Xdim() != 3) || (transform.Ydim() != 3)))
		throw std::invalid_argument("2D transformation matrix is not 3x3");
	if ((input.getDim() == 3)
			&& ((transform.Xdim() != 4) || (transform.Ydim() != 4)))
		throw std::invalid_argument("3D transformation matrix is not 4x4");
}
