
#include "cuda_gpu_geo_transformer.h"
#include <cuda_runtime_api.h>

template<typename T>
void GeoTransformer<T>::release() {
	cudaFree(d_coefs);
	cudaFree(d_in);
	cudaFree(d_out);
	cudaFree(d_trInv);
	d_coefs = d_in = d_out = d_trInv = NULL;
	isReady = false;
}

template<typename T>
void GeoTransformer<T>::init(int splineDegree, size_t x, size_t y, size_t z) {
	release();

	this->X = x;
	this->Y = y;
	this->Z = z;

	size_t matSize = (0 == z) ? 9 : 16;
	gpuErrchk(gpuMalloc((void**) &d_trInv, matSize * sizeof(T)));
	gpuErrchk(gpuMalloc((void**) &d_in, x * y * z * sizeof(T)));
	gpuErrchk(gpuMalloc((void**) &d_out, x * y * z * sizeof(T)));

	isReady = true;
}

template<typename T>
template<typename T_IN, typename T_MAT, typename T_COEFFS>
void GeoTransformer<T>::applyGeometry(int splineDegree,
	                   MultidimArray<T> &output,
	                   const MultidimArray<T_IN> &input,
	                   const Matrix2D<T_MAT> &transform, bool isInv,
	                   bool wrap, T outside, const MultidimArray<T_COEFFS> *bCoeffsPtr) {
	checkRestrictions(output, input, transform);
	if (transform.isIdentity()) {
		typeCast(input, output);
	}

	prepareAndLoadTransform(transform, isInv);
	prepareAndLoadOutput(output, outside);
	loadInput(input);

	if (NULL != bCoeffsPtr) {
		loadCoeffs(*bCoeffsPtr);
	} else {
		if (splineDegree > 1) {
			prepareAndLoadCoeffs(splineDegree);
		}
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
}

template<typename T>
template<typename T_MAT>
void GeoTransformer<T>::prepareTransform(const Matrix2D<T_MAT> &transform, bool isInv) {
	Matrix2D<T_MAT> trInv = isInv ? transform : transform.inv();
	Matrix2D<T>tmp;
	typeCast(trInv, tmp);
	gpuErrchk(cudaMemcpy(d_trInv, tmp.mdata, tmp.mdim * sizeof(T), cudaMemcpyHostToDevice));
}

template<typename T>
template<typename T_COEFFS>
void GeoTransformer<T>::loadCoeffs(const T_COEFFS &bCoeffsPtr) {
	MultidimArray<T> tmp;
	typeCast(bCoeffsPtr, tmp);

	size_t bytes = tmp.yxdim * sizeof(T);
	if (NULL == d_coefs) gpuErrchk(gpuMalloc((void**) &d_coefs, bytes));
	gpuErrchk(cudaMemcpy(d_coefs, tmp.data, bytes, cudaMemcpyHostToDevice));
}

template<typename T>
template<typename T_IN>
void GeoTransformer<T>::prepareAndLoadCoeffs(int splineDegree, const MultidimArray<T_IN> &input) {
	MultidimArray<double> tmp; // FIXME this should be T, once produceSplineCoefficients can handle it
	produceSplineCoefficients(SplineDegree, tmp, input);
	loadCoeffs(tmp);
}

template<typename T>
void GeoTransformer<T>::applyGeometry_2D_wrap(int SplineDegree) {
	std::cout << "I'm here" << std::endl;
}

template<typename T>
template<typename T_IN>
void GeoTransformer<T>::loadInput(MultidimArray<T_IN> &input) {
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
bool GeoTransformer<T>::checkRestrictions(MultidimArray<T> &output,
        const MultidimArray<T_IN> &input,
        const Matrix2D<T_MAT> &transform) {
	static_assert(isReady, "Transformer is not ready yet.");
	static_assert(input.xdim, "Input is empty");
	static_assert((X == input.xdim) && (Y == input.ydim) && (Z == inpyut.zdim),
			"Transformer has been initialized for different size of the input");
	static_assert(&input == (MultidimArray<T_INT>*)&output,
			"Input array cannot be the same as output array");
	static_assert((input.getDim() == 2)
			&& ((transform.Xdim() != 3) || (transform.Ydim() != 3)),
			"2D transformation matrix is not 3x3");
	static_assert((input.getDim() == 3)
				&& ((transform.Xdim() != 4) || (transform.Ydim() != 4)),
				"3D transformation matrix is not 4x4");
}
