
#include <assert.h>
#include <exception>
#include "data/multidim_array.h"
#include <iostream> // FIXME remove

template<typename T>
class GeoTransformer {
	GeoTransformer():
		X(X), Y(Y), Z(Z),
		isReady(false),
		d_coefs(NULL), d_in(NULL), d_out(NULL),
		d_trInv(NULL) {};

	~GeoTransformer() { release(); }

	void init(int splineDegree, size_t x, size_t y=0, size_t z=0);
	void release();

	template<typename T_IN, typename T_MAT, typename T_COEFFS>
	void applyGeometry(int splineDegree,
	                   MultidimArray<T> &output,
	                   const MultidimArray<T_IN> &input,
	                   const Matrix2D<T_MAT> &transform, bool isInv,
	                   bool wrap, T outside = 0, const MultidimArray<T_COEFFS> *bCoeffsPtr=NULL);
private:
	template<typename T_IN, typename T_MAT>
	bool checkRestrictions(MultidimArray<T> &output,
            const MultidimArray<T_IN> &input,
            const Matrix2D<T_MAT> &transform);

	void prepareAndLoadOutput(MultidimArray<T> &output, T outside);
	template<typename T_IN>
	void loadInput(MultidimArray<T_IN> &input);

	void applyGeometry_2D_wrap(int SplineDegree);
	template<typename T_COEFFS>
	void loadCoeffs(const T_COEFFS &bCoeffsPtr);

	template<typename T_IN>
	void prepareAndLoadCoeffs(int splineDegree, const MultidimArray<T_IN> &input);

	template<typename T_MAT>
	void prepareTransform(const Matrix2D<T_MAT> &transform, bool isInv);

private:
	bool isReady;

	T* d_coefs;
	T* d_trInv;
	T* d_in;
	T* d_out;

	size_t X, Y, Z;
};
