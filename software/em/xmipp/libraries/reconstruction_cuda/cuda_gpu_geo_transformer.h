
#include <assert.h>
#include <exception>
#include "data/multidim_array.h"
#include "data/transformations.h"

template<typename T>
class GeoTransformer {

public:
	GeoTransformer():
		X(0), Y(0), Z(0),
		isReady(false),
		d_in(NULL), d_out(NULL),
		d_trInv(NULL) {};

	~GeoTransformer() { release(); }

	void init(size_t x, size_t y, size_t z);
	void initLazy(size_t x, size_t y=1, size_t z=1);
	void release();

	template<typename T_IN, typename T_MAT, typename T_COEFFS>
	void applyGeometry(int splineDegree,
	                   MultidimArray<T> &output,
	                   const MultidimArray<T_IN> &input,
	                   const Matrix2D<T_MAT> &transform, bool isInv,
	                   bool wrap, T outside = 0, const MultidimArray<T_COEFFS> *bCoeffsPtr=NULL);

	template<typename T_IN, typename T_MAT>
	void applyGeometry(int splineDegree, // FIXME support scaling, shear
	                   MultidimArray<T> &output,
	                   const MultidimArray<T_IN> &input,
	                   const Matrix2D<T_MAT> &transform, bool isInv,
	                   bool wrap, T outside = 0);

	void test();

private:
	template<typename T_IN, typename T_MAT>
	void checkRestrictions(int splineDegree, MultidimArray<T> &output,
            const MultidimArray<T_IN> &input,
            const Matrix2D<T_MAT> &transform);

	void prepareAndLoadOutput(MultidimArray<T> &output, T outside);
	template<typename T_IN>
	void loadInput(const MultidimArray<T_IN> &input);

	void applyGeometry_2D_wrap(int SplineDegree);

	template<typename T_IN>
	void prepareAndLoadCoeffs(int splineDegree, const MultidimArray<T_IN> &input);

	template<typename T_MAT>
	void prepareAndLoadTransform(const Matrix2D<T_MAT> &transform, bool isInv);

	void test(const Matrix2D<T> &transform);

private:
	bool isReady;

	static const size_t BLOCK_DIM_X = 32;

	T* d_trInv;
	T* d_in;
	T* d_out;

	size_t X, Y, Z;

#undef BLOCK_DIM_X
};
