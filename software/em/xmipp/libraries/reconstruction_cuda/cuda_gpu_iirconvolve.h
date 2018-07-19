#ifndef IIRCONVOLVE_KERNEL
#define IIRCONVOLVE_KERNEL


__global__
void transposeNoBankConflicts(float *odata, const float *idata);

__global__
void iirConvolve2D_Cardinal_Bspline_3_MirrorOffBound(float* input, float* outpu,
		size_t xDim, size_t yDim);

#endif
