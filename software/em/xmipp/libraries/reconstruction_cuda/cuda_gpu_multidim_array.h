#ifndef GPU_MULTIDIM_ARRAY
#define GPU_MULTIDIM_ARRAY

#include "cuda_gpu_bilib_kernel.h"

template<typename T>
__device__
T interpolatedElementBSpline2D_Degree3(T x, T y, int xdim, int ydim, T* data);

#endif
