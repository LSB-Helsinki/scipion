#include "cuda_gpu_multidim_array.h"
#define LOOKUP_TABLE_LEN 6

template __device__ float interpolatedElementBSpline2D_Degree3<float>(
		float x, float y, int xdim, int ydim, float* data);


template<typename T>
__device__
T interpolatedElementBSpline2D_Degree3(T x, T y, int xdim, int ydim, T* data)
{
	bool firstTime=true;			// Inner loop first time execution flag.
	T	*ref;

	int l1 = (int)ceil(x - 2);
	int l2 = l1 + 3;
	int m1 = (int)ceil(y - 2);
	int m2 = m1 + 3;

	T columns = 0.0;
	T aux;

	int		equivalent_l_Array[LOOKUP_TABLE_LEN]; // = new int [l2 - l1 + 1];
	T 	aux_Array[LOOKUP_TABLE_LEN];// = new double [l2 - l1 + 1];

	for (int m = m1; m <= m2; m++)
	{
		int equivalent_m=m;
		if      (m<0)
			equivalent_m=-m-1;
		else if (m>=ydim)
			equivalent_m=2*ydim-m-1;
		T rows = 0.0;
		int	index=0;
//		ref = &DIRECT_A2D_ELEM(*this, equivalent_m,0);
		ref = data + (equivalent_m*xdim);
		for (int l = l1; l <= l2; l++)
		{
			int equivalent_l;
			// Check if it is first time executing inner loop.
			if (firstTime)
			{
				T xminusl = x - (T) l;
				equivalent_l=l;
				if (l<0)
				{
					equivalent_l=-l-1;
				}
				else if (l>=xdim)
				{
					equivalent_l=2*xdim-l-1;
				}

				equivalent_l_Array[index] = equivalent_l;
				aux = bspline03(xminusl);
				aux_Array[index] = aux;
				index++;
			}
			else
			{
				equivalent_l = equivalent_l_Array[index];
				aux = aux_Array[index];
				index++;
			}

			//double Coeff = DIRECT_A2D_ELEM(*this, equivalent_m,equivalent_l);
			T Coeff = ref[equivalent_l];
			rows += Coeff * aux;
		}

		// Set first time inner flag is executed to false.
		firstTime = false;

		T yminusm = y - (T) m;
		aux = bspline03(yminusm);
		columns += rows * aux;
	}

	return columns;
}
