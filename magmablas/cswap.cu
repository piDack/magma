/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023
       
       @author Mark Gates

       @generated from magmablas/zswap.cu, normal z -> c, Wed Feb 22 15:21:18 2023

*/
#include "magma_internal.h"

#define NB 64


/* Vector is divided into ceil(n/nb) blocks.
   Each thread swaps one element, x[tid] <---> y[tid].
*/
__global__ void cswap_kernel(
    int n,
    magmaFloatComplex *x, int incx,
    magmaFloatComplex *y, int incy )
{
    magmaFloatComplex tmp;
    int ind = threadIdx.x + blockDim.x*blockIdx.x;
    if ( ind < n ) {
        x += ind*incx;
        y += ind*incy;
        tmp = *x;
        *x  = *y;
        *y  = tmp;
    }
}


/***************************************************************************//**
    Purpose:
    =============
    Swap vector x and y; \f$ x <-> y \f$.

    @param[in]
    n       Number of elements in vector x and y. n >= 0.

    @param[in,out]
    dx      COMPLEX array on GPU device.
            The n element vector x of dimension (1 + (n-1)*incx).

    @param[in]
    incx    Stride between consecutive elements of dx. incx != 0.

    @param[in,out]
    dy      COMPLEX array on GPU device.
            The n element vector y of dimension (1 + (n-1)*incy).

    @param[in]
    incy    Stride between consecutive elements of dy. incy != 0.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_swap
*******************************************************************************/
extern "C" void 
magmablas_cswap(
    magma_int_t n,
    magmaFloatComplex_ptr dx, magma_int_t incx, 
    magmaFloatComplex_ptr dy, magma_int_t incy,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( n, NB ) );
    cswap_kernel<<< grid, threads, 0, queue->cuda_stream() >>>( n, dx, incx, dy, incy );
}