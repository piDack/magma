#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from magmablas_hip/zaxpycp.hip.cpp, normal z -> d, Wed Feb 22 15:19:49 2023

*/
#include "magma_internal.h"

#define NB 64

/******************************************************************************/
// adds   x += r  --and--
// copies r = b
// each thread does one index, x[i] and r[i]
__global__ void
daxpycp_kernel(
    int m,
    double *r,
    double *x,
    const double *b)
{
    const int i = threadIdx.x + blockIdx.x*NB;
    if ( i < m ) {
        x[i] = MAGMA_D_ADD( x[i], r[i] );
        r[i] = b[i];
    }
}


/***************************************************************************//**
    adds   x += r  --and--
    copies r = b
*******************************************************************************/
extern "C" void
magmablas_daxpycp(
    magma_int_t m,
    magmaDouble_ptr r,
    magmaDouble_ptr x,
    magmaDouble_const_ptr b,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    hipLaunchKernelGGL(daxpycp_kernel, dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, r, x, b );
}
