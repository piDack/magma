#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from magmablas_hip/zlarfbx.hip.cpp, normal z -> c, Wed Feb 22 15:19:53 2023

*/
#include "magma_internal.h"
#include "commonblas_c.h"
#include "magma_templates.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512


/******************************************************************************/
extern "C"
__global__ void 
magma_cgemv_kernel1(int m, const magmaFloatComplex * __restrict__ V, int ldv, 
                    const magmaFloatComplex * __restrict__ c, 
                    magmaFloatComplex *dwork)
{
    const int i = threadIdx.x;
    const magmaFloatComplex *dV = V + (blockIdx.x) * ldv;

    __shared__ magmaFloatComplex sum[ BLOCK_SIZE ];
    magmaFloatComplex lsum;

    /*  lsum := v**H * C  */
    lsum = MAGMA_C_ZERO;
    for (int j = i; j < m; j += BLOCK_SIZE)
       lsum += MAGMA_C_MUL( MAGMA_C_CONJ( dV[j] ), c[j] );
    
    sum[i] = lsum;
    magma_sum_reduce< BLOCK_SIZE >( i, sum );

    __syncthreads();
    if (i == 0)
       dwork [blockIdx.x] = sum[0];
}

/******************************************************************************/
/*
    Call 
        hipLaunchKernelGGL(magma_cgemv_kernel3, dim3(n), dim3(BLOCK_SIZE), 0, queue->hip_stream() , m, V, ldv, c, dwork, tau)
    to compute
        CGEMV( "Conjugate transpose", m, n, -tau[0], V, ldv, c, 1, zero, dwork, 1)
        and to set c[0] to 1.
    i.e., 
        work = -tau[0] V**H c
*/
extern "C"
__global__ void
magma_cgemv_kernel3(int m, const magmaFloatComplex * __restrict__ V, int ldv, magmaFloatComplex *c,
                    magmaFloatComplex *dwork, magmaFloatComplex *tau)
{
    const int i = threadIdx.x;
    const magmaFloatComplex *dV = V + (blockIdx.x) * ldv;

    __shared__ magmaFloatComplex sum[ BLOCK_SIZE ];
    magmaFloatComplex lsum;

    if (i == 0)
       c[0] = MAGMA_C_ONE;           

    /*  lsum := v**H * C  */
    lsum = MAGMA_C_ZERO;
    for (int j = i; j < m; j += BLOCK_SIZE)
       lsum += MAGMA_C_MUL( MAGMA_C_CONJ( dV[j] ), c[j] );

    sum[i] = lsum;
    magma_sum_reduce< BLOCK_SIZE >( i, sum );

    __syncthreads();
    if (i == 0)
       dwork [blockIdx.x] = -tau[0]*sum[0];
}


/******************************************************************************/
extern "C"
__global__ void
magma_cgemv_kernel2(int m, int n, const magmaFloatComplex * __restrict__ V, int ldv, 
                    const magmaFloatComplex * __restrict__ x, magmaFloatComplex *c)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    magmaFloatComplex lsum;

    V += j;

    lsum = MAGMA_C_ZERO;
    if (j < m) {
        for (int k=0; k < n; k++)
            lsum += MAGMA_C_MUL( V[k*ldv], x[k]);
        
        c[j] -= lsum;
    }
}


/******************************************************************************/
/*
    Apply a complex block reflector H to a complex vector C from the left
    (i.e., C = H C). H is represented in the form
          H = I - V T V**H
    where T is the complex k-by-k upper triangular matrix in the 
    representation of the block reflector, and V is a complex block of
    k elementary reflectors. 
*/
extern "C" void
magma_clarfbx_gpu(
    magma_int_t m, magma_int_t k,
    magmaFloatComplex_ptr V,  magma_int_t ldv,
    magmaFloatComplex_ptr dT, magma_int_t ldt,
    magmaFloatComplex_ptr c,
    magmaFloatComplex_ptr dwork,
    magma_queue_t queue )
{
    /* dwork = V**H c     */
    hipLaunchKernelGGL(magma_cgemv_kernel1, dim3(k), dim3(BLOCK_SIZE), 0, queue->hip_stream() , m, V, ldv, c, dwork); 

    /* dwork = T**H dwork */
    hipLaunchKernelGGL(magma_ctrmv_tkernel, dim3(k), dim3(k), 0, queue->hip_stream() ,  dT, ldt, dwork, dwork+k);
 
    /* c = c - V dwork    */
    dim3  blocks3( magma_ceildiv( m, BLOCK_SIZE ) );
    dim3 threads3( BLOCK_SIZE );     
    hipLaunchKernelGGL(magma_cgemv_kernel2, dim3(blocks3), dim3(threads3), 0, queue->hip_stream() ,  m, k, V, ldv, dwork+k, c);
}
