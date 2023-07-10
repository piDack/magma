#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from magmablas_hip/zlascl2.hip.cpp, normal z -> s, Wed Feb 22 15:19:54 2023

       @author Theo Mary
*/
#include "magma_internal.h"

#define NB 64


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right.
__global__ void
slascl2_full(int m, int n, const float* D, float* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    float mul = D[ind];
    A += ind;
    if (ind < m) {
        for (int j=0; j < n; j++ )
            A[j*lda] *= mul;
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__global__ void
slascl2_lower(int m, int n, const float* D, float* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    int break_d = (ind < n) ? ind : n-1;

    float mul = D[ind];
    A += ind;
    if (ind < m) {
        for (int j=0; j <= break_d; j++ )
            A[j*lda] *= mul;
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__global__ void
slascl2_upper(int m, int n, const float *D, float* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    float mul = D[ind];
    A += ind;
    if (ind < m) {
        for (int j=n-1; j >= ind; j--)
            A[j*lda] *= mul;
    }
}


/***************************************************************************//**
    Purpose
    -------
    SLASCL2 scales the M by N real matrix A by the real diagonal matrix dD.
    TYPE specifies that A may be full, upper triangular, lower triangular.

    Arguments
    ---------
    @param[in]
    type    magma_type_t
            TYPE indices the storage type of the input matrix A.
            = MagmaFull:   full matrix.
            = MagmaLower:  lower triangular matrix.
            = MagmaUpper:  upper triangular matrix.
            Other formats that LAPACK supports, MAGMA does not currently support.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in]
    dD      REAL vector, dimension (M)
            The diagonal matrix containing the scalar factors. Stored as a vector.

    @param[in,out]
    dA      REAL array, dimension (LDDA,N)
            The matrix to be scaled by dD.  See TYPE for the
            storage type.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @see magma_slascl_diag 
    @ingroup magma_lascl_diag
*******************************************************************************/
extern "C" void
magmablas_slascl2(
    magma_type_t type, magma_int_t m, magma_int_t n,
    magmaFloat_const_ptr dD,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info )
{
    *info = 0;
    if ( type != MagmaLower && type != MagmaUpper && type != MagmaFull )
        *info = -1;
    else if ( m < 0 )
        *info = -2;
    else if ( n < 0 )
        *info = -3;
    else if ( ldda < max(1,m) )
        *info = -5;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;  //info;
    }
    
    dim3 grid( magma_ceildiv( m, NB ) );
    dim3 threads( NB );
    
    if (type == MagmaLower) {
        hipLaunchKernelGGL(slascl2_lower, dim3(grid), dim3(threads), 0, queue->hip_stream() , m, n, dD, dA, ldda);
    }
    else if (type == MagmaUpper) {
        hipLaunchKernelGGL(slascl2_upper, dim3(grid), dim3(threads), 0, queue->hip_stream() , m, n, dD, dA, ldda);
    }
    else if (type == MagmaFull) {
        hipLaunchKernelGGL(slascl2_full, dim3(grid), dim3(threads), 0, queue->hip_stream() , m, n, dD, dA, ldda);
    }
}
