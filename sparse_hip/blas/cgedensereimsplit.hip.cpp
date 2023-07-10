#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from sparse_hip/blas/zgedensereimsplit.hip.cpp, normal z -> c, Wed Feb 22 15:20:41 2023

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


// axpy kernel for matrices stored in the MAGMA format
__global__ void 
cgedensereimsplit_kernel( 
    int num_rows, 
    int num_cols,  
    magma_index_t* rowidx,
    magmaFloatComplex * A, 
    magmaFloatComplex * ReA, 
    magmaFloatComplex * ImA )
{
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if( row<num_rows ){
        for( j=0; j<num_cols; j++ ){
            ReA[ j ] = MAGMA_C_MAKE( MAGMA_C_REAL( A[ j ] ), 0.0 );
            ImA[ j ] = MAGMA_C_MAKE( MAGMA_C_IMAG( A[ j ] ), 0.0 );
        }
    }
}

/**
    Purpose
    -------
    
    This routine takes an input matrix A in DENSE format and located on the GPU
    and splits it into two matrixes ReA and ImA containing the real and the 
    imaginary contributions of A.
    The output matrices are allocated within the routine.
    
    Arguments
    ---------

    @param[in]
    A           magma_c_matrix
                input matrix A.
                
    @param[out]
    ReA         magma_c_matrix*
                output matrix contaning real contributions.
                
    @param[out]
    ImA         magma_c_matrix*
                output matrix contaning complex contributions.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" 
magma_int_t
magma_cgedensereimsplit(
    magma_c_matrix A,
    magma_c_matrix *ReA,
    magma_c_matrix *ImA,
    magma_queue_t queue )
{
    magma_cmtransfer( A, ReA, Magma_DEV, Magma_DEV, queue );
    magma_cmtransfer( A, ImA, Magma_DEV, Magma_DEV, queue );
        
    int m = A.num_rows;
    int n = A.num_cols;
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    hipLaunchKernelGGL(cgedensereimsplit_kernel, dim3(grid), dim3(threads), 0, queue->hip_stream() ,  m, n, A.row, A.dval, ReA->dval, ImA->dval );
                    
    return MAGMA_SUCCESS;
}
