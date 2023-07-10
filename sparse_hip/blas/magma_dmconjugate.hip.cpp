#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from sparse_hip/blas/magma_zmconjugate.hip.cpp, normal z -> d, Wed Feb 22 15:20:41 2023
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


__global__ void 
magma_dmconjugate_kernel(  
    int num_rows,
    magma_index_t *rowptr, 
    double *values )
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < num_rows ){
        for( int i = rowptr[row]; i < rowptr[row+1]; i++){
            values[i] = MAGMA_D_CONJ( values[i] );
        }
    }
}



/**
    Purpose
    -------

    This function conjugates a matrix. For a real matrix, no value is changed.

    Arguments
    ---------

    @param[in,out]
    A           magma_d_matrix*
                input/output matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_daux
    ********************************************************************/

extern "C" magma_int_t
magma_dmconjugate(
    magma_d_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    dim3 grid( magma_ceildiv( A->num_rows, BLOCK_SIZE ));
    hipLaunchKernelGGL(magma_dmconjugate_kernel, dim3(grid), dim3(BLOCK_SIZE), 0, queue->hip_stream() ,  A->num_rows, A->drow, A->dval );
        
    return info;
}
