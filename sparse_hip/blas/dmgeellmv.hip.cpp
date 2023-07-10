#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from sparse_hip/blas/zmgeellmv.hip.cpp, normal z -> d, Wed Feb 22 15:20:37 2023

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512


__global__ void 
dmgeellmv_kernel( 
    int num_rows, 
    int num_cols,
    int num_vecs,
    int num_cols_per_row,
    double alpha, 
    double * dval, 
    magma_index_t * dcolind,
    double * dx,
    double beta, 
    double * dy)
{
int row = blockDim.x * blockIdx.x + threadIdx.x;

    HIP_DYNAMIC_SHARED( double, dot)

    if(row < num_rows ){
        for( int i=0; i<num_vecs; i++)
                dot[ threadIdx.x + i*blockDim.x ] = MAGMA_D_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_cols_per_row * row + n ];
            double val = dval [ num_cols_per_row * row + n ];
            if( val != 0){
                for( int i=0; i<num_vecs; i++)
                    dot[ threadIdx.x + i*blockDim.x ] += 
                                    val * dx[col + i * num_cols ];
            }
        }
        for( int i=0; i<num_vecs; i++)
                dy[ row + i*num_cols ] = dot[ threadIdx.x + i*blockDim.x ] 
                                * alpha + beta * dy [ row + i * num_cols ];
    }
}


/**
    Purpose
    -------
    
    This routine computes Y = alpha *  A *  X + beta * Y for X and Y sets of 
    num_vec vectors on the GPU. Input format is ELLPACK. 
    
    Arguments
    ---------

    @param[in]
    transA      magma_trans_t
                transposition parameter for A

    @param[in]
    m           magma_int_t
                number of rows in A

    @param[in]
    n           magma_int_t
                number of columns in A 
                              
    @param[in]
    num_vecs    mama_int_t
                number of vectors
                
    @param[in]
    nnz_per_row magma_int_t
                number of elements in the longest row 
                
    @param[in]
    alpha       double
                scalar multiplier

    @param[in]
    dval        magmaDouble_ptr
                array containing values of A in ELLPACK

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in ELLPACK

    @param[in]
    dx          magmaDouble_ptr
                input vector x

    @param[in]
    beta        double
                scalar multiplier

    @param[out]
    dy          magmaDouble_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dblas
    ********************************************************************/

extern "C" magma_int_t
magma_dmgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
    magma_int_t nnz_per_row,
    double alpha,
    magmaDouble_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDouble_ptr dx,
    double beta,
    magmaDouble_ptr dy,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    unsigned int MEM_SIZE =  num_vecs* BLOCK_SIZE 
                            * sizeof( double ); // num_vecs vectors 
    hipLaunchKernelGGL(dmgeellmv_kernel, dim3(grid), dim3(threads), MEM_SIZE, queue->hip_stream() ,  m, n, num_vecs, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );


    return MAGMA_SUCCESS;
}
