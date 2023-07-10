/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from sparse_hip/blas/magma_zcuspaxpy.cpp, normal z -> c, Wed Feb 22 15:20:38 2023
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_slamch( "E" )
#define ATOLERANCE     lapackf77_slamch( "E" )

// todo: alpha and beta passed, dvalA, dvalB, valC, colC; buf passed to next
#if CUDA_VERSION >= 11000
#define hipsparseXcsrgeamNnz(handle, m, n, descrA, nnzA, drowA, dcolA,                           \
                            descrB, nnzB, drowB, dcolB, descrC, rowC, nnzTotal )                \
    {                                                                                           \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        hipblasComplex alpha = MAGMA_C_ONE, beta = MAGMA_C_ZERO;                               \
        hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST);                             \
        hipsparseCcsrgeam2_bufferSizeExt(handle, m, n, &alpha,                                   \
                                        descrA, nnzA, NULL, drowA, dcolA,                       \
                                        &beta,                                                  \
                                        descrB, nnzB, NULL, drowB, dcolB,                       \
                                        descrC, NULL, rowC, NULL, &bufsize);                    \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseXcsrgeam2Nnz(handle, m, n,                                                      \
                             descrA, nnzA, drowA, dcolA,                                        \
                             descrB, nnzB, drowB, dcolB,                                        \
                             descrC, rowC, nnzTotal, buf);                                      \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }
#endif

// todo: buf is passed from above
#if CUDA_VERSION >= 11000
#define hipsparseCcsrgeam(handle, m, n, alpha, descrA, nnzA, dvalA, drowA, dcolA, beta,          \
                         descrB, nnzB, dvalB, drowB, dcolB, descrC, dvalC, drowC, dcolC )       \
    {                                                                                           \
        void *buf;                                                                              \
        printf("hipsparseCcsrgeam bufsize = %d\n", 0);\
        hipsparseCcsrgeam2(handle, m, n, alpha,                                                  \
                          descrA, nnzA, dvalA, drowA, dcolA,                                    \
                          beta,                                                                 \
                          descrB, nnzB, dvalB, drowB, dcolB,                                    \
                          descrC, dvalC, drowC, dcolC,                                          \
                          buf);                                                                 \
    }
#endif

/**
    Purpose
    -------

    This is an interface to the cuSPARSE routine csrgeam computing the sum
    of two sparse matrices stored in csr format:

        C = alpha * A + beta * B


    Arguments
    ---------

    @param[in]
    alpha       magmaFloatComplex*
                scalar

    @param[in]
    A           magma_c_matrix
                input matrix

    @param[in]
    beta        magmaFloatComplex*
                scalar

    @param[in]
    B           magma_c_matrix
                input matrix

    @param[out]
    AB          magma_c_matrix*
                output matrix AB = alpha * A + beta * B

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_cblas
    ********************************************************************/

extern "C" magma_int_t
magma_ccuspaxpy(
    magmaFloatComplex *alpha, magma_c_matrix A,
    magmaFloatComplex *beta, magma_c_matrix B,
    magma_c_matrix *AB,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_c_matrix C={Magma_CSR};
    C.num_rows = A.num_rows;
    C.num_cols = A.num_cols;
    C.storage_type = A.storage_type;
    C.memory_location = A.memory_location;
    C.val = NULL;
    C.col = NULL;
    C.row = NULL;
    C.rowidx = NULL;
    C.blockinfo = NULL;
    C.diag = NULL;
    C.dval = NULL;
    C.dcol = NULL;
    C.drow = NULL;
    C.drowidx = NULL;
    C.ddiag = NULL;
   
    magma_index_t base_t, nnz_t, baseC;
    
    hipsparseHandle_t handle=NULL;
    hipsparseMatDescr_t descrA=NULL;
    hipsparseMatDescr_t descrB=NULL;
    hipsparseMatDescr_t descrC=NULL;
                             
    if (    A.memory_location == Magma_DEV
        && B.memory_location == Magma_DEV
        && ( A.storage_type == Magma_CSR ||
             A.storage_type == Magma_CSRCOO )
        && ( B.storage_type == Magma_CSR ||
             B.storage_type == Magma_CSRCOO ) )
    {
        // CUSPARSE context //
        CHECK_CUSPARSE( hipsparseCreate( &handle ));
        CHECK_CUSPARSE( hipsparseSetStream( handle, queue->hip_stream() ));
        CHECK_CUSPARSE( hipsparseCreateMatDescr( &descrA ));
        CHECK_CUSPARSE( hipsparseCreateMatDescr( &descrB ));
        CHECK_CUSPARSE( hipsparseCreateMatDescr( &descrC ));
        CHECK_CUSPARSE( hipsparseSetMatType( descrA, HIPSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( hipsparseSetMatType( descrB, HIPSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( hipsparseSetMatType( descrC, HIPSPARSE_MATRIX_TYPE_GENERAL ));
        CHECK_CUSPARSE( hipsparseSetMatIndexBase( descrA, HIPSPARSE_INDEX_BASE_ZERO ));
        CHECK_CUSPARSE( hipsparseSetMatIndexBase( descrB, HIPSPARSE_INDEX_BASE_ZERO ));
        CHECK_CUSPARSE( hipsparseSetMatIndexBase( descrC, HIPSPARSE_INDEX_BASE_ZERO ));


        // nnzTotalDevHostPtr points to host memory
        magma_index_t *nnzTotalDevHostPtr = (magma_index_t*) &C.nnz;
        CHECK_CUSPARSE( hipsparseSetPointerMode( handle, HIPSPARSE_POINTER_MODE_HOST ));
        CHECK( magma_index_malloc( &C.drow, (A.num_rows + 1) ));
        hipsparseXcsrgeamNnz( handle, A.num_rows, A.num_cols,
                             descrA, A.nnz, A.drow, A.dcol,
                             descrB, B.nnz, B.drow, B.dcol,
                             descrC, C.row, nnzTotalDevHostPtr );

        if (NULL != nnzTotalDevHostPtr) {
            C.nnz = *nnzTotalDevHostPtr;
        } else {
            // workaround as nnz and base C are magma_int_t
            magma_index_getvector( 1, C.drow+C.num_rows, 1, &nnz_t, 1, queue );
            magma_index_getvector( 1, C.drow,   1, &base_t,    1, queue );
            C.nnz = (magma_int_t) nnz_t;
            baseC = (magma_int_t) base_t;
            C.nnz -= baseC;
        }
        CHECK( magma_index_malloc( &C.dcol, C.nnz ));
        CHECK( magma_cmalloc( &C.dval, C.nnz ));
        #ifdef MAGMA_HAVE_HIP
        hipsparseCcsrgeam( handle, A.num_rows, A.num_cols,
                          (const hipFloatComplex*)alpha,
                          descrA, A.nnz,
                          (const hipFloatComplex*)A.dval, A.drow, A.dcol,
                          (const hipFloatComplex*)beta,
                          descrB, B.nnz,
                          (const hipFloatComplex*)B.dval, B.drow, B.dcol,
                          descrC,
                          (hipFloatComplex*)C.dval, C.drow, C.dcol );
        #else
        hipsparseCcsrgeam( handle, A.num_rows, A.num_cols,
                          alpha,
                          descrA, A.nnz,
                          A.dval, A.drow, A.dcol,
                          beta,
                          descrB, B.nnz,
                          B.dval, B.drow, B.dcol,
                          descrC,
                          C.dval, C.drow, C.dcol );
        #endif

        // end CUSPARSE context //

        CHECK( magma_cmtransfer( C, AB, Magma_DEV, Magma_DEV, queue ));
    }
    else {
        info = MAGMA_ERR_NOT_SUPPORTED; 
    }
    
cleanup:
    hipsparseDestroyMatDescr( descrA );
    hipsparseDestroyMatDescr( descrB );
    hipsparseDestroyMatDescr( descrC );
    hipsparseDestroy( handle );
    magma_cmfree( &C, queue );
    return info;
}
