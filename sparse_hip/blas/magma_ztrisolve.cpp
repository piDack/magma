/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Tobias Ribizel

       @precisions normal z -> s d c
*/
#include "magma_trisolve.h"

#define PRECISION_z

/* For hipSPARSE, they use a separate complex type than for hipBLAS */
#if defined(MAGMA_HAVE_HIP)
  #ifdef PRECISION_z
    #define hipblasDoubleComplex hipDoubleComplex
  #elif defined(PRECISION_c)
    #define hipblasComplex hipComplex
  #endif
#endif

magma_int_t magma_ztrisolve_analysis(magma_z_matrix M, magma_solve_info_t *solve_info, bool upper_triangular, bool unit_diagonal, bool transpose, magma_queue_t queue)
{
    magma_int_t info = 0;

    hipsparseHandle_t hipsparseHandle = NULL;
    hipsparseFillMode_t fill_mode = upper_triangular ? HIPSPARSE_FILL_MODE_UPPER
                                                    : HIPSPARSE_FILL_MODE_LOWER;
    hipsparseDiagType_t diag_type = unit_diagonal ? HIPSPARSE_DIAG_TYPE_UNIT
                                                 : HIPSPARSE_DIAG_TYPE_NON_UNIT;
    magmaDoubleComplex one = MAGMA_Z_ONE;
    hipsparseOperation_t op = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t M_op = transpose ? HIPSPARSE_OPERATION_TRANSPOSE
                                         : HIPSPARSE_OPERATION_NON_TRANSPOSE;
    CHECK_CUSPARSE(hipsparseCreate(&hipsparseHandle));
    CHECK_CUSPARSE(hipsparseSetStream(hipsparseHandle, queue->hip_stream()));

#if CUDA_VERSION >= 11031
    hipsparseSpMatDescr_t descr;
    hipsparseDnMatDescr_t in;
    hipsparseDnMatDescr_t out;
    {
        hipsparseSpSMAlg_t alg = CUSPARSE_SPSM_ALG_DEFAULT;
        hipblasDatatype_t data_type = HIPBLAS_C_64F;
        CHECK_CUSPARSE(hipsparseCreateCsr(&descr, M.num_rows, M.num_rows, M.nnz,
                                         M.drow, M.dcol, M.dval,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         HIPSPARSE_INDEX_BASE_ZERO, data_type));
        CHECK_CUSPARSE(hipsparseSpMatSetAttribute(descr, CUSPARSE_SPMAT_FILL_MODE,
                                                 &fill_mode, sizeof(fill_mode)));
        CHECK_CUSPARSE(hipsparseSpMatSetAttribute(descr, CUSPARSE_SPMAT_DIAG_TYPE,
                                                 &diag_type, sizeof(diag_type)));
        // create dummy input and output vectors with distinct non-null pointers
        // otherwise cuSPARSE complains, even though it doesn't use the vectors
        CHECK_CUSPARSE(hipsparseCreateDnMat(&in, M.num_rows, 1, M.num_rows,
                                           (void *)0xF0, data_type,
                                           CUSPARSE_ORDER_COL));
        CHECK_CUSPARSE(hipsparseCreateDnMat(&out, M.num_rows, 1, M.num_rows,
                                           (void *)0xE0, data_type,
                                           CUSPARSE_ORDER_COL));
        CHECK_CUSPARSE(hipsparseSpSM_createDescr(&solve_info->descr));
        size_t buffer_size = 0;
        CHECK_CUSPARSE(hipsparseSpSM_bufferSize(hipsparseHandle, M_op, op, &one,
                                               descr, in, out, data_type, alg,
                                               solve_info->descr, &buffer_size));
        if (buffer_size > 0)
            magma_malloc(&solve_info->buffer, buffer_size);
        CHECK_CUSPARSE(hipsparseSpSM_analysis(hipsparseHandle, M_op, op, &one,
                                             descr, in, out, data_type, alg,
                                             solve_info->descr,
                                             solve_info->buffer));
    }

cleanup:
    hipsparseDestroyDnMat(out);
    hipsparseDestroyDnMat(in);
    hipsparseDestroySpMat(descr);
#else
    hipsparseMatDescr_t descr;
    CHECK_CUSPARSE(hipsparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(hipsparseSetMatDiagType(descr, diag_type));
    CHECK_CUSPARSE(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(hipsparseSetMatFillMode(descr, fill_mode));
    {
        int algo = 0;
        size_t buffer_size = 0;
        CHECK_CUSPARSE(hipsparseCreateCsrsm2Info(&solve_info->descr));
        CHECK_CUSPARSE(hipsparseZcsrsm2_bufferSizeExt(hipsparseHandle, algo, M_op, op,
                                                     M.num_rows, 1, M.nnz, (const hipblasDoubleComplex*)&one,
                                                     descr, (hipblasDoubleComplex*)M.dval, M.drow, M.dcol,
                                                     NULL, M.num_rows,
                                                     solve_info->descr,
                                                     HIPSPARSE_SOLVE_POLICY_NO_LEVEL,
                                                     &buffer_size));
        if (buffer_size > 0)
            magma_malloc(&solve_info->buffer, buffer_size);
        CHECK_CUSPARSE(hipsparseZcsrsm2_analysis(hipsparseHandle, algo, M_op, op,
                                                M.num_rows, 1, M.nnz, (const hipblasDoubleComplex*)&one, descr,
                                                (hipblasDoubleComplex*)M.dval, M.drow, M.dcol, NULL,
                                                M.num_rows, solve_info->descr,
                                                HIPSPARSE_SOLVE_POLICY_NO_LEVEL,
                                                solve_info->buffer));
    }

cleanup:
    hipsparseDestroyMatDescr(descr);
#endif
    hipsparseDestroy(hipsparseHandle);

    return info;
}

magma_int_t magma_ztrisolve(magma_z_matrix M, magma_solve_info_t solve_info, bool upper_triangular, bool unit_diagonal, bool transpose, magma_z_matrix b, magma_z_matrix x, magma_queue_t queue)
{
    magma_int_t info = 0;

    hipsparseHandle_t hipsparseHandle = NULL;
    hipsparseFillMode_t fill_mode = upper_triangular ? HIPSPARSE_FILL_MODE_UPPER
                                                    : HIPSPARSE_FILL_MODE_LOWER;
    hipsparseDiagType_t diag_type = unit_diagonal ? HIPSPARSE_DIAG_TYPE_UNIT
                                                 : HIPSPARSE_DIAG_TYPE_NON_UNIT;
    magmaDoubleComplex one = MAGMA_Z_ONE;
    hipsparseOperation_t op = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t M_op = transpose ? HIPSPARSE_OPERATION_TRANSPOSE
                                         : HIPSPARSE_OPERATION_NON_TRANSPOSE;
    CHECK_CUSPARSE(hipsparseCreate(&hipsparseHandle));
    CHECK_CUSPARSE(hipsparseSetStream(hipsparseHandle, queue->hip_stream()));

#if CUDA_VERSION >= 11031
    hipsparseSpMatDescr_t descr;
    hipsparseDnMatDescr_t in;
    hipsparseDnMatDescr_t out;
    {
        hipsparseSpSMAlg_t alg = CUSPARSE_SPSM_ALG_DEFAULT;
        hipblasDatatype_t data_type = HIPBLAS_C_64F;
        CHECK_CUSPARSE(hipsparseCreateCsr(&descr, M.num_rows, M.num_rows, M.nnz,
                                         M.drow, M.dcol, M.dval,
                                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                         HIPSPARSE_INDEX_BASE_ZERO, data_type));
        CHECK_CUSPARSE(hipsparseSpMatSetAttribute(descr, CUSPARSE_SPMAT_FILL_MODE,
                                                 &fill_mode, sizeof(fill_mode)));
        CHECK_CUSPARSE(hipsparseSpMatSetAttribute(descr, CUSPARSE_SPMAT_DIAG_TYPE,
                                                 &diag_type, sizeof(diag_type)));
        CHECK_CUSPARSE(hipsparseCreateDnMat(&in, b.num_rows, b.num_cols, b.num_rows,
                                           b.dval, data_type, CUSPARSE_ORDER_COL));
        CHECK_CUSPARSE(hipsparseCreateDnMat(&out, x.num_rows, x.num_cols, x.num_rows,
                                           x.dval, data_type, CUSPARSE_ORDER_COL));
        CHECK_CUSPARSE(hipsparseSpSM_solve(hipsparseHandle, M_op, op, &one, descr,
                                          in, out, data_type, alg,
                                          solve_info.descr));
    }

cleanup:
    hipsparseDestroyDnMat(out);
    hipsparseDestroyDnMat(in);
    hipsparseDestroySpMat(descr);
#else
    hipsparseMatDescr_t descr;
    CHECK_CUSPARSE(hipsparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(hipsparseSetMatType(descr, HIPSPARSE_MATRIX_TYPE_GENERAL));
    CHECK_CUSPARSE(hipsparseSetMatDiagType(descr, diag_type));
    CHECK_CUSPARSE(hipsparseSetMatIndexBase(descr, HIPSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(hipsparseSetMatFillMode(descr, fill_mode));
    {
        int algo = 0;
        magmablas_zlacpy(MagmaFull, M.num_rows, b.num_cols, b.dval, M.num_rows,
                         x.dval, M.num_rows, queue);
        CHECK_CUSPARSE(hipsparseZcsrsm2_solve(hipsparseHandle, algo, M_op, op,
                                             M.num_rows, b.num_cols, M.nnz, (const hipblasDoubleComplex*)&one,
                                             descr, (hipblasDoubleComplex*)M.dval, M.drow, M.dcol, (hipblasDoubleComplex*)x.dval,
                                             M.num_rows, solve_info.descr,
                                             HIPSPARSE_SOLVE_POLICY_NO_LEVEL,
                                             solve_info.buffer));
    }

cleanup:
    hipsparseDestroyMatDescr(descr);
#endif
    hipsparseDestroy(hipsparseHandle);

    return info;
}
