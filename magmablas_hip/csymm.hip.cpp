/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from magmablas_hip/zsymm.hip.cpp, normal z -> c, Wed Feb 22 15:19:56 2023

       @author Ahmad Abdelfattah
       
*/
#include "magma_internal.h"

#define PRECISION_c
#include "hemm_template_kernel.hip.hpp"

#if defined(PRECISION_z)
#define SYMM_LEFT    8, 16, 16, 0
#define SYMM_RIGHT   8, 16, 16, 0
#elif defined(PRECISION_c)
#define SYMM_LEFT    16, 32, 32, 0
#define SYMM_RIGHT   16, 32, 32, 0
#elif defined(PRECISION_d)
#define SYMM_LEFT    16, 32, 32, 0
#define SYMM_RIGHT   16, 32, 32, 0
#else
#define SYMM_LEFT    16, 32, 32, 0
#define SYMM_RIGHT   16, 32, 32, 0
#endif

/***************************************************************************//**
    Purpose
    -------
    CSYMM performs one of the matrix-matrix operations

        C := alpha*A*B + beta*C,
    or
        C := alpha*B*A + beta*C,

    where alpha and beta are scalars, A is a symmetric matrix, and
    B and C are m by n matrices.

    Arguments
    ---------
    @param[in]
    side    magma_side_t
            On entry, side specifies whether the symmetric matrix A
            appears on the left or right in the operation as follows:

            SIDE = MagmaLeft    C := alpha*A*B + beta*C,
            SIDE = MagmaRight   C := alpha*B*A + beta*C.


    @param[in]
    uplo    magma_uplo_t
            On entry, uplo specifies whether the upper or lower
            triangular part of the symmetric matrix A is to be
            referenced as follows:

            uplo = MagmaUpper   Only the upper triangular part of the
                                symmetric matrix is to be referenced.
            uplo = MagmaLower   Only the lower triangular part of the
                                symmetric matrix is to be referenced.

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of C.
            m >= 0.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of C.
            n >= 0.

    @param[in]
    alpha   COMPLEX
            On entry, alpha specifies the scalar alpha.

    @param[in]
    dA      COMPLEX array A of DIMENSION ( ldda, ka ), where ka is
            m when side = MagmaLower and is n otherwise.
            Before entry with side = MagmaLeft, the m by m part of
            the array A must contain the symmetric matrix, such that
            when uplo = MagmaUpper, the leading m by m upper triangular
            part of the array A must contain the upper triangular part
            of the symmetric matrix and the strictly lower triangular
            part of A is not referenced, and when uplo = MagmaLower,
            the leading m by m lower triangular part of the array A
            must contain the lower triangular part of the symmetric
            matrix and the strictly upper triangular part of A is not
            referenced.
            Before entry with side = MagmaRight, the n by n part of
            the array A must contain the symmetric matrix, such that
            when uplo = MagmaUpper, the leading n by n upper triangular
            part of the array A must contain the upper triangular part
            of the symmetric matrix and the strictly lower triangular
            part of A is not referenced, and when uplo = MagmaLower,
            the leading n by n lower triangular part of the array A
            must contain the lower triangular part of the symmetric
            matrix and the strictly upper triangular part of A is not
            referenced.
            Note that the imaginary parts of the diagonal elements need
            not be set, they are assumed to be zero.

    @param[in]
    ldda    INTEGER
            On entry, ldda specifies the first dimension of A as declared
            in the calling (sub) program.
            When side = MagmaLower then ldda >= max( 1, m ),
            otherwise                   ldda >= max( 1, n ).

    @param[in]
    dB      COMPLEX array B of DIMENSION ( lddb, n ).
            Before entry, the leading m by n part of the array B must
            contain the matrix B.

    @param[in]
    lddb    INTEGER
            On entry, lddb specifies the first dimension of B as declared
            in the calling (sub) program. LDDB >= max( 1, m ).

    @param[in]
    beta    COMPLEX
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then C need not be set on input.

    @param[in,out]
    dC      COMPLEX array C of DIMENSION ( lddc, n ).
            Before entry, the leading m by n part of the array C must
            contain the matrix C, except when beta is zero, in which
            case C need not be set on entry.
            On exit, the array C is overwritten by the m by n updated
            matrix.

    @param[in]
    lddc    INTEGER
            On entry, lddc specifies the first dimension of C as declared
            in the calling (sub) program. lddc >= max( 1, m ).

    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    

    @ingroup magma_hemm
*******************************************************************************/
extern "C" 
void
magmablas_csymm(
    magma_side_t side, magma_uplo_t uplo,
    magma_int_t m, magma_int_t n,
    magmaFloatComplex alpha,
    magmaFloatComplex_const_ptr dA, magma_int_t ldda,
    magmaFloatComplex_const_ptr dB, magma_int_t lddb,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr       dC, magma_int_t lddc,
    magma_queue_t queue )
{
    magma_int_t nrowa = (side == MagmaLeft ? m : n);
    magma_int_t info = 0;
    if ( side != MagmaLeft && side != MagmaRight ) {
        info = -1;
    } else if (uplo != MagmaLower && uplo != MagmaUpper ) {
        info = -2;
    } else if ( m < 0 ) {
        info = -3;
    } else if ( n < 0 ) {
        info = -4;
    } else if ( ldda < max(1,nrowa) ) {
        info = -7;
    } else if ( lddb < max(1,m) ) {
        info = -9;
    } else if (lddc < max(1,m)) {
        info = -12;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if( side == MagmaLeft ) { 
        hemm_template<magmaFloatComplex, SYMM_LEFT>
        (side, uplo, m, n, dA, ldda, dB, lddb, dC, lddc, alpha, beta, queue);
    }
    else {
        hemm_template<magmaFloatComplex, SYMM_RIGHT>
        (side, uplo, m, n, dA, ldda, dB, lddb, dC, lddc, alpha, beta, queue);
    }
}
