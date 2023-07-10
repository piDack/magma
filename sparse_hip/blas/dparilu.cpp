/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Hartwig Anzt

       @generated from sparse_hip/blas/zparilu.cpp, normal z -> d, Wed Feb 22 15:20:39 2023
*/
#include "magmasparse_internal.h"

#include "magma_trisolve.h"

#define PRECISION_d

// This file is deprecated and will be removed in future.
// The ParILU/ParIC functionality is provided by 
// src/dparilu_gpu.cpp and src/dparic_gpu.cpp
// 

/**
    Purpose
    -------

    Prepares the ILU preconditioner via the iterative ILU iteration.

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A
                
    @param[in]
    b           magma_d_matrix
                input RHS b

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_dparilusetup(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_d_matrix hAh={Magma_CSR}, hA={Magma_CSR}, hL={Magma_CSR}, hU={Magma_CSR},
    hAcopy={Magma_CSR}, hAL={Magma_CSR}, hAU={Magma_CSR}, hAUt={Magma_CSR},
    hUT={Magma_CSR}, hAtmp={Magma_CSR}, hACSRCOO={Magma_CSR}, dAinitguess={Magma_CSR},
    dL={Magma_CSR}, dU={Magma_CSR};

    // copy original matrix as CSRCOO to device
    CHECK( magma_dmtransfer(A, &hAh, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_dmconvert( hAh, &hA, hAh.storage_type, Magma_CSR , queue ));
    magma_dmfree(&hAh, queue );

    CHECK( magma_dmtransfer( hA, &hAcopy, Magma_CPU, Magma_CPU , queue ));

    // in case using fill-in
    CHECK( magma_dsymbilu( &hAcopy, precond->levels, &hAL, &hAUt,  queue ));
    // add a unit diagonal to L for the algorithm
    CHECK( magma_dmLdiagadd( &hAL , queue ));
    // transpose U for the algorithm
    CHECK( magma_d_cucsrtranspose(  hAUt, &hAU , queue ));
    magma_dmfree( &hAUt , queue );

    // ---------------- initial guess ------------------- //
    CHECK( magma_dmconvert( hAcopy, &hACSRCOO, Magma_CSR, Magma_CSRCOO , queue ));
    CHECK( magma_dmtransfer( hACSRCOO, &dAinitguess, Magma_CPU, Magma_DEV , queue ));
    magma_dmfree(&hACSRCOO, queue );
    magma_dmfree(&hAcopy, queue );

    // transfer the factor L and U
    CHECK( magma_dmtransfer( hAL, &dL, Magma_CPU, Magma_DEV , queue ));
    CHECK( magma_dmtransfer( hAU, &dU, Magma_CPU, Magma_DEV , queue ));
    magma_dmfree(&hAL, queue );
    magma_dmfree(&hAU, queue );

    for(int i=0; i<precond->sweeps; i++){
        CHECK( magma_dparilu_csr( dAinitguess, dL, dU , queue ));
    }

    CHECK( magma_dmtransfer( dL, &hL, Magma_DEV, Magma_CPU , queue ));
    CHECK( magma_dmtransfer( dU, &hU, Magma_DEV, Magma_CPU , queue ));
    CHECK( magma_d_cucsrtranspose(  hU, &hUT , queue ));

    magma_dmfree(&dL, queue );
    magma_dmfree(&dU, queue );
    magma_dmfree(&hU, queue );
    CHECK( magma_dmlumerge( hL, hUT, &hAtmp, queue ));

    magma_dmfree(&hL, queue );
    magma_dmfree(&hUT, queue );

    CHECK( magma_dmtransfer( hAtmp, &precond->M, Magma_CPU, Magma_DEV , queue ));

    hAL.diagorder_type = Magma_UNITY;
    CHECK( magma_dmconvert(hAtmp, &hAL, Magma_CSR, Magma_CSRL, queue ));
    hAL.storage_type = Magma_CSR;
    CHECK( magma_dmconvert(hAtmp, &hAU, Magma_CSR, Magma_CSRU, queue ));
    hAU.storage_type = Magma_CSR;

    magma_dmfree(&hAtmp, queue );

    // CHECK( magma_dcsrsplit( 0, 256, hAL, &DL, &RL , queue ));
    // CHECK( magma_dcsrsplit( 0, 256, hAU, &DU, &RU , queue ));
    // 
    // CHECK( magma_dmtransfer( DL, &precond->LD, Magma_CPU, Magma_DEV , queue ));
    // CHECK( magma_dmtransfer( DU, &precond->UD, Magma_CPU, Magma_DEV , queue ));

    // for hipsparse uncomment this
    CHECK( magma_dmtransfer( hAL, &precond->L, Magma_CPU, Magma_DEV , queue ));
    CHECK( magma_dmtransfer( hAU, &precond->U, Magma_CPU, Magma_DEV , queue ));
    
/*

    //-- for ba-solve uncomment this

    if( RL.nnz != 0 )
        CHECK( magma_dmtransfer( RL, &precond->L, Magma_CPU, Magma_DEV , queue ));
    else {
        precond->L.nnz = 0;
        precond->L.val = NULL;
        precond->L.col = NULL;
        precond->L.row = NULL;
        precond->L.blockinfo = NULL;
    }

    if( RU.nnz != 0 )
        CHECK( magma_dmtransfer( RU, &precond->U, Magma_CPU, Magma_DEV , queue ));
    else {
        precond->U.nnz = 0;
        precond->L.val = NULL;
        precond->L.col = NULL;
        precond->L.row = NULL;
        precond->L.blockinfo = NULL;
    }

    //-- for ba-solve uncomment this
*/

        // extract the diagonal of L into precond->d
    CHECK( magma_djacobisetup_diagscal( precond->L, &precond->d, queue ));
    CHECK( magma_dvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_D_ZERO, queue ));
    
    // extract the diagonal of U into precond->d2
    CHECK( magma_djacobisetup_diagscal( precond->U, &precond->d2, queue ));
    CHECK( magma_dvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_D_ZERO, queue ));

    magma_dmfree(&hAL, queue );
    magma_dmfree(&hAU, queue );
    // magma_dmfree(&DL, queue );
    // magma_dmfree(&RL, queue );
    // magma_dmfree(&DU, queue );
    // magma_dmfree(&RU, queue );

    CHECK(magma_dtrisolve_analysis(precond->L, &precond->cuinfoL, false, false, false, queue));
    CHECK(magma_dtrisolve_analysis(precond->U, &precond->cuinfoU, true, false, false, queue));
    
    
    
cleanup:
    magma_dmfree( &hAh, queue );
    magma_dmfree( &hA, queue );
    magma_dmfree( &hL, queue );
    magma_dmfree( &hU, queue );
    magma_dmfree( &hAcopy, queue );
    magma_dmfree( &hAL, queue );
    magma_dmfree( &hAU, queue );
    magma_dmfree( &hAUt, queue );
    magma_dmfree( &hUT, queue );
    magma_dmfree( &hAtmp, queue );
    magma_dmfree( &hACSRCOO, queue );
    magma_dmfree( &dAinitguess, queue );
    magma_dmfree( &dL, queue );
    magma_dmfree( &dU, queue );
    // magma_dmfree( &DL, queue );
    // magma_dmfree( &DU, queue );
    // magma_dmfree( &RL, queue );
    // magma_dmfree( &RU, queue );

    return info;
}



/**
    Purpose
    -------

    Updates an existing preconditioner via additional iterative ILU sweeps for
    previous factorization initial guess (PFIG).
    See  Anzt et al., Parallel Computing, 2015.

    Arguments
    ---------
    
    @param[in]
    A           magma_d_matrix
                input matrix A, current target system

    @param[in]
    precond     magma_d_preconditioner*
                preconditioner parameters

    @param[in]
    updates     magma_int_t 
                number of updates
    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_dhepr
    ********************************************************************/
extern "C"
magma_int_t
magma_dpariluupdate(
    magma_d_matrix A,
    magma_d_preconditioner *precond,
    magma_int_t updates,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_d_matrix hALt={Magma_CSR};
    magma_d_matrix d_h={Magma_CSR};
    
    magma_d_matrix hL={Magma_CSR}, hU={Magma_CSR},
    hAcopy={Magma_CSR}, hAL={Magma_CSR}, hAU={Magma_CSR}, hAUt={Magma_CSR},
    hUT={Magma_CSR}, hAtmp={Magma_CSR},
    dL={Magma_CSR}, dU={Magma_CSR};

    if ( updates > 0 ){
        CHECK( magma_dmtransfer( precond->M, &hAcopy, Magma_DEV, Magma_CPU , queue ));
        // in case using fill-in
        CHECK( magma_dsymbilu( &hAcopy, precond->levels, &hAL, &hAUt,  queue ));
        // add a unit diagonal to L for the algorithm
        CHECK( magma_dmLdiagadd( &hAL , queue ));
        // transpose U for the algorithm
        CHECK( magma_d_cucsrtranspose(  hAUt, &hAU , queue ));
        // transfer the factor L and U
        CHECK( magma_dmtransfer( hAL, &dL, Magma_CPU, Magma_DEV , queue ));
        CHECK( magma_dmtransfer( hAU, &dU, Magma_CPU, Magma_DEV , queue ));
        magma_dmfree(&hAL, queue );
        magma_dmfree(&hAU, queue );
        magma_dmfree(&hAUt, queue );
        magma_dmfree(&precond->M, queue );
        magma_dmfree(&hAcopy, queue );
        
        // copy original matrix as CSRCOO to device
        for(int i=0; i<updates; i++){
            CHECK( magma_dparilu_csr( A, dL, dU, queue ));
        }
        CHECK( magma_dmtransfer( dL, &hL, Magma_DEV, Magma_CPU , queue ));
        CHECK( magma_dmtransfer( dU, &hU, Magma_DEV, Magma_CPU , queue ));
        CHECK( magma_d_cucsrtranspose(  hU, &hUT , queue ));
        magma_dmfree(&dL, queue );
        magma_dmfree(&dU, queue );
        magma_dmfree(&hU, queue );
        CHECK( magma_dmlumerge( hL, hUT, &hAtmp, queue ));
        // for CUSPARSE
        CHECK( magma_dmtransfer( hAtmp, &precond->M, Magma_CPU, Magma_DEV , queue ));
        
        magma_dmfree(&hL, queue );
        magma_dmfree(&hUT, queue );
        hAL.diagorder_type = Magma_UNITY;
        CHECK( magma_dmconvert(hAtmp, &hAL, Magma_CSR, Magma_CSRL, queue ));
        hAL.storage_type = Magma_CSR;
        CHECK( magma_dmconvert(hAtmp, &hAU, Magma_CSR, Magma_CSRU, queue ));
        hAU.storage_type = Magma_CSR;
        
        magma_dmfree(&hAtmp, queue );
        CHECK( magma_dmtransfer( hAL, &precond->L, Magma_CPU, Magma_DEV , queue ));
        CHECK( magma_dmtransfer( hAU, &precond->U, Magma_CPU, Magma_DEV , queue ));
        magma_dmfree(&hAL, queue );
        magma_dmfree(&hAU, queue );
    
        magma_dmfree( &precond->d , queue );
        magma_dmfree( &precond->d2 , queue );
        
        CHECK( magma_djacobisetup_diagscal( precond->L, &precond->d, queue ));
        CHECK( magma_djacobisetup_diagscal( precond->U, &precond->d2, queue ));
    }

cleanup:
    magma_dmfree(&d_h, queue );
    magma_dmfree(&hALt, queue );
    
    return info;
}


/**
    Purpose
    -------

    Prepares the IC preconditioner via the iterative IC iteration.

    Arguments
    ---------

    @param[in]
    A           magma_d_matrix
                input matrix A
                
    @param[in]
    b           magma_d_matrix
                input RHS b

    @param[in,out]
    precond     magma_d_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_dhepr
    ********************************************************************/
extern "C"
magma_int_t
magma_dparicsetup(
    magma_d_matrix A,
    magma_d_matrix b,
    magma_d_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    hipsparseHandle_t hipsparseHandle=NULL;
    hipsparseMatDescr_t descrL=NULL;
    hipsparseMatDescr_t descrU=NULL;

    magma_d_matrix hAh={Magma_CSR}, hA={Magma_CSR}, hAtmp={Magma_CSR},
    hAL={Magma_CSR}, hAUt={Magma_CSR}, hALt={Magma_CSR}, hM={Magma_CSR},
    hACSRCOO={Magma_CSR}, dAinitguess={Magma_CSR}, dL={Magma_CSR};
    magma_d_matrix d_h={Magma_CSR};


    // copy original matrix as CSRCOO to device
    CHECK( magma_dmtransfer(A, &hAh, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_dmconvert( hAh, &hA, hAh.storage_type, Magma_CSR , queue ));
    magma_dmfree(&hAh, queue );

    // in case using fill-in
    CHECK( magma_dsymbilu( &hA, precond->levels, &hAL, &hAUt , queue ));

    // need only lower triangular
    magma_dmfree(&hAUt, queue );
    magma_dmfree(&hAL, queue );
    CHECK( magma_dmconvert( hA, &hAtmp, Magma_CSR, Magma_CSRL , queue ));
    magma_dmfree(&hA, queue );

    // ---------------- initial guess ------------------- //
    CHECK( magma_dmconvert( hAtmp, &hACSRCOO, Magma_CSR, Magma_CSRCOO , queue ));
    //int blocksize = 1;
    //magma_dmreorder( hACSRCOO, n, blocksize, blocksize, blocksize, &hAinitguess , queue );
    CHECK( magma_dmtransfer( hACSRCOO, &dAinitguess, Magma_CPU, Magma_DEV , queue ));
    magma_dmfree(&hACSRCOO, queue );
    CHECK( magma_dmtransfer( hAtmp, &dL, Magma_CPU, Magma_DEV , queue ));
    magma_dmfree(&hAtmp, queue );

    for(int i=0; i<precond->sweeps; i++){
        CHECK( magma_dparic_csr( dAinitguess, dL , queue ));
    }
    CHECK( magma_dmtransfer( dL, &hAL, Magma_DEV, Magma_CPU , queue ));
    magma_dmfree(&dL, queue );
    magma_dmfree(&dAinitguess, queue );


    // for CUSPARSE
    CHECK( magma_dmtransfer( hAL, &precond->M, Magma_CPU, Magma_DEV , queue ));

    // Jacobi setup
    CHECK( magma_djacobisetup_matrix( precond->M, &precond->L, &precond->d , queue ));

    // for Jacobi, we also need U
    CHECK( magma_d_cucsrtranspose(   hAL, &hALt , queue ));
    CHECK( magma_djacobisetup_matrix( hALt, &hM, &d_h , queue ));

    CHECK( magma_dmtransfer( hM, &precond->U, Magma_CPU, Magma_DEV , queue ));

    magma_dmfree(&hM, queue );

    magma_dmfree(&d_h, queue );


        // copy the matrix to precond->L and (transposed) to precond->U
    CHECK( magma_dmtransfer(precond->M, &(precond->L), Magma_DEV, Magma_DEV, queue ));
    CHECK( magma_dmtranspose( precond->L, &(precond->U), queue ));

    // extract the diagonal of L into precond->d
    CHECK( magma_djacobisetup_diagscal( precond->L, &precond->d, queue ));
    CHECK( magma_dvinit( &precond->work1, Magma_DEV, hAL.num_rows, 1, MAGMA_D_ZERO, queue ));

    // extract the diagonal of U into precond->d2
    CHECK( magma_djacobisetup_diagscal( precond->U, &precond->d2, queue ));
    CHECK( magma_dvinit( &precond->work2, Magma_DEV, hAL.num_rows, 1, MAGMA_D_ZERO, queue ));


    magma_dmfree(&hAL, queue );
    magma_dmfree(&hALt, queue );

    CHECK(magma_dtrisolve_analysis(precond->M, &precond->cuinfoL, false, false, false, queue));
    CHECK(magma_dtrisolve_analysis(precond->M, &precond->cuinfoU, false, false, true, queue));
    
    cleanup:
    hipsparseDestroy( hipsparseHandle );
    hipsparseDestroyMatDescr( descrL );
    hipsparseDestroyMatDescr( descrU );
    hipsparseHandle=NULL;
    descrL=NULL;
    descrU=NULL;    
    magma_dmfree( &hAh, queue );
    magma_dmfree( &hA, queue );
    magma_dmfree( &hAtmp, queue );
    magma_dmfree( &hAL, queue );
    magma_dmfree( &hAUt, queue );
    magma_dmfree( &hALt, queue );
    magma_dmfree( &hM, queue );
    magma_dmfree( &hACSRCOO, queue );
    magma_dmfree( &dAinitguess, queue );
    magma_dmfree( &dL, queue );
    magma_dmfree( &d_h, queue );
    
    return info;
}


/**
    Purpose
    -------

    Updates an existing preconditioner via additional iterative IC sweeps for
    previous factorization initial guess (PFIG).
    See  Anzt et al., Parallel Computing, 2015.

    Arguments
    ---------
    
    @param[in]
    A           magma_d_matrix
                input matrix A, current target system

    @param[in]
    precond     magma_d_preconditioner*
                preconditioner parameters

    @param[in]
    updates     magma_int_t 
                number of updates
    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_dhepr
    ********************************************************************/
extern "C"
magma_int_t
magma_dparicupdate(
    magma_d_matrix A,
    magma_d_preconditioner *precond,
    magma_int_t updates,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_d_matrix hALt={Magma_CSR};
    magma_d_matrix d_h={Magma_CSR};
        
    if( updates > 0 ){
        // copy original matrix as CSRCOO to device
        for(int i=0; i<updates; i++){
            CHECK( magma_dparic_csr( A, precond->M , queue ));
        }
        //magma_dmtransfer( precond->M, &hALt, Magma_DEV, Magma_CPU , queue );
        magma_dmfree(&precond->L, queue );
        magma_dmfree(&precond->U, queue );
        magma_dmfree( &precond->d , queue );
        magma_dmfree( &precond->d2 , queue );
        
        // copy the matrix to precond->L and (transposed) to precond->U
        CHECK( magma_dmtransfer(precond->M, &(precond->L), Magma_DEV, Magma_DEV, queue ));
        CHECK( magma_dmtranspose( precond->L, &(precond->U), queue ));

        CHECK( magma_djacobisetup_diagscal( precond->L, &precond->d, queue ));
        CHECK( magma_djacobisetup_diagscal( precond->U, &precond->d2, queue ));
    }
    
cleanup:
    magma_dmfree(&d_h, queue );
    magma_dmfree(&hALt, queue );
    
    return info;
}

