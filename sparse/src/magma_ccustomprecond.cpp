/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @generated from sparse/src/magma_zcustomprecond.cpp, normal z -> c, Wed Feb 22 15:22:00 2023
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define PRECISION_c



/**
    Purpose
    -------

    This is an interface to the left solve for any custom preconditioner.
    It should compute x = FUNCTION(b)
    The vectors are located on the device.

    Arguments
    ---------

    @param[in]
    b           magma_c_matrix
                RHS

    @param[in,out]
    x           magma_c_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_capplycustomprecond_l(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // vector access via x.dval, y->dval
    
    return info;
}


/**
    Purpose
    -------

    This is an interface to the right solve for any custom preconditioner.
    It should compute x = FUNCTION(b)
    The vectors are located on the device.

    Arguments
    ---------

    @param[in]
    b           magma_c_matrix
                RHS

    @param[in,out]
    x           magma_c_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_c_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_caux
    ********************************************************************/

extern "C" magma_int_t
magma_capplycustomprecond_r(
    magma_c_matrix b,
    magma_c_matrix *x,
    magma_c_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // vector access via x.dval, y->dval
    // sizes are x.num_rows, x.num_cols
    
    
    return info;
}
