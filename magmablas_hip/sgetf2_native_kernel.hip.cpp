#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @generated from magmablas_hip/zgetf2_native_kernel.hip.cpp, normal z -> s, Wed Feb 22 15:20:01 2023
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "shuffle.hip.hpp"
#include "sync.hip.hpp"
#include "atomics.hip.hpp"
#include "batched_kernel_param.h"

#define PRECISION_s

/**
    Purpose
    -------
    LU factorization of m-by-n matrix ( m >= n ).
    Each thread block caches an entire column in register.
    Thread blocks communicate and synchronize through global memory.
    Assumptions:
        1. dA is of size MxN such that N <= M.
        2. Thread block must be 1D, with TX multiple of 32 (warp size)
        3. TX must be >= n
        4. n must be less than the number of SMs on the GPU
**/

// =============================================================================
// init kernel
__global__ void
sgetf2_native_init_kernel( int n, int npages, magma_int_t *ipiv, int* update_flags)
{
    const int tx = threadIdx.x;
    if( tx < n){
        ipiv[ tx ] = 0;
    }
    if( tx < max(n,npages) ){
        update_flags[ tx ] = 0;
    }
}

// =============================================================================
// the main kernel
template<int TX, int NPAGES>
__global__ void
sgetf2_native_kernel( int m, int n,
                      magmaFloat_ptr dA, int ldda,
                      volatile magma_int_t *ipiv, int gbstep,
                      volatile int* update_flag,
                      volatile magma_int_t *info)
{
#ifdef MAGMA_HAVE_CUDA
    const int tx  = threadIdx.x;
    const int bx = blockIdx.x;
    float rA[NPAGES] = {MAGMA_S_ZERO};
    float rx, rx_max;
    magmaFloat_ptr da = dA;
    int rx_id, max_id, flag = 0, linfo;
    float  rx_abs = 0.0, rx_abs_max = 0.0;
    const int m_ = m-(NPAGES-1)*TX;
    if( bx >= n ) return;

    // read the info (if it is set to non-zero a previous panel, then we don't set it again)
    linfo = (int)(*info);

    __shared__ float sx[ TX ];
    __shared__ float sabs[ TX ];
    __shared__ int smax_id[ TX ];
    __shared__ float sreg;

    // read
    dA += bx * ldda + tx;
    #pragma unroll
    for(int i = 0; i < NPAGES-1; i++){
        rA[i] = dA[ i * TX ];
    }
    if( tx <  m_){
        rA[NPAGES-1] = dA[ (NPAGES-1) * TX ];
    }

    // main loop
    #pragma unroll
    for(int i = 0; i < n; i++){
        // isamax and write pivot for the ith thread block
        if(bx == i){
            rx_max     = rx     = (tx < i) ? MAGMA_S_ZERO : rA[0];
            rx_abs_max = rx_abs = fabs(MAGMA_S_REAL(rx)) + fabs(MAGMA_S_IMAG(rx));
            max_id = rx_id = tx;
            #pragma unroll
            for(int j = 1; j < NPAGES; j++){
                rx = rA[j];
                rx_abs = fabs(MAGMA_S_REAL(rx)) + fabs(MAGMA_S_IMAG(rx));
                if ( rx_abs  > rx_abs_max ){
                    rx_max = rx;
                    rx_abs_max = rx_abs;
                    max_id = j * TX + tx;
                }
            }
            sx[ tx ] = rx_max;
            sabs[ tx ] = rx_abs_max;
            smax_id[ tx ] = max_id;
            __syncthreads();

            // let the first warp do the final reduction step
            if(tx < 32){
                #pragma unroll
                for(int j = 0; j < TX; j+= 32){
                    rx     = sx[ j + tx ];
                    rx_abs = sabs[ j + tx ];
                    rx_id  = smax_id[ j + tx ];
                    if ( rx_abs  > rx_abs_max ){
                        rx_max = rx;
                        rx_abs_max = rx_abs;
                        max_id = rx_id;
                    }
                }
                magmablas_syncwarp();
                sx[ tx ] = rx_max;
                sabs[ tx ] = rx_abs_max;
                smax_id[ tx ] = max_id;
                magmablas_syncwarp();
                #pragma unroll
                for(int j = 0; j < 32; j++){
                    rx     = sx[j];
                    rx_abs = sabs[j];
                    rx_id  = smax_id[j];
                    if ( rx_abs  > rx_abs_max ){
                        rx_abs_max = rx_abs;
                        rx_max = rx;
                        max_id = rx_id;
                    }
                }
            }

            if(tx == 0){
                sx[ 0 ] = rx_max;
                sabs[ 0 ] = rx_abs_max;
                smax_id[ 0 ] = (rx_abs_max == MAGMA_D_ZERO) ? i : max_id;
            }
            __syncthreads();
            rx_max = sx[ 0 ];
            rx_abs_max = sabs[ 0 ];
            max_id = smax_id[ 0 ];
            __syncthreads();

            // now every thread in the i^th block has the maximum
            linfo = (rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (max_id+gbstep+1) : linfo;
            if( tx == 0){
                //printf("[%2d]: bx = %d, max_id, = %d, rx_abs_max = %f, linfo = %d\n", i, bx, max_id, rx_abs_max, linfo);
                magmablas_iatomic_exchange((magma_int_t*)info, (magma_int_t)(linfo) );
                magmablas_iatomic_exchange((magma_int_t*)&ipiv[i], (magma_int_t)(max_id+1) ); // fortran indexing
            }
            __syncthreads();
            //if( rx_abs_max == MAGMA_D_ZERO )return;
        }
        else{ // other thread blocks are waiting
            if(tx == 0){
                max_id = 0;
                while( max_id == 0 ){
                    max_id = ipiv[i];
                };
                smax_id[ 0 ] = max_id;
            }
            __syncthreads();
            max_id = smax_id[ 0 ];
            max_id -= 1; // revert fortran indexing
            linfo = (*info);
            __syncthreads();
            //if( (*info) != 0 ) return;
        }

        // swap
        // swap always happens between page 0 and page x
        // to avoid spilling rA to local memory, we use shared memory
        if( max_id != i){
            // all blocks swap in registers
            // for bx < i, the column is already written in memory,
            // but we have a copy in reg., so continue to swap in reg.,
            // and do one final write to memory
            #pragma unroll
            for(int j = 0; j < NPAGES; j++){
                if( j == (max_id/TX) ){
                    sx[ tx ] = rA[j];
                    __syncthreads();
                    if( tx == i ){
                        float tmp    = sx[ max_id%TX ];
                        sx[ max_id%TX ] = rA[0];
                        rA[0] = tmp;
                    }
                    __syncthreads();
                    if( tx == max_id%TX ){
                        rA[j] = sx[ tx ];
                    }
                    __syncthreads();
                }
            }
            //__syncthreads();
        }

        // the ith block does scal
        if(bx == i){
            float reg = (rx_max == MAGMA_S_ZERO) ? MAGMA_S_ONE : MAGMA_S_DIV(MAGMA_S_ONE, rx_max );
            // scal
            if( tx > i ){
                rA[0] *= reg;
            }
            #pragma unroll
            for(int j = 1; j < NPAGES; j++){
                rA[j] *= reg;
            }
            // write column i to global memory
            #pragma unroll
            for(int j = 0; j < NPAGES-1; j++){
                dA[ j * TX ] = rA[j];
            }
            if( tx <  m_){
                dA[ (NPAGES-1) * TX ] = rA[NPAGES-1];
            }
            __threadfence(); __syncthreads(); // after cuda 9.0, both are needed, not sure why
            if(tx == 0) magmablas_iatomic_exchange( (int *)&update_flag[ i ], 1);
        }

        // thread blocks with ID larger than i perform ger
        if(bx > i){
            if( tx == i ){
                sreg = rA[0];
            }
            // wait for scal
            if( tx == 0){
                flag = 0;
                while( flag == 0 ){
                    flag = update_flag[ i ];
                };
            }
            __syncthreads();

            float reg = sreg;
            if( NPAGES == 1){
                if(tx > i && tx < m_){
                    rA[0] -= da[ i * ldda + tx ] * reg;
                }
            }else{
                if(tx > i){
                    rA[0] -= da[ i * ldda + tx ] * reg;
                }
            }
            #pragma unroll
            for(int j = 1; j < NPAGES-1; j++){
                rA[j] -= da[ i * ldda + j * TX + tx ] * reg;
            }
            if( NPAGES > 1){
                if( tx < m_ ){
                    rA[ NPAGES-1 ] -= da[ i * ldda + (NPAGES-1)*TX + tx ] * reg;
                }
            }
        }
    }

    // all blocks write their columns again except the last one
    if( bx < n-1 ){
        #pragma unroll
        for(int i = 0; i < NPAGES-1; i++){
            dA[ i * TX ] = rA[i];
        }
        if( tx <  m_){
            dA[ (NPAGES-1) * TX ] = rA[NPAGES-1];
        }
    }

#endif    // MAGMA_HAVE_CUDA
}

// =============================================================================
extern "C" magma_int_t
magma_sgetf2_native_fused(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv, magma_int_t gbstep,
    magma_int_t *flags,
    magma_int_t *info, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    const magma_int_t ntx   = SGETF2_FUSED_NTH;

    if( m < n || m > SGETF2_FUSED_MAX_M ){
        arginfo = -1;
    }
    else if( n > magma_getdevice_multiprocessor_count() ){
        arginfo = -2;
    }
    else if( ldda < max(1, m) ){
        arginfo = -4;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    magma_int_t arch = magma_getdevice_arch();

    dim3 grid(n, 1, 1);
    dim3 threads(ntx, 1, 1);
    const magma_int_t npages = magma_ceildiv(m, ntx);
    // the kernel uses communication among thread blocks
    // as a safeguard, force one thread block per multiprocessor
    // by allocating more than half the shared memory
    magma_int_t shmem = magma_getdevice_shmem_block();
    shmem = (shmem / 2);
    int *update_flag = (int*) flags;    // update_flag is an int, not magma_int_t
    size_t max_n_npages = max(n,npages);
    hipLaunchKernelGGL(sgetf2_native_init_kernel, dim3(1), dim3(max_n_npages), 0, queue->hip_stream() ,  n, npages, ipiv, update_flag);
    // The case statement should cover up to ( xGETF2_CHAIN_MAX_M / ntx )
    switch(npages){
        case  1: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx,  1>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  2: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx,  2>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  3: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx,  3>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  4: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx,  4>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  5: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx,  5>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  6: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx,  6>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  7: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx,  7>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  8: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx,  8>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case  9: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx,  9>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 10: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 10>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 11: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 11>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 12: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 12>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 13: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 13>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 14: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 14>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 15: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 15>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 16: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 16>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 17: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 17>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 18: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 18>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 19: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 19>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 20: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 20>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        #if defined(PRECISION_s) || defined(PRECISION_d)
        case 21: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 21>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 22: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 22>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 23: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 23>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 24: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 24>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 25: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 25>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 26: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 26>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 27: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 27>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 28: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 28>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 29: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 29>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 30: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 30>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 31: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 31>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 32: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 32>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 33: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 33>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 34: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 34>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 35: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 35>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 36: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 36>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 37: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 37>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 38: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 38>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 39: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 39>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 40: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 40>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 41: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 41>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 42: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 42>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 43: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 43>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 44: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 44>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 45: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 45>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 46: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 46>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        #endif // defined(PRECISION_s) || defined(PRECISION_d)
        #if defined(PRECISION_s)
        case 47: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 47>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 48: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 48>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 49: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 49>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 50: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 50>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 51: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 51>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 52: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 52>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 53: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 53>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 54: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 54>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 55: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 55>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 56: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 56>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 57: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 57>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 58: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 58>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 59: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 59>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 60: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 60>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 61: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 61>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 62: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 62>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 63: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 63>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 64: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 64>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 65: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 65>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 66: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 66>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 67: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 67>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 68: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 68>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 69: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 69>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 70: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 70>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 71: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 71>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 72: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 72>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 73: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 73>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 74: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 74>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 75: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 75>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 76: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 76>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 77: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 77>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 78: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 78>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 79: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 79>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        case 80: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgetf2_native_kernel< ntx, 80>), dim3(grid), dim3(threads), shmem, queue->hip_stream() ,  m, n, dA, ldda, ipiv, gbstep, update_flag, info); break;
        #endif // defined(PRECISION_s)
        default: printf("size not supported \n");
    }
    return 0;
}
