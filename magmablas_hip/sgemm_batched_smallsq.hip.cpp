#include "hip/hip_runtime.h"
/*
    -- MAGMA (version 2.7.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date February 2023

       @author Ahmad Abdelfattah
       @author Azzam Haidar

       @generated from magmablas_hip/zgemm_batched_smallsq.hip.cpp, normal z -> s, Wed Feb 22 15:20:01 2023
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define SLDA(N)    ( (N==15||N==23||N==31)? N : (N+1) )

template<int N>
__global__ void
sgemm_batched_smallsq_kernel(
        const magma_trans_t transA, magma_trans_t transB, 
        const float alpha, float const * const * dA_array, int ai, int aj, int ldda, 
                                        float const * const * dB_array, int bi, int bj, int lddb, 
        const float beta,  float**               dC_array, int ci, int cj, int lddc, 
        const int batchCount)
{
    HIP_DYNAMIC_SHARED( float, zdata)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int bx = blockIdx.x;
    
    const int batchid = bx * blockDim.z + tz;
    if(batchid >= batchCount) return;
    
    const float* __restrict__ dA = dA_array[batchid] + aj * ldda + ai;
    const float* __restrict__ dB = dB_array[batchid] + bj * lddb + bi;
          float* __restrict__ dC = dC_array[batchid] + cj * lddc + ci;
    
    float rC = MAGMA_S_ZERO; 
    float rTmp = MAGMA_S_ZERO; 
    
    const int slda = SLDA(N);
    const int sldb = SLDA(N);
    float* sA = (float*)(zdata);
    float* sB = (float*)(zdata + blockDim.z * slda * N);
    
    sA += tz * slda * N;
    sB += tz * sldb * N;
    
    // read A & B 
    if(transA == MagmaNoTrans){
        sA[ty * slda + tx] = dA[ty * ldda + tx];
    }
    else{
        sA[tx * slda + ty] = (transA == MagmaTrans) ? dA[ty * ldda + tx] : MAGMA_S_CONJ( dA[ty * ldda + tx] );
    }

    if(transB == MagmaNoTrans){
        sB[ty * sldb + tx] = dB[ty * lddb + tx];
    }
    else{
        sB[tx * sldb + ty] = (transB == MagmaTrans) ? dB[ty * lddb + tx] : MAGMA_S_CONJ( dB[ty * lddb + tx] );
    }
    __syncthreads(); 

    if(beta != MAGMA_S_ZERO){
        rC = beta * dC[ty * lddc + tx];
    }

    // multiply
    rTmp = MAGMA_S_ZERO;
    #pragma unroll
    for(int j = 0; j < N; j++){
        rTmp += sA[j * slda + tx] * sB[ty * sldb + j]; 
    }
    rC += alpha * rTmp;

    // write from rC
    dC[ty * lddc + tx] = rC;
}


extern "C" void 
magmablas_sgemm_batched_smallsq(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k, 
    float alpha,
    float const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, 
    float const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    float beta,
    float **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;
    
    if( !(m == n  && n == k) ){
        printf("Only square sizes are supported\n");
        info = -1;
    }

    if( m > 32){
        printf("Only square sizes of up to 32 are supported\n");
        info = -1;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
    }

    if ( m <= 0 || n <= 0 || k <= 0 ) return;
    
    magma_int_t ntcol  = magma_get_sgemm_batched_ntcol( m );
    magma_int_t shmem  = ( SLDA(m)*m + SLDA(n)*n ) * sizeof(float);
                shmem *= ntcol;

    const int nblocks = magma_ceildiv(batchCount, ntcol);
    dim3 grid(nblocks, 1, 1);
    dim3 threads(m, m, ntcol);

    switch(m){
        case  1: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel< 1>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  2: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel< 2>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  3: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel< 3>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  4: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel< 4>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  5: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel< 5>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  6: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel< 6>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  7: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel< 7>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  8: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel< 8>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  9: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel< 9>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 10: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<10>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 11: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<11>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 12: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<12>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 13: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<13>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 14: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<14>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 15: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<15>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 16: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<16>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 17: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<17>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 18: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<18>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 19: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<19>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 20: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<20>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 21: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<21>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 22: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<22>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 23: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<23>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 24: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<24>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 25: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<25>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 26: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<26>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 27: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<27>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 28: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<28>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 29: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<29>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 30: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<30>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 31: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<31>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 32: hipLaunchKernelGGL(HIP_KERNEL_NAME(sgemm_batched_smallsq_kernel<32>), dim3(grid), dim3(threads), shmem, queue->hip_stream(), transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        default:;
    }
}
