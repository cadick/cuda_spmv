// final comparison: c++, parallel mv multiplication, sparse mv multiplication with fft

// CentOS compiling
// export LD_LIBRARY_PATH="/usr/local/cuda-11.0/targets/x86_64-linux/lib/"
// nvcc -std=c++11 comparison.cu -o comparison -I/usr/local/cuda/targets/x86_64-linux/include/ -L/usr/local/cuda/targets/x86_64-linux/lib/ -lcufft
// ./comparison

// Windows compiling:
// nvcc -std=c++11 comparison.cu -o comparison -I/usr/local/cuda/targets/x86_64-linux/include/ -L/usr/local/cuda/targets/x86_64-linux/lib/ -lcufft
// comparison.exe

#include <iostream>
#include <chrono>
#include <cufft.h>
#include "make_print.h"
#include "cuComplex.h"

#define M 500        // number of Matrices we want to multiply
#define V 100         // number of Vectors we want to multiply
#define dimU 16      // number of senders = rows of Matrix h_m_W
#define dimB 128     // number of receivers = columns of Matrix h_m_W
#define K 12         // number of zero values per row
#define MATRIX_SIZE dimU*dimB
#define MATRIXW_BYTES MATRIX_SIZE*sizeof(cuDoubleComplex)
#define VECTORY_BYTES dimB*sizeof(cuDoubleComplex)
#define VECTORR_BYTES dimU*sizeof(cuDoubleComplex)

dim3 getDimGrid(const int m, const int n) {
    dim3 dimGrid(m, n, 1);
    return dimGrid;
}

dim3 getDimBlock(const int m, const int n) {
    dim3 dimBlock(m, n, 1);
    return dimBlock;
}

/********************************** normal MV *********************************/

// matrix/vector multiplication, one thread per row
// cuDoubleComplex is complex number for cuda (cuFloatComplex does NOT work/exist)
__global__ void matrixVectorMultiplication(cuDoubleComplex *d_m_W, cuDoubleComplex *d_v_y, cuDoubleComplex *d_v_r) {
    // each thread represents one row in one of the Matrices
    int threadId = ((gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * 1) + threadIdx.x;

    cuDoubleComplex sum = make_cuDoubleComplex(0, 0);
    
    // each thread does one row of multiplications
    for(int i = 0; i < dimB; i++) {
        sum = cuCadd(sum, cuCmul(d_m_W[(threadId%(M*dimU))* dimB  + i], d_v_y[ i + (threadId / (M * dimU)) * dimB]));
    }
    d_v_r[threadId] = sum;

}

/********************************** normal MV *********************************/

/************************************ SPMV ************************************/

// matrix/vector multiplication, one thread per row
// cuDoubleComplex is complex number for cuda (cuFloatComplex does NOT work/exist)
__global__ void sparseMatrixVectorMultiplication(cuDoubleComplex* d_a, int* d_ia, int* d_ja, cuDoubleComplex *d_v_y, cuDoubleComplex* d_a_r) {
    // each thread represents one row in one of the Matrices
    int threadId = ((gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x * 1) + threadIdx.x;

    cuDoubleComplex sum = make_cuDoubleComplex(0, 0);

    // each thread does one row of non zero multiplications
    for (int i = d_ia[(threadId%(M*dimU))]; i < d_ia[(threadId%(M*dimU)) +1]; i++) {
        sum = cuCadd(sum, cuCmul(d_a[i], d_v_y[d_ja[i]+(threadId / (M * dimU)) * dimB]));
    }
    d_a_r[threadId] = sum;
}

/************************************ SPMV ************************************/

int main() {
    // declare host matrices and vectors
    cuDoubleComplex *h_m_W, *h_v_y, *h_v_r, *h_a_r, *h_a_rfft, *h_a;
    int *h_ia, *h_ja;
    int total_nnz = 0;

    // allocate Memory
    h_m_W = (cuDoubleComplex*)malloc(MATRIXW_BYTES * M);
    h_v_y = (cuDoubleComplex*)malloc(VECTORY_BYTES * V);
    h_v_r = (cuDoubleComplex*)malloc(VECTORR_BYTES * M*V);
    h_a_r = (cuDoubleComplex*)malloc(VECTORR_BYTES * M*V);
    h_a_rfft = (cuDoubleComplex*)malloc(VECTORR_BYTES * M*V);
    h_a = (cuDoubleComplex*)malloc(MATRIXW_BYTES * M);
	h_ia = (int*)malloc((dimU*M+1) * sizeof(int));
	h_ja = (int*)malloc(M * dimU*dimB*sizeof(int));

    // declare GPU memory pointers
    cuDoubleComplex  *d_m_W, *d_v_y, *d_v_r, *d_a_r, *d_a_rfft, *d_a;
    int *d_ia, *d_ja;

    // filling matrices with rendom amount of zeros per row
    fillMatrix(h_m_W, MATRIX_SIZE * M);
    // printMatrices(h_m_W, MATRIX_SIZE, dimB, M);

    //fill matrices with k zeros per row
    //fillStructuredMatrix(h_m_W, dimU*M, dimB, K);
    //printMatrices(h_m_W, MATRIX_SIZE, dimB, M);

    fillVector(h_v_y, dimB * V);
    //printVectors(h_v_y, dimB, V);

    // make matrices CSR
    //auto start_csr = std::chrono::high_resolution_clock::now();
    makeCSR(h_m_W, dimU, dimB, h_a, h_ia, h_ja, &total_nnz, M);
    //auto finish_csr = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> elapsed_csr = finish_csr - start_csr;
    //printCSR(h_a, h_ia, h_ja, dimU, &total_nnz, M);

    // allocate GPU memory pointers
    cudaMalloc((void **) &d_m_W, MATRIXW_BYTES*M);
    cudaMalloc((void **) &d_v_y, VECTORY_BYTES*V);
    cudaMalloc((void **) &d_v_r, VECTORR_BYTES*M*V);
    cudaMalloc((void **) &d_a_r, VECTORR_BYTES*M*V);
    cudaMalloc((void **) &d_a_rfft, VECTORR_BYTES*M*V);
    cudaMalloc((void **) &d_a, total_nnz*sizeof(cuDoubleComplex));
    cudaMalloc((void **) &d_ia, (dimU*M+1) * sizeof(int));
    cudaMalloc((void **) &d_ja, total_nnz*sizeof(int));

    // calculate the necessary space (same fo MV and SPMV)
    dim3 dimGrid = getDimGrid(M, V);
    dim3 dimBlock = getDimBlock(dimU, 1);

    /********************************** normal MV *********************************/

    // transfer the array to the GPU
    cudaMemcpy(d_m_W, h_m_W, M*MATRIXW_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_y, h_v_y, V*VECTORY_BYTES, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    auto start_mv = std::chrono::high_resolution_clock::now();

    // launch the kernel
    matrixVectorMultiplication <<<dimGrid, dimBlock>>>(d_m_W, d_v_y, d_v_r);
    cudaDeviceSynchronize();
    auto finish_mv = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_mv = finish_mv - start_mv;

    // copy back the result array to the CPU
    cudaMemcpy(h_v_r, d_v_r, M*V*VECTORR_BYTES, cudaMemcpyDeviceToHost);

    // free GPU memory allocation
    cudaFree(d_m_W);
    cudaFree(d_v_y);
    cudaFree(d_v_r);

    /********************************** normal MV *********************************/

    cudaDeviceSynchronize();

    /************************************ SPMV ************************************/

    //transfer the array to the GPU
    cudaMemcpy(d_v_y, h_v_y, V*VECTORY_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a, h_a, total_nnz*sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ia, h_ia, (dimU*M+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ja, h_ja,total_nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_r, h_a_r, VECTORR_BYTES*M*V, cudaMemcpyHostToDevice);

    // create plan for FFT
    cufftHandle plan;
    cufftPlan1d(&plan, dimU, CUFFT_Z2Z, M*V);
    cudaDeviceSynchronize();

    auto start_spmv = std::chrono::high_resolution_clock::now();

    // launch the kernel
    sparseMatrixVectorMultiplication <<<dimGrid, dimBlock >>> (d_a, d_ia, d_ja, d_v_y, d_a_r);
    cudaDeviceSynchronize();
    auto finish_spmv = std::chrono::high_resolution_clock::now();

    // FFT
	cufftExecZ2Z(plan, d_a_r, d_a_rfft, CUFFT_FORWARD);
    cudaDeviceSynchronize();
    auto finish_spmvfft = std::chrono::high_resolution_clock::now();
	
	cufftDestroy(plan);

    // copy back the result array to the CPU
    cudaMemcpy(h_a_r, d_a_r, M * V * VECTORR_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_a_rfft, d_a_rfft, VECTORR_BYTES*M*V, cudaMemcpyDeviceToHost);

    //printVectors(h_a_r, dimU, M*V);
    //printVectors(h_a_rfft, dimU, M*V);

    std::chrono::duration<double> elapsed_spmv = finish_spmv - start_spmv;
    std::chrono::duration<double> elapsed_spmvfft = finish_spmvfft - start_spmv;


    // free GPU memory allocation
    cudaFree(d_a_rfft);
    cudaFree(d_a_r);
    cudaFree(d_a);
    cudaFree(d_ia);
    cudaFree(d_ja);

    /************************************ SPMV ************************************/

    /********************************** c++ part **********************************/

    cuDoubleComplex* cppResult;
    cppResult = (cuDoubleComplex*)malloc(M * V * VECTORR_BYTES);

    auto start_cpp = std::chrono::high_resolution_clock::now();
    // loops through the result Array, same memory access as cuda implementation
    for (int i = 0; i < dimU * M * V; i++) {
        cppResult[i] = make_cuDoubleComplex(0, 0);
        for (int j = 0; j < dimB; j++) {
            cppResult[i] = cuCadd(cppResult[i], cuCmul(h_m_W[((i % (dimU * M)) * dimB) + j], h_v_y[((i / (dimU * M))) * dimB + j]));
        }
    }
    auto finish_cpp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_cpp = finish_cpp - start_cpp;

    /********************************** c++ part **********************************/
    

    std::cout << "\n\nRESULTS:\n\n";
    //printVectors(a, dimU, V*M);
    //printVectors(cppResult, dimU, V * M);

    // Check if MV == SPMV result
    int errorCount = 0;
    for (int i = 0; i < dimU * M * V; i++) {
        if (cuCreal(h_v_r[i]) == cuCreal(h_a_r[i]) && cuCimag(h_v_r[i]) == cuCimag(h_a_r[i])
            /*&& cuCreal(cppResult[i]) == cuCreal(h_a_r[i]) && cuCimag(cppResult[i]) == cuCimag(h_a_r[i])*/) {
            continue;
        }
        else {
            errorCount++;
        }
    }

    if (errorCount == 0) {
        std::cout << "SPMV results equal MV result \n";
    }
    else {
        std::cout << "SPMV result not equal to MV, number of Errors: " << errorCount << " \n";
    }

    std::cout << "\nComputation time:\nMV:       " << elapsed_mv.count() << " s\nSPMV:     " << elapsed_spmv.count()<< " s\nSPMV+FFT: " << elapsed_spmvfft.count() << " s\n";
    //std::cout <<"c++:      " << elapsed_cpp.count() << " s\n";
    std::cout << "rows: " << dimU << ", cols: " << dimB << ", matrices: " << M << ", vectors: " << V << ", zeros: " << K << "\n";
    //std::cout << "csr:      " << elapsed_csr.count() << " s\n";
    return 0;
}
