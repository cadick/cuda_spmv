# cuda_spmv
comparison of spmv implementations in cuda

**C++:** CPU implementation

**MV:** one thread per row parallel GPU implementation ignoring sparsity

**SPMV:** one thread per row sparse matrix vector implementation by using CSR representation which enables to only execute multiplications of non zero values


### CentOS compiling

export LD_LIBRARY_PATH="/usr/local/cuda-11.0/targets/x86_64-linux/lib/"
nvcc -std=c++11 comparison.cu -o comparison -I/usr/local/cuda/targets/x86_64-linux/include/ -L/usr/local/cuda/targets/x86_64-linux/lib/ -lcufft
./comparison

### Windows compiling:

nvcc -std=c++11 comparison.cu -o comparison -I/usr/local/cuda/targets/x86_64-linux/include/ -L/usr/local/cuda/targets/x86_64-linux/lib/ -lcufft
comparison.exe
