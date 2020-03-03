#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include "../inperfect/gridify_up-inl.h"
#include <cooperative_groups.h>
#include <math.h>
#include <stdlib.h>
#include <curand_kernel.h>

using namespace cooperative_groups;
//using namespace std;

#define ndim 3
#define data_ndim 4

namespace mshadow {
    namespace cuda {


// compilation workaround
// in python when use,
// import ctypes
//_ = ctypes.CDLL('additional.so')
//    __device__ double atomicAdd(double *address, double val) {
//            return 0.0;
//        }
//
//        __device__  half::half_t
//        atomicAdd(half::half_t
//        * address,
//        half::half_t val
//        ) {
//        return (half::half_t)0.0;
//    }
//
//    __device__ float atomicAdd(float *address, float val) {
//        return ::atomicAdd(address, val);
//    }
//
//    __device__ float atomicAdd(int *address, int val) {
//        return ::atomicAdd(address, val);
//    }
//
//    __device__ float cas(double *addr, double compare, double val) {
//        unsigned long long int *address_as_ull = (unsigned long long int *) addr;
//        return __longlong_as_double(atomicCAS(address_as_ull,
//                                              __double_as_longlong(compare),
//                                              __double_as_longlong(val)));
//
//    }
//
//    __device__ float cas(float *addr, float compare, float val) {
//        unsigned int *address_as_uint = (unsigned int *) addr;
//        return __uint_as_float(atomicCAS(address_as_uint,
//                                         __float_as_uint(compare),
//                                         __float_as_uint(val)));
//    }
//
//    __device__ half::half_t
//    cas(half::half_t
//    * addr,
//    half::half_t compare, half::half_t
//    val) {
//    // NOT IMPLEMENTED YET!
//    return 0;
//    }

template<typename Dtype>
__global__ void gridify_kernel_build_index(
        Dtype* out_nebidx, Dtype* out_nebidxmsk,
        const Dtype* in_downdata, const Dtype* in_updata,
        const Dtype* in_down_actual_numpoints, const Dtype* in_up_actual_numpoints,
        const int B,
        const int N,
        const int max_o,
        const int P,
        const int kernel_size,
        const float *d_coord_shift,
        const float *d_voxel_size,
        const float *d_grid_size,
        const int grid_size_vol,
        int *d_coor_to_downpidx,
        int *d_coor_downp_counter,
        int *d_upp_counter
) {
//    grid_group grid = this_grid();i_ogrid
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
//    if(index>8190){
//        printf("index: %d;  ", index);
//    }
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - N * i_batch;
    if (i_pt < in_down_actual_numpoints[i_batch]) {
        int coor[ndim];
        const Dtype *p_pt = in_downdata + index * data_ndim;
        for (int j = 0; j < ndim; j++) {
            int c = floor((p_pt[j] + d_coord_shift[j]) / d_voxel_size[j]);
            if (c < 0 || c >= d_grid_size[j]) {
                return;
            }
            coor[j] = c;
        }
//        printf("value: p_pt[0] + d_coord_shift[0]: %f ;coor[0]: %d ",p_pt[0] + d_coord_shift[0], coor[0]);
        int coor_index = coor[2] * (d_grid_size[0] * d_grid_size[1]) + coor[1] * d_grid_size[0] + coor[0];
        int coor_indx_b =  i_batch * grid_size_vol + coor_index;
//        printf("grid_size_vol: %d, coor_index: %d, i_batch %d, voxel_idx: %d ; \n", grid_size_vol, coor_index, i_batch, voxel_idx);
        if (d_coor_downp_counter[coor_indx_b] < P){
            int grid_pntidx = atomicAdd(d_coor_downp_counter+coor_indx_b, 1);
            if (grid_pntidx < P) {
                d_coor_to_downpidx[coor_indx_b * P + grid_pntidx] = i_pt;
            } else {
                d_coor_downp_counter[coor_indx_b] = P;
            }
        }
    }
//    grid.sync();
}

template<typename Dtype>
__global__ void gridify_kernel_query_neighs(
        Dtype* out_nebidx, Dtype* out_nebidxmsk,
        const Dtype* in_downdata, const Dtype* in_updata,
        const Dtype* in_down_actual_numpoints, const Dtype* in_up_actual_numpoints,
        const int B,
        const int N,
        const int max_o,
        const int P,
        const int kernel_size,
        const float *d_coord_shift,
        const float *d_voxel_size,
        const float *d_grid_size,
        const int *d_coor_shift,
        const int *h_coor_shift,
        const int *w_coor_shift,
        const int grid_size_vol,
        const int size,
        int *d_coor_to_downpidx,
        int *d_coor_downp_counter,
        int *d_upp_counter
){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    int i_ogrid = index - i_batch * max_o;
    if (i_ogrid < in_up_actual_numpoints[i_batch]) {
        curandState state;
        int coor[ndim];
        const Dtype *p_pt = in_updata + index * data_ndim;
        for (int j = 0; j < ndim; j++) {
            int c = floor((p_pt[j] + d_coord_shift[j]) / d_voxel_size[j]);
            if (c < 0 || c >= d_grid_size[j]) {
                return;
            }
            coor[j] = c;
        }
        int* idx = new int[P];
        int d_coor, h_coor, w_coor, coor_indx_b, num_pnt=0, insrtidx=0;
//        int sum_num = 0, num_per_vox=0;
//        float P_f = (float)P;
//        for (int i = 0; i < size; i++){
//            d_coor = d_coor_shift[i] + coor[2];
//            h_coor = h_coor_shift[i] + coor[1];
//            w_coor = w_coor_shift[i] + coor[0];
//            if (d_coor >= 0 && d_coor < d_grid_size[2] && h_coor >= 0 && h_coor < d_grid_size[1] && w_coor >= 0 && w_coor < d_grid_size[0]){
//                coor_indx_b = i_batch * grid_size_vol + d_coor * (d_grid_size[0] * d_grid_size[1])
//                              + h_coor * d_grid_size[0] + w_coor;
//                sum_num += d_coor_downp_counter[coor_indx_b];
//            }
//        }
//        float ratio = (P_f / sum_num);
        for (int i = 0; i < size; i++){
            d_coor = d_coor_shift[i] + coor[2];
            h_coor = h_coor_shift[i] + coor[1];
            w_coor = w_coor_shift[i] + coor[0];
            if (d_coor >= 0 && d_coor < d_grid_size[2] && h_coor >= 0 && h_coor < d_grid_size[1] && w_coor >= 0 && w_coor < d_grid_size[0]){
                coor_indx_b = i_batch * grid_size_vol + d_coor * (d_grid_size[0] * d_grid_size[1])
                              + h_coor * d_grid_size[0] + w_coor;
                for (int j = 0; j < d_coor_downp_counter[coor_indx_b]; j++){
                    if(num_pnt < P){
                        idx[num_pnt++] = d_coor_to_downpidx[coor_indx_b * P + j];
                    }else{
                        curand_init(index + i*P + j, 0, 0, &state);
                        insrtidx = ceilf(curand_uniform(&state) * (++num_pnt)) - 1;
                        if(insrtidx < P){
                            idx[insrtidx] = d_coor_to_downpidx[coor_indx_b * P + j];
//                            if (index == 70) {
//                                printf("insrtidx:%d, num_pnt %d i=%d; d_coor_shift[i] %d,"
//                                       " h_coor_shift[i] %d, w_coor_shift[i] %d \n", insrtidx, num_pnt, i,
//                                       d_coor_shift[i], h_coor_shift[i], w_coor_shift[i]);
//                            }
                        }
                    }
                }
            }
        }

        d_upp_counter[i_ogrid]= min(P, num_pnt);
        for (int i = 0; i < d_upp_counter[i_ogrid]; i++){
            out_nebidx[i_batch * max_o * P + i_ogrid * P + i] = idx[i];
            out_nebidxmsk[i_batch * max_o * P + i_ogrid * P + i] = 1.0;
        }
        delete[] idx;
    }
}
}  // namespace cuda

template<typename Dtype>
inline void GridifyUpForward(Tensor<gpu, 3, Dtype> &nebidx, // B * O * P
                             Tensor<gpu, 3, Dtype> &nebidxmsk, // B * O * P
                             const Tensor<gpu, 3, Dtype> &downdata,
                             const Tensor<gpu, 3, Dtype> &updata,
                             const Tensor<gpu, 1, Dtype> &down_actual_numpoints,
                             const Tensor<gpu, 1, Dtype> &up_actual_numpoints,
                             const mxnet::op::GridifyUpParam &param
) {
    const int B = downdata.size(0);
    const int N = downdata.size(1);
    const int O = nebidx.size(1);
    const int P = nebidx.size(2);
//    printf("B: %d, N: %d, O: %d, P: %d; \n", B,N,O,P);
    // 3d voxel, ndim=3
    Dtype *out_nebidx = nebidx.dptr_;
    Dtype *out_nebidxmsk = nebidxmsk.dptr_;

    const Dtype *in_downdata = downdata.dptr_;
    const Dtype *in_updata = updata.dptr_;
    const Dtype *in_down_actual_numpoints = down_actual_numpoints.dptr_;
    const Dtype *in_up_actual_numpoints = up_actual_numpoints.dptr_;

//    index_t max_p_grid;
//    index_t max_o_grid;
//    index_t kernel_size;
//    index_t stride;
//    index_t width;
//    index_t height;
//    index_t depth;
//    nnvm::Tuple<float> coord_shift;
//    nnvm::Tuple<float> voxel_size;
    int grid_size_vol = (int)(param.grid_size[0] * param.grid_size[1] * param.grid_size[2]);
    const int size = param.kernel_size * param.kernel_size * param.kernel_size;
    float *coord_shift = new float[3];
    float *voxel_size = new float[3];
    float *grid_size = new float[3];
    int *coor_to_downpidx = new int[B * grid_size_vol * P];
    int *coor_downp_counter = new int[B * grid_size_vol];
    int *upp_counter = new int[B * O];
    int *d_coor_shift = new int[size];
    int *h_coor_shift = new int[size];
    int *w_coor_shift = new int[size];

    for(int i = 0; i < size; i++){
        d_coor_shift[i] = i / (param.kernel_size * param.kernel_size) - (param.kernel_size-1) / 2;
        h_coor_shift[i] = (i % (param.kernel_size * param.kernel_size)) / param.kernel_size - (param.kernel_size-1) / 2;
        w_coor_shift[i] =  i % param.kernel_size - (param.kernel_size-1) / 2;
    }

    memset(coor_to_downpidx, 0, B * grid_size_vol * P * sizeof(int));
    memset(coor_downp_counter, 0, B * grid_size_vol * sizeof(int));
    memset(upp_counter, 0, B * O * sizeof(int));

//    printf("grid_size_vol, %d \n", grid_size_vol);
//    printf("coor_to_voxelidx[0]: %d \n", coor_to_voxelidx[0]);
//    printf("coor_to_voxelidx[B * grid_size_vol-1]: %d \n", coor_to_voxelidx[B * grid_size_vol-1]);

    float *d_coord_shift, *d_voxel_size, *d_grid_size;
    int *d_coor_to_downpidx, *d_coor_downp_counter, *d_upp_counter,
            *d_d_coor_shift, *d_h_coor_shift, *d_w_coor_shift;

    for (int i = 0; i < 3; ++i) {
        coord_shift[i] = param.coord_shift[i];
        voxel_size[i] = param.voxel_size[i];
        grid_size[i] = param.grid_size[i];
    }
    cudaMalloc(&d_coord_shift, 3 * sizeof(float));
    cudaMalloc(&d_voxel_size, 3 * sizeof(float));
    cudaMalloc(&d_grid_size, 3 * sizeof(float));
    cudaMalloc(&d_d_coor_shift, size * sizeof(int));
    cudaMalloc(&d_h_coor_shift, size * sizeof(int));
    cudaMalloc(&d_w_coor_shift, size * sizeof(int));
    cudaMalloc(&d_coor_to_downpidx, B * grid_size_vol * P * sizeof(int));
    cudaMalloc(&d_coor_downp_counter, B * grid_size_vol * sizeof(int));
    cudaMalloc(&d_upp_counter, B * O * sizeof(int));

    cudaMemcpy(d_coord_shift, coord_shift, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_voxel_size, voxel_size, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_size, grid_size, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_d_coor_shift, d_coor_shift, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h_coor_shift, h_coor_shift, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w_coor_shift, w_coor_shift, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coor_to_downpidx, coor_to_downpidx,  B * grid_size_vol * P * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_coor_downp_counter, coor_downp_counter, B * grid_size_vol * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_upp_counter, upp_counter, B * O * sizeof(int), cudaMemcpyHostToDevice);

    const int gridSize = (B * N + cuda::kMaxThreadsPerBlock - 1) / cuda::kMaxThreadsPerBlock;
    dim3 dimGrid(gridSize);
    dim3 dimBlock(cuda::kMaxThreadsPerBlock);
//    printf("dimGrid: %d ; dimBlock %d ;", gridSize, cuda::kMaxThreadsPerBlock);
    cuda::CheckLaunchParam(dimGrid, dimBlock, "GridifyUpForward");
    cudaStream_t stream = Stream<gpu>::GetStream(nebidx.stream_);
    cuda::gridify_kernel_build_index<Dtype><<<dimGrid, dimBlock>>>(
            out_nebidx, out_nebidxmsk, in_downdata, in_updata, in_down_actual_numpoints, in_up_actual_numpoints,
            B, N, O, P, param.kernel_size, d_coord_shift, d_voxel_size, d_grid_size,
            grid_size_vol, d_coor_to_downpidx, d_coor_downp_counter, d_upp_counter);

    const int o_gridSize = (B * O + cuda::kMaxThreadsPerBlock - 1) / cuda::kMaxThreadsPerBlock;
    dim3 o_dimGrid(o_gridSize);
    dim3 o_dimBlock(cuda::kMaxThreadsPerBlock);
    MSHADOW_CUDA_POST_KERNEL_CHECK(cuda::gridify_kernel_build_index);
    cuda::gridify_kernel_query_neighs<Dtype><<<o_dimGrid, o_dimBlock>>>(
            out_nebidx, out_nebidxmsk, in_downdata, in_updata, in_down_actual_numpoints, in_up_actual_numpoints,
            B, N, O, P, param.kernel_size, d_coord_shift, d_voxel_size, d_grid_size,
            d_d_coor_shift, d_h_coor_shift, d_w_coor_shift, grid_size_vol, size,
            d_coor_to_downpidx, d_coor_downp_counter, d_upp_counter);
    MSHADOW_CUDA_POST_KERNEL_CHECK(cuda::gridify_kernel_query_neighs);

    delete coord_shift;
    delete voxel_size;
    delete grid_size;
    delete coor_to_downpidx;
    delete coor_downp_counter;
    delete upp_counter;
    cudaFree(d_coord_shift);
    cudaFree(d_voxel_size);
    cudaFree(d_grid_size);
    cudaFree(d_coor_to_downpidx);
    cudaFree(d_coor_downp_counter);
    cudaFree(d_upp_counter);
}

}  // namespace mshadow

namespace mxnet {
    namespace op {

        template<>
        Operator *CreateOp<gpu>(GridifyUpParam param, int dtype) {
            Operator *op = NULL;
            MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
                    op = new GridifyUpOp<gpu, DType>(param);
            });
            return op;
        }

    }  // namespace op
}  // namespace mxnet
