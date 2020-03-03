#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include "./gridify_up-inl.h"
#include <cooperative_groups.h>
#include <math.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <sys/time.h>

using namespace cooperative_groups;
//using namespace std;

#define ndim 3
#define data_ndim 4
#define CUDA_CHECK_ERROR() __cuda_check_errors(__FILE__, __LINE__)
#define CUDA_SAFE_CALL(err) __cuda_safe_call(err, __FILE__, __LINE__)


// See: http://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
inline void
__cuda_check_errors (const char *filename, const int line_number)
{
    cudaError err = cudaDeviceSynchronize ();
    if (err != cudaSuccess)
    {
        printf ("CUDA error %i at %s:%i: %s\n",
                err, filename, line_number, cudaGetErrorString (err));
        exit (-1);
    }
}

inline void
__cuda_safe_call (cudaError err, const char *filename, const int line_number)
{
    if (err != cudaSuccess)
    {
        printf ("CUDA error %i at %s:%i: %s\n",
                err, filename, line_number, cudaGetErrorString (err));
        exit (-1);
    }
}


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
                int* out_nebidx, Dtype* out_nebidxmsk,
                const Dtype* in_downdata, const Dtype* in_updata,
                const int* in_down_actual_numpoints, const int* in_up_actual_numpoints,
                const int B,
                const int N,
                const int max_o,
                const int P,
                const int kernel_size,
                const float *d_coord_shift,
                const float *d_voxel_size,
                const float *d_grid_size,
                const int grid_size_vol,
                const int size,
                int *d_coor_to_downpidx,
                int *d_coor_downp_counter,
                unsigned long seconds
        ) {
            int threadindex = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
            int nei_idx = threadindex % size;
            int index = threadindex / size;
            int i_batch = index / N;  // index of batch

            if (i_batch >= B) { return; }
            int i_pt = index - N * i_batch;

            if (i_pt < in_down_actual_numpoints[i_batch]) {
                int coor[ndim];
                curandState state;
                const Dtype *p_pt = in_downdata + index * data_ndim;
                for (int j = 0; j < ndim; j++) {
                    int c = floor((p_pt[j] + d_coord_shift[j]) / d_voxel_size[j]);
                    if (c < 0 || c >= d_grid_size[j]) {
                        return;
                    }
                    coor[j] = c;
                }
//        printf("value: p_pt[0] + d_coord_shift[0]: %f ;coor[0]: %d ",p_pt[0] + d_coord_shift[0], coor[0]);
                int d_coor = nei_idx / (kernel_size * kernel_size) - (kernel_size-1) / 2 + coor[2];
                int h_coor = (nei_idx % (kernel_size * kernel_size)) / kernel_size - (kernel_size-1) / 2 + coor[1];
                int w_coor = nei_idx % kernel_size - (kernel_size-1) / 2 + coor[0];

                if (d_coor >= 0 && d_coor < d_grid_size[2] && h_coor >= 0 && h_coor < d_grid_size[1] && w_coor >= 0 &&
                    w_coor < d_grid_size[0]) {
                    int coor_indx_b = i_batch * grid_size_vol + d_coor * (d_grid_size[0] * d_grid_size[1])
                                      + h_coor * d_grid_size[0] + w_coor;
//            if (d_coor_downp_counter[coor_indx_b] < P) {
//                int grid_pntidx = atomicAdd(d_coor_downp_counter + coor_indx_b, 1);
//                if (grid_pntidx < P) {
//                    d_coor_to_downpidx[coor_indx_b * P + grid_pntidx] = i_pt;
//                } else {
//                    d_coor_downp_counter[coor_indx_b] = P;
//                }
//            }

                    int grid_pntidx = atomicAdd(d_coor_downp_counter + coor_indx_b, 1);
                    if (grid_pntidx < P) {
                        d_coor_to_downpidx[coor_indx_b * P + grid_pntidx] = i_pt;
                    } else {
                        curand_init(seconds + threadindex, 0, 0, &state);
                        int insrtidx = ceilf(curand_uniform(&state) * (grid_pntidx+1)) - 1;
                        if(insrtidx < P){
                            d_coor_to_downpidx[coor_indx_b * P + insrtidx] = i_pt;
                        }
                    }
                }
            }
        }

        template<typename Dtype>
        __global__ void gridify_kernel_query_neighs(
                int* out_nebidx, Dtype* out_nebidxmsk,
                const Dtype* in_downdata, const Dtype* in_updata,
                const int* in_down_actual_numpoints, const int* in_up_actual_numpoints,
                const int B,
                const int N,
                const int max_o,
                const int P,
                const int kernel_size,
                const float *d_coord_shift,
                const float *d_voxel_size,
                const float *d_grid_size,
                const int grid_size_vol,
                const int size,
                int *d_coor_to_downpidx,
                int *d_coor_downp_counter
        ){
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int i_batch = index / max_o;  // index of batch
            if (i_batch >= B) { return; }
            int i_ogrid = index - i_batch * max_o;
            if (i_ogrid < in_up_actual_numpoints[i_batch]) {
//        printf("in_up_actual_numpoints %d \n",in_up_actual_numpoints[i_batch]);
                int coor[ndim];
                const Dtype *p_pt = in_updata + index * data_ndim;
                for (int j = 0; j < ndim; j++) {
                    int c = floor((p_pt[j] + d_coord_shift[j]) / d_voxel_size[j]);
                    if (c < 0 || c >= d_grid_size[j]) {
                        return;
                    }
                    coor[j] = c;
                }
                if (coor[2] >= 0 && coor[2] < d_grid_size[2] && coor[1] >= 0 && coor[1] < d_grid_size[1]
                    && coor[0] >= 0 && coor[0] < d_grid_size[0]){
                    int coor_indx_b = i_batch * grid_size_vol + coor[2] * (d_grid_size[0] * d_grid_size[1])
                                      + coor[1] * d_grid_size[0] + coor[0];
//            memcpy(out_nebidx + index * P, d_coor_to_downpidx + coor_indx_b * P,
//                    d_coor_downp_counter[coor_indx_b] * sizeof(int));
                    int initID, idx, countlimit;
                    countlimit = d_coor_downp_counter[coor_indx_b];
                    for (int j = 0; j < P; j++){
                        if (j < countlimit){
                            idx= d_coor_to_downpidx[coor_indx_b * P + j];
                            if (j == 0){initID = idx;}
                            out_nebidx[index * P + j] = idx;
                            out_nebidxmsk[index * P + j] = 1.0;
                        } else {
                            out_nebidx[index * P + j] = initID;
                        }
                    }
                }
            }
        }
    }  // namespace cuda

    template<typename Dtype>
    inline void GridifyUpForward(Tensor<gpu, 3, int> &nebidx, // B * O * P
                                 Tensor<gpu, 3, Dtype> &nebidxmsk, // B * O * P
                                 const Tensor<gpu, 3, Dtype> &downdata,
                                 const Tensor<gpu, 3, Dtype> &updata,
                                 const Tensor<gpu, 2, int> &down_actual_numpoints,
                                 const Tensor<gpu, 2, int> &up_actual_numpoints,
                                 const mxnet::op::GridifyUpParam &param
    ) {
        const int B = downdata.size(0);
        const int N = downdata.size(1);
        const int O = nebidx.size(1);
        const int P = nebidx.size(2);
//    printf("B: %d, N: %d, O: %d, P: %d; \n", B,N,O,P);
        // 3d voxel, ndim=3
        int *out_nebidx = nebidx.dptr_;
        Dtype *out_nebidxmsk = nebidxmsk.dptr_;

        const Dtype *in_downdata = downdata.dptr_;
        const Dtype *in_updata = updata.dptr_;
        const int *in_down_actual_numpoints = down_actual_numpoints.dptr_;
        const int *in_up_actual_numpoints = up_actual_numpoints.dptr_;

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


        float *d_coord_shift, *d_voxel_size, *d_grid_size;
        int *d_coor_to_downpidx, *d_coor_downp_counter;

        for (int i = 0; i < 3; ++i) {
            coord_shift[i] = param.coord_shift[i];
            voxel_size[i] = param.voxel_size[i];
            grid_size[i] = param.grid_size[i];
        }
        cudaMalloc(&d_coord_shift, 3 * sizeof(float));
        cudaMalloc(&d_voxel_size, 3 * sizeof(float));
        cudaMalloc(&d_grid_size, 3 * sizeof(float));
        cudaMalloc(&d_coor_to_downpidx, B * grid_size_vol * P * sizeof(int));
        cudaMalloc(&d_coor_downp_counter, B * grid_size_vol * sizeof(int));

        cudaMemcpy(d_coord_shift, coord_shift, 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_voxel_size, voxel_size, 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_grid_size, grid_size, 3 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_coor_to_downpidx, 0, B * grid_size_vol * P * sizeof(int));
        cudaMemset(d_coor_downp_counter, 0, B * grid_size_vol * sizeof(int));


        const int gridSize = (B * N * size + cuda::kMaxThreadsPerBlock - 1) / cuda::kMaxThreadsPerBlock;
        timeval curTime;
        gettimeofday(&curTime, NULL);
        unsigned long seconds = curTime.tv_usec;
        dim3 dimGrid(gridSize);
        dim3 dimBlock(cuda::kMaxThreadsPerBlock);
//    printf("dimGrid: %d ; dimBlock %d ;", gridSize, cuda::kMaxThreadsPerBlock);
        cuda::CheckLaunchParam(dimGrid, dimBlock, "GridifyUpForward");
        cudaStream_t stream = Stream<gpu>::GetStream(nebidx.stream_);
        cuda::gridify_kernel_build_index<Dtype><<<dimGrid, dimBlock>>>(
                out_nebidx, out_nebidxmsk, in_downdata, in_updata, in_down_actual_numpoints, in_up_actual_numpoints,
                        B, N, O, P, param.kernel_size, d_coord_shift, d_voxel_size, d_grid_size,
                        grid_size_vol, size,
                        d_coor_to_downpidx, d_coor_downp_counter, seconds);

        const int o_gridSize = (B * O + cuda::kMaxThreadsPerBlock - 1) / cuda::kMaxThreadsPerBlock;
        dim3 o_dimGrid(o_gridSize);
        dim3 o_dimBlock(cuda::kMaxThreadsPerBlock);
        MSHADOW_CUDA_POST_KERNEL_CHECK(cuda::gridify_kernel_build_index);

        cuda::gridify_kernel_query_neighs<Dtype><<<o_dimGrid, o_dimBlock>>>(
                out_nebidx, out_nebidxmsk, in_downdata, in_updata, in_down_actual_numpoints, in_up_actual_numpoints,
                        B, N, O, P, param.kernel_size, d_coord_shift, d_voxel_size, d_grid_size,
                        grid_size_vol, size,
                        d_coor_to_downpidx, d_coor_downp_counter);
        MSHADOW_CUDA_POST_KERNEL_CHECK(cuda::gridify_kernel_query_neighs);

        delete coord_shift;
        delete voxel_size;
        delete grid_size;
        cudaFree(d_coord_shift);
        cudaFree(d_voxel_size);
        cudaFree(d_grid_size);
        cudaFree(d_coor_to_downpidx);
        cudaFree(d_coor_downp_counter);
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
