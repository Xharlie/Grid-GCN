#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include "./gridifyknn-inl.h"
#include <cooperative_groups.h>
#include <math.h>
#include <stdlib.h>
#include <curand_kernel.h>
#include <sys/time.h>
//using namespace cooperative_groups;

#define ndim 3
#define data_ndim 4
#define CUDA_CHECK_ERROR() __cuda_check_errors(__FILE__, __LINE__)
#define CUDA_SAFE_CALL(err) __cuda_safe_call(err, __FILE__, __LINE__)
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

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
    static __device__ double atomicAdd(double *address, double val) {
            return 0.0;
        }

    static    __device__  half::half_t
        atomicAdd(half::half_t
        * address,
        half::half_t val
        ) {
        return (half::half_t)0.0;
    }

    static __device__ float atomicAdd(float *address, float val) {
        return ::atomicAdd(address, val);
    }

    static __device__ float atomicAdd(int *address, int val) {
        return ::atomicAdd(address, val);
    }

    static __device__ float cas(double *addr, double compare, double val) {
        unsigned long long int *address_as_ull = (unsigned long long int *) addr;
        return __longlong_as_double(atomicCAS(address_as_ull,
                                              __double_as_longlong(compare),
                                              __double_as_longlong(val)));

    }

    static __device__ float cas(float *addr, float compare, float val) {
        unsigned int *address_as_uint = (unsigned int *) addr;
        return __uint_as_float(atomicCAS(address_as_uint,
                                         __float_as_uint(compare),
                                         __float_as_uint(val)));
    }

    static __device__ half::half_t
    cas(half::half_t * addr,
    half::half_t compare, half::half_t
    val) {
    // NOT IMPLEMENTED YET!
    return 0;
}




template<typename Dtype>
__global__ void gridifyKNN_kernel_build_index(
        int* out_nebidx, Dtype* out_nebidxmsk, Dtype* out_cent,
        Dtype* out_centmsk, int* out_actual_centnum, int* actual_centcount,
        const Dtype* in_data, const int* in_actual_numpoints,
        const int B,
        const int N,
        const int max_o,
        const int P,
        const int kernel_size,
        const int stride,
        const int loc,
        const float *d_coord_shift,
        const float *d_voxel_size,
        const float *d_grid_size,
        const int grid_size_vol,
        const int size,
        int *coor_to_voxelidx,
        int *voxelidx_to_coor,
        int *coor_to_pntidx,
        float *coor_to_locxyzw,
        int *coor_counter,
        unsigned long seconds
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / N;  // index of batch
    if (i_batch >= B) { return; }
    int i_pt = index - N * i_batch;
    if (i_pt < in_actual_numpoints[i_batch]) {
        curandState state;
        int coor[ndim];
        const Dtype *p_pt = in_data + index * data_ndim;
        for (int j = 0; j < ndim; j++) {
            int c = floor((p_pt[j] + d_coord_shift[j]) / d_voxel_size[j]);
            if (c < 0 || c >= d_grid_size[j]) {
                return;
            }
            coor[j] = c;
        }
        int coor_indx = coor[2] * (d_grid_size[0] * d_grid_size[1])
                        + coor[1] * d_grid_size[0] + coor[0];
        int coor_indx_b = i_batch * grid_size_vol + coor_indx;

        int grid_pntidx = atomicAdd(coor_counter+coor_indx_b, 1);
        if (grid_pntidx < P) {
            coor_to_pntidx[coor_indx_b * P + grid_pntidx] = i_pt;
        } else {
            curand_init(index+seconds, 0, 0, &state);
            int insrtidx = ceilf(curand_uniform(&state) * (grid_pntidx+1)) - 1;
            if(insrtidx < P){
                coor_to_pntidx[coor_indx_b * P + insrtidx] = i_pt;
            }
        }
        if(loc == 1){
            int coor_b_idx = coor_indx_b * data_ndim;
            float weight = p_pt[3];
            atomicAdd(coor_to_locxyzw + coor_b_idx, p_pt[0] * weight);
            atomicAdd(coor_to_locxyzw + coor_b_idx + 1, p_pt[1] * weight);
            atomicAdd(coor_to_locxyzw + coor_b_idx + 2, p_pt[2] * weight);
            atomicAdd(coor_to_locxyzw + coor_b_idx + 3, weight);
        }


        int voxel_idx = coor_to_voxelidx[coor_indx_b];
        //        printf("grid_size_vol: %d, coor_index: %d, i_batch %d, voxel_idx: %d ; \n", grid_size_vol, coor_index, i_batch, voxel_idx);
        if (voxel_idx == -1) {  // found an empty voxel
            Dtype old_voxel_num = atomicCAS(
                    &coor_to_voxelidx[coor_indx_b],
                    -1, 0
            );
            if (old_voxel_num == -1) {
                // CAS -> old val, if old val is -1
                // if we get -1, this thread is the one who obtain a new voxel
                // so only this thread should do the increase operator below
                int tmp = atomicAdd(out_actual_centnum + i_batch, 1); // increase the counter, return old counter
                 // increase the counter, return old counter
                if (tmp < max_o) {
                    voxelidx_to_coor[i_batch * max_o + tmp] = coor_indx;
                    out_centmsk[i_batch * max_o + tmp] = 1.0; // change center mask to 1 at new occupied voxel
                } else {
                    curand_init(index+2*seconds, 0, 0, &state);
                    int insrtidx = ceilf(curand_uniform(&state) * (tmp+1)) - 1;
                    if(insrtidx < max_o){
                        voxelidx_to_coor[i_batch * max_o + insrtidx] = coor_indx;
                    }
                }
            }
        }
    }
}

template<typename Dtype>
__global__ void gridifyKNN_kernel_query_neighs(
        int* out_nebidx, Dtype* out_nebidxmsk, Dtype* out_cent,
        Dtype* out_centmsk, int* out_actual_centnum,
        const Dtype* in_data, const int* in_actual_numpoints,
        const int B,
        const int N,
        const int max_o,
        const int P,
        const int kernel_size,
        const int stride,
        const int loc,
        const float *d_coord_shift,
        const float *d_voxel_size,
        const float *d_grid_size,
        const int grid_size_vol,
        const int size,
        int *coor_to_voxelidx,
        int *voxelidx_to_coor,
        int *coor_to_pntidx,
        float *coor_to_locxyzw,
        int *coor_counter,
        int *voxelidx_counter,
        unsigned long seconds
){
    int index = blockIdx.x * blockDim.x + threadIdx.x; // index of gpu thread
    int i_batch = index / max_o;  // index of batch
    if (i_batch >= B) { return; }
    int i_ogrid = index - i_batch * max_o;
    if(out_actual_centnum[i_batch] > max_o){
        out_actual_centnum[i_batch] = max_o;
    }
    if (i_ogrid < out_actual_centnum[i_batch]) {
        int coor_indx_b, in_data_eleweight, idx, idx_b_data;
        float xsum = 0.0, ysum = 0.0, zsum = 0.0, countweightsum = 0;
//        int coor_indx = coor[2] * (d_grid_size[0] * d_grid_size[1])
//                        + coor[1] * d_grid_size[0] + coor[0];
        int coor = voxelidx_to_coor[index];
        int coor2 = coor / (d_grid_size[0] * d_grid_size[1]);
        int coor1 = (coor - coor2* (d_grid_size[0] * d_grid_size[1])) / d_grid_size[0];
        int coor0 = coor - coor2* (d_grid_size[0] * d_grid_size[1]) - coor1 * d_grid_size[0];
        int d_coor, h_coor, w_coor;
        float total_weight = 0;
        int index_data = index * data_ndim;
        int index_P = index * P;
        int coor_indx_b_origin;

        float ux = (coor0 + 0.5) * d_voxel_size[0];
        float uy = (coor1 + 0.5) * d_voxel_size[1];
        float uz = (coor2 + 0.5) * d_voxel_size[2];

        float best[128];
        int besti[128];
        for (int l = 0; l < 64; l++){
            best[l] = FLT_MAX;
        }
        int need_P = P, amount_layer=0, amount;
        float x,y,z,dst;
        for (int layer = 0; layer < (kernel_size+1)/2; layer++){
            amount_layer = 0;
            for (int w = -layer; w < layer+1; w++) {
                for (int h = -layer; h < layer + 1; h++) {
                    for (int d = -layer; d < layer + 1; d++) {
                        if (max(max(abs(w),abs(h)),abs(d)) != layer) continue;

                        d_coor = d + coor2;
                        h_coor = h + coor1;
                        w_coor = w + coor0;
                        if (d_coor >= 0 && d_coor < d_grid_size[2] && h_coor >= 0 &&
                            h_coor < d_grid_size[1] && w_coor >= 0 && w_coor < d_grid_size[0]) {
                            coor_indx_b = i_batch * grid_size_vol + d_coor * (d_grid_size[0] * d_grid_size[1])
                                          + h_coor * d_grid_size[0] + w_coor;
                            if (layer == 0) coor_indx_b_origin = coor_indx_b;
                            amount = min(P, coor_counter[coor_indx_b]);
                            amount_layer += amount;
                            for (int g = 0; g < amount; g++) {
                                idx = coor_to_pntidx[coor_indx_b * P + g];
                                idx_b_data = (idx + N * i_batch) * data_ndim;
                                x = in_data[idx_b_data];
                                y = in_data[idx_b_data + 1];
                                z = in_data[idx_b_data + 2];
                                dst = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
                                for (int l = 0; l < P; l++) {
                                    if (dst < best[l]) {
                                        for (int j = P - 1; j > l; j--) {
                                            best[j] = best[j - 1];
                                            besti[j] = besti[j - 1];
                                        }
                                        best[l] = dst;
                                        besti[l] = idx;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            need_P = need_P - amount_layer;
            if (need_P<=0) break;
        }

        for (int l = 0; l < P; l++){
            out_nebidx[index_P + l] = besti[l];
            idx_b_data = (besti[l] + N * i_batch) * data_ndim;
            in_data_eleweight = in_data[idx_b_data + 3];
            out_nebidxmsk[index_P + l] = 1.0;
            total_weight += in_data_eleweight;
        }

        out_cent[index_data + 3] = total_weight;
        if (need_P > 0){
            for (int j = P-need_P; j < P; j++){
                out_nebidx[index_P + j] = besti[0];
            }
        }
        if(loc==1){
            int coor_indx_b_data = coor_indx_b_origin * data_ndim;
            xsum = coor_to_locxyzw[coor_indx_b_data];
            ysum = coor_to_locxyzw[coor_indx_b_data +1];
            zsum = coor_to_locxyzw[coor_indx_b_data +2];
            countweightsum = coor_to_locxyzw[coor_indx_b_data + 3];
            out_cent[index_data] = xsum / countweightsum;
            out_cent[index_data + 1] = ysum / countweightsum;
            out_cent[index_data + 2] = zsum / countweightsum;
        }
    }
}
}  // namespace cuda

template<typename Dtype>
inline void GridifyKNNForward(Tensor<gpu, 3, int> &nebidx, // B * O * P
                           Tensor<gpu, 3, Dtype> &nebidxmsk, // B * O * P
                           Tensor<gpu, 3, Dtype> &cent, // B * O * 4
                           Tensor<gpu, 2, Dtype> &centmsk, // B * O
                           Tensor<gpu, 2, int> &actual_centnum, // B
                           const Tensor<gpu, 3, Dtype> &data,   // B * N * 4
                           const Tensor<gpu, 2, int> &actual_numpoints, // B
                           const mxnet::op::GridifyKNNParam &param
) {
    const int B = data.size(0);
    const int N = data.size(1);
    const int O = nebidx.size(1);
    const int P = nebidx.size(2);
//    printf("B: %d, N: %d, O: %d, P: %d; \n", B,N,O,P);
    // 3d voxel, ndim=3
    int *out_nebidx = nebidx.dptr_;
    Dtype *out_nebidxmsk = nebidxmsk.dptr_;
    Dtype *out_cent = cent.dptr_;
    Dtype *out_centmsk = centmsk.dptr_;
    int *out_actual_centnum = actual_centnum.dptr_;

    const Dtype *in_data = data.dptr_;
    const int *in_actual_numpoints = actual_numpoints.dptr_;

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
//
//    for(int i = 0; i < size; i++){
//        d_coor_shift[i] = i / (param.kernel_size * param.kernel_size) - (param.kernel_size-1) / 2;
//        h_coor_shift[i] = (i % (param.kernel_size * param.kernel_size)) / param.kernel_size - (param.kernel_size-1) / 2;
//        w_coor_shift[i] =  i % param.kernel_size - (param.kernel_size-1) / 2;
//    }

    float *d_coord_shift, *d_voxel_size, *d_grid_size, *d_coor_to_locxyzw;
    int *d_coor_to_voxelidx, *d_voxelidx_to_coor, *d_coor_to_pntidx, *d_coor_counter, *d_voxelidx_counter, *actual_centcount;

    for (int i = 0; i < 3; ++i) {
        coord_shift[i] = param.coord_shift[i];
        voxel_size[i] = param.voxel_size[i];
        grid_size[i] = param.grid_size[i];
    }
    cudaMalloc(&d_coord_shift, 3 * sizeof(float));
    cudaMalloc(&d_voxel_size, 3 * sizeof(float));
    cudaMalloc(&d_grid_size, 3 * sizeof(float));

    cudaMemcpy(d_coord_shift, coord_shift, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_voxel_size, voxel_size, 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_size, grid_size, 3 * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_coor_to_locxyzw, B * grid_size_vol * data_ndim * sizeof(float));
    cudaMalloc(&d_coor_to_pntidx, B * grid_size_vol * P * sizeof(int));
    cudaMalloc(&d_coor_counter, B * grid_size_vol * sizeof(int));
    cudaMalloc(&d_coor_to_voxelidx, B * grid_size_vol * sizeof(int));
    cudaMalloc(&d_voxelidx_to_coor, B * O * sizeof(int));
    cudaMalloc(&d_voxelidx_counter, B * O * sizeof(int));
    cudaMalloc(&actual_centcount, B * sizeof(int));
    cudaMemset(d_coor_to_locxyzw, 0, B * grid_size_vol * data_ndim * sizeof(float));
    cudaMemset(d_coor_counter, 0, B * grid_size_vol * sizeof(int));
    cudaMemset(d_voxelidx_counter, 0, B * O * sizeof(int));
    cudaMemset(d_coor_to_voxelidx, -1, B * grid_size_vol * sizeof(int));
    cudaMemset(actual_centcount, 0, B * sizeof(int));


    const int gridSize = (B * N + cuda::kMaxThreadsPerBlock - 1) / cuda::kMaxThreadsPerBlock;
//    dim3 dimGrid(sqrt(gridSize), gridSize / sqrt(gridSize) + 1);
    dim3 dimGrid(gridSize);
    dim3 dimBlock(cuda::kMaxThreadsPerBlock);
//    printf("dimGrid: %d ; dimBlock %d ;", gridSize, cuda::kMaxThreadsPerBlock);
    cuda::CheckLaunchParam(dimGrid, dimBlock, "GridifyKNNForward");
    cudaStream_t stream = Stream<gpu>::GetStream(nebidx.stream_);
    timeval curTime;
    gettimeofday(&curTime, NULL);
    unsigned long seconds = curTime.tv_usec;
//    printf("seconds: %lu", seconds);
    cuda::gridifyKNN_kernel_build_index<Dtype><<<dimGrid, dimBlock>>>(
            out_nebidx, out_nebidxmsk, out_cent, out_centmsk, out_actual_centnum, actual_centcount, in_data, in_actual_numpoints,
            B, N, O, P, param.kernel_size, param.stride, param.loc, d_coord_shift, d_voxel_size, d_grid_size,
            grid_size_vol, size, d_coor_to_voxelidx, d_voxelidx_to_coor, d_coor_to_pntidx, d_coor_to_locxyzw, d_coor_counter,
            seconds);
    MSHADOW_CUDA_POST_KERNEL_CHECK(cuda::gridifyKNN_kernel_build_index);


    const int o_gridSize = (B * O + cuda::kMaxThreadsPerBlock - 1) / cuda::kMaxThreadsPerBlock;
    dim3 o_dimGrid(o_gridSize);
    dim3 o_dimBlock(cuda::kMaxThreadsPerBlock);
    cuda::gridifyKNN_kernel_query_neighs<Dtype><<<o_dimGrid, o_dimBlock>>>(
            out_nebidx, out_nebidxmsk, out_cent, out_centmsk, out_actual_centnum, in_data, in_actual_numpoints,
            B, N, O, P, param.kernel_size, param.stride, param.loc, d_coord_shift, d_voxel_size, d_grid_size,
            grid_size_vol, size, d_coor_to_voxelidx, d_voxelidx_to_coor, d_coor_to_pntidx, d_coor_to_locxyzw,
            d_coor_counter, d_voxelidx_counter, seconds);
    MSHADOW_CUDA_POST_KERNEL_CHECK(cuda::gridifyKNN_kernel_query_neighs);


    delete coord_shift;
    delete voxel_size;
    delete grid_size;
    cudaFree(d_coord_shift);
    cudaFree(d_voxel_size);
    cudaFree(d_grid_size);
    cudaFree(d_coor_to_voxelidx);
    cudaFree(d_voxelidx_to_coor);
    cudaFree(d_coor_to_pntidx);
    cudaFree(d_coor_to_locxyzw);
    cudaFree(d_coor_counter);
    cudaFree(d_voxelidx_counter);
    cudaFree(actual_centcount);
}

}  // namespace mshadow

namespace mxnet {
    namespace op {

        template<>
        Operator *CreateOp<gpu>(GridifyKNNParam param, int dtype) {
            Operator *op = NULL;
            MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
                    op = new GridifyKNNOp<gpu, DType>(param);
            });
            return op;
        }

    }  // namespace op
}  // namespace mxnet
