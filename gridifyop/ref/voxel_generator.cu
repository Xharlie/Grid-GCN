#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>

#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <stdio.h>
#include "./voxel_generator-inl.h"

using namespace std;

#define ndim 3
#define data_ndim 4

namespace mshadow {
    namespace cuda {


// compilation workaround
        __device__ double atomicAdd(double *address, double val) {
            return 0.0;
        }

        __device__  half::half_t
        atomicAdd(half::half_t
        * address,
        half::half_t val
        ) {
        return (half::half_t)0.0;
    }

    __device__ float atomicAdd(float *address, float val) {
        return ::atomicAdd(address, val);
    }

    __device__ float cas(double *addr, double compare, double val) {
        unsigned long long int *address_as_ull = (unsigned long long int *) addr;
        return __longlong_as_double(atomicCAS(address_as_ull,
                                              __double_as_longlong(compare),
                                              __double_as_longlong(val)));

    }

    __device__ float cas(float *addr, float compare, float val) {
        unsigned int *address_as_uint = (unsigned int *) addr;
        return __uint_as_float(atomicCAS(address_as_uint,
                                         __float_as_uint(compare),
                                         __float_as_uint(val)));
    }

    __device__ half::half_t
    cas(half::half_t
    * addr,
    half::half_t compare, half::half_t
    val) {
    // NOT IMPLEMENTED YET!
    return 0;
}

template<typename Dtype>
__global__ void voxel_gerenator_kernel_step1(
        Dtype *voxels,
        Dtype *coors,
        Dtype *coor_to_voxelidx,
        Dtype *num_pt_per_voxel,
        Dtype *actual_voxel_num,
        const Dtype *data,
        const Dtype *actual_points,
        const int point_size,
        const int max_points,
        const int max_voxels,
        const int height,
        const int width,
        const float *coors_range,
        const float *voxel_size,
        const float *grid_size,
        const int batch_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i_batch = index / point_size;
    if (i_batch >= batch_size) { return; }
    int i_pt = index - point_size * i_batch;
    if (i_pt < actual_points[i_batch]) {
        int coor[ndim];
        const Dtype *p_pt = data + (i_batch * point_size + i_pt) * data_ndim; //move to data dim (4)
        bool failed = false;
        for (int j = 0; j < ndim; j++) {
            int c = floor((p_pt[j] - coors_range[j]) / voxel_size[j]);
            if (c < 0 || c >= grid_size[j]) {
                failed = true;
                break;
            }
            coor[ndim - 1 - j] = c;
        }
        if (failed) {
            return;
        }

        int coor_index = coor[0] * (height * width) + coor[1] * width + coor[2];

        Dtype voxel_idx = coor_to_voxelidx[i_batch * height * width + coor_index];
        if (voxel_idx == max_voxels) {  // found an empty voxel
            if (actual_voxel_num[i_batch] >= max_voxels) {
                return; // we don't have more voxel to assign
            }
            // coors[ (i_batch*max_voxels+(int)*p_voxel_idx)*3+0 ] = *p_voxel_idx;
            Dtype old_voxel_num = cas(
                    &coor_to_voxelidx[i_batch * height * width + coor_index],
                    max_voxels,
                    actual_voxel_num[i_batch]
            );
            if (old_voxel_num == max_voxels) {
                // CAS -> old val, if old val is max_voxel
                // if we get max_voxel, this thread is the one who obtain a new voxel
                // so only this thread should do the increase operator below
                Dtype tmp = atomicAdd(actual_voxel_num + i_batch, (Dtype) 1.0); // increase the counter
                voxel_idx = tmp;
                if (voxel_idx < max_voxels) {
                    for (int k = 0; k < ndim; ++k) {
                        coors[(i_batch * max_voxels + (int) voxel_idx) * 3 + k] = coor[k];
                    }
                }
                // here atomic op will help us determin the order, tmp should be the acquired idx
                // coors[ (i_batch*max_voxels+(int)tmp)*3+1 ] = tmp;
                coor_to_voxelidx[i_batch * height * width + coor_index] = tmp;
            } else {
                // this thread get a new value
                // old_voxel_num should be the index, since CAS get the old value
                voxel_idx = old_voxel_num;
            }
            // coors[ (i_batch*max_voxels+(int)*p_voxel_idx)*3+2 ] = actual_voxel_num[i_batch];
        } else {
        } // we encounter a non-empty voxel, do nothing
    }
} // end of __global__ void voxel_gerenator_kernel_step1

template<typename Dtype>
__global__ void voxel_gerenator_kernel_step2(
        Dtype *voxels,
        Dtype *coors,
        Dtype *coor_to_voxelidx,
        Dtype *num_pt_per_voxel,
        Dtype *actual_voxel_num,
        const Dtype *data,
        const Dtype *actual_points,
        const int point_size,
        const int max_points,
        const int max_voxels,
        const int height,
        const int width,
        const float *coors_range,
        const float *voxel_size,
        const float *grid_size,
        const int batch_size
) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int i_batch = index / point_size;
    if (i_batch >= batch_size) { return; }

    int i_pt = index - point_size * i_batch;
    if (i_pt < actual_points[i_batch]) {
        int coor[ndim];
        const Dtype *p_pt = data + (i_batch * point_size + i_pt) * data_ndim;
        bool failed = false;
        for (int j = 0; j < ndim; j++) {
            int c = floor((p_pt[j] - coors_range[j]) / voxel_size[j]);
            if (c < 0 || c >= grid_size[j]) {
                failed = true;
                break;
            }
            coor[ndim - 1 - j] = c;
        }
        if (failed) {
            return;
        }

        int coor_index = coor[0] * (height * width) + coor[1] * width + coor[2];

        Dtype voxel_idx = coor_to_voxelidx[i_batch * height * width + coor_index];
        if (voxel_idx >= max_voxels) {
            // skip voxel that exceed max_voxel(invalid voxel)
            // set back to max_voxels
            coor_to_voxelidx[i_batch * height * width + coor_index] = max_voxels;
            return;
        }

        // num should be the old value in num_pt_per_voxel
        int num = (int) atomicAdd(num_pt_per_voxel + i_batch * max_voxels + (int) voxel_idx, (Dtype) 1.0);

        if (num >= max_points) {
            // do nothing
            // voxels[i_pt] = -1.0;
            // voxel is full
            num_pt_per_voxel[i_batch * max_voxels + (int) voxel_idx] = max_points;
        } else {
            int batch_stride = data_ndim * max_voxels * max_points;
            int voxel_stride = data_ndim * max_points;
            int pt_stride = data_ndim;
            // voxels[i_pt*4] = 1.0;
            // copy pt into the slot
            for (int k = 0; k < data_ndim; ++k) {
                voxels[i_batch * batch_stride + (int) voxel_idx * voxel_stride + num * pt_stride + k] =
                        data[i_batch * max_points * data_ndim + i_pt * data_ndim + k];
            }
        }
    }
} // end of __global__ void voxel_gerenator_kernel_step2

}  // namespace cuda

template<typename Dtype>
inline void VoxelGeneratorForward(Tensor<gpu, 4, Dtype> &voxels,
                                  Tensor<gpu, 3, Dtype> &coors,
                                  Tensor<gpu, 2, Dtype> &coor_to_voxelidx,
                                  Tensor<gpu, 2, Dtype> &num_points_per_voxel,
                                  Tensor<gpu, 1, Dtype> &actual_voxel_num,
                                  const Tensor<gpu, 3, Dtype> &data,
                                  const Tensor<gpu, 1, Dtype> &actual_points,
                                  const mxnet::op::VoxelGeneratorParam &param
) {
    const int batch_size_ = data.size(0);
    const int point_size_ = data.size(1);
    // 3d voxel, ndim=3

    Dtype *out_voxels = voxels.dptr_;
    Dtype *out_coors = coors.dptr_;
    Dtype *out_coor_to_voxelidx = coor_to_voxelidx.dptr_;
    Dtype *out_num_points_per_voxel = num_points_per_voxel.dptr_;
    Dtype *out_actual_voxel_num = actual_voxel_num.dptr_;

    const Dtype *in_data = data.dptr_;
    const Dtype *in_actual_points = actual_points.dptr_;

    float *coors_range = new float[6];
    float *voxel_size = new float[3];
    float *grid_size = new float[3];
    float *d_coors_range, *d_voxel_size, *d_grid_size;

    for (int i = 0; i < ndim; ++i) {
        coors_range[i] = param.coors_range[i];
        coors_range[i + ndim] = param.coors_range[i + ndim];
        voxel_size[i] = param.voxel_size[i];
        grid_size[i] = std::round((param.coors_range[i + ndim] - param.coors_range[i]) / param.voxel_size[i]);
    }
    // perpare params@device
//  if(_d_params == 0) {
//    cudaMalloc(&_d_params, sizeof(cuda::kernel_params));
//  }
    cudaMalloc(&d_coors_range, 2 * ndim * sizeof(float));
    cudaMalloc(&d_voxel_size, ndim * sizeof(float));
    cudaMalloc(&d_grid_size, ndim * sizeof(float));
    cudaMemcpy(d_coors_range, coors_range, 2 * ndim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_voxel_size, voxel_size, ndim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grid_size, grid_size, ndim * sizeof(float), cudaMemcpyHostToDevice);

    const int gridSize = (data.size(0) * data.size(1) + cuda::kMaxThreadsPerBlock - 1) / cuda::kMaxThreadsPerBlock;
    dim3 dimGrid(gridSize);
    dim3 dimBlock(cuda::kMaxThreadsPerBlock);
    cuda::CheckLaunchParam(dimGrid, dimBlock, "VoxelGeneratorForward");
    cudaStream_t stream = Stream<gpu>::GetStream(voxels.stream_);
    cuda::voxel_gerenator_kernel_step1 < Dtype ><<<dimGrid, dimBlock>>>(
            out_voxels, out_coors, out_coor_to_voxelidx, out_num_points_per_voxel, out_actual_voxel_num,
                    in_data, in_actual_points,
                    point_size_, param.max_points, param.max_voxels, param.height, param.width, d_coors_range, d_voxel_size, d_grid_size,
                    batch_size_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(cuda::voxel_gerenator_kernel_step1);

    cuda::voxel_gerenator_kernel_step2 < Dtype ><<<dimGrid, dimBlock>>>(
            out_voxels, out_coors, out_coor_to_voxelidx, out_num_points_per_voxel, out_actual_voxel_num,
                    in_data, in_actual_points,
                    point_size_, param.max_points, param.max_voxels, param.height, param.width, d_coors_range, d_voxel_size, d_grid_size,
                    batch_size_);
    MSHADOW_CUDA_POST_KERNEL_CHECK(cuda::voxel_gerenator_kernel_step2);

    delete coors_range;
    delete voxel_size;
    delete grid_size;
    cudaFree(d_coors_range);
    cudaFree(d_voxel_size);
    cudaFree(d_grid_size);
}

}  // namespace mshadow

namespace mxnet {
    namespace op {

        template<>
        Operator *CreateOp<gpu>(VoxelGeneratorParam param, int dtype) {
            Operator *op = NULL;
            MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
                    op = new VoxelGeneratorOp<gpu, DType>(param);
            });
            return op;
        }

    }  // namespace op
}  // namespace mxnet
