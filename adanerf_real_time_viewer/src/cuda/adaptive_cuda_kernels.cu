#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "device_launch_parameters.h"

#include "../include/cuda/adanerf_cuda_kernels.cuh"
#include "../include/cuda/helper_math.h"
#include "../include/cuda/adanerf_cuda_helper.h"

#include <cuda.h>
#include <cub/cub.cuh>
#include <iostream>
#include <assert.h> 

constexpr int WARP_SIZE = 32;

// Each ray gets its offset in the z_vals via atomic operations on this global counter
__device__ int global_offset_counter;

template<int N_DEPTHS, int MAX_OUTPUT, int NUM_WARPS>
__global__ void sampleAdaptiveWarpMax(const float* __restrict depth_vals, const int batch_size,
  const int n_max_ray_samples, const float adaptive_sampling_threshold,
  int* __restrict ray_offsets, int* __restrict ray_n_samples,
  int* __restrict d_ray_idx_per_z_val, int* __restrict d_sample_idx_per_z_val, float* __restrict z_vals, int* __restrict d_sample_depth_idx)
{
  int warp_id = threadIdx.x / WARP_SIZE;
  int ray_id = blockIdx.x * NUM_WARPS + warp_id;
  int laneid = lane_id();

  if (ray_id >= batch_size)
    return;

  __shared__ float densities[MAX_OUTPUT * NUM_WARPS];
  __shared__ int ids[MAX_OUTPUT * NUM_WARPS];

  if (laneid < MAX_OUTPUT)
  {
    densities[warp_id * MAX_OUTPUT + laneid] = -1.f;
    ids[warp_id * MAX_OUTPUT + laneid] = N_DEPTHS - 1;
  }

  int has_samples = 0;
  float current_threshold = adaptive_sampling_threshold;

  float max_density_val = -1000.0f;
  int max_myid = 0;

  //if(ray_id == 28712)
#pragma unroll
  for (int sample_offset = 0; sample_offset < N_DEPTHS; sample_offset += WARP_SIZE)
  {
    float density_val = -1.0f;
    int myid = sample_offset + laneid;
    if (sample_offset + WARP_SIZE > N_DEPTHS)
    {
      if (sample_offset + laneid < N_DEPTHS)
        density_val = depth_vals[ray_id * N_DEPTHS + sample_offset + laneid] - 0.00001f * (sample_offset + laneid); // make sure that we take the closer one when we have multiple with the same value
    }
    else
    {
      density_val = depth_vals[ray_id * N_DEPTHS + sample_offset + laneid] - 0.00001f * (sample_offset + laneid); // make sure that we take the closer one when we have multiple with the same value
    }
    bool exceeds_thresh = density_val > current_threshold;

    unsigned exmask = __ballot_sync(0xFFFFFFFF, exceeds_thresh);

    if (exmask != 0)
    { 
      int tsamples = __popc(exmask);
      if (tsamples > MAX_OUTPUT)
      {
        unsigned votemask = exmask;

        // warp find
        while(votemask != 0)
        { 
          int cur = __ffs(votemask) - 1;
          float tval = __shfl_sync(0xFFFFFFFF, density_val, cur, WARP_SIZE);
          unsigned tmask = __ballot_sync(0xFFFFFFFF, density_val > tval);

          //printf("%d %d shfl %d: %x %x  %f > %f ? %x\n", ray_id, laneid, cur, exmask, votemask, density_val, tval, tmask);


          int tcount = __popc(tmask);
          if (tcount > MAX_OUTPUT)
          {
            // still too many, but maybe removed some already
            exmask = tmask;
            votemask = votemask & tmask;
          }
          else if (tcount == MAX_OUTPUT)
          {
            // all above the threshold are right
            current_threshold = tval;
            exmask = tmask;
            break;
          }
          else if (tcount == MAX_OUTPUT - 1)
          {
            // actually found the right number including the voter
            current_threshold = tval;
            exmask = tmask | (1u << cur);
            break;
          }
          else
          {
            unsigned int equalmask = __ballot_sync(0xFFFFFFFF, density_val == tval);
            if (tcount + __popc(equalmask) >= MAX_OUTPUT)
            {
              // there are multiple above the threashold that get us where we need to be
            /*  if(laneid == 0)
                  printf("%d fixing %x %x %x\n", ray_id, exmask, votemask, equalmask);*/
              // determine how many equals we need
              bool needed = (density_val > tval) ||
                  (density_val == tval && (tcount + (equalmask & lanemask_lt()) < MAX_OUTPUT));
              current_threshold = tval;
              exmask = __ballot_sync(0xFFFFFFFF, needed);
              break;
            }

            // delete from voters and carry on
            votemask = votemask ^ (1u << cur);
          }
        }
        //if (__popc(exmask) > MAX_OUTPUT && laneid == 0)
        //     printf("%d broken %x %x\n", ray_id, exmask, votemask);
        tsamples = MAX_OUTPUT;
      }

      unsigned int prefix = __popc((exmask) & lanemask_lt());
      if ((lanemask_eq() & exmask) && prefix < MAX_OUTPUT)
      {
        // combine with last
        if (sample_offset != 0)
        {
          if (has_samples != 0)
          {
            unsigned activemask = __activemask();
            #pragma unroll
            for (int i = 0; i < MAX_OUTPUT; ++i)
            {
              __threadfence_block();
              float temp_density = densities[warp_id * MAX_OUTPUT + prefix];
              if (temp_density < density_val)
              {
                densities[warp_id * MAX_OUTPUT + prefix] = density_val;
                int temp_id = ids[warp_id * MAX_OUTPUT + prefix];
                ids[warp_id * MAX_OUTPUT + prefix] = myid;
                myid = temp_id;
                density_val = temp_density;
              }
              __threadfence_block();
              prefix = (prefix + 1) % MAX_OUTPUT;
              __syncwarp(activemask);
            }
          }
          else
          {
            densities[warp_id * MAX_OUTPUT + prefix] = density_val;
            ids[warp_id * MAX_OUTPUT + prefix] = myid;
          }
        }
        else
        {
          densities[warp_id * MAX_OUTPUT + prefix] = density_val;
          ids[warp_id * MAX_OUTPUT + prefix] = myid;
        }
        //densities[warp_id * MAX_OUTPUT + prefix] = density_val;
        //ids[warp_id * MAX_OUTPUT + prefix] = myid;
      }
      has_samples = min(has_samples + tsamples, MAX_OUTPUT);
    }
    else
    {
      if (density_val > max_density_val)
      {
        max_density_val = density_val;
        max_myid = myid;
      }
    }
  }

  if (has_samples == 0)
  {
    float wmax = myWarpMax(max_density_val);
    if(max_density_val == wmax)
        ids[warp_id * MAX_OUTPUT] = max_myid;

    has_samples = 1;
  }
  
  __syncwarp(0xFFFFFFFF);

  int ray_offset;
  if (laneid == 0)
  {
    // sort
    for(int i = 0; i < MAX_OUTPUT-1; ++i)
      for (int j = i; j < MAX_OUTPUT-1; ++j)
      {
        int sid = warp_id * MAX_OUTPUT + j;
        if (ids[sid] > ids[sid + 1])
        {
          //swap(densities[sid], densities[sid + 1]);
          swap(ids[sid], ids[sid + 1]);
        }
      }
    // Increase the global offset counter by the number of samples that will be used
    ray_offset = atomicAdd(&global_offset_counter, has_samples);

    // Store offset and number of samples in global storage for use in later kernels
    ray_offsets[ray_id] = ray_offset;
    ray_n_samples[ray_id] = has_samples;
  }
  ray_offset = __shfl_sync(0xFFFFFFFF, ray_offset, 0, WARP_SIZE);

  if (laneid < has_samples)
  {
    int sample_id = ids[warp_id * MAX_OUTPUT + laneid];
    float sample = getBinMidSimple(N_DEPTHS, sample_id);
    z_vals[ray_offset + laneid] = sample;
    d_ray_idx_per_z_val[ray_offset + laneid] = ray_id;
    d_sample_idx_per_z_val[ray_offset + laneid] = laneid;
    d_sample_depth_idx[ray_offset + laneid] = sample_id;
  }
}

template<int N_DEPTHS, int NUM_WARPS>
__global__ void sampleAdaptiveWarpReductionMax1(const float* __restrict depth_vals, const int batch_size,
  const int n_max_ray_samples, const float adaptive_sampling_threshold,
  int* __restrict ray_offsets, int* __restrict ray_n_samples,
  int* __restrict d_ray_idx_per_z_val, int* __restrict d_sample_idx_per_z_val, float* __restrict z_vals, int* __restrict d_sample_depth_idx)
{
  int warp_id = threadIdx.x / WARP_SIZE;
  int ray_id = blockIdx.x * NUM_WARPS + warp_id;
  int laneid = lane_id();

  if (ray_id >= batch_size)
    return;

  float my_density;
  int my_id;

#pragma unroll
  for (int sample_round = 0; sample_round * WARP_SIZE < N_DEPTHS; ++sample_round)
  {
    float density_val = -1000.0f;
    int myid = sample_round * WARP_SIZE + laneid;
    if (sample_round * WARP_SIZE > N_DEPTHS)
    {
      if (sample_round * WARP_SIZE + laneid < N_DEPTHS)
        density_val = depth_vals[ray_id * N_DEPTHS + sample_round * WARP_SIZE + laneid];
    }
    else
    {
      density_val = depth_vals[ray_id * N_DEPTHS + sample_round * WARP_SIZE + laneid];
    }

    if (sample_round == 0)
    {
      my_density = density_val;
      my_id = myid;
    }
    else
    {
      if (my_density < density_val)
      {
        my_density = density_val;
        my_id = myid;
      }
    }
  }

  // find current max in warp
  float tmax = myWarpMax(my_density);

  bool is_max = my_density == tmax;
  unsigned int equalmask = __ballot_sync(0xFFFFFFFF, is_max);
  int fin = __ffs(equalmask) - 1;
  int sample_id = __shfl_sync(0xFFFFFFFF, my_id, fin);

  if (laneid == 0)
  {
    // Increase the global offset counter by the number of samples that will be used
    int ray_offset = atomicAdd(&global_offset_counter, 1);

    // Store offset and number of samples in global storage for use in later kernels
    ray_offsets[ray_id] = ray_offset;
    ray_n_samples[ray_id] = 1;

    float sample = getBinMidSimple(N_DEPTHS, sample_id);
    z_vals[ray_offset] = sample;
    d_ray_idx_per_z_val[ray_offset] = ray_id;
    d_sample_idx_per_z_val[ray_offset] = 0;
    d_sample_depth_idx[ray_offset] = sample_id;
  }
}


template<int N_DEPTHS, int MAX_OUTPUT, int NUM_WARPS>
__global__ void sampleAdaptiveWarpReductionMax(const float* __restrict depth_vals, const int batch_size,
    const int n_max_ray_samples, const float adaptive_sampling_threshold,
    int* __restrict ray_offsets, int* __restrict ray_n_samples,
    int* __restrict d_ray_idx_per_z_val, int* __restrict d_sample_idx_per_z_val, float* __restrict z_vals, int* __restrict d_sample_depth_idx)
{
  constexpr int MAX_PER_THREAD = MAX_OUTPUT < N_DEPTHS / (WARP_SIZE) ? MAX_OUTPUT : N_DEPTHS / (WARP_SIZE);
  int warp_id = threadIdx.x / WARP_SIZE;
  int ray_id = blockIdx.x * NUM_WARPS + warp_id;
  int laneid = lane_id();

  if (ray_id >= batch_size)
    return;

  __shared__ int ids[MAX_OUTPUT * NUM_WARPS];
  
  if (laneid < MAX_OUTPUT)
    ids[warp_id * MAX_OUTPUT + laneid] = N_DEPTHS - 1;

  float my_densitites[MAX_PER_THREAD];
  int my_ids[MAX_PER_THREAD];

#pragma unroll
  for (int sample_round = 0; sample_round * WARP_SIZE < N_DEPTHS; ++sample_round)
  {
    float density_val = -1000.0f;
    int myid = sample_round * WARP_SIZE + laneid;
    if (sample_round * WARP_SIZE > N_DEPTHS)
    {
      if (sample_round * WARP_SIZE + laneid < N_DEPTHS)
        density_val = depth_vals[ray_id * N_DEPTHS + sample_round * WARP_SIZE + laneid] - 0.00001f * (sample_round * WARP_SIZE + laneid); // make sure that we take the closer one when we have multiple with the same value
    }
    else
    {
      density_val = depth_vals[ray_id * N_DEPTHS + sample_round * WARP_SIZE + laneid] - 0.00001f * (sample_round * WARP_SIZE + laneid); // make sure that we take the closer one when we have multiple with the same value
    }

#pragma unroll
    for (int i = 0; i < sample_round; ++i)
    {
      if (my_densitites[i] < density_val)
      {
        swap(my_densitites[i], density_val);
        swap(my_ids[i], myid);
      }
    }
    if (sample_round < MAX_PER_THREAD)
    {
      my_densitites[sample_round] = density_val;
      my_ids[sample_round] = myid;
    }
  }

  // every thread now holds MAX_PER_THREAD densitites in a sorted manner
  int num_samples = 0;
#pragma unroll
  for (int it = 0; it < MAX_OUTPUT; ++it)
  {
    // find current max in warp
    float tmax = myWarpMax(my_densitites[0]);

    if (it > 0)
    {
      // if we are a later iteration and we moved below the threshold, we are done
      if (tmax <= adaptive_sampling_threshold)
        break;
    }

    // all that equal the max add themselves
    bool is_max = my_densitites[0] == tmax;
    unsigned int equalmask = __ballot_sync(0xFFFFFFFF, is_max);
    int pos = num_samples + __popc(equalmask & lanemask_lt());
    if (is_max && pos < MAX_OUTPUT)
    {
      ids[warp_id * MAX_OUTPUT + pos] = my_ids[0];
    }

    if (it == 0)
    {
      // if we are at the first iteration and below the threshold, we end with the highest
      if (tmax <= adaptive_sampling_threshold)
      {
        num_samples = 1;
        break;
      }
    }

    // the number of samples we have added
    num_samples += __popc(equalmask);
    if (it < MAX_PER_THREAD - 1)
    {
      // if there is still another round, we need to overwrite the max
      if (num_samples < MAX_OUTPUT)
      {
        if (is_max)
        {
          // move as many as we may still need to the front
          #pragma unroll
          for (int j = 1; j < min(MAX_OUTPUT - it, MAX_PER_THREAD); ++j)
          {
              my_densitites[j - 1] = my_densitites[j];
              my_ids[j - 1] = my_ids[j];
          }
          if (MAX_OUTPUT - it > MAX_PER_THREAD)
              my_densitites[MAX_PER_THREAD - 1] = -1000.f;
        }
      }
      else
      {
        break;
      }
    }
  } 

  num_samples = min(MAX_OUTPUT, num_samples);
  
  __syncwarp(0xFFFFFFFF);
  int ray_offset;
  if (laneid == 0)
  {
    // sort
    for (int i = 0; i < MAX_OUTPUT - 1; ++i)
      for (int j = i; j < MAX_OUTPUT - 1; ++j)
      {
        int sid = warp_id * MAX_OUTPUT + j;
        if (ids[sid] > ids[sid + 1])
        {
          //swap(densities[sid], densities[sid + 1]);
          swap(ids[sid], ids[sid + 1]);
        }
      }
    // Increase the global offset counter by the number of samples that will be used
    ray_offset = atomicAdd(&global_offset_counter, num_samples);

    // Store offset and number of samples in global storage for use in later kernels
    ray_offsets[ray_id] = ray_offset;
    ray_n_samples[ray_id] = num_samples;
  }
  ray_offset = __shfl_sync(0xFFFFFFFF, ray_offset, 0, WARP_SIZE);

  if (laneid < num_samples)
  {
    int sample_id = ids[warp_id * MAX_OUTPUT + laneid];
    float sample = getBinMidSimple(N_DEPTHS, sample_id);
    z_vals[ray_offset + laneid] = sample;
    d_ray_idx_per_z_val[ray_offset + laneid] = ray_id;
    d_sample_idx_per_z_val[ray_offset + laneid] = laneid;
    d_sample_depth_idx[ray_offset + laneid] = sample_id;
  }
}

template<int N_DEPTHS>
__global__ void sampleAdaptive(const float* __restrict depth_vals, const int batch_size, 
  const int n_max_ray_samples, const float adaptive_sampling_threshold, 
  int* __restrict ray_offsets, int* __restrict ray_n_samples, 
  int* __restrict d_ray_idx_per_z_val, int* __restrict d_sample_idx_per_z_val, float* __restrict z_vals,
  int* __restrict d_sample_depth_idx) 
{
  int ray_id = blockIdx.x;
  int sample_id = threadIdx.x;
  int id = ray_id * blockDim.x + sample_id;

  if (ray_id >= batch_size || sample_id >= N_DEPTHS)
    return;

  float depth_val = depth_vals[id];
  const int exceeds_thresh = depth_val > adaptive_sampling_threshold;

  // Calculate the number of samples that exceed the threshold
  int n_exceeds;
  {
    typedef cub::BlockReduce<int, N_DEPTHS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    n_exceeds = BlockReduce(temp_storage).Sum(exceeds_thresh);
  }

  __shared__ int n_exceeds_thresh;
  __shared__ int ray_offset;
  __shared__ int n_ray_samples;

  if (sample_id == 0)
  {
    n_exceeds_thresh = n_exceeds;

    // Increase the global offset counter by the number of samples that will be used
    n_ray_samples = max(1, min(n_exceeds, n_max_ray_samples));
    ray_offset = atomicAdd(&global_offset_counter, n_ray_samples);

    // Store offset and number of samples in global storage for use in later kernels
    ray_offsets[ray_id] = ray_offset;
    ray_n_samples[ray_id] = n_ray_samples;
  }
    
  __syncthreads();

  if (n_exceeds_thresh == 0)
  {
    // CASE 1: No samples exceed threshold
    // -> Take sample with maximum value

    __shared__ float s_depth_reduce_max[N_DEPTHS];
    __shared__ int s_index_depth_reduce_max[N_DEPTHS];

    s_depth_reduce_max[sample_id] = depth_val;
    s_index_depth_reduce_max[sample_id] = sample_id;
    __syncthreads();

    // Do custom reduction that gives the maximum value and the corresponding index
    // The final result of the reduction will be held by thread_0
    float d_max = depth_val;
    int d_max_i = sample_id;
    for (int s = N_DEPTHS / 2; s > 0; s /= 2)
    {
      if (sample_id < s)
      {
        float d2 = s_depth_reduce_max[sample_id + s];
        
        if (d_max < d2) 
        {
          d_max = d2;
          d_max_i = s_index_depth_reduce_max[sample_id + s];
          
          s_depth_reduce_max[sample_id] = d2;
          s_index_depth_reduce_max[sample_id] = d_max_i;
        }
      }
      __syncthreads();
    }

    if (sample_id == 0)
    {      
      // thread_0 writes out the final result
      float sample = getBinMidSimple(N_DEPTHS, d_max_i);
      
      z_vals[ray_offset] = sample;
      d_ray_idx_per_z_val[ray_offset] = ray_id;
      d_sample_idx_per_z_val[ray_offset] = 0;
      d_sample_depth_idx[ray_offset] = d_max_i;
    }
  }
  else if (n_exceeds_thresh > n_max_ray_samples)
  {
    // CASE 2: More than maximum number of samples exceed threshold
    // -> Order and take top <n_max_ray_samples> samples

    __shared__ int s_ordering[N_DEPTHS];

    float thread_key[1]; thread_key[0] = depth_val;
    int thread_value[1]; thread_value[0] = sample_id;
    {
      typedef cub::BlockRadixSort<float, N_DEPTHS, 1, int> BlockRadixSort;
      __shared__ typename BlockRadixSort::TempStorage temp_storage;

      // Sort the key-value pair, where key = depth_value, value = thread_id
      BlockRadixSort(temp_storage).SortDescending(thread_key, thread_value);
    }

    // write a 1 to all samples that are in the top <n_max_ray_samples>, else 0
    int val_to_insert = sample_id < n_max_ray_samples ? 1 : 0;
    s_ordering[thread_value[0]] = val_to_insert;
    __syncthreads();

    int is_one_of_top_samples = s_ordering[sample_id];
    int final_sample_offset;
    {
      typedef cub::BlockScan<int, N_DEPTHS> BlockScan;
      __shared__ typename BlockScan::TempStorage temp_storage;

      // Do an inclusive sum, so that each of the top samples knows its final offset in the result
      BlockScan(temp_storage).InclusiveSum(is_one_of_top_samples, final_sample_offset);
      final_sample_offset -= 1;
    }

    if (is_one_of_top_samples)
    {
      float sample = getBinMidSimple(N_DEPTHS, sample_id);
      z_vals[ray_offset + final_sample_offset] = sample;
      d_ray_idx_per_z_val[ray_offset + final_sample_offset] = ray_id;
      d_sample_idx_per_z_val[ray_offset + final_sample_offset] = final_sample_offset;
      d_sample_depth_idx[ray_offset + final_sample_offset] = sample_id;
    }
  }
  else // if (n_exceeds_thresh > 0 && n_exceeds_thresh <= n_max_ray_samples) -> no sorting needed; just take
  {
    // CASE 3: Number of exceeding samples is in valid range
    // -> Just take all samples that exceed the threshold

    int final_sample_offset;
    {
      typedef cub::BlockScan<int, N_DEPTHS> BlockScan;
      __shared__ typename BlockScan::TempStorage temp_storage;

      BlockScan(temp_storage).InclusiveSum(exceeds_thresh, final_sample_offset);
      final_sample_offset -= 1;
    }

    if (exceeds_thresh)
    {
      float sample = getBinMidSimple(N_DEPTHS, sample_id);
      z_vals[ray_offset + final_sample_offset] = sample;
      d_ray_idx_per_z_val[ray_offset + final_sample_offset] = ray_id;
      d_sample_idx_per_z_val[ray_offset + final_sample_offset] = final_sample_offset;
      d_sample_depth_idx[ray_offset + final_sample_offset] = sample_id;
    }
  }
}

template<int N_FIXED_FEATURE_SIZE, int N_SAMPLES_PER_BLOCK>
__global__ void rayMarchFromPosesAdaptiveNDC(const float* __restrict z_vals, const int* __restrict ray_offsets, 
  const int* __restrict d_ray_idx_per_z_val, const int* __restrict d_sample_idx_per_z_val, 
  const int batch_size, const int n_max_ray_samples, const int n_actual_entries,
  const int width, const int height, const float3 center, const float focal, 
  const int num_dir_freqs, const int num_pos_freqs, const float* __restrict freq_bands,
  const float* __restrict features_last, float* __restrict features)
{
  int feature_idx = threadIdx.x & (N_FIXED_FEATURE_SIZE - 1);
  int idx = blockIdx.x * N_SAMPLES_PER_BLOCK + (threadIdx.x / N_FIXED_FEATURE_SIZE);

  if (feature_idx >= num_pos_freqs + 1 || idx >= n_actual_entries)
    return;

  int ray_idx = d_ray_idx_per_z_val[idx];
  int sample_idx = d_sample_idx_per_z_val[idx];

  const float3* features_last_r = reinterpret_cast<const float3*>(features_last);
  float3 pos = features_last_r[ray_idx * 2 + 1];
  float3 dir = features_last_r[ray_idx * 2 + 0];
  dir = normalize(dir);

  float3* features_r = reinterpret_cast<float3*>(features);

  int z_val_offset = ray_offsets[ray_idx];

  int n_features_r = (1 + num_dir_freqs * 2 + 1 + num_pos_freqs * 2);
  int feature_idx_r = z_val_offset * n_features_r;

  int pos_start_r = feature_idx_r + sample_idx * n_features_r;
  int dir_start_r = feature_idx_r + sample_idx * n_features_r + 1 + num_pos_freqs * 2;
  
  const float near = 1.0f;
  float t = -(near + pos.z) / dir.z;
  float3 pos_ndc = pos + t * dir;
  
  float o0 = -1.0f / ( width / (2.0f * focal)) * pos_ndc.x / pos_ndc.z;
  float o1 = -1.0f / (height / (2.0f * focal)) * pos_ndc.y / pos_ndc.z;
  float o2 =  1.0f + 2.0f * near / pos_ndc.z;
  
  float d0 = -1.0f / ( width / (2.0f * focal)) * (dir.x / dir.z - pos_ndc.x / pos_ndc.z);
  float d1 = -1.0f / (height / (2.0f * focal)) * (dir.y / dir.z - pos_ndc.y / pos_ndc.z);
  float d2 = -2.0f * near / pos_ndc.z;
  
  pos_ndc = make_float3(o0, o1, o2);
  float3 dir_ndc = make_float3(d0, d1, d2);

  float z_val = z_vals[z_val_offset + sample_idx];
  
  float3 pos_world = pos_ndc + dir_ndc * z_val;
  
  if (feature_idx == 0)
  {
    features_r[pos_start_r] = pos_world;
    features_r[dir_start_r] = dir_ndc;
    return;
  }
  feature_idx -= 1;

  float fbv = freq_bands[feature_idx];

  float3 v_f = pos_world * fbv;
  features_r[pos_start_r + 1 + feature_idx * 2 + 0] = make_float3(sin(v_f.x), sin(v_f.y), sin(v_f.z));
  features_r[pos_start_r + 1 + feature_idx * 2 + 1] = make_float3(cos(v_f.x), cos(v_f.y), cos(v_f.z));

  if (feature_idx < num_dir_freqs)
  {
    v_f = normalize(dir_ndc) * fbv;
    features_r[dir_start_r + 1 + feature_idx * 2 + 0] = make_float3(sin(v_f.x), sin(v_f.y), sin(v_f.z));
    features_r[dir_start_r + 1 + feature_idx * 2 + 1] = make_float3(cos(v_f.x), cos(v_f.y), cos(v_f.z));
  }
}

template<int N_FIXED_FEATURE_SIZE, int N_SAMPLES_PER_BLOCK>
__global__ void rayMarchFromPosesAdaptive(const float* __restrict z_vals, const int* __restrict ray_offsets, 
  const int* __restrict d_ray_idx_per_z_val, const int* __restrict d_sample_idx_per_z_val, 
  const int batch_size, const int n_max_ray_samples, const int n_actual_entries,
  const float min_d_range, const float max_d_range, const float3 center, const float max_depth, 
  const int num_dir_freqs, const int num_pos_freqs, const float* __restrict freq_bands,
  const float* __restrict features_last, float* __restrict features)
{
  int feature_idx = threadIdx.x & (N_FIXED_FEATURE_SIZE - 1);
  int idx = blockIdx.x * N_SAMPLES_PER_BLOCK + (threadIdx.x / N_FIXED_FEATURE_SIZE);

  if (feature_idx >= num_pos_freqs + 1 || idx >= n_actual_entries)
    return;

  int ray_idx = d_ray_idx_per_z_val[idx];
  int sample_idx = d_sample_idx_per_z_val[idx];

  const float3* features_last_r = reinterpret_cast<const float3*>(features_last);
  float3 pos = features_last_r[ray_idx * 2 + 1];
  float3 dir = features_last_r[ray_idx * 2 + 0];
  dir = normalize(dir);

  float3* features_r = reinterpret_cast<float3*>(features);

  int z_val_offset = ray_offsets[ray_idx];

  int n_features_r = (1 + num_dir_freqs * 2 + 1 + num_pos_freqs * 2);
  int feature_idx_r = z_val_offset * n_features_r;

  int pos_start_r = feature_idx_r + sample_idx * n_features_r;
  int dir_start_r = feature_idx_r + sample_idx * n_features_r + 1 + num_pos_freqs * 2;

  float z_val = logtToWorld(z_vals[z_val_offset + sample_idx], min_d_range, max_d_range);

  float3 pos_world = pos + dir * z_val;
  float3 norm_pos = normalizationInverseSqrtDistCentered(pos_world, center, max_depth);
  
  if (feature_idx == 0)
  {
    features_r[pos_start_r] = norm_pos;
    features_r[dir_start_r] = dir;
    return;
  }
  feature_idx -= 1;

  float fbv = freq_bands[feature_idx];

  float3 v_f = norm_pos * fbv;
  features_r[pos_start_r + 1 + feature_idx * 2 + 0] = make_float3(sin(v_f.x), sin(v_f.y), sin(v_f.z));
  features_r[pos_start_r + 1 + feature_idx * 2 + 1] = make_float3(cos(v_f.x), cos(v_f.y), cos(v_f.z));

  if (feature_idx < num_dir_freqs)
  {
    v_f = dir * fbv;
    features_r[dir_start_r + 1 + feature_idx * 2 + 0] = make_float3(sin(v_f.x), sin(v_f.y), sin(v_f.z));
    features_r[dir_start_r + 1 + feature_idx * 2 + 1] = make_float3(cos(v_f.x), cos(v_f.y), cos(v_f.z));
  }
}


__global__ void nerf_raw_2_output_adaptive(float* __restrict shading_output, cudaSurfaceObject_t out_fb_surf,
  const int batch_size, const int batch_offset, const int width, const int n_max_ray_samples,
  const int* __restrict ray_offsets, const int* __restrict ray_n_samples)
{
  unsigned int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray_idx >= batch_size)
    return;

  int x_ = (ray_idx + batch_offset) % width;
  int y_ = (ray_idx + batch_offset) / width;

  int z_val_offset = ray_offsets[ray_idx];
  int n_ray_samples = ray_n_samples[ray_idx];


  int out_offset = z_val_offset * NUM_SHADING_OUTPUTS;
  float last_prod = 1.0f;

  float nr = 0.0f;
  float ng = 0.0f;
  float nb = 0.0f;

  #pragma unroll
  for (int sample_idx = 0; sample_idx < n_ray_samples; sample_idx++)
  {
    float r = sigmoid(shading_output[out_offset + 0]);
    float g = sigmoid(shading_output[out_offset + 1]);
    float b = sigmoid(shading_output[out_offset + 2]);

    float alpha = sigmoid(shading_output[out_offset + 3]);

    float prod =  last_prod * (1.0f - alpha + 1e-10f);
    float weight = alpha * last_prod;

    nr = nr + weight * r;
    ng = ng + weight * g;
    nb = nb + weight * b;

    last_prod = prod;

    out_offset += NUM_SHADING_OUTPUTS;
  }

  uchar4 data;
  data.x = clamp(nr, 0.0f, 1.0f) * 255.0f;
  data.y = clamp(ng, 0.0f, 1.0f) * 255.0f;
  data.z = clamp(nb, 0.0f, 1.0f) * 255.0f;
  data.w = 255.0f;
  surf2Dwrite(data, out_fb_surf, x_ * NUM_SHADING_OUTPUTS, y_);
}

__global__ void nerf_raw_2_output_adaptive_mult_depth(float* __restrict shading_output, cudaSurfaceObject_t out_fb_surf,
  const int batch_size, const int batch_offset, const int width, const int n_max_ray_samples,
  const int* __restrict ray_offsets, const int* __restrict ray_n_samples, float* __restrict depth_values, int* __restrict d_sample_depth_idx,
  const int mult_location)
{
  unsigned int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray_idx >= batch_size)
    return;

  int x_ = (ray_idx + batch_offset) % width;
  int y_ = (ray_idx + batch_offset) / width;

  int z_val_offset = ray_offsets[ray_idx];
  int n_ray_samples = ray_n_samples[ray_idx];


  int out_offset = z_val_offset * NUM_SHADING_OUTPUTS;
  float last_prod = 1.0f;

  float nr = 0.0f;
  float ng = 0.0f;
  float nb = 0.0f;

#pragma unroll
  for (int sample_idx = 0; sample_idx < n_ray_samples; sample_idx++)
  {
    float r = sigmoid(shading_output[out_offset + 0]);
    float g = sigmoid(shading_output[out_offset + 1]);
    float b = sigmoid(shading_output[out_offset + 2]);

    float alpha = sigmoid(shading_output[out_offset + 3]);

    // Multiply weight on top
    float depth_val = (depth_values[ray_idx * 128 + d_sample_depth_idx[z_val_offset + sample_idx]]);
    
    if (mult_location == 1)
      alpha = depth_val * alpha;

    float prod = last_prod * (1.0f - alpha + 1e-10f);
    float weight = alpha * last_prod;

    if (mult_location == 2)
      weight = depth_val * weight;

    nr = nr + weight * r;
    ng = ng + weight * g;
    nb = nb + weight * b;

    last_prod = prod;

    out_offset += NUM_SHADING_OUTPUTS;
  }

  uchar4 data;
  data.x = clamp(nr, 0.0f, 1.0f) * 255.0f;
  data.y = clamp(ng, 0.0f, 1.0f) * 255.0f;
  data.z = clamp(nb, 0.0f, 1.0f) * 255.0f;
  data.w = 255.0f;
  surf2Dwrite(data, out_fb_surf, x_ * NUM_SHADING_OUTPUTS, y_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST CODE 

#define SampleAdaptiveWarpReductionMaxPaster(NUM) \
 else if (n_max_ray_samples == NUM) \
  { \
  sampleAdaptiveWarpReductionMax< N_DEPTHS, NUM, NUM_WARPS> << <divup(batch_size, NUM_WARPS), NUM_WARPS * 32 >> > (depth_vals, batch_size, n_max_ray_samples, \
      adaptive_sampling_threshold, ray_offsets, ray_n_samples, d_ray_idx_per_z_val, d_sample_idx_per_z_val, z_vals, d_sample_depth_idx); \
  }


int updateRayMarchFromPosesAdaptive(const int batch_size, const int n_max_ray_samples, 
  const float adaptive_sampling_threshold, const int num_dir_enc, const int num_pos_enc,  
  const float min_d_range, const float max_d_range, const float max_depth,
  const float vc_center_x, const float vc_center_y, const float vc_center_z, 
  float* depth_vals, const float* features_last, const float* freq_bands, 
  int* ray_offsets, int* ray_n_samples, 
  int* d_ray_idx_per_z_val, int* d_sample_idx_per_z_val, float* z_vals, float* features,
  int* d_sample_depth_idx, const int width, const int height, const float focal)
{
  const unsigned int N_DEPTHS = 128;
  const int NUM_WARPS = 8;

  int zero = 0;
  cudaMemcpyToSymbol(global_offset_counter, &zero, sizeof(int));
  // cudaDeviceSynchronize();

  if (n_max_ray_samples == 1)
  {
    sampleAdaptiveWarpReductionMax1< N_DEPTHS, NUM_WARPS> << <divup(batch_size, NUM_WARPS), NUM_WARPS * 32 >> > (depth_vals, batch_size, n_max_ray_samples,
      adaptive_sampling_threshold, ray_offsets, ray_n_samples, d_ray_idx_per_z_val, d_sample_idx_per_z_val, z_vals, d_sample_depth_idx);

  }
  SampleAdaptiveWarpReductionMaxPaster(2)
  SampleAdaptiveWarpReductionMaxPaster(3)
  SampleAdaptiveWarpReductionMaxPaster(4)
  SampleAdaptiveWarpReductionMaxPaster(5)
  SampleAdaptiveWarpReductionMaxPaster(6)
  SampleAdaptiveWarpReductionMaxPaster(7)
  SampleAdaptiveWarpReductionMaxPaster(8)
  //else if (n_max_ray_samples == 4)
  //{
  //    //sampleAdaptiveWarpMax< N_DEPTHS, 4, NUM_WARPS> << <divup(batch_size, NUM_WARPS), NUM_WARPS * 32 >> > (depth_vals, batch_size, n_max_ray_samples,
  //    //    adaptive_sampling_threshold, ray_offsets, ray_n_samples, d_ray_idx_per_z_val, d_sample_idx_per_z_val, z_vals, d_sample_depth_idx);

  //    sampleAdaptiveWarpReductionMax< N_DEPTHS, 4, NUM_WARPS> << <divup(batch_size, NUM_WARPS), NUM_WARPS * 32 >> > (depth_vals, batch_size, n_max_ray_samples,
  //        adaptive_sampling_threshold, ray_offsets, ray_n_samples, d_ray_idx_per_z_val, d_sample_idx_per_z_val, z_vals, d_sample_depth_idx);
  //}
  else
  {
    dim3 dimBlock(N_DEPTHS);
    dim3 dimGrid(batch_size);
    sampleAdaptive<N_DEPTHS> << <dimGrid, dimBlock >> > (depth_vals, batch_size, n_max_ray_samples,
      adaptive_sampling_threshold, ray_offsets, ray_n_samples, d_ray_idx_per_z_val, d_sample_idx_per_z_val, z_vals, d_sample_depth_idx);
  }

  //cudaDeviceSynchronize();

  int num_act_inputs;
  cudaMemcpyFromSymbol(&num_act_inputs, global_offset_counter, sizeof(int));

  CUDA_CHECK;

  int n_l_features = num_pos_enc + 1;
  const int N_BLOCK_SIZE = 128;
  const int N_FIXED_FEATURE_SIZE = 16;
  const int N_SAMPLES_PER_BLOCK = N_BLOCK_SIZE / N_FIXED_FEATURE_SIZE;
  dim3 dimBlock2(N_BLOCK_SIZE);
  dim3 dimGrid2(num_act_inputs / N_SAMPLES_PER_BLOCK + 1);

  if (n_l_features > N_FIXED_FEATURE_SIZE)
    assert("currently for performance reasons, not more than fixed number allowed for this kernel");

  float3 center = make_float3(vc_center_x, vc_center_y, vc_center_z);
  if (focal < 0.0f)
    rayMarchFromPosesAdaptive<N_FIXED_FEATURE_SIZE, N_SAMPLES_PER_BLOCK>
      <<<dimGrid2, dimBlock2>>>(z_vals, ray_offsets, d_ray_idx_per_z_val, d_sample_idx_per_z_val, 
      batch_size, n_max_ray_samples, num_act_inputs, min_d_range, max_d_range, center, max_depth, 
      num_dir_enc, num_pos_enc, freq_bands, features_last, features);
  else
    rayMarchFromPosesAdaptiveNDC<N_FIXED_FEATURE_SIZE, N_SAMPLES_PER_BLOCK>
      <<<dimGrid2, dimBlock2>>>(z_vals, ray_offsets, d_ray_idx_per_z_val, d_sample_idx_per_z_val, 
      batch_size, n_max_ray_samples, num_act_inputs, width, height, center, focal, 
      num_dir_enc, num_pos_enc, freq_bands, features_last, features);
  //cudaDeviceSynchronize();
  CUDA_CHECK;

  return num_act_inputs;
}

void copyResultRaymarchAdaptive(void* shading_output, cudaSurfaceObject_t fb_surf, 
  const int batch_size, const int batch_offset, const int width, const int n_max_ray_samples,
  const int* ray_offsets, const int* ray_n_samples)
{
  dim3 dimBlock(16);
  dim3 dimGrid((batch_size + dimBlock.x - 1) / dimBlock.x);
  nerf_raw_2_output_adaptive<<<dimGrid, dimBlock>>>((float*) shading_output, fb_surf, 
    batch_size, batch_offset, width, n_max_ray_samples,
    ray_offsets, ray_n_samples);
  
  CUDA_CHECK;
}

void copyResultRaymarchAdaptiveMultDepth(void* shading_output, cudaSurfaceObject_t fb_surf,
  const int batch_size, const int batch_offset, const int width, const int n_max_ray_samples,
  const int* ray_offsets, const int* ray_n_samples, void* depth_values, int* d_sample_depth_idx,
  const int mult_location)
{
    dim3 dimBlock(16);
    dim3 dimGrid((batch_size + dimBlock.x - 1) / dimBlock.x);
    nerf_raw_2_output_adaptive_mult_depth<<<dimGrid, dimBlock>>>((float*)shading_output, fb_surf,
        batch_size, batch_offset, width, n_max_ray_samples,
        ray_offsets, ray_n_samples, (float*)depth_values, d_sample_depth_idx, mult_location);

    CUDA_CHECK;
}
