
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "device_launch_parameters.h"

#include "../include/cuda/adanerf_cuda_kernels.cuh"
#include "../include/cuda/helper_math.h"
#include "../include/cuda/adanerf_cuda_helper.h"

#include <cuda.h>
#include <iostream>
#include <assert.h> 
#include <cub/cub.cuh>

constexpr int WARP_SIZE = 32;

__global__ void setSpherePosDirBatchedUnrolledEnc(float* __restrict features, 
  const float* __restrict features_clean, float* __restrict features_mod, const float* __restrict rot_mat, 
  float px, float py, float pz, float cx, float cy, float cz, 
  float radius, float min_d, float max_d, int batch_offset, int batch_size,
  int num_dir_freqs, int num_pos_freqs, const float* __restrict dir_freq_bands, 
  const float* __restrict pos_freq_bands, const int width, const int height)
{
  unsigned int feature_idx = threadIdx.x / 2;
  unsigned int ray_idx = blockIdx.x;
  
  if (ray_idx >= batch_size || feature_idx >= num_pos_freqs + num_dir_freqs + 1)
    return;
  
  const float3 fc = reinterpret_cast<const float3*>(features_clean)[batch_offset + ray_idx];
  const float3 rt1 = reinterpret_cast<const float3*>(rot_mat)[0];
  const float3 rt2 = reinterpret_cast<const float3*>(rot_mat)[1];
  const float3 rt3 = reinterpret_cast<const float3*>(rot_mat)[2];

  float3 d;
  d.x = fc.x * rt1.x + fc.y * rt1.y + fc.z * rt1.z;
  d.y = fc.x * rt2.x + fc.y * rt2.y + fc.z * rt2.z;
  d.z = fc.x * rt3.x + fc.y * rt3.y + fc.z * rt3.z;

  d = normalize(d);

  float3 np = raySphereIntersect(make_float3(px, py, pz), d, make_float3(cx, cy, cz), radius);
  
  int n_features_r = 1 + num_dir_freqs * 2 + 1 + num_pos_freqs * 2;
  int feature_idx_r = ray_idx * n_features_r;
  
  // seems that we do dir before pos in SpherePosDir, and the other way around in RayMarchFromPoses
  int p_start_r = feature_idx_r + 1 + num_dir_freqs * 2;
  int d_start_r = feature_idx_r;
  
  float3* r_feat = reinterpret_cast<float3*>(features);
  
  if (feature_idx == 0)
  {
    float3* m_feat = reinterpret_cast<float3*>(features_mod);
    
    if (threadIdx.x % 2 == 0)
    {
      r_feat[p_start_r] = np;
      m_feat[ray_idx * 2 + 1] = np;
    }
    else
    {
      r_feat[d_start_r] = d;
      m_feat[ray_idx * 2 + 0] = d;
    }
    
    return;
  }
  feature_idx -= 1;
  
  if (feature_idx < num_pos_freqs)
  {
    float fbv = pos_freq_bands[feature_idx];
    float3 v_f = np * fbv;

    if (threadIdx.x % 2 == 0)
      r_feat[p_start_r + 1 + feature_idx * 2 + 0] = make_float3(sin(v_f.x), sin(v_f.y), sin(v_f.z));
    else
      r_feat[p_start_r + 1 + feature_idx * 2 + 1] = make_float3(cos(v_f.x), cos(v_f.y), cos(v_f.z));
  }
  else
  {
    feature_idx -= num_pos_freqs;
    
    // freq_bands are the same for pos and dir
    float fbv = pos_freq_bands[feature_idx];
    float3 v_f = d * fbv;

    if (threadIdx.x % 2 == 0)
      r_feat[d_start_r + 1 + feature_idx * 2 + 0] = make_float3(sin(v_f.x), sin(v_f.y), sin(v_f.z));
    else
      r_feat[d_start_r + 1 + feature_idx * 2 + 1] = make_float3(cos(v_f.x), cos(v_f.y), cos(v_f.z));
  }
}

template<int N_DEPTHS>
__global__ void setSpherePosDirBatchedUnrolledNoEnc(float* __restrict features, 
  const float* __restrict features_clean, float* __restrict features_mod, const float* __restrict rot_mat, 
  float px, float py, float pz, float cx, float cy, float cz, 
  float radius, float min_d, float max_d, int batch_offset, int batch_size)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int ray_sample_idx = idx & (N_DEPTHS - 1);
  unsigned int ray_idx = idx / N_DEPTHS;

  if ((ray_idx >= batch_size) || (ray_sample_idx >= N_DEPTHS))
    return;

  const float3 fc = reinterpret_cast<const float3*>(features_clean)[batch_offset + ray_idx];
  const float3 rt1 = reinterpret_cast<const float3*>(rot_mat)[0];
  const float3 rt2 = reinterpret_cast<const float3*>(rot_mat)[1];
  const float3 rt3 = reinterpret_cast<const float3*>(rot_mat)[2];

  float3 d;
  d.x = fc.x * rt1.x + fc.y * rt1.y + fc.z * rt1.z;
  d.y = fc.x * rt2.x + fc.y * rt2.y + fc.z * rt2.z;
  d.z = fc.x * rt3.x + fc.y * rt3.y + fc.z * rt3.z;

  d = normalize(d);

  float3 np = raySphereIntersect(make_float3(px,py,pz), d, make_float3(cx,cy,cz), radius);

  float3* r_feat = reinterpret_cast<float3*>(features);
  float3* m_feat = reinterpret_cast<float3*>(features_mod);
  int idx_r = (ray_idx) * (2 + N_DEPTHS);

  float step = 1.0f / N_DEPTHS;
  float cur_step = step / 2.0f + step * ray_sample_idx;

  r_feat[idx_r + 2 + ray_sample_idx] = np + d * logtToWorld(cur_step, min_d, max_d);

  if (ray_sample_idx > 3)
    return;
  
  float3 add_write = d;
  if (ray_sample_idx % 2 == 1)
    add_write = np;
  if (ray_sample_idx < 2)
    r_feat[idx_r + ray_sample_idx] = add_write;
  else if (ray_sample_idx < 4)
    m_feat[ray_idx * 2 + ray_sample_idx % 2] = add_write;
}

template<int N_DEPTHS, int RAYS_PER_BLOCK>
__global__ void setSpherePosDirBatchedUnrolledNoEncMulti(float* __restrict features,
    const float* __restrict features_clean, float* __restrict features_mod, const float* __restrict rot_mat,
    float px, float py, float pz, float cx, float cy, float cz,
    float radius, float min_d, float max_d, int batch_offset, int batch_size)
{
  __shared__ float all_ds_x[RAYS_PER_BLOCK],
      all_ds_y[RAYS_PER_BLOCK],
      all_ds_z[RAYS_PER_BLOCK],
      all_dists[RAYS_PER_BLOCK];

  int ray_id_base = blockIdx.x * RAYS_PER_BLOCK;

  int t_ray_id = ray_id_base + threadIdx.x;
  if (threadIdx.x < RAYS_PER_BLOCK && t_ray_id < batch_size)
  {
    const float3 fc = reinterpret_cast<const float3*>(features_clean)[batch_offset + t_ray_id];
    const float3 rt1 = reinterpret_cast<const float3*>(rot_mat)[0];
    const float3 rt2 = reinterpret_cast<const float3*>(rot_mat)[1];
    const float3 rt3 = reinterpret_cast<const float3*>(rot_mat)[2];

    float3 d;
    d.x = fc.x * rt1.x + fc.y * rt1.y + fc.z * rt1.z;
    d.y = fc.x * rt2.x + fc.y * rt2.y + fc.z * rt2.z;
    d.z = fc.x * rt3.x + fc.y * rt3.y + fc.z * rt3.z;

    d = normalize(d);
    float dist = raySphereIntersectD(make_float3(px, py, pz), d, make_float3(cx, cy, cz), radius);
    float3 np = make_float3(px, py, pz) + d * dist;

    all_dists[threadIdx.x] = dist;
    all_ds_x[threadIdx.x] = d.x;
    all_ds_y[threadIdx.x] = d.y;
    all_ds_z[threadIdx.x] = d.z;


    //// write first features
    //float3* r_feat = reinterpret_cast<float3*>(features);
    //int idx_r = (t_ray_id) * (2 + N_DEPTHS);
    //r_feat[idx_r] = d;
    //r_feat[idx_r + 1] = np;
  }

  __syncthreads();

  float step = 1.0f / N_DEPTHS;
  float cur_step = step * (0.5f + threadIdx.x);
  float my_log_to_world = logtToWorld(cur_step, min_d, max_d);

  for (int ray_id_offset = 0; ray_id_offset < RAYS_PER_BLOCK; ++ray_id_offset)
  {
    int ray_id = ray_id_base + ray_id_offset;
    if (ray_id >= batch_size)
        return;

    float3 d = make_float3(all_ds_x[ray_id_offset], all_ds_y[ray_id_offset], all_ds_z[ray_id_offset]);
    float dist = all_dists[ray_id_offset];
    float3 np = make_float3(px, py, pz) + d * dist;
    float3 feature = np + d * my_log_to_world;

    float3* r_feat = reinterpret_cast<float3*>(features);
    float3* m_feat = reinterpret_cast<float3*>(features_mod);
    int idx_r = (ray_id) * (2 + N_DEPTHS);
    r_feat[idx_r + 2 + threadIdx.x] = feature;

    float3 add_write = d;
    if(threadIdx.x % 2 == 1)
      add_write = np;
    if (threadIdx.x < 2)
      r_feat[idx_r + threadIdx.x] = add_write;
    else if (threadIdx.x < 4)
      m_feat[ray_id * 2 + threadIdx.x % 2] = add_write;
  }
}


template<int N_DEPTHS, int RAYS_PER_WARP, int NUM_WARPS>
__global__ void setSpherePosDirBatchedUnrolledNoEncMultiWarp(float* __restrict features,
  const float* __restrict features_clean, float* __restrict features_mod, const float* __restrict rot_mat,
  float px, float py, float pz, float cx, float cy, float cz,
  float radius, float min_d, float max_d, int batch_offset, int batch_size)
{
  constexpr int RAYS_PER_BLOCK = RAYS_PER_WARP * NUM_WARPS;
  int ray_id_base = blockIdx.x * RAYS_PER_BLOCK + threadIdx.x / WARP_SIZE * RAYS_PER_WARP;

  const float3 rt1 = reinterpret_cast<const float3*>(rot_mat)[0];
  const float3 rt2 = reinterpret_cast<const float3*>(rot_mat)[1];
  const float3 rt3 = reinterpret_cast<const float3*>(rot_mat)[2];

  int laneid = threadIdx.x % WARP_SIZE;
  for (int rays = 0; rays < RAYS_PER_WARP; rays += WARP_SIZE)
  {
    float3 d;
    float dist;
    int t_ray_id = ray_id_base + rays + laneid;
    if (t_ray_id < batch_size)
    {
      const float3 fc = reinterpret_cast<const float3*>(features_clean)[batch_offset + t_ray_id];

      d.x = fc.x * rt1.x + fc.y * rt1.y + fc.z * rt1.z;
      d.y = fc.x * rt2.x + fc.y * rt2.y + fc.z * rt2.z;
      d.z = fc.x * rt3.x + fc.y * rt3.y + fc.z * rt3.z;

      d = normalize(d);
      dist = raySphereIntersectD(make_float3(px, py, pz), d, make_float3(cx, cy, cz), radius);
    }

    for (int i = 0; i < WARP_SIZE; ++i)
    {
      int t_ray_id = ray_id_base + rays + i;
      if (t_ray_id >= batch_size)
        return;

      float3 td;
      td.x = __shfl_sync(0xFFFFFFFF, d.x, i);
      td.y = __shfl_sync(0xFFFFFFFF, d.y, i);
      td.z = __shfl_sync(0xFFFFFFFF, d.z, i);
      float tdist = __shfl_sync(0xFFFFFFFF, dist, i);
      float3 np = make_float3(px, py, pz) + td * tdist;

      float3* r_feat = reinterpret_cast<float3*>(features);
      float3* m_feat = reinterpret_cast<float3*>(features_mod);
      int idx_r = (t_ray_id) * (2 + N_DEPTHS);

      for (int round = 0; round < N_DEPTHS; round += WARP_SIZE)
      {
        int s_id = round + laneid;
        if (s_id < N_DEPTHS)
        {
          float step = 1.0f / N_DEPTHS;
          float cur_step = step * (0.5f + s_id);
          float my_log_to_world = logtToWorld(cur_step, min_d, max_d);

          float3 feature = np + td * my_log_to_world;
          r_feat[idx_r + 2 + s_id] = feature;
        }
      }

      float3 add_write = td;
      if (laneid % 2 == 1)
        add_write = np;
      if (laneid < 2)
        r_feat[idx_r + laneid] = add_write;
      else if (laneid < 4)
        m_feat[t_ray_id * 2 + laneid % 2] = add_write;
    }
  }
}

__global__ void samplePDF(int batch_size, int num_ray_samples, float* z_vals, float* depth, int num_depths)
{
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= batch_size)
    return;

  int idx = id;// x + y * size;

  float depth_sum = 0.0f;

  float current_depth = 0.0f;

  int cdf_size = num_depths + 1;
  float cdfs_[129];
  float depths_[129];

  int n_samples = num_ray_samples + 2;

  for (int k = 0; k < num_depths; k++)
  {
    current_depth = sigmoid(depth[idx * num_depths + k]) + 1e-5f; // depth transform;   weight += 1e-5
    depths_[k] = current_depth;
    depth_sum += current_depth; //  torch.sum(weights, -1, keepdim=True)
  }

  float cdf = 0.0f;
  float u = 0.0f;
  float u_step = 1.0f / (n_samples - 1);

  int below[10];
  int above[10];
  int ind_c = 0;

  for (int k = 0; k < cdf_size; k++)
  {
    float pdf = 0;
    if (k < num_depths)
      pdf = depths_[k] / depth_sum;

    while (cdf > u && ind_c < 10)
    {
      below[ind_c] = max(0, k - 1);
      above[ind_c] = min(cdf_size - 1, k);

      ind_c++;
      u += u_step;
    }
    cdfs_[k] = cdf;
    cdf += pdf;
  }

  for (int l = ind_c; l < 10; l++)
  {
    below[l] = max(0, cdf_size - 2);
    above[l] = min(cdf_size - 1, cdf_size - 1);
  }
  u = 0;
  for (int l = 0; l < n_samples; l++)
  {
    float cdf_g0 = cdfs_[below[l]];
    float cdf_g1 = cdfs_[above[l]];

    float bin_g0 = getBin(num_depths, below[l]); // num_bins = num_depths+1 ; torch.linspace(0., 1., 128+1, device="cpu", dtype=torch.float32)
    float bin_g1 = getBin(num_depths, above[l]);

    float denom = cdf_g1 - cdf_g0;
    if (denom < 1e-5f)
      denom = 1.0f;

    float t = (u - cdf_g0) / denom;
    float sample = bin_g0 + t * (bin_g1 - bin_g0);

    if (l > 0 && l < num_ray_samples + 1)
      z_vals[idx * num_ray_samples + l - 1] = sample;
    u += u_step;
  }
}

__global__ void rayMarchFromPoses(int size, int batch_size, int batch_offset, 
  float* features, float* features_last, float z_step, float noise_amplitude, int num_ray_samples, 
  int depth_transform, float z_near, float z_far, float min_d_range, float max_d_range, float* z_vals,
  float3 center, float max_depth, int num_dir_freqs, int num_pos_freqs, 
  float* dir_freq_bands, float* pos_freq_bands, float* depth, int num_depths)
{
  unsigned int feature_idx = threadIdx.x;
  unsigned int ray_idx = blockIdx.x / num_ray_samples;
  unsigned int sample_idx = blockIdx.x % num_ray_samples;

  if (ray_idx >= batch_size || feature_idx >= num_pos_freqs + 1 || sample_idx >= num_ray_samples)
    return;

  float3* features_r = reinterpret_cast<float3*>(features);
  float3* features_last_r = reinterpret_cast<float3*>(features_last);

  float3 p = features_last_r[ray_idx * 2 + 1];
  float3 d = features_last_r[ray_idx * 2 + 0];

  d = normalize(d);

  int n_features_r = (1 + num_dir_freqs * 2 + 1 + num_pos_freqs * 2);
  int feature_idx_r = ray_idx * num_ray_samples * n_features_r;

  float z_val = logtToWorld(z_vals[ray_idx * num_ray_samples + sample_idx], min_d_range, max_d_range);

  float3 p_ = p + d * z_val;

  float3 np = normalizationInverseSqrtDistCentered(p_, center, max_depth);

  int p_start_r = feature_idx_r + sample_idx * n_features_r;
  int d_start_r = feature_idx_r + sample_idx * n_features_r + 1 + num_pos_freqs * 2;


  if (feature_idx == 0)
  {
    features_r[p_start_r] = np;
    features_r[d_start_r] = d;
    z_vals[ray_idx * num_ray_samples + sample_idx] = z_val;
    return;
  }
  feature_idx -= 1;

  float fbv = pos_freq_bands[feature_idx];
  float3 v_f = np * fbv;
  features_r[p_start_r + 1 + feature_idx * 2 + 0] = make_float3(sin(v_f.x), sin(v_f.y), sin(v_f.z));
  features_r[p_start_r + 1 + feature_idx * 2 + 1] = make_float3(cos(v_f.x), cos(v_f.y), cos(v_f.z));

  if (feature_idx >= num_dir_freqs)
    return;

  v_f = d * fbv;
  features_r[d_start_r + 1 + feature_idx * 2 + 0] = make_float3(sin(v_f.x), sin(v_f.y), sin(v_f.z));
  features_r[d_start_r + 1 + feature_idx * 2 + 1] = make_float3(cos(v_f.x), cos(v_f.y), cos(v_f.z));
}

__global__ void nerf_raw_2_output(float* shading_output, cudaSurfaceObject_t out_fb_surf, 
  float* z_vals, float* features, int size, int num_samples, int batch_size, int batch_offset)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= batch_size)
    return;

  int x_ = (idx + batch_offset) % size;
  int y_ = (idx + batch_offset) / size;

  if (x_ >= size || y_ >= size)
    return;

  int start = idx * 4 * num_samples;
  int start_2 = idx * num_samples;

  float last_prod = 1.0f;

  float nr = 0.0f;
  float ng = 0.0f;
  float nb = 0.0f;
  
  for (int k = 0; k < num_samples; k++) 
  {
    // dists = z_vals[..., 1:] - z_vals[..., :-1];    
    // dists = torch.cat([dists, (torch.ones(1, device = raw.device) * 1e10).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
    float dist = (k < num_samples - 1)
      ? z_vals[start_2 + k + 1] - z_vals[start_2 + k]
      : 1e10f;

    // rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    float r = sigmoid(shading_output[start + 0 + k * 4]);
    float g = sigmoid(shading_output[start + 1 + k * 4]);
    float b = sigmoid(shading_output[start + 2 + k * 4]);

    float alpha = raw2alpha(shading_output[start + 3 + k * 4], dist);

    // weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device = raw.device), 1. - alpha + 1e-10], -1), -1)[:,:-1]
    float prod =  last_prod * (1.0f - alpha + 1e-10f);
    float weight = alpha * last_prod;  

    // rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
    nr = nr + weight * r;
    ng = ng + weight * g;
    nb = nb + weight * b;

    last_prod = prod;
  }

  uchar4 data;
  data.x = clamp(nr, 0.0f, 1.0f) * 255.0f;
  data.y = clamp(ng, 0.0f, 1.0f) * 255.0f;
  data.z = clamp(nb, 0.0f, 1.0f) * 255.0f;
  data.w = 255.0f;
  surf2Dwrite(data, out_fb_surf, x_ * 4, y_);
}

__global__ void samplesToImage(float* __restrict output, cudaSurfaceObject_t out_fb_surf, const int batch_size, 
  const int batch_offset, const int width, const int num_outputs)
{
  int ray_id = blockIdx.x;
  int sample_id = threadIdx.x;

  if (ray_id >= batch_size || sample_id >= num_outputs)
    return;
  
  float sample_value = output[ray_id * num_outputs + sample_id];
  
  float thread_key[1];
  thread_key[0] = sample_value;
  int thread_value[1];
  thread_value[0] = sample_id;
  {
    typedef cub::BlockRadixSort<float, 128, 1, int> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // Sort the key-value pair, where key = sample value, value = sample id
    BlockRadixSort(temp_storage).SortDescending(thread_key, thread_value);
  }
  
  if (sample_id > 2)
    return;
  
  __shared__ float ordered[3];
  ordered[sample_id] = (0.5f + thread_value[0]) / 128.0f;
  
  if (sample_id != 0)
    return;
  
  int x_ = (ray_id + batch_offset) % width;
  int y_ = (ray_id + batch_offset) / width;
  
  uchar4 data;
  data.x = clamp(ordered[0], 0.0f, 1.0f) * 255.0f;
  data.y = clamp(ordered[1], 0.0f, 1.0f) * 255.0f;
  data.z = clamp(ordered[2], 0.0f, 1.0f) * 255.0f;
  data.w = 255.0f;
  surf2Dwrite(data, out_fb_surf, x_ * 4, y_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// HOST CODE 

void updateSpherePosDirBatchedUnrolledNoEnc(float* features, float* features_clean, float* features_mod, float x, float y, float z,
  float* rotation_matrix, int width, int height, int num_dir_enc, int num_pos_enc, float* dir_freq_bands, float* pos_freq_bands,
  float center_x, float center_y, float center_z, float radius, int additional_samples, float min_d_range, float max_d_range,
  int batch_offset, int batch_size)
{
  const int N_DEPTHS = 128;
  
  assert(additional_samples == N_DEPTHS && "use constant for better performance");

#if 1
  // this one does a lot of duplicated work but seems to be the fastest / however maybe not if the batch size becomes very larger
  const int N_RAYS_PER_BLOCK = 4;
  dim3 dimBlock(N_DEPTHS * N_RAYS_PER_BLOCK);
  dim3 dimGrid((batch_size + N_RAYS_PER_BLOCK - 1) / N_RAYS_PER_BLOCK);

  setSpherePosDirBatchedUnrolledNoEnc<N_DEPTHS><<<dimGrid, dimBlock>>>(features, features_clean, features_mod, rotation_matrix, 
    x, y, z, center_x, center_y, center_z , radius, min_d_range, max_d_range, batch_offset, batch_size);
#elif 1
  // this one does no duplicated work, but writes more data per block, overall that leads to slightly worse 
  // memory access patterns considern block distances, but may be faster for larger batch sizes
  constexpr int N_DEPTHS = 128;
  constexpr int RAYS_PER_BLOCK = 128;

  setSpherePosDirBatchedUnrolledNoEncMulti<N_DEPTHS, RAYS_PER_BLOCK> <<<divup(batch_size, RAYS_PER_BLOCK), N_DEPTHS >>> (features, features_clean, 
    rotation_matrix, features_mod, x, y, z, center_x, center_y, center_z, radius, min_d_range, max_d_range, batch_offset, batch_size);
#else
  // this on gets rid of __syncthreads but worsen the memory access pattern further, so dont use it
  constexpr int N_DEPTHS = 128;
  constexpr int RAYS_PER_WARP = 32;
  constexpr int WARPS_PER_BLOCK = 16;

  setSpherePosDirBatchedUnrolledNoEncMultiWarp<N_DEPTHS, RAYS_PER_WARP, WARPS_PER_BLOCK> << <divup(batch_size, RAYS_PER_WARP * WARPS_PER_BLOCK), RAYS_PER_WARP * WARPS_PER_BLOCK >> > (features, features_clean, features_mod, rotation_matrix,
    x, y, z, center_x, center_y, center_z, radius, min_d_range, max_d_range, batch_offset, batch_size);
#endif
  CUDA_CHECK;
}

void updateSpherePosDirBatchedUnrolledEnc(float* features, float* features_clean, float* features_mod, float x, float y, float z, float* rotation_matrix, 
  int width, int height, int num_dir_enc, int num_pos_enc, float* dir_freq_bands, float* pos_freq_bands,
  float center_x, float center_y, float center_z, float radius, int additional_samples, float min_d_range, float max_d_range,
  int batch_offset, int batch_size)
{
  const int ZERO = 0;
  assert(additional_samples == ZERO && "use constant for better performance");
  
  int n_l_features = num_pos_enc + num_dir_enc + 1;
  dim3 dimBlock(n_l_features * 2);
  dim3 dimGrid(batch_size);
  
  setSpherePosDirBatchedUnrolledEnc<<<dimGrid, dimBlock>>>(features, features_clean, features_mod, rotation_matrix, x, y, z, center_x, center_y, center_z,
    radius, min_d_range, max_d_range, batch_offset, batch_size, num_dir_enc, num_pos_enc, dir_freq_bands, pos_freq_bands, width, height);
}

void updateRayMarchFromPoses(float* features, float* features_clean, float* features_last, 
  float x, float y, float z, float* rotation_matrix, int width, int height,
  int num_dir_enc, int num_pos_enc, float* dir_freq_bands, float* pos_freq_bands,
  float z_step, float noise_amplitude, int num_ray_samples, int depth_transform, 
  float z_near, float z_far, float min_d_range, float max_d_range, float* z_vals,
  float vc_center_x, float vc_center_y, float vc_center_z, float max_depth,
  int batch_offset, int batch_size,  float* depth, float* cdf, int num_depths, std::string type)
{
  dim3 dimBlock(1024);
  dim3 dimGrid((batch_size + dimBlock.x - 1) / dimBlock.x);
  samplePDF<<<dimGrid, dimBlock>>>(batch_size, num_ray_samples, z_vals, depth, num_depths);
  
  // cudaDeviceSynchronize();
  CUDA_CHECK;
  assert(depth_transform && "ERROR: need to change function because of depth transform");

  int n_l_features = num_pos_enc + 1;
  dim3 dimBlock2(n_l_features);
  dim3 dimGrid2(batch_size * num_ray_samples);

  rayMarchFromPoses<<<dimGrid2, dimBlock2>>>(width, batch_size, batch_offset, features, features_last, 
    z_step, noise_amplitude, num_ray_samples, depth_transform, 
    z_near, z_far, min_d_range, max_d_range, z_vals,
    make_float3(vc_center_x, vc_center_y, vc_center_z), max_depth,
    num_dir_enc, num_pos_enc, dir_freq_bands, pos_freq_bands, depth, num_depths);

  // cudaDeviceSynchronize();
}

void copyResultSamplingNetwork(float* output, cudaSurfaceObject_t out_fb_surf, const int batch_size, 
  const int batch_offset, const int width, const int num_outputs)
{
  dim3 dimBlock(num_outputs);
  dim3 dimGrid(batch_size);
  samplesToImage<<<dimGrid, dimBlock>>>(output, out_fb_surf, batch_size, batch_offset, width, num_outputs);
}

void copyResultRaymarch(void* shading_output, cudaSurfaceObject_t fb_surf, int width, int height, 
  float* z_vals, float* features, int num_samples, int batch_size, int batch_offset)
{
  dim3 dimBlock(16);
  dim3 dimGrid((batch_size + dimBlock.x - 1) / dimBlock.x);
  nerf_raw_2_output<<<dimGrid, dimBlock>>>((float*) shading_output, fb_surf, z_vals, features, width, num_samples, batch_size, batch_offset);
  
  CUDA_CHECK;
}
