#include "../include/featureset.h"

#include "../include/config.h"
#include "../include/cuda/adanerf_cuda_kernels.cuh"
#include "../include/encoding.h"
#include "../include/settings.h"

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <locale>
#include <functional>
#include <math.h>

FeatureSet::FeatureSet(Settings& settings, Config& config) 
  : settings(settings), config(config)
{ 
  if (d_features != nullptr) 
    cudaFree(d_features);
}

// SpherePosDir
bool SpherePosDir::create(int width, int height, int batch_size, Encoding* enc_)
{
  enc = enc_;
  type = "SpherePosDir";
  additional_samples = config.raySampleInput[0];

  vc_center[0] = config.viewcellCenter[0];
  vc_center[1] = config.viewcellCenter[1];
  vc_center[2] = config.viewcellCenter[2];
  vc_size[0] = config.viewcellSize[0];
  vc_size[1] = config.viewcellSize[1];
  vc_size[2] = config.viewcellSize[2];

  std::cout << "view cell center: " << vc_center[0]<<", "<< vc_center[1]<<", "<< vc_center[2] << std::endl;
  std::cout << "view cell size: " << vc_size[0]<<", "<< vc_size[1]<<", "<< vc_size[2] << std::endl;

  vc_radius = sqrtf(vc_size[0] * vc_size[0] + vc_size[1] * vc_size[1] + vc_size[2] * vc_size[2]) * 0.5f;

  int s = 6;
  s += (enc->getNumDirFreqs() + enc->getNumPosFreqs()) * 3 * 2 + additional_samples * 3;
  int f_size_f = height * width * s * sizeof(float);
  cudaMalloc(&d_features, f_size_f);

  max_drange = config.depthRange[1];
  min_drange = config.depthRange[0];

  return true;
}

void SpherePosDir::updateFeaturesBatched(glm::vec3 point, float* rot_mat, float* features_clean, float* features_mod, 
                                         int width, int height, int batch_offset, int batch_size)
{
  if (additional_samples > 0)
    updateSpherePosDirBatchedUnrolledNoEnc(d_features, features_clean, features_mod, point.x, point.y, point.z, rot_mat, 
      width, height, enc->getNumDirFreqs(), enc->getNumPosFreqs(), enc->d_dir_freq_bands, enc->d_pos_freq_bands,
      vc_center[0], vc_center[1], vc_center[2], vc_radius, additional_samples, min_drange, max_drange, batch_offset, batch_size);
  else
    updateSpherePosDirBatchedUnrolledEnc(d_features, features_clean, features_mod, point.x, point.y, point.z, rot_mat,
      width, height, enc->getNumDirFreqs(), enc->getNumPosFreqs(), enc->d_dir_freq_bands, enc->d_pos_freq_bands,
      vc_center[0], vc_center[1], vc_center[2], vc_radius, additional_samples, min_drange, max_drange, batch_offset, batch_size);
}

bool RayMarchFromPoses::create(int width, int height, int batch_size, Encoding* enc_, int count)
{
  enc = enc_;
 
  type = "RayMarchFromPoses";

  z_step = config.rayMarchSamplingStep[count]; 
  noise_amplitude = config.rayMarchSamplingNoise[count];
  num_ray_samples = config.numRaymarchSamples[count];
  min_d = config.zNear[count];
  max_d = config.zFar[count];
  max_depth = config.max_depth;
  
  adaptiveSamplingThreshold = config.adaptiveSamplingThreshold;
  
  ndc = config.useNDC;
  if (ndc)
    focal = 0.5f * width / tan(0.5f * config.fov);

  if (config.accumulationMult == "alpha")
    mult_location = 1;
  else if (config.accumulationMult == "weights")
    mult_location = 2;

  depth_transform = 0;
  if (config.depthTransform == "log")
    depth_transform = 1;

  sampler = config.rayMarchSampler[count];

  max_drange = config.depthRange[1];
  min_drange = config.depthRange[0];

  vc_x = config.viewcellCenter[0];
  vc_y = config.viewcellCenter[1];
  vc_z = config.viewcellCenter[2];

  vc_center[0] = config.viewcellCenter[0];
  vc_center[1] = config.viewcellCenter[1];
  vc_center[2] = config.viewcellCenter[2];
  vc_size[0] = config.viewcellSize[0];
  vc_size[1] = config.viewcellSize[1];
  vc_size[2] = config.viewcellSize[2];

  std::cout << "view cell center: " << vc_center[0] << ", " << vc_center[1] << ", " << vc_center[2] << std::endl;
  std::cout << "view cell size: " << vc_size[0] << ", " << vc_size[1] << ", " << vc_size[2] << std::endl;

  vc_radius = sqrtf(vc_size[0] * vc_size[0] + vc_size[1] * vc_size[1] + vc_size[2] * vc_size[2]) * 0.5f;

  int s = 6;
  s += (enc->getNumDirFreqs() + enc->getNumPosFreqs()) * 3 * 2;
  int f_size_f = batch_size * s * num_ray_samples * sizeof(float);
  cudaMalloc(&d_features, f_size_f);

  cudaMalloc(&d_z_vals, batch_size * num_ray_samples * sizeof(float));

  if (sampler == "FromClassifiedDepthAdaptive" || sampler == "FromClassifiedDepthAdaptiveNoDepthRange")
  {
    cudaMalloc(&d_ray_idx_per_z_val, batch_size * num_ray_samples * sizeof(int));
    cudaMalloc(&d_sample_idx_per_z_val, batch_size * num_ray_samples * sizeof(int));
    cudaMalloc(&d_sample_depth_idx, batch_size * num_ray_samples * sizeof(int));

    cudaMalloc(&d_ray_offsets, batch_size * sizeof(int));
    cudaMalloc(&d_ray_n_samples, batch_size * sizeof(int));
    
    adaptive = true;
  }

  if(config.multiDepthFeatures.size() > 0)
    cudaMalloc(&d_cdf, batch_size * (config.multiDepthFeatures[count-1]+1) * sizeof(float));
  CUDA_CHECK;

  return true;
}

int RayMarchFromPoses::updateFeatures(glm::vec3 point, float* rot_mat, float* features_clean, float* features_mod, 
                                      int width, int height, int batch_offset, int batch_size, float* depth, int num_depths)
{
  cudaError_t cu_error = cudaGetLastError();
  if (cu_error != cudaSuccess)
  {
    std::cout << "Cuda error: " << cudaGetErrorString(cu_error) << std::endl;
  }

  if (adaptive)
  {
    return updateRayMarchFromPosesAdaptive(batch_size, num_ray_samples, adaptiveSamplingThreshold, 
      enc->getNumDirFreqs(), enc->getNumPosFreqs(), min_drange, max_drange, max_depth, vc_x, vc_y, vc_z, 
      depth, features_mod, enc->d_pos_freq_bands, 
      d_ray_offsets, d_ray_n_samples, d_ray_idx_per_z_val, d_sample_idx_per_z_val, d_z_vals, d_features, d_sample_depth_idx,
      width, height, focal);
  }
  else
  {
    if (sampler == "FromClassifiedDepth")
    {
      updateRayMarchFromPoses(d_features, features_clean, features_mod, point.x, point.y, point.z, rot_mat, width, height,
        enc->getNumDirFreqs(), enc->getNumPosFreqs(), enc->d_dir_freq_bands, enc->d_pos_freq_bands,
        z_step, noise_amplitude, num_ray_samples, depth_transform, min_d, max_d, min_drange, max_drange, d_z_vals,
        vc_x, vc_y, vc_z, max_depth, batch_offset, batch_size, depth, d_cdf, num_depths, sampler);
    }
    else
    {
      int feature_encoding = enc->type == "nerf" ? 1 : 0;
      updateRayMarchCoarse(d_features, features_clean, features_mod, point.x, point.y, point.z, rot_mat, width, height,
        enc->getNumDirFreqs(), enc->getNumPosFreqs(), feature_encoding, enc->d_dir_freq_bands, enc->d_pos_freq_bands,
        z_step, num_ray_samples, depth_transform, min_d, max_d, min_drange, max_drange, d_z_vals,
        vc_x, vc_y, vc_z, max_depth, batch_offset, batch_size);
    }

    return batch_size * num_ray_samples;
  }
}

void RayMarchFromPoses::raymarch(void* shading_output, cudaSurfaceObject_t fb_surf, int batch_size, int batch_offset, 
                                 int width, int height, float* features, void* depth_values)
{
  if (adaptive)
  {
    if (mult_location == 0)
    {
      copyResultRaymarchAdaptive(shading_output, fb_surf, batch_size, batch_offset, width, num_ray_samples, 
                                 d_ray_offsets, d_ray_n_samples);
    }
    else
    {
      copyResultRaymarchAdaptiveMultDepth(shading_output, fb_surf, batch_size, batch_offset, width, num_ray_samples,
                                          d_ray_offsets, d_ray_n_samples, depth_values, d_sample_depth_idx, mult_location);
    }
  }
  else
  {
      copyResultRaymarch(shading_output, fb_surf, width, height, d_z_vals, features, num_ray_samples, batch_size, batch_offset);
  }
}

bool RayMarchFromCoarse::create(int width, int height, int batch_size, Encoding* enc_, int count)
{
  enc = enc_;
  type = "RayMarchFromCoarse";

  z_step = config.rayMarchSamplingStep[0];
  noise_amplitude = config.rayMarchSamplingNoise[0];
  num_ray_samples = config.numRaymarchSamples[count];
  num_ray_samples_pre = config.numRaymarchSamples[count-1];
  min_d = config.zNear[0];
  max_d = config.zFar[0];
  max_depth = config.max_depth;

  depth_transform = 0;
  if (config.depthTransform == "log")
    depth_transform = 1;

  sampler = config.rayMarchSampler[count];
  max_drange = config.depthRange[1];
  min_drange = config.depthRange[0];

  vc_x = config.viewcellCenter[0];
  vc_y = config.viewcellCenter[1];
  vc_z = config.viewcellCenter[2];

  vc_center[0] = config.viewcellCenter[0];
  vc_center[1] = config.viewcellCenter[1];
  vc_center[2] = config.viewcellCenter[2];
  vc_size[0] = config.viewcellSize[0];
  vc_size[1] = config.viewcellSize[1];
  vc_size[2] = config.viewcellSize[2];

  std::cout << "view cell center: " << vc_center[0] << ", " << vc_center[1] << ", " << vc_center[2] << std::endl;
  std::cout << "view cell size: " << vc_size[0] << ", " << vc_size[1] << ", " << vc_size[2] << std::endl;

  vc_radius = sqrtf(vc_size[0] * vc_size[0] + vc_size[1] * vc_size[1] + vc_size[2] * vc_size[2]) * 0.5f;

  int s = 6;
  s += (enc->getNumDirFreqs() + enc->getNumPosFreqs()) * 3 * 2;
  int f_size_f = batch_size * s * (num_ray_samples + num_ray_samples_pre) * sizeof(float);
  cudaMalloc(&d_features, f_size_f);

  cudaMalloc(&d_z_vals, batch_size * (num_ray_samples + num_ray_samples_pre) * sizeof(float));
  cudaMalloc(&d_weights, batch_size * (num_ray_samples_pre) * sizeof(float));

  cudaError_t cu_error = cudaGetLastError();
  if (cu_error != cudaSuccess)
  {
    std::cout << "Cuda error: " << cudaGetErrorString(cu_error) << std::endl;
    return false;
  }

  return true;
}

void RayMarchFromCoarse::updateFeatures(glm::vec3 point, float* rot_mat, float* features_clean, float* features_last,
                                        int width, int height, int batch_offset, int batch_size, float* prev_z_vals,
                                        float* prev_weights, int prev_num_weights)
{
  int feature_encoding = enc->type == "nerf" ? 1 : 0;

  updateRayMarchFromCoarse(d_features, features_clean, features_last, point.x, point.y, point.z, rot_mat, width, height,
    enc->getNumDirFreqs(), enc->getNumPosFreqs(), feature_encoding, enc->d_dir_freq_bands, enc->d_pos_freq_bands,
    z_step, noise_amplitude, num_ray_samples, depth_transform, min_d, max_d, min_drange, max_drange, d_z_vals,
    vc_x, vc_y, vc_z, max_depth, batch_offset, batch_size, sampler, d_cdf, prev_z_vals, d_weights, prev_weights, prev_num_weights);
}
