
#pragma once

#include <string>
#include <iostream>

#ifdef NDEBUG
# define CUDA_CHECK // No op
#else
# define CUDA_CHECK {\
    cudaError_t  cu_error = cudaGetLastError();                                 \
    if (cu_error != cudaSuccess) {                                              \
      std::cout << "Cuda error: " << cudaGetErrorString(cu_error) << std::endl; \
    }                                                                           \
  }
#endif

static constexpr int NUM_SHADING_OUTPUTS{4};

void copyResultSamplingNetwork(float* output, cudaSurfaceObject_t out_fb_surf, const int batch_size, const int batch_offset,
  const int width, const int num_outputs);

void copyResultRaymarch(void* shading_output, cudaSurfaceObject_t out_fb_surf, int width, int height, 
  float* z_vals, float* features, int num_samples,  int batch_offset, int batch_size);

void copyResultRaymarchAdaptive(void* shading_output, cudaSurfaceObject_t out_fb_surf, 
  const int batch_size, const int batch_offset, const int width, const int n_max_ray_samples,
  const int* ray_offsets, const int* ray_n_samples);

void copyResultRaymarchAdaptiveMultDepth(void* shading_output, cudaSurfaceObject_t out_fb_surf,
  const int batch_size, const int batch_offset, const int width, const int n_max_ray_samples,
  const int* ray_offsets, const int* ray_n_samples, void* depth_value, int* d_sample_depth_idx, const int mult_location);

void updateSpherePosDirBatchedUnrolledNoEnc(float* features, float* features_clean, float* features_mod, float x, float y, float z, float* rotation_matrix, 
  int width, int height, int num_dir_enc, int num_pos_enc, float* dir_freq_bands, float* pos_freq_bands,
  float center_x, float center_y, float center_z, float radius, int additional_samples, float min_d_range, float max_d_range,
  int batch_offset, int batch_size);

void updateSpherePosDirBatchedUnrolledEnc(float* features, float* features_clean, float* features_mod, float x, float y, float z, float* rotation_matrix, 
  int width, int height, int num_dir_enc, int num_pos_enc, float* dir_freq_bands, float* pos_freq_bands,
  float center_x, float center_y, float center_z, float radius, int additional_samples, float min_d_range, float max_d_range,
  int batch_offset, int batch_size);

void updateRayMarchFromPoses(float* features, float* features_clean, float* features_last, 
  float x, float y, float z, float* rotation_matrix, int width, int height,
  int num_dir_enc, int num_pos_enc, float* dir_freq_bands, float* pos_freq_bands,
  float z_step, float noise_amplitude, int num_ray_samples, int depth_transform, 
  float z_near, float z_far, float min_d_range, float max_d_range, float* z_vals,
  float vc_center_x, float vc_center_y, float vc_center_z, float max_depth,
  int batch_offset, int batch_size, float* depth, float* cdf, int num_depths, std::string sampler);

int updateRayMarchFromPosesAdaptive(const int batch_size, const int n_max_ray_samples, 
  const float adaptive_sampling_threshold, const int num_dir_enc, const int num_pos_enc,  
  const float min_d_range, const float max_d_range, const float max_depth,
  const float vc_center_x, const float vc_center_y, const float vc_center_z, 
  float* depth_vals, const float* features_last, const float* freq_bands, 
  int* ray_offsets, int* ray_n_samples, int* d_ray_idx_per_z_val, int* d_sample_idx_per_z_val, float* z_vals, float* features,
  int* d_sample_depth_idx, const int width, const int height, const float focal);

void updateRayMarchCoarse(float* features, float* features_clean, float* features_last, 
  float x, float y, float z, float* rotation_matrix, int width, int height,
  int num_dir_enc, int num_pos_enc, int encoding, float* dir_freq_bands, float* pos_freq_bands,
  float z_step, int num_ray_samples, int depth_transform, 
  float z_near, float z_far, float min_d_range, float max_d_range, float* z_vals,
  float vc_center_x, float vc_center_y, float vc_center_z, float max_depth,
  int batch_offset, int batch_size);

void updateRayMarchFromCoarse(float* features, float* features_clean, float* features_last, 
  float x, float y, float z, float* rotation_matrix, int width, int height,
  int num_dir_enc, int num_pos_enc, int encoding, float* dir_freq_bands, float* pos_freq_bands,
  float z_step, float noise_amplitude, int num_ray_samples, int depth_transform, 
  float z_near, float z_far, float min_d_range, float max_d_range, float* z_vals,
  float vc_center_x, float vc_center_y, float vc_center_z, float max_depth,
  int batch_offset, int batch_size, std::string type, float* cdf, float* prev_z_vals, float* weights, float* out_prev, int out_prev_size);
