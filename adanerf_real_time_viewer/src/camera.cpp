#include "../include/camera.h"

#include <math.h> 
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <cuda_runtime.h>

#include "../include/encoding.h"
#include "../include/featureset.h"
#include "../include/config.h"
#include "../include/cuda/adanerf_cuda_kernels.cuh"
#include "../include/settings.h"


Camera::Camera(Settings& settings, Config& config) 
  : settings(settings), config(config)
{
}

Camera::~Camera() 
{
  if (d_rotation_matrix != nullptr)
    cudaFree(d_rotation_matrix);
}

bool Camera::init(std::string feature_type, float fov_, Encoding* enc_)
{
  enc = enc_;
  if (enc->type == "nerf")
    feature_encoding = 1;
  if (enc->type == "fourier")
    feature_encoding = 2;
  fov = fov_;

  int w = settings.width;
  int h = settings.height;
  focal = 0.5f * w / tan(0.5f * fov);

  float x_dist = tan(fov / 2.0f) * focal;
  float y_dist = x_dist * (w / h);
  float x_dist_pp = x_dist / (w / 2.0f);
  float y_dist_pp = y_dist / (h / 2.0f);

  features_clean = new float[w * h * 3];

  glm::vec3 vc_center = glm::vec3(config.viewcellCenter[0], config.viewcellCenter[1], config.viewcellCenter[2]);
  glm::vec3 vc_size = glm::vec3(config.viewcellSize[0] / 2, config.viewcellSize[1] / 2, config.viewcellSize[2] / 2);
  
  // speed is now proportional to view cell size
  speed_mult = std::max(vc_size[0], std::max(vc_size[1], vc_size[2]));

  pos = vc_center;
  dir = glm::vec3(30, 50, 5) - pos;

  for (int x = 0; x < w; x++)
  {
    for (int y = 0; y < h; y++)
    {
      int idx = (y * w + x) * 3;

      features_clean[idx + 0] = ( y_dist - x_dist_pp * x) * -1;
      features_clean[idx + 1] = (-x_dist + y_dist_pp * y) * -1;
      features_clean[idx + 2] = -focal;
    }
  }
  int f_size = h * w * 3 * sizeof(float);

  cudaMalloc(&d_features_clean, f_size);
  cudaMalloc(&d_rotation_matrix, 3 * 3 * sizeof(float));
  
  // stores SpherePosDir modified pos and dir, so we can access them from kernels
  // withouzt worrying about additional inputs / encoding size
  int mod_size = settings.batch_size * 6 * sizeof(float);
  cudaMalloc(&d_features_mod, mod_size);

  cudaMemcpy(d_features_clean, features_clean, f_size, cudaMemcpyHostToDevice);

  if (feature_type == "SpherePosDir")
  {
    input_features = (FeatureSet*) new SpherePosDir(settings, config);
    ((SpherePosDir*) input_features)->create(w, h, settings.batch_size, enc);
  }
  if (feature_type == "RayMarchFromPoses")
  {
    input_features = (FeatureSet*) new RayMarchFromPoses(settings, config);
    ((RayMarchFromPoses*) input_features)->create(w, h, settings.batch_size, enc, 0);
  }

  pos_changed = rot_changed = true;
  yaw = -80.f;
  pitch = 0.f;

  return true;
}

void Camera::MovementKeyPressed(unsigned char key)
{
  if (key == 'w')
    move_fwd = 1;
  if (key == 's')
    move_fwd = -1;
  if (key == 'a')
    move_right = -1;
  if (key == 'd')
    move_right = 1;
  if (key == 'q')
    move_up = 1;
  if (key == 'e')
    move_up = -1;
}

void Camera::MovementKeyReleased(unsigned char key)
{
  if (key == 'w')
    move_fwd = 0;
  if (key == 's')
    move_fwd = 0;
  if (key == 'a')
    move_right = 0;
  if (key == 'd')
    move_right = 0;
  if (key == 'q')
    move_up = 0;
  if (key == 'e')
    move_up = 0;
}

void Camera::MouseDrag(const glm::ivec2& offset)
{
  if (offset.x != 0 || offset.y != 0)
  {  
    float sensitivity = 0.15f;
    yaw -= (float) offset.x * sensitivity;
    pitch -= (float) offset.y * sensitivity;
    if (pitch > 89.0f)
      pitch = 89.0f;
    if (pitch < -89.0f)
      pitch = -89.0f;
    rot_changed = true;
  }
}

void Camera::UpdateFeatureRot()
{
  dir.x = cos(glm::radians(yaw)) * cos(glm::radians(pitch));
  dir.z = sin(glm::radians(pitch));
  dir.y = sin(glm::radians(yaw)) * cos(glm::radians(pitch));

  dir = glm::normalize(dir);
  
  right = glm::cross(dir, glm::vec3(0, 0, 1));
  up = glm::cross(right, dir);

  glm::mat3 rot = glm::lookAt(pos, pos + dir, up);
  
  cudaMemcpy(d_rotation_matrix, &rot, 3 * 3 * sizeof(float), cudaMemcpyHostToDevice);
  CUDA_CHECK;
}

void Camera::UpdateFeaturesBatch(int batch_offset, int batch_size)
{
  if (rot_changed)
  {
    UpdateFeatureRot();
  }

  pos_changed = false;
  float speed = 0.005f * speed_mult;

  if (move_fwd != 0)
  {
    pos += dir * speed * (float)move_fwd;
    pos_changed = true;
  }

  if (move_right != 0)
  {
    pos += right * speed * (float)move_right;
    pos_changed = true;
  }

  if (move_up != 0)
  {
    pos += up * speed * (float)move_up;
    pos_changed = true;
  }

  if (pos_changed || rot_changed)
  {
    if (input_features->type == "RayMarchFromPoses")
    {
      ((RayMarchFromPoses*)input_features)->updateFeatures(pos, d_rotation_matrix, d_features_clean, nullptr,
                                                           settings.width, settings.height, batch_offset, batch_size, nullptr, 0);
    }
    else if (input_features->type == "SpherePosDir")
    {
      ((SpherePosDir*)input_features)->updateFeaturesBatched(pos, d_rotation_matrix, d_features_clean, d_features_mod,
                                                             settings.width, settings.height, batch_offset, batch_size);
    }
  }
}
