// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.


#pragma once

#include "colmap/mvs/depth_map.h"
#include "colmap/mvs/mat.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_backend.h"

#include <vector>

// Target OpenCL 3.0: the Adreno X1 driver exposes OpenCL 3.0, and this also
// unhides the clCreateCommandQueueWithProperties (2.0+) entry point. Define
// before including the headers so every translation unit that pulls in this
// header agrees on the API version.
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif
#include <CL/cl.h>

namespace colmap {
namespace mvs {

// OpenCL implementation of the PatchMatch stereo algorithm, targeting the
// Qualcomm Adreno GPU on Windows-on-ARM (and any other OpenCL GPU exposed
// through the system ICD).
//
// PHASE 2b: photometric PatchMatch on the GPU. Device discovery prefers the
// native Qualcomm platform (name contains "QUALCOMM"/"SNAPDRAGON") and a GPU
// device with image support, avoiding the D3D12-translation "OpenCLOn12"
// platform. The compute is a faithful port of the validated CPU backend
// (patch_match_cpu.cc): identical PCG32 PRNG and cost function, one work-item
// per column, sequential rows. The geometric-consistency term is NOT yet
// implemented (Phase 2c); Run() throws if PatchMatchOptions::geom_consistency
// is set.
class PatchMatchOpenCL : public PatchMatchBackend {
 public:
  PatchMatchOpenCL(const PatchMatchOptions& options,
                   const PatchMatch::Problem& problem);
  ~PatchMatchOpenCL() override;

  void Run() override;

  DepthMap GetDepthMap() const override;
  NormalMap GetNormalMap() const override;
  Mat<float> GetSelProbMap() const override;
  std::vector<int> GetConsistentImageIdxs() const override;

 private:
  // Discover, select and report the OpenCL device, then create a context and
  // command queue on it. Throws if no suitable GPU device is found.
  void InitDevice();

  // Compile the kernel program for the selected device. Throws on build error
  // (logging the OpenCL build log).
  void BuildProgram();

  // Host-side initialization mirroring PatchMatchCpu: reference-image bilateral
  // filtering, source-image packing, transforms, and random depth/normal/PRNG
  // initialization. Allocates and uploads all device buffers.
  void InitHostDataAndUpload();

  // Run num_iterations * 4 sweeps with a 90-degree CCW rotation between each
  // (column-banded launches for the Windows TDR watchdog).
  void RunSweeps();

  // Rotate all reference-frame maps 90 degrees CCW on the device, mirroring
  // PatchMatchCpu::Rotate. `rotate_cmask` is set on the final rotation.
  void RotateMaps(bool rotate_cmask);

  // Read depth/normal/sel_prob/consistency mask back into host Mats.
  void ReadbackResults();

  const PatchMatchOptions options_;
  const PatchMatch::Problem problem_;

  // Selected OpenCL handles (owned; released in the destructor).
  cl_platform_id platform_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;

  cl_program program_ = nullptr;
  cl_kernel k_initial_cost_ = nullptr;
  cl_kernel k_sweep_ = nullptr;
  cl_kernel k_rot_f_ = nullptr;
  cl_kernel k_rot_u8_ = nullptr;
  cl_kernel k_rot_rand_ = nullptr;
  cl_kernel k_rot_normal_ = nullptr;

  // Problem dimensions.
  int ref_width_ = 0;
  int ref_height_ = 0;
  int num_src_ = 0;
  int src_max_w_ = 0;
  int src_max_h_ = 0;
  int window_radius_ = 0;
  int window_step_ = 0;

  // Current (rotated) dimensions and rotation index, updated during the sweep.
  int cur_width_ = 0;
  int cur_height_ = 0;
  int rotation_ = 0;

  // Per-rotation reference calibration {1/fx, -cx/fx, 1/fy, -cy/fy} (16 floats).
  // Uploaded once to ref_inv_K_buf_.
  float ref_inv_K_host_[4][4] = {{0}};
  float ref_K_host_[4][4] = {{0}};

  // Device buffers. Reference-frame maps are physically rotated (via a scratch
  // buffer) between sweeps; source data and transforms are fixed.
  cl_mem depth_buf_ = nullptr;          // float W*H
  cl_mem normal_buf_ = nullptr;         // float 3*W*H
  cl_mem ref_img_buf_ = nullptr;        // uchar W*H
  cl_mem ref_sum_buf_ = nullptr;        // float W*H
  cl_mem ref_sqsum_buf_ = nullptr;      // float W*H
  cl_mem cost_buf_ = nullptr;           // float num_src*W*H
  cl_mem rand_buf_ = nullptr;           // ulong2 W*H
  cl_mem sel_prob_buf_ = nullptr;       // float num_src*W*H
  cl_mem prev_sel_prob_buf_ = nullptr;  // float num_src*W*H
  cl_mem cmask_buf_ = nullptr;          // uchar num_src*W*H
  cl_mem scratch_buf_ = nullptr;        // generic rotation scratch (>= num_src*W*H floats)
  cl_mem src_images_buf_ = nullptr;     // uchar src_max_w*src_max_h*num_src
  cl_mem src_depth_buf_ = nullptr;      // float src_max_w*src_max_h*num_src (geom)
  cl_mem poses_buf_ = nullptr;          // float 4*num_src*43
  cl_mem ref_inv_K_buf_ = nullptr;      // float 16
  cl_mem ref_K_buf_ = nullptr;          // float 16 (geom)
  cl_mem bilateral_spatial_buf_ = nullptr;
  cl_mem bilateral_color_buf_ = nullptr;

  // Results (filled by Run()).
  DepthMap result_depth_;
  NormalMap result_normal_;
  Mat<float> result_sel_prob_;
  Mat<uint8_t> result_cmask_;
};

}  // namespace mvs
}  // namespace colmap
