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
// PHASE 2a SCAFFOLD: this discovers, selects and reports a GPU device and
// creates a context + command queue on it, but does not yet run any kernels;
// Run() throws. The photometric/geometric sweep kernels are Phase 2b+.
//
// Device selection prefers the native Qualcomm platform (name contains
// "QUALCOMM"/"SNAPDRAGON") and a GPU device with image support, and explicitly
// avoids the D3D12-translation "OpenCLOn12" platform.
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

  const PatchMatchOptions options_;
  const PatchMatch::Problem problem_;

  // Selected OpenCL handles (owned; released in the destructor).
  cl_platform_id platform_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
};

}  // namespace mvs
}  // namespace colmap
