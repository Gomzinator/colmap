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

#include <memory>
#include <string>
#include <vector>

namespace colmap {
namespace mvs {

// Abstract interface for PatchMatch stereo compute backends. Implementations
// compute depth, normal, and selection probability maps for one reference
// image given a set of source images (see PatchMatch::Problem).
class PatchMatchBackend {
 public:
  virtual ~PatchMatchBackend() = default;

  // Run the PatchMatch optimization.
  virtual void Run() = 0;

  // Get the computed values after running the algorithm.
  virtual DepthMap GetDepthMap() const = 0;
  virtual NormalMap GetNormalMap() const = 0;
  virtual Mat<float> GetSelProbMap() const = 0;
  virtual std::vector<int> GetConsistentImageIdxs() const = 0;
};

enum class PatchMatchBackendType {
  kCuda,
  kCpu,
  kOpenCL,
};

// Resolve the backend type from the option string {auto, cuda, cpu, opencl}.
// For "auto", CUDA is selected if compiled in and at least one device is
// available, otherwise the CPU backend is selected (the OpenCL backend is not
// auto-selected yet, since it is still being brought up). Throws for invalid
// values or if an explicitly requested backend is unavailable.
PatchMatchBackendType ResolvePatchMatchBackend(const std::string& backend);

}  // namespace mvs
}  // namespace colmap
