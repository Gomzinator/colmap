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

#include "colmap/mvs/patch_match_opencl.h"

#include "colmap/util/logging.h"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace colmap {
namespace mvs {
namespace {

// Map a subset of the common OpenCL error codes to a readable name; fall back
// to the numeric code for anything not listed.
std::string CLErrorString(cl_int err) {
  switch (err) {
    case CL_SUCCESS:
      return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND:
      return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE:
      return "CL_DEVICE_NOT_AVAILABLE";
    case CL_OUT_OF_RESOURCES:
      return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY:
      return "CL_OUT_OF_HOST_MEMORY";
    case CL_INVALID_VALUE:
      return "CL_INVALID_VALUE";
    case CL_INVALID_PLATFORM:
      return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE:
      return "CL_INVALID_DEVICE";
    case CL_INVALID_DEVICE_TYPE:
      return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_CONTEXT:
      return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES:
      return "CL_INVALID_QUEUE_PROPERTIES";
    default:
      return "CL error " + std::to_string(err);
  }
}

void CheckCL(cl_int err, const std::string& what) {
  if (err != CL_SUCCESS) {
    LOG(FATAL_THROW) << "OpenCL error in " << what << ": "
                     << CLErrorString(err);
  }
}

template <typename T>
T GetDeviceInfo(cl_device_id device, cl_device_info param) {
  T value{};
  CheckCL(clGetDeviceInfo(device, param, sizeof(T), &value, nullptr),
          "clGetDeviceInfo");
  return value;
}

std::string GetDeviceInfoString(cl_device_id device, cl_device_info param) {
  size_t size = 0;
  if (clGetDeviceInfo(device, param, 0, nullptr, &size) != CL_SUCCESS ||
      size == 0) {
    return std::string();
  }
  std::string value(size, '\0');
  if (clGetDeviceInfo(device, param, size, value.data(), nullptr) !=
      CL_SUCCESS) {
    return std::string();
  }
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();  // Drop the trailing NUL the API includes in `size`.
  }
  return value;
}

std::string GetPlatformInfoString(cl_platform_id platform,
                                  cl_platform_info param) {
  size_t size = 0;
  if (clGetPlatformInfo(platform, param, 0, nullptr, &size) != CL_SUCCESS ||
      size == 0) {
    return std::string();
  }
  std::string value(size, '\0');
  if (clGetPlatformInfo(platform, param, size, value.data(), nullptr) !=
      CL_SUCCESS) {
    return std::string();
  }
  if (!value.empty() && value.back() == '\0') {
    value.pop_back();
  }
  return value;
}

std::string ToLowerCopy(std::string str) {
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return str;
}

bool ContainsCI(const std::string& haystack, const std::string& needle_lower) {
  return ToLowerCopy(haystack).find(needle_lower) != std::string::npos;
}

}  // namespace

PatchMatchOpenCL::PatchMatchOpenCL(const PatchMatchOptions& options,
                                   const PatchMatch::Problem& problem)
    : options_(options), problem_(problem) {
  InitDevice();
}

PatchMatchOpenCL::~PatchMatchOpenCL() {
  if (queue_ != nullptr) {
    clReleaseCommandQueue(queue_);
  }
  if (context_ != nullptr) {
    clReleaseContext(context_);
  }
}

void PatchMatchOpenCL::InitDevice() {
  cl_uint num_platforms = 0;
  const cl_int platforms_err = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (platforms_err != CL_SUCCESS || num_platforms == 0) {
    LOG(FATAL_THROW) << "No OpenCL platforms found (clGetPlatformIDs: "
                     << CLErrorString(platforms_err) << ").";
  }
  std::vector<cl_platform_id> platforms(num_platforms);
  CheckCL(clGetPlatformIDs(num_platforms, platforms.data(), nullptr),
          "clGetPlatformIDs");

  // Enumerate all GPU devices with image support. Prefer the native Qualcomm
  // platform; always skip the OpenCLOn12 D3D12-translation platform (which
  // also exposes a Microsoft Basic Render CPU fallback).
  struct Candidate {
    cl_platform_id platform;
    cl_device_id device;
    bool qualcomm;
  };
  std::vector<Candidate> candidates;

  for (cl_platform_id platform : platforms) {
    const std::string platform_name =
        GetPlatformInfoString(platform, CL_PLATFORM_NAME);
    const std::string platform_version =
        GetPlatformInfoString(platform, CL_PLATFORM_VERSION);
    const bool is_on12 = ContainsCI(platform_name, "openclon12");
    const bool is_qualcomm = ContainsCI(platform_name, "qualcomm") ||
                             ContainsCI(platform_name, "snapdragon");

    LOG(INFO) << "OpenCL platform: \"" << platform_name << "\" ("
              << platform_version << ")"
              << (is_on12 ? "  [skipped: D3D12 translation layer]" : "");
    if (is_on12) {
      continue;
    }

    cl_uint num_devices = 0;
    if (clGetDeviceIDs(
            platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) !=
            CL_SUCCESS ||
        num_devices == 0) {
      continue;
    }
    std::vector<cl_device_id> devices(num_devices);
    CheckCL(clGetDeviceIDs(platform,
                           CL_DEVICE_TYPE_GPU,
                           num_devices,
                           devices.data(),
                           nullptr),
            "clGetDeviceIDs");

    for (cl_device_id device : devices) {
      const cl_bool image_support =
          GetDeviceInfo<cl_bool>(device, CL_DEVICE_IMAGE_SUPPORT);
      LOG(INFO) << "  GPU device: \""
                << GetDeviceInfoString(device, CL_DEVICE_NAME)
                << "\"  image_support="
                << (image_support ? "yes" : "no");
      if (image_support == CL_FALSE) {
        // The PatchMatch kernels (Phase 2b) require image2d_array sampling.
        continue;
      }
      candidates.push_back({platform, device, is_qualcomm});
    }
  }

  if (candidates.empty()) {
    LOG(FATAL_THROW) << "No suitable OpenCL GPU device with image support was "
                        "found (the OpenCLOn12 translation layer is excluded). "
                        "On Snapdragon, ensure the Adreno OpenCL ICD is "
                        "installed.";
  }

  const Candidate* chosen = nullptr;
  for (const Candidate& candidate : candidates) {
    if (candidate.qualcomm) {
      chosen = &candidate;
      break;
    }
  }
  if (chosen == nullptr) {
    chosen = &candidates.front();
    LOG(WARNING) << "No Qualcomm OpenCL platform found; falling back to the "
                    "first GPU device with image support.";
  }
  platform_ = chosen->platform;
  device_ = chosen->device;

  // Report the selected device for reproducibility.
  LOG(INFO) << "Selected OpenCL device:";
  LOG(INFO) << "  platform:       "
            << GetPlatformInfoString(platform_, CL_PLATFORM_NAME);
  LOG(INFO) << "  device:         "
            << GetDeviceInfoString(device_, CL_DEVICE_NAME);
  LOG(INFO) << "  OpenCL version: "
            << GetDeviceInfoString(device_, CL_DEVICE_VERSION);
  LOG(INFO) << "  driver version: "
            << GetDeviceInfoString(device_, CL_DRIVER_VERSION);
  LOG(INFO) << "  compute units:  "
            << GetDeviceInfo<cl_uint>(device_, CL_DEVICE_MAX_COMPUTE_UNITS);
  LOG(INFO) << "  global memory:  "
            << (GetDeviceInfo<cl_ulong>(device_, CL_DEVICE_GLOBAL_MEM_SIZE) >>
                20)
            << " MB";
  LOG(INFO) << "  local memory:   "
            << (GetDeviceInfo<cl_ulong>(device_, CL_DEVICE_LOCAL_MEM_SIZE) >> 10)
            << " KB";
  LOG(INFO) << "  max work-group: "
            << GetDeviceInfo<size_t>(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE);
  LOG(INFO) << "  image support:  "
            << (GetDeviceInfo<cl_bool>(device_, CL_DEVICE_IMAGE_SUPPORT)
                    ? "yes"
                    : "no");

  // Create a context and command queue on the selected device. This exercises
  // the ICD path now (Phase 2a), so any loader/driver problem surfaces at
  // device selection rather than later when kernels are added (Phase 2b).
  cl_int err = CL_SUCCESS;
  const cl_context_properties context_props[] = {
      CL_CONTEXT_PLATFORM,
      reinterpret_cast<cl_context_properties>(platform_),
      0};
  context_ = clCreateContext(
      context_props, 1, &device_, nullptr, nullptr, &err);
  CheckCL(err, "clCreateContext");

  queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr, &err);
  CheckCL(err, "clCreateCommandQueueWithProperties");
}

void PatchMatchOpenCL::Run() {
  LOG(FATAL_THROW)
      << "OpenCL kernels not yet implemented. Phase 2a is the OpenCL backend "
         "scaffold (device discovery only); the selected GPU device has been "
         "reported above. The photometric/geometric PatchMatch sweep kernels "
         "are Phase 2b. Use --PatchMatchStereo.backend cpu for a working "
         "CUDA-less dense reconstruction in the meantime.";
}

DepthMap PatchMatchOpenCL::GetDepthMap() const { return DepthMap(); }

NormalMap PatchMatchOpenCL::GetNormalMap() const { return NormalMap(); }

Mat<float> PatchMatchOpenCL::GetSelProbMap() const { return Mat<float>(); }

std::vector<int> PatchMatchOpenCL::GetConsistentImageIdxs() const {
  return std::vector<int>();
}

}  // namespace mvs
}  // namespace colmap
