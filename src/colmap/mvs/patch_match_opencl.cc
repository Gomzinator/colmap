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

#include "colmap/mvs/image.h"
#include "colmap/mvs/patch_match_opencl_kernels.h"
#include "colmap/util/logging.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// cl_khr_priority_hints / cl_khr_throttle_hints queue-property values. Defined
// here (with guards) so we don't depend on a particular cl_ext.h variant; they
// are only passed when the device advertises the corresponding extension.
#ifndef CL_QUEUE_PRIORITY_KHR
#define CL_QUEUE_PRIORITY_KHR 0x1096
#endif
#ifndef CL_QUEUE_PRIORITY_LOW_KHR
#define CL_QUEUE_PRIORITY_LOW_KHR (1 << 2)
#endif
#ifndef CL_QUEUE_THROTTLE_KHR
#define CL_QUEUE_THROTTLE_KHR 0x1097
#endif
#ifndef CL_QUEUE_THROTTLE_LOW_KHR
#define CL_QUEUE_THROTTLE_LOW_KHR (1 << 2)
#endif

namespace colmap {
namespace mvs {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = 0.0174532925199432f;
constexpr int kNumTformParams = 4 + 9 + 3 + 3 + 12 + 12;  // 43
constexpr int kNumIntensityBins = 256;
constexpr float kInvMaxIntensity = 1.0f / 255.0f;
// Default number of columns processed per sweep kernel launch. This is the
// main throughput knob: too small starves the GPU of parallel work-items (each
// column is one work-item) and tanks throughput; large keeps occupancy up but
// lengthens each dispatch (and the desktop stalls for the dispatch duration,
// since the Adreno is the display GPU and the driver exposes no preemption
// hint). 48 is a throughput/latency balance; lower it if dispatch stalls are
// annoying (costs speed). Override with COLMAP_OPENCL_SWEEP_BAND. Also keeps
// each dispatch below the Windows TDR watchdog.
constexpr int kDefaultSweepColumnBand = 48;
// Default GPU duty-cycle. Vanilla behavior is full speed (1.0 = GPU ~100%, like
// the CUDA backend); throttling is opt-in. Set COLMAP_OPENCL_DUTY < 1 (e.g. 0.5
// from a launch script) to make the host sleep band_ms*(1-duty)/duty after each
// sweep band, leaving the shared Adreno idle ~(1-duty) of the time so the
// desktop stays responsive.
constexpr float kDefaultGpuDuty = 1.0f;
// Default per-work-item (per-column) work budget for ONE forward-sweep dispatch,
// in photo-cost window-samples. The Adreno silently garbages a sweep work-item
// above ~1.5M such samples per column (see OPENCL_LARGE_IMAGE_BUG.md); we chunk
// the per-column sweep over the row dimension so each launch stays under this.
// ~1.0M leaves margin below the empirical threshold (1.40M was still clean).
// Override with COLMAP_OPENCL_ROW_CHUNK_WORK, or set rows directly with
// COLMAP_OPENCL_ROW_CHUNK.
constexpr int kDefaultRowChunkWork = 1000000;

int EnvInt(const char* name, int fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr) return fallback;
  const int parsed = std::atoi(v);
  return parsed > 0 ? parsed : fallback;
}

float EnvFloat(const char* name, float fallback) {
  const char* v = std::getenv(name);
  if (v == nullptr) return fallback;
  const float parsed = static_cast<float>(std::atof(v));
  return (parsed > 0.0f && parsed <= 1.0f) ? parsed : fallback;
}

std::string CLErrorString(cl_int err) {
  switch (err) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    default: return "CL error " + std::to_string(err);
  }
}

void CheckCL(cl_int err, const std::string& what) {
  if (err != CL_SUCCESS) {
    LOG(FATAL_THROW) << "OpenCL error in " << what << ": " << CLErrorString(err);
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
    value.pop_back();
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

void SetArg(cl_kernel kernel, cl_uint index, size_t size, const void* value) {
  CheckCL(clSetKernelArg(kernel, index, size, value), "clSetKernelArg");
}

// Enqueue a 2D kernel over a (w x h) domain with an EXPLICIT local work-group
// size and the global size padded up to a multiple of it. The kernels bounds-
// check (col>=w || row>=h => return), so padded work-items are no-ops. This
// avoids relying on the driver's local-size choice for a NULL local, which on
// the Adreno silently drops work-items for awkward dimensions (e.g. height 598)
// and corrupts the maps.
void Enqueue2D(cl_command_queue queue,
               cl_device_id device,
               cl_kernel kernel,
               int w,
               int h,
               const std::string& what) {
  size_t kernel_wg = 0;
  CheckCL(clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE,
                                   sizeof(size_t), &kernel_wg, nullptr),
          "clGetKernelWorkGroupInfo");
  size_t l = 16;
  while (l > 1 && l * l > kernel_wg) {
    l /= 2;
  }
  const size_t lws[2] = {l, l};
  const size_t gws[2] = {((static_cast<size_t>(w) + l - 1) / l) * l,
                         ((static_cast<size_t>(h) + l - 1) / l) * l};
  CheckCL(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, gws, lws, 0, nullptr,
                                 nullptr),
          what);
}

cl_mem CreateBuffer(cl_context context,
                    cl_mem_flags flags,
                    size_t bytes,
                    const void* host_ptr) {
  cl_int err = CL_SUCCESS;
  cl_mem mem = clCreateBuffer(
      context, flags, bytes, const_cast<void*>(host_ptr), &err);
  CheckCL(err, "clCreateBuffer");
  return mem;
}

////////////////////////////////////////////////////////////////////////////////
// Host PCG32 (identical to patch_match_cpu.cc) for the random initialization,
// so the OpenCL backend starts from the same depth/normal/PRNG state.
////////////////////////////////////////////////////////////////////////////////

struct HostPcg32 {
  uint64_t state = 0;
  uint64_t inc = 0;
};

uint64_t SplitMix64(uint64_t& x) {
  x += 0x9e3779b97f4a7c15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

void SeedPcg32(uint64_t seed, HostPcg32* state) {
  uint64_t sm = seed;
  state->state = SplitMix64(sm);
  state->inc = SplitMix64(sm) | 1ULL;
}

uint32_t Pcg32Next(HostPcg32* state) {
  const uint64_t old_state = state->state;
  state->state = old_state * 6364136223846793005ULL + state->inc;
  const uint32_t xorshifted =
      static_cast<uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
  const uint32_t rot = static_cast<uint32_t>(old_state >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

float RandUniform(HostPcg32* state) {
  return ((Pcg32Next(state) >> 8) + 1) * 0x1p-24f;
}

float GenRandDepth(float depth_min, float depth_max, HostPcg32* state) {
  return RandUniform(state) * (depth_max - depth_min) + depth_min;
}

void GenRandNormal(int row,
                   int col,
                   const float ref_inv_K[4],
                   HostPcg32* state,
                   float normal[3]) {
  float v1 = 0.0f;
  float v2 = 0.0f;
  float s = 2.0f;
  while (s >= 1.0f) {
    v1 = 2.0f * RandUniform(state) - 1.0f;
    v2 = 2.0f * RandUniform(state) - 1.0f;
    s = v1 * v1 + v2 * v2;
  }
  const float s_norm = std::sqrt(1.0f - s);
  normal[0] = 2.0f * v1 * s_norm;
  normal[1] = 2.0f * v2 * s_norm;
  normal[2] = 1.0f - 2.0f * s;
  const float view_ray[3] = {ref_inv_K[0] * col + ref_inv_K[1],
                             ref_inv_K[2] * row + ref_inv_K[3], 1.0f};
  if (normal[0] * view_ray[0] + normal[1] * view_ray[1] +
          normal[2] * view_ray[2] >
      0.0f) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }
}

}  // namespace

PatchMatchOpenCL::PatchMatchOpenCL(const PatchMatchOptions& options,
                                   const PatchMatch::Problem& problem)
    : options_(options), problem_(problem) {
  const Image& ref_image = problem_.images->at(problem_.ref_image_idx);
  ref_width_ = static_cast<int>(ref_image.GetWidth());
  ref_height_ = static_cast<int>(ref_image.GetHeight());
  num_src_ = static_cast<int>(problem_.src_image_idxs.size());
  window_radius_ = options_.window_radius;
  window_step_ = options_.window_step;
  cur_width_ = ref_width_;
  cur_height_ = ref_height_;
  rotation_ = 0;

  if (num_src_ > 32) {
    LOG(FATAL_THROW) << "The OpenCL backend supports at most 32 source images "
                        "per problem (got "
                     << num_src_ << "); reduce --PatchMatchStereo.* source "
                        "image count.";
  }

  // NOTE: the earlier "~32 MB Adreno single-buffer addressing limit" was WRONG
  // (S6's premise). S7 ctypes tests proved no buffer/malloc ceiling (single
  // 512 MB and 8x64 MB concurrent buffers verified clean). The real large-image
  // failure was per-WORK-ITEM (per-column) WORK in the sweep, now bounded by
  // row-chunking in RunSweeps. No buffer-size guard is needed: an allocation
  // beyond the device's max single allocation is simply rejected by
  // clCreateBuffer (CheckCL throws).

  InitDevice();
  BuildProgram();
  InitHostDataAndUpload();
}

PatchMatchOpenCL::~PatchMatchOpenCL() {
  const cl_mem buffers[] = {
      depth_buf_,        normal_buf_,    ref_img_buf_,       ref_sum_buf_,
      ref_sqsum_buf_,    cost_buf_,      rand_buf_,          sel_prob_buf_,
      prev_sel_prob_buf_, cmask_buf_,    scratch_buf_,       src_images_buf_,
      src_depth_buf_,    poses_buf_,     ref_inv_K_buf_,     ref_K_buf_,
      bilateral_spatial_buf_, bilateral_color_buf_,
      fwd_prev_depth_buf_, fwd_prev_normal_buf_, fwd_message_buf_};
  for (cl_mem buf : buffers) {
    if (buf != nullptr) clReleaseMemObject(buf);
  }
  const cl_kernel kernels[] = {k_initial_cost_, k_sweep_bwd_,  k_sweep_fwd_,
                               k_rot_f_,        k_rot_u8_,     k_rot_rand_,
                               k_rot_normal_};
  for (cl_kernel kernel : kernels) {
    if (kernel != nullptr) clReleaseKernel(kernel);
  }
  if (program_ != nullptr) clReleaseProgram(program_);
  if (queue_ != nullptr) clReleaseCommandQueue(queue_);
  if (context_ != nullptr) clReleaseContext(context_);
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
    CheckCL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices,
                           devices.data(), nullptr),
            "clGetDeviceIDs");
    for (cl_device_id device : devices) {
      const cl_bool image_support =
          GetDeviceInfo<cl_bool>(device, CL_DEVICE_IMAGE_SUPPORT);
      if (image_support == CL_FALSE) {
        continue;
      }
      candidates.push_back({platform, device, is_qualcomm});
    }
  }

  if (candidates.empty()) {
    LOG(FATAL_THROW) << "No suitable OpenCL GPU device with image support was "
                        "found (the OpenCLOn12 translation layer is excluded).";
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

  LOG(INFO) << "Selected OpenCL device: "
            << GetDeviceInfoString(device_, CL_DEVICE_NAME) << " ("
            << GetDeviceInfoString(device_, CL_DEVICE_VERSION) << ", "
            << GetDeviceInfo<cl_uint>(device_, CL_DEVICE_MAX_COMPUTE_UNITS)
            << " CUs, "
            << (GetDeviceInfo<cl_ulong>(device_, CL_DEVICE_LOCAL_MEM_SIZE) >> 10)
            << " KB local)";

  cl_int err = CL_SUCCESS;
  const cl_context_properties context_props[] = {
      CL_CONTEXT_PLATFORM,
      reinterpret_cast<cl_context_properties>(platform_), 0};
  context_ = clCreateContext(context_props, 1, &device_, nullptr, nullptr, &err);
  CheckCL(err, "clCreateContext");

  // Request a low-priority / low-throttle command queue when the device
  // advertises the hints, so the GPU scheduler favors the desktop compositor
  // (the Adreno is the display GPU). Only passed when supported, otherwise
  // clCreateCommandQueueWithProperties would return CL_INVALID_QUEUE_PROPERTIES.
  const std::string extensions =
      GetDeviceInfoString(device_, CL_DEVICE_EXTENSIONS);
  const bool no_hints = std::getenv("COLMAP_OPENCL_NO_QUEUE_HINTS") != nullptr;
  std::vector<cl_queue_properties> queue_props;
  bool priority_low = false;
  bool throttle_low = false;
  if (!no_hints) {
    if (extensions.find("cl_khr_priority_hints") != std::string::npos) {
      queue_props.push_back(CL_QUEUE_PRIORITY_KHR);
      queue_props.push_back(CL_QUEUE_PRIORITY_LOW_KHR);
      priority_low = true;
    }
    if (extensions.find("cl_khr_throttle_hints") != std::string::npos) {
      queue_props.push_back(CL_QUEUE_THROTTLE_KHR);
      queue_props.push_back(CL_QUEUE_THROTTLE_LOW_KHR);
      throttle_low = true;
    }
  }
  queue_props.push_back(0);
  queue_ = clCreateCommandQueueWithProperties(context_, device_,
                                              queue_props.data(), &err);
  if (err != CL_SUCCESS && (priority_low || throttle_low)) {
    // Fall back to a default queue if the hints are rejected.
    LOG(WARNING) << "OpenCL low-priority queue rejected (" << CLErrorString(err)
                 << "); using a default-priority queue.";
    queue_ = clCreateCommandQueueWithProperties(context_, device_, nullptr,
                                                &err);
    priority_low = throttle_low = false;
  }
  CheckCL(err, "clCreateCommandQueueWithProperties");
  LOG(INFO) << "OpenCL queue scheduling hints: priority_low=" << priority_low
            << " throttle_low=" << throttle_low
            << (priority_low || throttle_low
                    ? ""
                    : " (not advertised by the device)");
}

void PatchMatchOpenCL::BuildProgram() {
  cl_int err = CL_SUCCESS;
  const char* source = kPatchMatchOpenCLKernelSource;
  const size_t length = std::strlen(source);
  program_ = clCreateProgramWithSource(context_, 1, &source, &length, &err);
  CheckCL(err, "clCreateProgramWithSource");

  const char* build_options = "-cl-std=CL1.2";
  err = clBuildProgram(program_, 1, &device_, build_options, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                          &log_size);
    std::string log(log_size, '\0');
    clGetProgramBuildInfo(program_, device_, CL_PROGRAM_BUILD_LOG, log_size,
                          log.data(), nullptr);
    LOG(FATAL_THROW) << "OpenCL kernel build failed (" << CLErrorString(err)
                     << "):\n" << log;
  }

  k_initial_cost_ = clCreateKernel(program_, "compute_initial_cost", &err);
  CheckCL(err, "clCreateKernel(compute_initial_cost)");
  k_sweep_bwd_ = clCreateKernel(program_, "sweep_backward", &err);
  CheckCL(err, "clCreateKernel(sweep_backward)");
  k_sweep_fwd_ = clCreateKernel(program_, "sweep_forward", &err);
  CheckCL(err, "clCreateKernel(sweep_forward)");
  k_rot_f_ = clCreateKernel(program_, "rotate_ccw_f", &err);
  CheckCL(err, "clCreateKernel(rotate_ccw_f)");
  k_rot_u8_ = clCreateKernel(program_, "rotate_ccw_u8", &err);
  CheckCL(err, "clCreateKernel(rotate_ccw_u8)");
  k_rot_rand_ = clCreateKernel(program_, "rotate_ccw_rand", &err);
  CheckCL(err, "clCreateKernel(rotate_ccw_rand)");
  k_rot_normal_ = clCreateKernel(program_, "rotate_ccw_normal", &err);
  CheckCL(err, "clCreateKernel(rotate_ccw_normal)");
}

void PatchMatchOpenCL::InitHostDataAndUpload() {
  const int W = ref_width_;
  const int H = ref_height_;
  const int wr = window_radius_;
  const int ws = window_step_;
  const int window_size = 2 * wr + 1;

  //////////////////////////////////////////////////////////////////////////////
  // Bilateral weight tables.
  //////////////////////////////////////////////////////////////////////////////
  const float spatial_norm =
      1.0f / (2.0f * options_.sigma_spatial * options_.sigma_spatial);
  const float color_norm =
      1.0f / (2.0f * options_.sigma_color * options_.sigma_color);
  std::vector<float> bilateral_spatial(window_size * window_size);
  for (int dr = -wr; dr <= wr; ++dr) {
    for (int dc = -wr; dc <= wr; ++dc) {
      const float d2 = static_cast<float>(dr * dr + dc * dc);
      bilateral_spatial[(dr + wr) * window_size + (dc + wr)] =
          std::exp(-d2 * spatial_norm);
    }
  }
  std::vector<float> bilateral_color(kNumIntensityBins);
  for (int d = 0; d < kNumIntensityBins; ++d) {
    const float cd = d * kInvMaxIntensity;
    bilateral_color[d] = std::exp(-cd * cd * color_norm);
  }

  //////////////////////////////////////////////////////////////////////////////
  // Reference image bilateral local sums (mirror PatchMatchCpu::FilterRefImage).
  //////////////////////////////////////////////////////////////////////////////
  const Image& ref_image = problem_.images->at(problem_.ref_image_idx);
  const std::vector<uint8_t>& ref_data = ref_image.GetBitmap().RowMajorData();
  std::vector<uint8_t> ref_img_host(W * H);
  std::vector<float> ref_sum_host(W * H);
  std::vector<float> ref_sqsum_host(W * H);
  for (int row = 0; row < H; ++row) {
    for (int col = 0; col < W; ++col) {
      const int center_q = ref_data[row * W + col];
      float color_sum = 0.0f;
      float color_sq_sum = 0.0f;
      float weight_sum = 0.0f;
      for (int wrow = -wr; wrow <= wr; wrow += ws) {
        for (int wcol = -wr; wcol <= wr; wcol += ws) {
          const int r = row + wrow;
          const int c = col + wcol;
          const int q = (r >= 0 && r < H && c >= 0 && c < W) ? ref_data[r * W + c]
                                                             : 0;
          const float color = q * kInvMaxIntensity;
          const float weight =
              bilateral_spatial[(wrow + wr) * window_size + (wcol + wr)] *
              bilateral_color[std::abs(center_q - q)];
          color_sum += weight * color;
          color_sq_sum += weight * color * color;
          weight_sum += weight;
        }
      }
      ref_img_host[row * W + col] = static_cast<uint8_t>(center_q);
      ref_sum_host[row * W + col] = color_sum / weight_sum;
      ref_sqsum_host[row * W + col] = color_sq_sum / weight_sum;
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Source images, zero-padded to common max dimensions (layer-major).
  //////////////////////////////////////////////////////////////////////////////
  int max_w = 0;
  int max_h = 0;
  for (const int image_idx : problem_.src_image_idxs) {
    const Image& image = problem_.images->at(image_idx);
    max_w = std::max(max_w, static_cast<int>(image.GetWidth()));
    max_h = std::max(max_h, static_cast<int>(image.GetHeight()));
  }
  src_max_w_ = max_w;
  src_max_h_ = max_h;
  std::vector<uint8_t> src_images_host(
      static_cast<size_t>(max_w) * max_h * num_src_, 0);
  for (int i = 0; i < num_src_; ++i) {
    const Image& image = problem_.images->at(problem_.src_image_idxs[i]);
    const std::vector<uint8_t>& src = image.GetBitmap().RowMajorData();
    uint8_t* dest =
        src_images_host.data() + static_cast<size_t>(max_w) * max_h * i;
    for (int r = 0; r < static_cast<int>(image.GetHeight()); ++r) {
      std::memcpy(dest + static_cast<size_t>(r) * max_w,
                  src.data() + static_cast<size_t>(r) * image.GetWidth(),
                  image.GetWidth());
    }
  }

  // Source depth maps for the geometric-consistency term (layer-major,
  // zero-padded), mirroring PatchMatchCpu::InitSourceImages.
  std::vector<float> src_depth_host;
  if (options_.geom_consistency) {
    src_depth_host.assign(static_cast<size_t>(max_w) * max_h * num_src_, 0.0f);
    for (int i = 0; i < num_src_; ++i) {
      const DepthMap& dm = problem_.depth_maps->at(problem_.src_image_idxs[i]);
      float* dest =
          src_depth_host.data() + static_cast<size_t>(max_w) * max_h * i;
      for (int r = 0; r < static_cast<int>(dm.GetHeight()); ++r) {
        std::memcpy(dest + static_cast<size_t>(r) * max_w,
                    dm.GetPtr() + static_cast<size_t>(r) * dm.GetWidth(),
                    dm.GetWidth() * sizeof(float));
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Transforms: per-rotation reference calibration and source poses
  // (mirror PatchMatchCpu::InitTransforms).
  //////////////////////////////////////////////////////////////////////////////
  for (int i = 0; i < 4; ++i) {
    ref_K_host_[i][0] = ref_image.GetK()[0];
    ref_K_host_[i][1] = ref_image.GetK()[2];
    ref_K_host_[i][2] = ref_image.GetK()[4];
    ref_K_host_[i][3] = ref_image.GetK()[5];
  }
  std::swap(ref_K_host_[1][0], ref_K_host_[1][2]);
  std::swap(ref_K_host_[1][1], ref_K_host_[1][3]);
  ref_K_host_[1][3] = ref_width_ - 1 - ref_K_host_[1][3];
  ref_K_host_[2][1] = ref_width_ - 1 - ref_K_host_[2][1];
  ref_K_host_[2][3] = ref_height_ - 1 - ref_K_host_[2][3];
  std::swap(ref_K_host_[3][0], ref_K_host_[3][2]);
  std::swap(ref_K_host_[3][1], ref_K_host_[3][3]);
  ref_K_host_[3][1] = ref_height_ - 1 - ref_K_host_[3][1];
  for (int i = 0; i < 4; ++i) {
    ref_inv_K_host_[i][0] = 1.0f / ref_K_host_[i][0];
    ref_inv_K_host_[i][1] = -ref_K_host_[i][1] / ref_K_host_[i][0];
    ref_inv_K_host_[i][2] = 1.0f / ref_K_host_[i][2];
    ref_inv_K_host_[i][3] = -ref_K_host_[i][3] / ref_K_host_[i][2];
  }

  float rotated_R[9];
  std::memcpy(rotated_R, ref_image.GetR(), 9 * sizeof(float));
  float rotated_T[3];
  std::memcpy(rotated_T, ref_image.GetT(), 3 * sizeof(float));
  const float R_z90[9] = {0, 1, 0, -1, 0, 0, 0, 0, 1};

  std::vector<float> poses_host(
      static_cast<size_t>(4) * num_src_ * kNumTformParams, 0.0f);
  for (int i = 0; i < 4; ++i) {
    for (int s = 0; s < num_src_; ++s) {
      const Image& image = problem_.images->at(problem_.src_image_idxs[s]);
      float* p = poses_host.data() +
                 (static_cast<size_t>(i) * num_src_ + s) * kNumTformParams;
      int off = 0;
      const float K[4] = {image.GetK()[0], image.GetK()[2], image.GetK()[4],
                          image.GetK()[5]};
      std::memcpy(p + off, K, 4 * sizeof(float));
      off += 4;
      float rel_R[9];
      float rel_T[3];
      ComputeRelativePose(rotated_R, rotated_T, image.GetR(), image.GetT(),
                          rel_R, rel_T);
      std::memcpy(p + off, rel_R, 9 * sizeof(float));
      off += 9;
      std::memcpy(p + off, rel_T, 3 * sizeof(float));
      off += 3;
      float C[3];
      ComputeProjectionCenter(rel_R, rel_T, C);
      std::memcpy(p + off, C, 3 * sizeof(float));
      off += 3;
      float P[12];
      ComposeProjectionMatrix(image.GetK(), rel_R, rel_T, P);
      std::memcpy(p + off, P, 12 * sizeof(float));
      off += 12;
      float inv_P[12];
      ComposeInverseProjectionMatrix(image.GetK(), rel_R, rel_T, inv_P);
      std::memcpy(p + off, inv_P, 12 * sizeof(float));
      off += 12;
    }
    RotatePose(R_z90, rotated_R, rotated_T);
  }

  std::vector<float> ref_inv_K_flat(16);
  std::vector<float> ref_K_flat(16);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      ref_inv_K_flat[i * 4 + j] = ref_inv_K_host_[i][j];
      ref_K_flat[i * 4 + j] = ref_K_host_[i][j];
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Random initialization of depth / normal / PRNG states (slice-major), then
  // consume the per-pixel states exactly as the CPU does.
  //////////////////////////////////////////////////////////////////////////////
  std::vector<uint64_t> rand_host(static_cast<size_t>(2) * W * H);
  std::vector<float> depth_host(static_cast<size_t>(W) * H);
  std::vector<float> normal_host(static_cast<size_t>(3) * W * H);
  const size_t plane = static_cast<size_t>(W) * H;
  if (options_.geom_consistency) {
    // Geometric pass: initialize depth/normal from the photometric maps (loaded
    // by the controller) and seed the PRNG states without consuming them, like
    // PatchMatchCpu::InitWorkspaceMemory.
    const DepthMap& init_depth =
        problem_.depth_maps->at(problem_.ref_image_idx);
    const NormalMap& init_normal =
        problem_.normal_maps->at(problem_.ref_image_idx);
    std::memcpy(depth_host.data(), init_depth.GetPtr(), plane * sizeof(float));
    std::memcpy(normal_host.data(), init_normal.GetPtr(),
                3 * plane * sizeof(float));
    for (int row = 0; row < H; ++row) {
      for (int col = 0; col < W; ++col) {
        const size_t idx = static_cast<size_t>(row) * W + col;
        HostPcg32 st;
        SeedPcg32(idx, &st);
        rand_host[2 * idx + 0] = st.state;
        rand_host[2 * idx + 1] = st.inc;
      }
    }
  } else {
    for (int row = 0; row < H; ++row) {
      for (int col = 0; col < W; ++col) {
        const size_t idx = static_cast<size_t>(row) * W + col;
        HostPcg32 st;
        SeedPcg32(idx, &st);
        depth_host[idx] =
            GenRandDepth(options_.depth_min, options_.depth_max, &st);
        float normal[3];
        GenRandNormal(row, col, ref_inv_K_host_[0], &st, normal);
        normal_host[0 * plane + idx] = normal[0];
        normal_host[1 * plane + idx] = normal[1];
        normal_host[2 * plane + idx] = normal[2];
        rand_host[2 * idx + 0] = st.state;
        rand_host[2 * idx + 1] = st.inc;
      }
    }
  }

  //////////////////////////////////////////////////////////////////////////////
  // Allocate and upload device buffers.
  //////////////////////////////////////////////////////////////////////////////
  const cl_mem_flags rw = CL_MEM_READ_WRITE;
  const cl_mem_flags ro_copy = CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
  const cl_mem_flags rw_copy = CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR;

  depth_buf_ = CreateBuffer(context_, rw_copy, plane * sizeof(float),
                            depth_host.data());
  normal_buf_ = CreateBuffer(context_, rw_copy, 3 * plane * sizeof(float),
                             normal_host.data());
  ref_img_buf_ =
      CreateBuffer(context_, rw_copy, plane * sizeof(uint8_t), ref_img_host.data());
  ref_sum_buf_ = CreateBuffer(context_, rw_copy, plane * sizeof(float),
                              ref_sum_host.data());
  ref_sqsum_buf_ = CreateBuffer(context_, rw_copy, plane * sizeof(float),
                                ref_sqsum_host.data());
  cost_buf_ =
      CreateBuffer(context_, rw, static_cast<size_t>(num_src_) * plane *
                                     sizeof(float), nullptr);
  rand_buf_ = CreateBuffer(context_, rw_copy, plane * 2 * sizeof(uint64_t),
                           rand_host.data());
  sel_prob_buf_ = CreateBuffer(
      context_, rw, static_cast<size_t>(num_src_) * plane * sizeof(float),
      nullptr);
  prev_sel_prob_buf_ = CreateBuffer(
      context_, rw, static_cast<size_t>(num_src_) * plane * sizeof(float),
      nullptr);
  cmask_buf_ = CreateBuffer(
      context_, rw, static_cast<size_t>(num_src_) * plane * sizeof(uint8_t),
      nullptr);
  // Rotation scratch must hold the largest rotated map: num_src float channels
  // (cost/sel) OR the ulong2 PRNG map (2*8 bytes/pixel) OR 3 normal channels.
  const size_t scratch_bytes =
      plane * std::max<size_t>(static_cast<size_t>(num_src_) * sizeof(float),
                               2 * sizeof(cl_ulong));
  scratch_buf_ = CreateBuffer(context_, rw, scratch_bytes, nullptr);
  src_images_buf_ = CreateBuffer(context_, ro_copy, src_images_host.size(),
                                 src_images_host.data());
  const size_t src_depth_bytes =
      static_cast<size_t>(src_max_w_) * src_max_h_ * num_src_ * sizeof(float);
  if (options_.geom_consistency) {
    src_depth_buf_ = CreateBuffer(context_, ro_copy,
                                  src_depth_host.size() * sizeof(float),
                                  src_depth_host.data());
  } else {
    // Allocated but unused in the photometric pass (kernel arg must be valid).
    src_depth_buf_ =
        CreateBuffer(context_, CL_MEM_READ_ONLY, src_depth_bytes, nullptr);
  }
  poses_buf_ = CreateBuffer(context_, ro_copy, poses_host.size() * sizeof(float),
                            poses_host.data());
  ref_inv_K_buf_ = CreateBuffer(context_, ro_copy, 16 * sizeof(float),
                                ref_inv_K_flat.data());
  ref_K_buf_ = CreateBuffer(context_, ro_copy, 16 * sizeof(float),
                            ref_K_flat.data());
  bilateral_spatial_buf_ = CreateBuffer(
      context_, ro_copy, bilateral_spatial.size() * sizeof(float),
      bilateral_spatial.data());
  bilateral_color_buf_ = CreateBuffer(
      context_, ro_copy, bilateral_color.size() * sizeof(float),
      bilateral_color.data());

  // Per-column forward-pass carry buffers (tiny). Sized for the larger image
  // dimension so they fit both rotation orientations; the forward kernel indexes
  // them with the current width (<= max_dim) and they are fully consumed within
  // a single forward pass, so no rotation or cross-sweep persistence is needed.
  const size_t max_dim =
      static_cast<size_t>(std::max(ref_width_, ref_height_));
  fwd_prev_depth_buf_ = CreateBuffer(context_, rw, max_dim * sizeof(float),
                                     nullptr);
  fwd_prev_normal_buf_ = CreateBuffer(context_, rw, 3 * max_dim * sizeof(float),
                                      nullptr);
  fwd_message_buf_ = CreateBuffer(
      context_, rw, static_cast<size_t>(num_src_) * max_dim * sizeof(float),
      nullptr);

  // Previous selection probabilities start at 0.5 (CPU InitWorkspaceMemory).
  const float half = 0.5f;
  CheckCL(clEnqueueFillBuffer(queue_, prev_sel_prob_buf_, &half, sizeof(float),
                              0, static_cast<size_t>(num_src_) * plane *
                                     sizeof(float),
                              0, nullptr, nullptr),
          "clEnqueueFillBuffer(prev_sel_prob)");
}

void PatchMatchOpenCL::RotateMaps(bool rotate_cmask) {
  const int iw = cur_width_;
  const int ih = cur_height_;
  const size_t plane = static_cast<size_t>(iw) * ih;

  auto run2d = [&](cl_kernel kernel) {
    Enqueue2D(queue_, device_, kernel, iw, ih, "clEnqueueNDRangeKernel(rotate)");
  };
  auto copy_back = [&](cl_mem dst, size_t bytes) {
    CheckCL(clEnqueueCopyBuffer(queue_, scratch_buf_, dst, 0, 0, bytes, 0,
                                nullptr, nullptr),
            "clEnqueueCopyBuffer(rotate)");
  };

  // rand (ulong2)
  SetArg(k_rot_rand_, 0, sizeof(cl_mem), &rand_buf_);
  SetArg(k_rot_rand_, 1, sizeof(cl_mem), &scratch_buf_);
  SetArg(k_rot_rand_, 2, sizeof(int), &iw);
  SetArg(k_rot_rand_, 3, sizeof(int), &ih);
  run2d(k_rot_rand_);
  copy_back(rand_buf_, plane * 2 * sizeof(uint64_t));

  // float maps via rotate_ccw_f(in, out, iw, ih, channels)
  auto rotate_f = [&](cl_mem buf, int channels) {
    SetArg(k_rot_f_, 0, sizeof(cl_mem), &buf);
    SetArg(k_rot_f_, 1, sizeof(cl_mem), &scratch_buf_);
    SetArg(k_rot_f_, 2, sizeof(int), &iw);
    SetArg(k_rot_f_, 3, sizeof(int), &ih);
    SetArg(k_rot_f_, 4, sizeof(int), &channels);
    run2d(k_rot_f_);
    copy_back(buf, static_cast<size_t>(channels) * plane * sizeof(float));
  };
  rotate_f(depth_buf_, 1);
  rotate_f(ref_sum_buf_, 1);
  rotate_f(ref_sqsum_buf_, 1);
  rotate_f(cost_buf_, num_src_);

  // normal (3 channels + vector rotation)
  SetArg(k_rot_normal_, 0, sizeof(cl_mem), &normal_buf_);
  SetArg(k_rot_normal_, 1, sizeof(cl_mem), &scratch_buf_);
  SetArg(k_rot_normal_, 2, sizeof(int), &iw);
  SetArg(k_rot_normal_, 3, sizeof(int), &ih);
  run2d(k_rot_normal_);
  copy_back(normal_buf_, 3 * plane * sizeof(float));

  // ref image (uchar, 1 channel)
  {
    const int channels = 1;
    SetArg(k_rot_u8_, 0, sizeof(cl_mem), &ref_img_buf_);
    SetArg(k_rot_u8_, 1, sizeof(cl_mem), &scratch_buf_);
    SetArg(k_rot_u8_, 2, sizeof(int), &iw);
    SetArg(k_rot_u8_, 3, sizeof(int), &ih);
    SetArg(k_rot_u8_, 4, sizeof(int), &channels);
    run2d(k_rot_u8_);
    copy_back(ref_img_buf_, plane * sizeof(uint8_t));
  }

  // sel_prob -> prev_sel_prob (direct; different buffers, no copy-back).
  SetArg(k_rot_f_, 0, sizeof(cl_mem), &sel_prob_buf_);
  SetArg(k_rot_f_, 1, sizeof(cl_mem), &prev_sel_prob_buf_);
  SetArg(k_rot_f_, 2, sizeof(int), &iw);
  SetArg(k_rot_f_, 3, sizeof(int), &ih);
  SetArg(k_rot_f_, 4, sizeof(int), &num_src_);
  run2d(k_rot_f_);

  // consistency mask (uchar, num_src channels) on the final rotation.
  if (rotate_cmask) {
    SetArg(k_rot_u8_, 0, sizeof(cl_mem), &cmask_buf_);
    SetArg(k_rot_u8_, 1, sizeof(cl_mem), &scratch_buf_);
    SetArg(k_rot_u8_, 2, sizeof(int), &iw);
    SetArg(k_rot_u8_, 3, sizeof(int), &ih);
    SetArg(k_rot_u8_, 4, sizeof(int), &num_src_);
    run2d(k_rot_u8_);
    copy_back(cmask_buf_, static_cast<size_t>(num_src_) * plane *
                              sizeof(uint8_t));
  }

  cur_width_ = ih;
  cur_height_ = iw;
  rotation_ = (rotation_ + 1) % 4;
}

void PatchMatchOpenCL::RunSweeps() {
  // Likelihood-computer constants (constant across sweeps).
  const float ncc_sigma = options_.ncc_sigma;
  const float inv_ncc_sigma_sq = -0.5f / (ncc_sigma * ncc_sigma);
  const float ncc_norm_factor =
      2.0f / (std::sqrt(2.0f * kPi) * ncc_sigma *
              std::erf(2.0f / (ncc_sigma * 1.414213562f)));
  const float inc_sigma = options_.incident_angle_sigma;
  const float inv_inc_sigma_sq = -0.5f / (inc_sigma * inc_sigma);
  const float cos_min_tri =
      std::cos(static_cast<float>(options_.min_triangulation_angle) * kDegToRad);
  const float filter_cos_min_tri = std::cos(
      static_cast<float>(options_.filter_min_triangulation_angle) * kDegToRad);
  const float one_minus_min_ncc = 1.0f - options_.filter_min_ncc;
  const float filter_min_ncc_prob =
      std::exp(one_minus_min_ncc * one_minus_min_ncc * inv_ncc_sigma_sq) *
      ncc_norm_factor;
  const int filter_min_num_consistent = options_.filter_min_num_consistent;
  const int num_samples = options_.num_samples;

  // Geometric-consistency term (Phase 2c). Active for every sweep of the
  // geometric pass; the geometric filter additionally runs on the last sweep.
  const int geom_consistency_term = options_.geom_consistency ? 1 : 0;
  const float geom_regularizer = options_.geom_consistency_regularizer;
  const float geom_max_cost = options_.geom_consistency_max_cost;
  const float filter_geom_max_cost = options_.filter_geom_consistency_max_cost;

  const float total_num_steps = options_.num_iterations * 4.0f;

  // GPU throttling so the shared Adreno does not starve the desktop.
  const int sweep_band =
      EnvInt("COLMAP_OPENCL_SWEEP_BAND", kDefaultSweepColumnBand);
  const float gpu_duty = EnvFloat("COLMAP_OPENCL_DUTY", kDefaultGpuDuty);

  // Row-chunking: bound per-work-item (per-column) work per dispatch. Each
  // sweep work-item processes a full image column; above ~1.5M photo-cost
  // window-samples per column the Adreno silently garbages the result (NOT a
  // buffer/TDR limit -- see OPENCL_LARGE_IMAGE_BUG.md). Split the column sweep
  // over the row dimension into multiple launches that persist per-column state
  // between them (prev depth/normal, forward messages, PRNG via rand_state, and
  // the backward betas which already live in sel_prob_map). chunk_rows is sized
  // so each forward dispatch stays under the work budget; the chunked result is
  // bit-identical to the original single-dispatch sweep.
  const long window_steps =
      (2L * window_radius_) / std::max(1, window_step_) + 1;
  const long window_samples = window_steps * window_steps;
  const long work_per_row =
      static_cast<long>(num_samples * 4 + num_src_) * window_samples;
  const long row_chunk_budget =
      EnvInt("COLMAP_OPENCL_ROW_CHUNK_WORK", kDefaultRowChunkWork);
  int chunk_rows = static_cast<int>(
      std::max<long>(1, row_chunk_budget / std::max<long>(1, work_per_row)));
  chunk_rows = EnvInt("COLMAP_OPENCL_ROW_CHUNK", chunk_rows);
  chunk_rows = std::max(1, chunk_rows);

  if (gpu_duty < 1.0f) {
    LOG(INFO) << "OpenCL GPU throttle ON: duty=" << gpu_duty << ", band="
              << sweep_band << " cols (GPU idle ~" << (1.0f - gpu_duty) * 100.0f
              << "% of the time for desktop responsiveness). Row chunk="
              << chunk_rows << " rows (~" << (chunk_rows * work_per_row)
              << " work/col/dispatch).";
  } else {
    LOG(INFO) << "OpenCL GPU: full speed (band=" << sweep_band
              << ", row chunk=" << chunk_rows << " rows). Set "
                 "COLMAP_OPENCL_DUTY=0.5 (or lower) to throttle and keep the "
                 "desktop responsive.";
  }

  // Enqueue a 1D kernel over the columns [0, W) in bands (one work-item per
  // column), with a clFinish + optional duty-cycle sleep after each band so the
  // shared Adreno does not starve the desktop and each dispatch stays under the
  // Windows TDR watchdog.
  auto dispatch_banded = [&](cl_kernel kernel, int W, const char* what) {
    for (int c0 = 0; c0 < W; c0 += sweep_band) {
      const auto band_start = std::chrono::steady_clock::now();
      const size_t offset = static_cast<size_t>(c0);
      const size_t gsize = static_cast<size_t>(std::min(sweep_band, W - c0));
      CheckCL(clEnqueueNDRangeKernel(queue_, kernel, 1, &offset, &gsize, nullptr,
                                     0, nullptr, nullptr),
              what);
      CheckCL(clFinish(queue_), "clFinish(sweep band)");
      if (gpu_duty < 1.0f) {
        const double band_ms = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - band_start)
                                   .count();
        const long sleep_ms =
            static_cast<long>(band_ms * (1.0 - gpu_duty) / gpu_duty);
        if (sleep_ms > 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }
      }
    }
  };

  for (int iter = 0; iter < options_.num_iterations; ++iter) {
    Timer iter_timer;
    iter_timer.Start();
    for (int sweep = 0; sweep < 4; ++sweep) {
      const float perturbation =
          1.0f / std::pow(2.0f, iter + sweep / 4.0f);
      const float prev_sel_prob_weight =
          static_cast<float>(iter * 4 + sweep) / total_num_steps;
      const bool last_sweep =
          iter == options_.num_iterations - 1 && sweep == 3;
      const int filter_photo = (last_sweep && options_.filter) ? 1 : 0;
      const int filter_geom =
          (last_sweep && options_.filter && options_.geom_consistency) ? 1 : 0;

      const int W = cur_width_;
      const int H = cur_height_;

      if (filter_photo) {
        const uint8_t zero = 0;
        CheckCL(clEnqueueFillBuffer(
                    queue_, cmask_buf_, &zero, sizeof(uint8_t), 0,
                    static_cast<size_t>(num_src_) * W * H * sizeof(uint8_t),
                    0, nullptr, nullptr),
                "clEnqueueFillBuffer(cmask)");
      }

      //////////////////////////////////////////////////////////////////////////
      // Backward message pass: write the whole column's betas into
      // sel_prob_map. Chunked high->low; must fully complete (every chunk, every
      // column) before the forward pass reads the betas.
      //////////////////////////////////////////////////////////////////////////
      {
        cl_uint a = 0;
        SetArg(k_sweep_bwd_, a++, sizeof(cl_mem), &cost_buf_);
        SetArg(k_sweep_bwd_, a++, sizeof(cl_mem), &sel_prob_buf_);
        SetArg(k_sweep_bwd_, a++, sizeof(int), &cur_width_);
        SetArg(k_sweep_bwd_, a++, sizeof(int), &cur_height_);
        SetArg(k_sweep_bwd_, a++, sizeof(int), &num_src_);
        SetArg(k_sweep_bwd_, a++, sizeof(float), &inv_ncc_sigma_sq);
        SetArg(k_sweep_bwd_, a++, sizeof(float), &ncc_norm_factor);
        const cl_uint bwd_row_begin = a++;
        const cl_uint bwd_row_end = a++;
        for (int r_hi = H - 1; r_hi >= 0; r_hi -= chunk_rows) {
          const int row_begin = std::max(0, r_hi - chunk_rows + 1);
          const int row_end = r_hi + 1;
          SetArg(k_sweep_bwd_, bwd_row_begin, sizeof(int), &row_begin);
          SetArg(k_sweep_bwd_, bwd_row_end, sizeof(int), &row_end);
          dispatch_banded(k_sweep_bwd_, W,
                          "clEnqueueNDRangeKernel(sweep_backward)");
        }
      }

      //////////////////////////////////////////////////////////////////////////
      // Forward sweep: chunked low->high, persisting per-column state between
      // chunks. Set the (mostly constant) args once, vary only the row range.
      //////////////////////////////////////////////////////////////////////////
      {
        cl_uint a = 0;
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &depth_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &normal_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &ref_img_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &ref_sum_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &ref_sqsum_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &src_images_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &poses_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &ref_inv_K_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &bilateral_spatial_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &bilateral_color_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &rand_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &sel_prob_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &prev_sel_prob_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &cost_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &cmask_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &cur_width_);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &cur_height_);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &num_src_);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &src_max_w_);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &src_max_h_);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &window_radius_);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &window_step_);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &rotation_);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &num_samples);
        SetArg(k_sweep_fwd_, a++, sizeof(float), &perturbation);
        SetArg(k_sweep_fwd_, a++, sizeof(float), &prev_sel_prob_weight);
        SetArg(k_sweep_fwd_, a++, sizeof(float), &inv_ncc_sigma_sq);
        SetArg(k_sweep_fwd_, a++, sizeof(float), &ncc_norm_factor);
        SetArg(k_sweep_fwd_, a++, sizeof(float), &cos_min_tri);
        SetArg(k_sweep_fwd_, a++, sizeof(float), &inv_inc_sigma_sq);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &filter_photo);
        SetArg(k_sweep_fwd_, a++, sizeof(float), &filter_min_ncc_prob);
        SetArg(k_sweep_fwd_, a++, sizeof(float), &filter_cos_min_tri);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &filter_min_num_consistent);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &src_depth_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &ref_K_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &geom_consistency_term);
        SetArg(k_sweep_fwd_, a++, sizeof(float), &geom_regularizer);
        SetArg(k_sweep_fwd_, a++, sizeof(float), &geom_max_cost);
        SetArg(k_sweep_fwd_, a++, sizeof(int), &filter_geom);
        SetArg(k_sweep_fwd_, a++, sizeof(float), &filter_geom_max_cost);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &fwd_prev_depth_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &fwd_prev_normal_buf_);
        SetArg(k_sweep_fwd_, a++, sizeof(cl_mem), &fwd_message_buf_);
        const cl_uint fwd_row_begin = a++;
        const cl_uint fwd_row_end = a++;
        for (int r_lo = 0; r_lo < H; r_lo += chunk_rows) {
          const int row_begin = r_lo;
          const int row_end = std::min(H, r_lo + chunk_rows);
          SetArg(k_sweep_fwd_, fwd_row_begin, sizeof(int), &row_begin);
          SetArg(k_sweep_fwd_, fwd_row_end, sizeof(int), &row_end);
          dispatch_banded(k_sweep_fwd_, W,
                          "clEnqueueNDRangeKernel(sweep_forward)");
        }
      }

      RotateMaps(/*rotate_cmask=*/last_sweep && options_.filter);
    }
    CheckCL(clFinish(queue_), "clFinish(iter)");
    LOG(INFO) << "Iteration " << iter + 1 << ": "
              << iter_timer.ElapsedSeconds() << " [s]";
  }
}

void PatchMatchOpenCL::ReadbackResults() {
  const size_t plane = static_cast<size_t>(ref_width_) * ref_height_;

  Mat<float> depth_mat(ref_width_, ref_height_, 1);
  CheckCL(clEnqueueReadBuffer(queue_, depth_buf_, CL_TRUE, 0,
                              plane * sizeof(float), depth_mat.GetPtr(), 0,
                              nullptr, nullptr),
          "clEnqueueReadBuffer(depth)");
  result_depth_ = DepthMap(depth_mat, options_.depth_min, options_.depth_max);

  Mat<float> normal_mat(ref_width_, ref_height_, 3);
  CheckCL(clEnqueueReadBuffer(queue_, normal_buf_, CL_TRUE, 0,
                              3 * plane * sizeof(float), normal_mat.GetPtr(), 0,
                              nullptr, nullptr),
          "clEnqueueReadBuffer(normal)");
  result_normal_ = NormalMap(normal_mat);

  result_sel_prob_ = Mat<float>(ref_width_, ref_height_, num_src_);
  CheckCL(clEnqueueReadBuffer(queue_, prev_sel_prob_buf_, CL_TRUE, 0,
                              static_cast<size_t>(num_src_) * plane *
                                  sizeof(float),
                              result_sel_prob_.GetPtr(), 0, nullptr, nullptr),
          "clEnqueueReadBuffer(sel_prob)");

  if (options_.filter) {
    result_cmask_ = Mat<uint8_t>(ref_width_, ref_height_, num_src_);
    CheckCL(clEnqueueReadBuffer(queue_, cmask_buf_, CL_TRUE, 0,
                                static_cast<size_t>(num_src_) * plane *
                                    sizeof(uint8_t),
                                result_cmask_.GetPtr(), 0, nullptr, nullptr),
            "clEnqueueReadBuffer(cmask)");
  }
}

void PatchMatchOpenCL::Run() {
  Timer total_timer;
  total_timer.Start();
  LOG(INFO) << "Running PatchMatch stereo ("
            << (options_.geom_consistency ? "geometric" : "photometric")
            << ") on the GPU via OpenCL with "
            << GetDeviceInfoString(device_, CL_DEVICE_NAME);

  // Initial cost (2D grid over the reference image, rotation 0).
  {
    cl_uint a = 0;
    SetArg(k_initial_cost_, a++, sizeof(cl_mem), &depth_buf_);
    SetArg(k_initial_cost_, a++, sizeof(cl_mem), &normal_buf_);
    SetArg(k_initial_cost_, a++, sizeof(cl_mem), &ref_img_buf_);
    SetArg(k_initial_cost_, a++, sizeof(cl_mem), &ref_sum_buf_);
    SetArg(k_initial_cost_, a++, sizeof(cl_mem), &ref_sqsum_buf_);
    SetArg(k_initial_cost_, a++, sizeof(cl_mem), &src_images_buf_);
    SetArg(k_initial_cost_, a++, sizeof(cl_mem), &poses_buf_);
    SetArg(k_initial_cost_, a++, sizeof(cl_mem), &ref_inv_K_buf_);
    SetArg(k_initial_cost_, a++, sizeof(cl_mem), &bilateral_spatial_buf_);
    SetArg(k_initial_cost_, a++, sizeof(cl_mem), &bilateral_color_buf_);
    SetArg(k_initial_cost_, a++, sizeof(cl_mem), &cost_buf_);
    SetArg(k_initial_cost_, a++, sizeof(int), &cur_width_);
    SetArg(k_initial_cost_, a++, sizeof(int), &cur_height_);
    SetArg(k_initial_cost_, a++, sizeof(int), &num_src_);
    SetArg(k_initial_cost_, a++, sizeof(int), &src_max_w_);
    SetArg(k_initial_cost_, a++, sizeof(int), &src_max_h_);
    SetArg(k_initial_cost_, a++, sizeof(int), &window_radius_);
    SetArg(k_initial_cost_, a++, sizeof(int), &window_step_);
    SetArg(k_initial_cost_, a++, sizeof(int), &rotation_);
    // The initial-cost kernel is compute-heavy (num_src bilaterally-weighted NCC
    // costs per pixel) and was launched as ONE 2D dispatch over the whole image.
    // The Adreno silently truncates any single dispatch running longer than the
    // ~2.4 s GPU watchdog (TDR) and returns garbage with no error, so band it by
    // rows; per-dispatch work ~ width*rows*num_src is kept well under the limit.
    {
      size_t kwg = 0;
      clGetKernelWorkGroupInfo(k_initial_cost_, device_,
                               CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &kwg,
                               nullptr);
      size_t L = 16;
      while (L > 1 && L * L > kwg) L /= 2;
      const long init_budget = 1500000;  // width*rows*num_src per dispatch
      const int init_rows = static_cast<int>(std::max<long>(
          static_cast<long>(L),
          init_budget / (static_cast<long>(cur_width_) * num_src_)));
      for (int r0 = 0; r0 < cur_height_; r0 += init_rows) {
        const int rows = std::min(init_rows, cur_height_ - r0);
        const size_t off[2] = {0, static_cast<size_t>(r0)};
        const size_t gws[2] = {
            ((static_cast<size_t>(cur_width_) + L - 1) / L) * L,
            ((static_cast<size_t>(rows) + L - 1) / L) * L};
        const size_t lws[2] = {L, L};
        CheckCL(clEnqueueNDRangeKernel(queue_, k_initial_cost_, 2, off, gws, lws,
                                       0, nullptr, nullptr),
                "clEnqueueNDRangeKernel(compute_initial_cost)");
        CheckCL(clFinish(queue_), "clFinish(initial_cost band)");
      }
    }
  }

  RunSweeps();
  ReadbackResults();

  LOG(INFO) << "Total: " << total_timer.ElapsedSeconds() << " [s]";
}

DepthMap PatchMatchOpenCL::GetDepthMap() const { return result_depth_; }

NormalMap PatchMatchOpenCL::GetNormalMap() const { return result_normal_; }

Mat<float> PatchMatchOpenCL::GetSelProbMap() const { return result_sel_prob_; }

std::vector<int> PatchMatchOpenCL::GetConsistentImageIdxs() const {
  const Mat<uint8_t>& mask = result_cmask_;
  std::vector<int> consistent_image_idxs;
  std::vector<int> pixel_consistent_image_idxs;
  pixel_consistent_image_idxs.reserve(mask.GetDepth());
  for (size_t r = 0; r < mask.GetHeight(); ++r) {
    for (size_t c = 0; c < mask.GetWidth(); ++c) {
      pixel_consistent_image_idxs.clear();
      for (size_t d = 0; d < mask.GetDepth(); ++d) {
        if (mask.Get(r, c, d)) {
          pixel_consistent_image_idxs.push_back(problem_.src_image_idxs[d]);
        }
      }
      if (!pixel_consistent_image_idxs.empty()) {
        consistent_image_idxs.push_back(c);
        consistent_image_idxs.push_back(r);
        consistent_image_idxs.push_back(pixel_consistent_image_idxs.size());
        consistent_image_idxs.insert(consistent_image_idxs.end(),
                                     pixel_consistent_image_idxs.begin(),
                                     pixel_consistent_image_idxs.end());
      }
    }
  }
  return consistent_image_idxs;
}

}  // namespace mvs
}  // namespace colmap
