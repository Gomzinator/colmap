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
#include "colmap/mvs/image.h"
#include "colmap/mvs/mat.h"
#include "colmap/mvs/normal_map.h"
#include "colmap/mvs/patch_match.h"
#include "colmap/mvs/patch_match_backend.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace colmap {
namespace mvs {

// CPU implementation of the PatchMatch stereo algorithm. This is an
// algorithmically faithful port of the CUDA implementation in
// patch_match_cuda.cu: identical cost function (bilaterally weighted NCC with
// optional geometric consistency term), identical sweeping scheme (four 90
// degree rotations per iteration with a sequential top-to-bottom sweep), and
// identical message passing for the source image selection probabilities.
// Results are statistically equivalent but not bit-identical to the CUDA
// backend, since a different pseudo-random number generator is used.
//
// Parallelization: columns are processed in parallel within each sweep
// (OpenMP), since the sequential dependency of the propagation runs along
// rows only. The initial cost computation is parallelized over rows.
class PatchMatchCpu : public PatchMatchBackend {
 public:
  // PCG32 PRNG (O'Neill, 2014). One state per pixel, mirroring the curand
  // state map of the CUDA implementation.
  struct Pcg32State {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc = 0xda3e39cb94b95bdbULL;
  };

  PatchMatchCpu(const PatchMatchOptions& options,
                const PatchMatch::Problem& problem);

  void Run() override;

  DepthMap GetDepthMap() const override;
  NormalMap GetNormalMap() const override;
  Mat<float> GetSelProbMap() const override;
  std::vector<int> GetConsistentImageIdxs() const override;

 private:
  void InitRefImage();
  void InitSourceImages();
  void InitTransforms();
  void InitWorkspaceMemory();

  void ComputeInitialCost();

  struct SweepOptions;
  void SweepFromTopToBottom(const SweepOptions& sweep_options,
                            bool geom_consistency_term,
                            bool filter_photo_consistency,
                            bool filter_geom_consistency);

  // Rotate all maps by 90 degrees in counter-clockwise direction.
  void Rotate();

  // Filter reference image to compute bilaterally weighted local sums of
  // (squared) intensities, mirroring GpuMatRefImage::Filter.
  void FilterRefImage(const uint8_t* image_data);

  // Sampling helpers replicating the CUDA texture semantics.
  inline float SampleRefImage(int row, int col) const;
  inline float SampleSrcImageBilinear(int image_idx, float x, float y) const;
  inline float SampleSrcDepthNearest(int image_idx, float x, float y) const;

  // Cost computation helpers.
  inline float ComputePhotoConsistencyCost(int row,
                                           int col,
                                           float depth,
                                           const float normal[3],
                                           int src_image_idx,
                                           float ref_sum,
                                           float ref_squared_sum) const;
  inline float ComputeGeomConsistencyCost(float row,
                                          float col,
                                          float depth,
                                          int image_idx,
                                          float max_cost) const;
  inline void ComputeViewingAngles(const float point[3],
                                   const float normal[3],
                                   int image_idx,
                                   float* cos_triangulation_angle,
                                   float* cos_incident_angle) const;
  inline void ComposeHomography(int image_idx,
                                int row,
                                int col,
                                float depth,
                                const float normal[3],
                                float H[9]) const;

  const PatchMatchOptions options_;
  const PatchMatch::Problem problem_;

  // Number of OpenMP threads used within sweeps.
  int num_threads_ = 1;

  // Original (not rotated) dimensions of the reference image.
  size_t ref_width_ = 0;
  size_t ref_height_ = 0;

  // Current dimensions (swapped for odd rotations).
  size_t width_ = 0;
  size_t height_ = 0;

  // Rotation of reference image in pi/2, i.e. number of calls to Rotate()
  // modulo 4.
  int rotation_in_half_pi_ = 0;

  // Reference image data (quantized intensities and bilaterally weighted
  // local sums), rotated alongside the other maps.
  Mat<uint8_t> ref_image_;
  Mat<float> ref_sum_image_;
  Mat<float> ref_squared_sum_image_;

  // Source images / depth maps, zero-padded to common max dimensions
  // (replicates the layered CUDA textures with border addressing).
  size_t src_max_width_ = 0;
  size_t src_max_height_ = 0;
  std::vector<uint8_t> src_images_;
  std::vector<float> src_depth_maps_;

  // Relative poses from the rotated versions of the reference image to the
  // source images. Layout per source image (43 floats):
  //   [K(4), R(9), T(3), C(3), P(12), inv_P(12)]
  // See PatchMatchCuda::InitTransforms for the rotation conventions.
  static constexpr int kNumTformParams = 4 + 9 + 3 + 3 + 12 + 12;
  std::vector<float> poses_[4];

  // Calibration of (rotated) reference image as {fx, cx, fy, cy} and
  // {1/fx, -cx/fx, 1/fy, -cy/fy}.
  float ref_K_host_[4][4];
  float ref_inv_K_host_[4][4];
  const float* ref_K_ = nullptr;
  const float* ref_inv_K_ = nullptr;

  // Per-pixel PRNG states (linearized, current width_/height_ dims).
  std::vector<Pcg32State> rand_state_map_;

  // Working maps (current rotation).
  Mat<float> depth_map_;
  Mat<float> normal_map_;
  Mat<float> sel_prob_map_;
  Mat<float> prev_sel_prob_map_;
  Mat<float> cost_map_;
  Mat<uint8_t> consistency_mask_;

  // Precomputed bilateral weight tables:
  //   weight(dr, dc, c1, c2) = spatial_table[(dr+R)*(2R+1)+(dc+R)]
  //                          * color_table[|q1 - q2|]
  // where q = round(255 * c) is the quantized intensity. Exact w.r.t. the
  // CUDA computation since reference intensities are uint8-quantized.
  std::vector<float> bilateral_spatial_table_;
  std::vector<float> bilateral_color_table_;
};

}  // namespace mvs
}  // namespace colmap
