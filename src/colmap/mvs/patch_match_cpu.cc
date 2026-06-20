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


// CPU port of the PatchMatch stereo CUDA implementation. See
// patch_match_cuda.cu for the reference implementation. The port replicates
// the CUDA texture semantics (border addressing with zero outside the image,
// bilinear filtering at pixel centers for source images, nearest neighbor
// for source depth maps, uint8 quantization of the reference image) so that
// results are statistically equivalent to the CUDA backend. The PRNG differs
// (PCG32 instead of curand), so results are not bit-identical.

#include "colmap/mvs/patch_match_cpu.h"

#include "colmap/mvs/consistency_graph.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <vector>

namespace colmap {
namespace mvs {
namespace {

////////////////////////////////////////////////////////////////////////////////
// Constants (centralized; mirror the values in patch_match_cuda.cu).
////////////////////////////////////////////////////////////////////////////////

constexpr float kPi = 3.14159265358979323846f;
// Degree-to-radian conversion factor (same constant as DEG2RAD in
// patch_match_cuda.cu for exact parity).
constexpr float kDegToRad = 0.0174532925199432f;
// Probability for boundary pixels in the message passing.
constexpr float kUniformProb = 0.5f;
// Maximum photo consistency cost as 1 - min(NCC).
constexpr float kMaxPhotoCost = 2.0f;
// Minimum variance for the NCC computation (Jensen's inequality bound).
constexpr float kMinVar = 1e-5f;
// Number of cost hypotheses evaluated per pixel and sweep.
constexpr int kNumCosts = 5;
// Maximum number of retrials when perturbing the normal.
constexpr int kMaxPerturbNormalTrials = 3;
// Quantization of image intensities (uint8).
constexpr int kNumIntensityBins = 256;
constexpr float kInvMaxIntensity = 1.0f / 255.0f;

////////////////////////////////////////////////////////////////////////////////
// PCG32 PRNG, replacing curand. Uniform() returns a float in (0, 1] to match
// the semantics of curand_uniform().
////////////////////////////////////////////////////////////////////////////////

inline uint64_t SplitMix64(uint64_t& x) {
  x += 0x9e3779b97f4a7c15ULL;
  uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

inline void SeedPcg32(uint64_t seed, PatchMatchCpu::Pcg32State* state) {
  uint64_t sm = seed;
  state->state = SplitMix64(sm);
  state->inc = SplitMix64(sm) | 1ULL;
}

inline uint32_t Pcg32Next(PatchMatchCpu::Pcg32State* state) {
  const uint64_t old_state = state->state;
  state->state = old_state * 6364136223846793005ULL + state->inc;
  const uint32_t xorshifted =
      static_cast<uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
  const uint32_t rot = static_cast<uint32_t>(old_state >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31u));
}

// Uniform float in (0, 1], matching curand_uniform().
inline float RandUniform(PatchMatchCpu::Pcg32State* state) {
  return ((Pcg32Next(state) >> 8) + 1) * 0x1p-24f;
}

////////////////////////////////////////////////////////////////////////////////
// Small math helpers (verbatim ports from patch_match_cuda.cu).
////////////////////////////////////////////////////////////////////////////////

inline void Mat33DotVec3(const float mat[9],
                         const float vec[3],
                         float result[3]) {
  result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

inline void Mat33DotVec3Homogeneous(const float mat[9],
                                    const float vec[2],
                                    float result[2]) {
  const float inv_z = 1.0f / (mat[6] * vec[0] + mat[7] * vec[1] + mat[8]);
  result[0] = inv_z * (mat[0] * vec[0] + mat[1] * vec[1] + mat[2]);
  result[1] = inv_z * (mat[3] * vec[0] + mat[4] * vec[1] + mat[5]);
}

inline float DotProduct3(const float vec1[3], const float vec2[3]) {
  return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

inline float GenerateRandomDepth(const float depth_min,
                                 const float depth_max,
                                 PatchMatchCpu::Pcg32State* rand_state) {
  return RandUniform(rand_state) * (depth_max - depth_min) + depth_min;
}

inline void GenerateRandomNormal(const int row,
                                 const int col,
                                 const float ref_inv_K[4],
                                 PatchMatchCpu::Pcg32State* rand_state,
                                 float normal[3]) {
  // Unbiased sampling of normal, according to George Marsaglia, "Choosing a
  // Point from the Surface of a Sphere", 1972.
  float v1 = 0.0f;
  float v2 = 0.0f;
  float s = 2.0f;
  while (s >= 1.0f) {
    v1 = 2.0f * RandUniform(rand_state) - 1.0f;
    v2 = 2.0f * RandUniform(rand_state) - 1.0f;
    s = v1 * v1 + v2 * v2;
  }

  const float s_norm = std::sqrt(1.0f - s);
  normal[0] = 2.0f * v1 * s_norm;
  normal[1] = 2.0f * v2 * s_norm;
  normal[2] = 1.0f - 2.0f * s;

  // Make sure normal is looking away from camera.
  const float view_ray[3] = {ref_inv_K[0] * col + ref_inv_K[1],
                             ref_inv_K[2] * row + ref_inv_K[3],
                             1.0f};
  if (DotProduct3(normal, view_ray) > 0) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }
}

inline float PerturbDepth(const float perturbation,
                          const float depth,
                          PatchMatchCpu::Pcg32State* rand_state) {
  const float depth_min = (1.0f - perturbation) * depth;
  const float depth_max = (1.0f + perturbation) * depth;
  return GenerateRandomDepth(depth_min, depth_max, rand_state);
}

inline void PerturbNormal(const int row,
                          const int col,
                          const float perturbation,
                          const float ref_inv_K[4],
                          const float normal[3],
                          PatchMatchCpu::Pcg32State* rand_state,
                          float perturbed_normal[3],
                          const int num_trials = 0) {
  // Perturbation rotation angles.
  const float a1 = (RandUniform(rand_state) - 0.5f) * perturbation;
  const float a2 = (RandUniform(rand_state) - 0.5f) * perturbation;
  const float a3 = (RandUniform(rand_state) - 0.5f) * perturbation;

  const float sin_a1 = std::sin(a1);
  const float sin_a2 = std::sin(a2);
  const float sin_a3 = std::sin(a3);
  const float cos_a1 = std::cos(a1);
  const float cos_a2 = std::cos(a2);
  const float cos_a3 = std::cos(a3);

  // R = Rx * Ry * Rz
  float R[9];
  R[0] = cos_a2 * cos_a3;
  R[1] = -cos_a2 * sin_a3;
  R[2] = sin_a2;
  R[3] = cos_a1 * sin_a3 + cos_a3 * sin_a1 * sin_a2;
  R[4] = cos_a1 * cos_a3 - sin_a1 * sin_a2 * sin_a3;
  R[5] = -cos_a2 * sin_a1;
  R[6] = sin_a1 * sin_a3 - cos_a1 * cos_a3 * sin_a2;
  R[7] = cos_a3 * sin_a1 + cos_a1 * sin_a2 * sin_a3;
  R[8] = cos_a1 * cos_a2;

  // Perturb the normal vector.
  Mat33DotVec3(R, normal, perturbed_normal);

  // Make sure the perturbed normal is still looking in the same direction as
  // the viewing direction, otherwise try again but with smaller perturbation.
  const float view_ray[3] = {ref_inv_K[0] * col + ref_inv_K[1],
                             ref_inv_K[2] * row + ref_inv_K[3],
                             1.0f};
  if (DotProduct3(perturbed_normal, view_ray) >= 0.0f) {
    if (num_trials < kMaxPerturbNormalTrials) {
      PerturbNormal(row,
                    col,
                    0.5f * perturbation,
                    ref_inv_K,
                    normal,
                    rand_state,
                    perturbed_normal,
                    num_trials + 1);
      return;
    } else {
      perturbed_normal[0] = normal[0];
      perturbed_normal[1] = normal[1];
      perturbed_normal[2] = normal[2];
      return;
    }
  }

  // Make sure normal has unit norm.
  const float inv_norm =
      1.0f / std::sqrt(DotProduct3(perturbed_normal, perturbed_normal));
  perturbed_normal[0] *= inv_norm;
  perturbed_normal[1] *= inv_norm;
  perturbed_normal[2] *= inv_norm;
}

inline void ComputePointAtDepth(const float row,
                                const float col,
                                const float ref_inv_K[4],
                                const float depth,
                                float point[3]) {
  point[0] = depth * (ref_inv_K[0] * col + ref_inv_K[1]);
  point[1] = depth * (ref_inv_K[2] * row + ref_inv_K[3]);
  point[2] = depth;
}

// Transfer depth on plane from viewing ray at row1 to row2. The returned
// depth is the intersection of the viewing ray through row2 with the plane
// at row1 defined by the given depth and normal.
inline float PropagateDepth(const float ref_inv_K[4],
                            const float depth1,
                            const float normal1[3],
                            const float row1,
                            const float row2) {
  // Point along first viewing ray.
  const float x1 = depth1 * (ref_inv_K[2] * row1 + ref_inv_K[3]);
  const float y1 = depth1;
  // Point on plane defined by point along first viewing ray and plane normal1.
  const float x2 = x1 + normal1[2];
  const float y2 = y1 - normal1[1];

  // Origin of second viewing ray.
  // const float x3 = 0.0f;
  // const float y3 = 0.0f;
  // Point on second viewing ray.
  const float x4 = ref_inv_K[2] * row2 + ref_inv_K[3];
  // const float y4 = 1.0f;

  // Intersection of the lines ((x1, y1), (x2, y2)) and ((x3, y3), (x4, y4)).
  const float denom = x2 - x1 + x4 * (y1 - y2);
  constexpr float kEps = 1e-5f;
  if (std::abs(denom) < kEps) {
    return depth1;
  }
  const float nom = y1 * x2 - x1 * y2;
  return nom / denom;
}

// Find index of minimum in given values.
inline int FindMinCost(const float costs[kNumCosts]) {
  float min_cost = costs[0];
  int min_cost_idx = 0;
  for (int idx = 1; idx < kNumCosts; ++idx) {
    if (costs[idx] <= min_cost) {
      min_cost = costs[idx];
      min_cost_idx = idx;
    }
  }
  return min_cost_idx;
}

inline void TransformPDFToCDF(float* probs, const int num_probs) {
  float prob_sum = 0.0f;
  for (int i = 0; i < num_probs; ++i) {
    prob_sum += probs[i];
  }
  const float inv_prob_sum = 1.0f / prob_sum;

  float cum_prob = 0.0f;
  for (int i = 0; i < num_probs; ++i) {
    const float prob = probs[i] * inv_prob_sum;
    cum_prob += prob;
    probs[i] = cum_prob;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Likelihood computer (verbatim port from patch_match_cuda.cu).
////////////////////////////////////////////////////////////////////////////////

class LikelihoodComputer {
 public:
  LikelihoodComputer(const float ncc_sigma,
                     const float min_triangulation_angle,
                     const float incident_angle_sigma)
      : cos_min_triangulation_angle_(std::cos(min_triangulation_angle)),
        inv_incident_angle_sigma_square_(
            -0.5f / (incident_angle_sigma * incident_angle_sigma)),
        inv_ncc_sigma_square_(-0.5f / (ncc_sigma * ncc_sigma)),
        ncc_norm_factor_(ComputeNCCCostNormFactor(ncc_sigma)) {}

  // Compute forward message from current cost and forward message of
  // previous / neighboring pixel.
  float ComputeForwardMessage(const float cost, const float prev) const {
    return ComputeMessage<true>(cost, prev);
  }

  // Compute backward message from current cost and backward message of
  // previous / neighboring pixel.
  float ComputeBackwardMessage(const float cost, const float prev) const {
    return ComputeMessage<false>(cost, prev);
  }

  // Compute the selection probability from the forward and backward message.
  inline float ComputeSelProb(const float alpha,
                              const float beta,
                              const float prev,
                              const float prev_weight) const {
    const float zn0 = (1.0f - alpha) * (1.0f - beta);
    const float zn1 = alpha * beta;
    const float curr = zn1 / (zn0 + zn1);
    return prev_weight * prev + (1.0f - prev_weight) * curr;
  }

  // Compute NCC probability. Note that cost = 1 - NCC.
  inline float ComputeNCCProb(const float cost) const {
    return std::exp(cost * cost * inv_ncc_sigma_square_) * ncc_norm_factor_;
  }

  // Compute the triangulation angle probability.
  inline float ComputeTriProb(const float cos_triangulation_angle) const {
    if (cos_triangulation_angle > cos_min_triangulation_angle_) {
      const float scaled = 1.0f - (1.0f - cos_triangulation_angle) /
                                      (1.0f - cos_min_triangulation_angle_);
      const float likelihood = 1.0f - scaled * scaled;
      return std::min(1.0f, std::max(0.0f, likelihood));
    } else {
      return 1.0f;
    }
  }

  // Compute the incident angle probability.
  inline float ComputeIncProb(const float cos_incident_angle) const {
    const float x = 1.0f - std::max(0.0f, cos_incident_angle);
    return std::exp(x * x * inv_incident_angle_sigma_square_);
  }

  // Compute the warping/resolution prior probability.
  inline float ComputeResolutionProb(const float H[9],
                                     const float row,
                                     const float col,
                                     const int window_size) const {
    const int window_radius = window_size / 2;

    // Warp corners of patch in reference image to source image.
    float src1[2];
    const float ref1[2] = {col - window_radius, row - window_radius};
    Mat33DotVec3Homogeneous(H, ref1, src1);
    float src2[2];
    const float ref2[2] = {col - window_radius, row + window_radius};
    Mat33DotVec3Homogeneous(H, ref2, src2);
    float src3[2];
    const float ref3[2] = {col + window_radius, row + window_radius};
    Mat33DotVec3Homogeneous(H, ref3, src3);
    float src4[2];
    const float ref4[2] = {col + window_radius, row - window_radius};
    Mat33DotVec3Homogeneous(H, ref4, src4);

    // Compute area of patches in reference and source image.
    const float ref_area = window_size * window_size;
    const float src_area =
        std::abs(0.5f * (src1[0] * src2[1] - src2[0] * src1[1] -
                         src1[0] * src4[1] + src2[0] * src3[1] -
                         src3[0] * src2[1] + src4[0] * src1[1] +
                         src3[0] * src4[1] - src4[0] * src3[1]));

    if (ref_area > src_area) {
      return src_area / ref_area;
    } else {
      return ref_area / src_area;
    }
  }

 private:
  // The normalization for the likelihood function, i.e. the normalization for
  // the prior on the matching cost.
  static inline float ComputeNCCCostNormFactor(const float ncc_sigma) {
    // A = sqrt(2pi)*sigma/2*erf(sqrt(2)/sigma)
    // erf(x) = 2/sqrt(pi) * integral from 0 to x of exp(-t^2) dt
    return 2.0f / (std::sqrt(2.0f * kPi) * ncc_sigma *
                   std::erf(2.0f / (ncc_sigma * 1.414213562f)));
  }

  // Compute the forward or backward message.
  template <bool kForward>
  inline float ComputeMessage(const float cost, const float prev) const {
    constexpr float kNoChangeProb = 0.99999f;
    constexpr float kChangeProb = 1.0f - kNoChangeProb;
    const float emission = ComputeNCCProb(cost);

    float zn0;  // Message for selection probability = 0.
    float zn1;  // Message for selection probability = 1.
    if (kForward) {
      zn0 = (prev * kChangeProb + (1.0f - prev) * kNoChangeProb) * kUniformProb;
      zn1 = (prev * kNoChangeProb + (1.0f - prev) * kChangeProb) * emission;
    } else {
      zn0 = prev * emission * kChangeProb +
            (1.0f - prev) * kUniformProb * kNoChangeProb;
      zn1 = prev * emission * kNoChangeProb +
            (1.0f - prev) * kUniformProb * kChangeProb;
    }

    return zn1 / (zn0 + zn1);
  }

  const float cos_min_triangulation_angle_;
  const float inv_incident_angle_sigma_square_;
  const float inv_ncc_sigma_square_;
  const float ncc_norm_factor_;
};

////////////////////////////////////////////////////////////////////////////////
// Rotation helpers. Counter-clockwise rotation with the same index mapping as
// CudaRotate: output(width_in - 1 - col_in, row_in) = input(row_in, col_in),
// with swapped output dimensions.
////////////////////////////////////////////////////////////////////////////////

// Mat<T> provides GetSlice but no SetSlice; helper for 3-channel maps.
inline void SetSlice3(Mat<float>& mat,
                      const size_t row,
                      const size_t col,
                      const float values[3]) {
  mat.Set(row, col, 0, values[0]);
  mat.Set(row, col, 1, values[1]);
  mat.Set(row, col, 2, values[2]);
}

template <typename T>
Mat<T> RotateMatCCW(const Mat<T>& input) {
  Mat<T> output(input.GetHeight(), input.GetWidth(), input.GetDepth());
  for (size_t slice = 0; slice < input.GetDepth(); ++slice) {
    for (size_t row = 0; row < input.GetHeight(); ++row) {
      for (size_t col = 0; col < input.GetWidth(); ++col) {
        output.Set(
            input.GetWidth() - 1 - col, row, slice, input.Get(row, col, slice));
      }
    }
  }
  return output;
}

template <typename T>
std::vector<T> RotateVecCCW(const std::vector<T>& input,
                            const size_t width,
                            const size_t height) {
  std::vector<T> output(input.size());
  // Output dimensions: width_out = height, height_out = width.
  const size_t width_out = height;
  for (size_t row = 0; row < height; ++row) {
    for (size_t col = 0; col < width; ++col) {
      const size_t out_row = width - 1 - col;
      const size_t out_col = row;
      output[out_row * width_out + out_col] = input[row * width + col];
    }
  }
  return output;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// Sweep options (mirror of the CUDA SweepOptions).
////////////////////////////////////////////////////////////////////////////////

struct PatchMatchCpu::SweepOptions {
  float perturbation = 1.0f;
  float depth_min = 0.0f;
  float depth_max = 1.0f;
  int num_samples = 15;
  float sigma_spatial = 3.0f;
  float sigma_color = 0.3f;
  float ncc_sigma = 0.6f;
  float min_triangulation_angle = 0.5f;
  float incident_angle_sigma = 0.9f;
  float prev_sel_prob_weight = 0.0f;
  float geom_consistency_regularizer = 0.1f;
  float geom_consistency_max_cost = 5.0f;
  float filter_min_ncc = 0.1f;
  float filter_min_triangulation_angle = 3.0f;
  int filter_min_num_consistent = 2;
  float filter_geom_consistency_max_cost = 1.0f;
};

////////////////////////////////////////////////////////////////////////////////
// Construction & initialization.
////////////////////////////////////////////////////////////////////////////////

PatchMatchCpu::PatchMatchCpu(const PatchMatchOptions& options,
                             const PatchMatch::Problem& problem)
    : options_(options), problem_(problem) {
  num_threads_ = GetEffectiveNumThreads(options_.num_threads);

  // Precompute bilateral weight tables. Exact w.r.t. the CUDA computation,
  // since all reference intensities are uint8-quantized in [0, 1].
  const int window_size = 2 * options_.window_radius + 1;
  const float spatial_normalization =
      1.0f / (2.0f * options_.sigma_spatial * options_.sigma_spatial);
  const float color_normalization =
      1.0f / (2.0f * options_.sigma_color * options_.sigma_color);
  bilateral_spatial_table_.resize(window_size * window_size);
  for (int dr = -options_.window_radius; dr <= options_.window_radius; ++dr) {
    for (int dc = -options_.window_radius; dc <= options_.window_radius;
         ++dc) {
      const float spatial_dist_squared =
          static_cast<float>(dr * dr + dc * dc);
      bilateral_spatial_table_[(dr + options_.window_radius) * window_size +
                               (dc + options_.window_radius)] =
          std::exp(-spatial_dist_squared * spatial_normalization);
    }
  }
  bilateral_color_table_.resize(kNumIntensityBins);
  for (int d = 0; d < kNumIntensityBins; ++d) {
    const float color_dist = d * kInvMaxIntensity;
    bilateral_color_table_[d] =
        std::exp(-color_dist * color_dist * color_normalization);
  }

  InitRefImage();
  InitSourceImages();
  InitTransforms();
  InitWorkspaceMemory();
}

void PatchMatchCpu::FilterRefImage(const uint8_t* image_data) {
  const int width = static_cast<int>(width_);
  const int height = static_cast<int>(height_);
  const int window_radius = options_.window_radius;
  const int window_step = options_.window_step;
  const int window_size = 2 * window_radius + 1;

  ref_image_ = Mat<uint8_t>(width_, height_, 1);
  ref_sum_image_ = Mat<float>(width_, height_, 1);
  ref_squared_sum_image_ = Mat<float>(width_, height_, 1);

#pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      const int center_q = image_data[row * width + col];

      float color_sum = 0.0f;
      float color_squared_sum = 0.0f;
      float bilateral_weight_sum = 0.0f;

      for (int window_row = -window_radius; window_row <= window_radius;
           window_row += window_step) {
        for (int window_col = -window_radius; window_col <= window_radius;
             window_col += window_step) {
          // CUDA texture border addressing: pixels outside the image have
          // color 0 and still contribute their bilateral weight.
          const int r = row + window_row;
          const int c = col + window_col;
          const int q = (r >= 0 && r < height && c >= 0 && c < width)
                            ? image_data[r * width + c]
                            : 0;
          const float color = q * kInvMaxIntensity;
          const float bilateral_weight =
              bilateral_spatial_table_[(window_row + window_radius) *
                                           window_size +
                                       (window_col + window_radius)] *
              bilateral_color_table_[std::abs(center_q - q)];
          color_sum += bilateral_weight * color;
          color_squared_sum += bilateral_weight * color * color;
          bilateral_weight_sum += bilateral_weight;
        }
      }

      color_sum /= bilateral_weight_sum;
      color_squared_sum /= bilateral_weight_sum;

      ref_image_.Set(row, col, static_cast<uint8_t>(center_q));
      ref_sum_image_.Set(row, col, color_sum);
      ref_squared_sum_image_.Set(row, col, color_squared_sum);
    }
  }
}

void PatchMatchCpu::InitRefImage() {
  const Image& ref_image = problem_.images->at(problem_.ref_image_idx);

  ref_width_ = ref_image.GetWidth();
  ref_height_ = ref_image.GetHeight();
  width_ = ref_width_;
  height_ = ref_height_;

  FilterRefImage(ref_image.GetBitmap().RowMajorData().data());
}

void PatchMatchCpu::InitSourceImages() {
  // Determine maximum image size.
  size_t max_width = 0;
  size_t max_height = 0;
  for (const auto image_idx : problem_.src_image_idxs) {
    const Image& image = problem_.images->at(image_idx);
    max_width = std::max(max_width, image.GetWidth());
    max_height = std::max(max_height, image.GetHeight());
  }
  src_max_width_ = max_width;
  src_max_height_ = max_height;

  // Copy source images to a zero-padded contiguous block, equivalent to the
  // layered CUDA texture with border addressing. Note: rows are copied with
  // the padded stride (the CUDA implementation copies images contiguously,
  // which scrambles layers for heterogeneous image sizes - this port uses the
  // row-aligned layout, like the CUDA code does for the depth maps).
  src_images_.assign(max_width * max_height * problem_.src_image_idxs.size(),
                     0);
  for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
    const Image& image = problem_.images->at(problem_.src_image_idxs[i]);
    const Bitmap& bitmap = image.GetBitmap();
    const uint8_t* src = bitmap.RowMajorData().data();
    uint8_t* dest = src_images_.data() + max_width * max_height * i;
    for (size_t r = 0; r < image.GetHeight(); ++r) {
      std::memcpy(dest + r * max_width, src + r * image.GetWidth(),
                  image.GetWidth());
    }
  }

  // Copy source depth maps for the geometric consistency term.
  if (options_.geom_consistency) {
    src_depth_maps_.assign(
        max_width * max_height * problem_.src_image_idxs.size(), 0.0f);
    for (size_t i = 0; i < problem_.src_image_idxs.size(); ++i) {
      const DepthMap& depth_map =
          problem_.depth_maps->at(problem_.src_image_idxs[i]);
      float* dest = src_depth_maps_.data() + max_width * max_height * i;
      for (size_t r = 0; r < depth_map.GetHeight(); ++r) {
        std::memcpy(dest + r * max_width,
                    depth_map.GetPtr() + r * depth_map.GetWidth(),
                    depth_map.GetWidth() * sizeof(float));
      }
    }
  }
}

void PatchMatchCpu::InitTransforms() {
  const Image& ref_image = problem_.images->at(problem_.ref_image_idx);

  //////////////////////////////////////////////////////////////////////////////
  // Generate rotated versions (counter-clockwise) of calibration matrix.
  //////////////////////////////////////////////////////////////////////////////

  for (size_t i = 0; i < 4; ++i) {
    ref_K_host_[i][0] = ref_image.GetK()[0];
    ref_K_host_[i][1] = ref_image.GetK()[2];
    ref_K_host_[i][2] = ref_image.GetK()[4];
    ref_K_host_[i][3] = ref_image.GetK()[5];
  }

  // Rotated by 90 degrees.
  std::swap(ref_K_host_[1][0], ref_K_host_[1][2]);
  std::swap(ref_K_host_[1][1], ref_K_host_[1][3]);
  ref_K_host_[1][3] = ref_width_ - 1 - ref_K_host_[1][3];

  // Rotated by 180 degrees.
  ref_K_host_[2][1] = ref_width_ - 1 - ref_K_host_[2][1];
  ref_K_host_[2][3] = ref_height_ - 1 - ref_K_host_[2][3];

  // Rotated by 270 degrees.
  std::swap(ref_K_host_[3][0], ref_K_host_[3][2]);
  std::swap(ref_K_host_[3][1], ref_K_host_[3][3]);
  ref_K_host_[3][1] = ref_height_ - 1 - ref_K_host_[3][1];

  // Extract 1/fx, -cx/fx, 1/fy, -cy/fy.
  for (size_t i = 0; i < 4; ++i) {
    ref_inv_K_host_[i][0] = 1.0f / ref_K_host_[i][0];
    ref_inv_K_host_[i][1] = -ref_K_host_[i][1] / ref_K_host_[i][0];
    ref_inv_K_host_[i][2] = 1.0f / ref_K_host_[i][2];
    ref_inv_K_host_[i][3] = -ref_K_host_[i][3] / ref_K_host_[i][2];
  }

  ref_K_ = ref_K_host_[0];
  ref_inv_K_ = ref_inv_K_host_[0];

  //////////////////////////////////////////////////////////////////////////////
  // Generate rotated versions of camera poses.
  //////////////////////////////////////////////////////////////////////////////

  float rotated_R[9];
  std::memcpy(rotated_R, ref_image.GetR(), 9 * sizeof(float));

  float rotated_T[3];
  std::memcpy(rotated_T, ref_image.GetT(), 3 * sizeof(float));

  // Matrix for 90deg rotation around Z-axis in counter-clockwise direction.
  const float R_z90[9] = {0, 1, 0, -1, 0, 0, 0, 0, 1};

  for (size_t i = 0; i < 4; ++i) {
    poses_[i].resize(kNumTformParams * problem_.src_image_idxs.size());
    int offset = 0;
    for (const auto image_idx : problem_.src_image_idxs) {
      const Image& image = problem_.images->at(image_idx);

      const float K[4] = {
          image.GetK()[0], image.GetK()[2], image.GetK()[4], image.GetK()[5]};
      std::memcpy(poses_[i].data() + offset, K, 4 * sizeof(float));
      offset += 4;

      float rel_R[9];
      float rel_T[3];
      ComputeRelativePose(
          rotated_R, rotated_T, image.GetR(), image.GetT(), rel_R, rel_T);
      std::memcpy(poses_[i].data() + offset, rel_R, 9 * sizeof(float));
      offset += 9;
      std::memcpy(poses_[i].data() + offset, rel_T, 3 * sizeof(float));
      offset += 3;

      float C[3];
      ComputeProjectionCenter(rel_R, rel_T, C);
      std::memcpy(poses_[i].data() + offset, C, 3 * sizeof(float));
      offset += 3;

      float P[12];
      ComposeProjectionMatrix(image.GetK(), rel_R, rel_T, P);
      std::memcpy(poses_[i].data() + offset, P, 12 * sizeof(float));
      offset += 12;

      float inv_P[12];
      ComposeInverseProjectionMatrix(image.GetK(), rel_R, rel_T, inv_P);
      std::memcpy(poses_[i].data() + offset, inv_P, 12 * sizeof(float));
      offset += 12;
    }

    RotatePose(R_z90, rotated_R, rotated_T);
  }
}

void PatchMatchCpu::InitWorkspaceMemory() {
  const size_t num_src = problem_.src_image_idxs.size();

  // Per-pixel PRNG states.
  rand_state_map_.resize(width_ * height_);
  for (size_t i = 0; i < rand_state_map_.size(); ++i) {
    SeedPcg32(static_cast<uint64_t>(i), &rand_state_map_[i]);
  }

  // Depth map.
  depth_map_ = Mat<float>(width_, height_, 1);
  if (options_.geom_consistency) {
    const DepthMap& init_depth_map =
        problem_.depth_maps->at(problem_.ref_image_idx);
    std::memcpy(depth_map_.GetPtr(),
                init_depth_map.GetPtr(),
                width_ * height_ * sizeof(float));
  } else {
    for (size_t row = 0; row < height_; ++row) {
      for (size_t col = 0; col < width_; ++col) {
        depth_map_.Set(row,
                       col,
                       GenerateRandomDepth(options_.depth_min,
                                           options_.depth_max,
                                           &rand_state_map_[row * width_ +
                                                            col]));
      }
    }
  }

  // Normal map.
  normal_map_ = Mat<float>(width_, height_, 3);
  if (options_.geom_consistency) {
    const NormalMap& init_normal_map =
        problem_.normal_maps->at(problem_.ref_image_idx);
    std::memcpy(normal_map_.GetPtr(),
                init_normal_map.GetPtr(),
                3 * width_ * height_ * sizeof(float));
  } else {
    for (size_t row = 0; row < height_; ++row) {
      for (size_t col = 0; col < width_; ++col) {
        float normal[3];
        GenerateRandomNormal(static_cast<int>(row),
                             static_cast<int>(col),
                             ref_inv_K_,
                             &rand_state_map_[row * width_ + col],
                             normal);
        SetSlice3(normal_map_, row, col, normal);
      }
    }
  }

  // Selection probability maps, cost map, consistency mask.
  sel_prob_map_ = Mat<float>(width_, height_, num_src);
  prev_sel_prob_map_ = Mat<float>(width_, height_, num_src);
  prev_sel_prob_map_.Fill(0.5f);
  cost_map_ = Mat<float>(width_, height_, num_src);
  consistency_mask_ = Mat<uint8_t>(0, 0, 0);
}

////////////////////////////////////////////////////////////////////////////////
// Sampling helpers replicating the CUDA texture semantics.
////////////////////////////////////////////////////////////////////////////////

// Reference image: point sampling, normalized uint8, border addressing.
inline float PatchMatchCpu::SampleRefImage(const int row, const int col) const {
  if (row < 0 || col < 0 || row >= static_cast<int>(height_) ||
      col >= static_cast<int>(width_)) {
    return 0.0f;
  }
  return ref_image_.Get(row, col) * kInvMaxIntensity;
}

// Source images: bilinear sampling at pixel centers, normalized uint8,
// border addressing (texels outside the image contribute 0). Equivalent to
// tex2DLayered with cudaFilterModeLinear at coordinate (x + 0.5, y + 0.5).
inline float PatchMatchCpu::SampleSrcImageBilinear(const int image_idx,
                                                   const float x,
                                                   const float y) const {
  const int width = static_cast<int>(src_max_width_);
  const int height = static_cast<int>(src_max_height_);
  const float x0f = std::floor(x);
  const float y0f = std::floor(y);
  const int x0 = static_cast<int>(x0f);
  const int y0 = static_cast<int>(y0f);
  const float fx = x - x0f;
  const float fy = y - y0f;

  const uint8_t* layer =
      src_images_.data() +
      static_cast<size_t>(image_idx) * src_max_width_ * src_max_height_;

  const auto fetch = [&](const int xi, const int yi) -> float {
    if (xi < 0 || yi < 0 || xi >= width || yi >= height) {
      return 0.0f;
    }
    return layer[static_cast<size_t>(yi) * src_max_width_ + xi] *
           kInvMaxIntensity;
  };

  const float v00 = fetch(x0, y0);
  const float v10 = fetch(x0 + 1, y0);
  const float v01 = fetch(x0, y0 + 1);
  const float v11 = fetch(x0 + 1, y0 + 1);

  return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v10) +
         fy * ((1.0f - fx) * v01 + fx * v11);
}

// Source depth maps: nearest neighbor sampling, border addressing (returns
// 0 outside, which is treated as invalid depth). Equivalent to tex2DLayered
// with cudaFilterModePoint at coordinate (x + 0.5, y + 0.5).
inline float PatchMatchCpu::SampleSrcDepthNearest(const int image_idx,
                                                  const float x,
                                                  const float y) const {
  const int xi = static_cast<int>(std::floor(x + 0.5f));
  const int yi = static_cast<int>(std::floor(y + 0.5f));
  if (xi < 0 || yi < 0 || xi >= static_cast<int>(src_max_width_) ||
      yi >= static_cast<int>(src_max_height_)) {
    return 0.0f;
  }
  return src_depth_maps_[static_cast<size_t>(image_idx) * src_max_width_ *
                             src_max_height_ +
                         static_cast<size_t>(yi) * src_max_width_ + xi];
}

////////////////////////////////////////////////////////////////////////////////
// Cost computation.
////////////////////////////////////////////////////////////////////////////////

inline void PatchMatchCpu::ComposeHomography(const int image_idx,
                                             const int row,
                                             const int col,
                                             const float depth,
                                             const float normal[3],
                                             float H[9]) const {
  const float* pose =
      poses_[rotation_in_half_pi_].data() + image_idx * kNumTformParams;

  // Calibration of source image.
  const float* K = pose;
  // Relative rotation between reference and source image.
  const float* R = pose + 4;
  // Relative translation between reference and source image.
  const float* T = pose + 13;

  const float* ref_inv_K = ref_inv_K_;

  // Distance to the plane.
  const float dist =
      depth * (normal[0] * (ref_inv_K[0] * col + ref_inv_K[1]) +
               normal[1] * (ref_inv_K[2] * row + ref_inv_K[3]) + normal[2]);
  const float inv_dist = 1.0f / dist;

  const float inv_dist_N0 = inv_dist * normal[0];
  const float inv_dist_N1 = inv_dist * normal[1];
  const float inv_dist_N2 = inv_dist * normal[2];

  // Homography as H = K * (R - T * n' / d) * Kref^-1.
  H[0] = ref_inv_K[0] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
                         K[1] * (R[6] + inv_dist_N0 * T[2]));
  H[1] = ref_inv_K[2] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
                         K[1] * (R[7] + inv_dist_N1 * T[2]));
  H[2] = K[0] * (R[2] + inv_dist_N2 * T[0]) +
         K[1] * (R[8] + inv_dist_N2 * T[2]) +
         ref_inv_K[1] * (K[0] * (R[0] + inv_dist_N0 * T[0]) +
                         K[1] * (R[6] + inv_dist_N0 * T[2])) +
         ref_inv_K[3] * (K[0] * (R[1] + inv_dist_N1 * T[0]) +
                         K[1] * (R[7] + inv_dist_N1 * T[2]));
  H[3] = ref_inv_K[0] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
                         K[3] * (R[6] + inv_dist_N0 * T[2]));
  H[4] = ref_inv_K[2] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
                         K[3] * (R[7] + inv_dist_N1 * T[2]));
  H[5] = K[2] * (R[5] + inv_dist_N2 * T[1]) +
         K[3] * (R[8] + inv_dist_N2 * T[2]) +
         ref_inv_K[1] * (K[2] * (R[3] + inv_dist_N0 * T[1]) +
                         K[3] * (R[6] + inv_dist_N0 * T[2])) +
         ref_inv_K[3] * (K[2] * (R[4] + inv_dist_N1 * T[1]) +
                         K[3] * (R[7] + inv_dist_N1 * T[2]));
  H[6] = ref_inv_K[0] * (R[6] + inv_dist_N0 * T[2]);
  H[7] = ref_inv_K[2] * (R[7] + inv_dist_N1 * T[2]);
  H[8] = R[8] + ref_inv_K[1] * (R[6] + inv_dist_N0 * T[2]) +
         ref_inv_K[3] * (R[7] + inv_dist_N1 * T[2]) + inv_dist_N2 * T[2];
}

inline void PatchMatchCpu::ComputeViewingAngles(
    const float point[3],
    const float normal[3],
    const int image_idx,
    float* cos_triangulation_angle,
    float* cos_incident_angle) const {
  *cos_triangulation_angle = 0.0f;
  *cos_incident_angle = 0.0f;

  // Projection center of source image.
  const float* C =
      poses_[rotation_in_half_pi_].data() + image_idx * kNumTformParams + 16;

  // Ray from point to camera.
  const float SX[3] = {C[0] - point[0], C[1] - point[1], C[2] - point[2]};

  // Length of ray from reference image to point.
  const float RX_inv_norm = 1.0f / std::sqrt(DotProduct3(point, point));

  // Length of ray from point to source image.
  const float SX_inv_norm = 1.0f / std::sqrt(DotProduct3(SX, SX));

  *cos_incident_angle = DotProduct3(SX, normal) * SX_inv_norm;
  *cos_triangulation_angle =
      -DotProduct3(SX, point) * RX_inv_norm * SX_inv_norm;
}

inline float PatchMatchCpu::ComputePhotoConsistencyCost(
    const int row,
    const int col,
    const float depth,
    const float normal[3],
    const int src_image_idx,
    const float ref_sum,
    const float ref_squared_sum) const {
  const int window_radius = options_.window_radius;
  const int window_step = options_.window_step;
  const int window_size = 2 * window_radius + 1;

  float tform[9];
  ComposeHomography(src_image_idx, row, col, depth, normal, tform);

  float tform_step[8];
  for (int i = 0; i < 8; ++i) {
    tform_step[i] = window_step * tform[i];
  }

  const int row_start = row - window_radius;
  const int col_start = col - window_radius;

  float col_src = tform[0] * col_start + tform[1] * row_start + tform[2];
  float row_src = tform[3] * col_start + tform[4] * row_start + tform[5];
  float z = tform[6] * col_start + tform[7] * row_start + tform[8];
  float base_col_src = col_src;
  float base_row_src = row_src;
  float base_z = z;

  const int ref_center_q =
      (row >= 0 && row < static_cast<int>(height_) && col >= 0 &&
       col < static_cast<int>(width_))
          ? ref_image_.Get(row, col)
          : 0;
  const float ref_color_sum = ref_sum;
  const float ref_color_squared_sum = ref_squared_sum;
  float src_color_sum = 0.0f;
  float src_color_squared_sum = 0.0f;
  float src_ref_color_sum = 0.0f;
  float bilateral_weight_sum = 0.0f;

  for (int win_row = -window_radius; win_row <= window_radius;
       win_row += window_step) {
    const int ref_row = row + win_row;
    const bool row_in_bounds =
        ref_row >= 0 && ref_row < static_cast<int>(height_);
    for (int win_col = -window_radius; win_col <= window_radius;
         win_col += window_step) {
      const float inv_z = 1.0f / z;
      const float norm_col_src = inv_z * col_src;
      const float norm_row_src = inv_z * row_src;

      const int ref_col = col + win_col;
      const int ref_q =
          (row_in_bounds && ref_col >= 0 && ref_col < static_cast<int>(width_))
              ? ref_image_.Get(ref_row, ref_col)
              : 0;
      const float ref_color = ref_q * kInvMaxIntensity;
      const float src_color =
          SampleSrcImageBilinear(src_image_idx, norm_col_src, norm_row_src);

      const float bilateral_weight =
          bilateral_spatial_table_[(win_row + window_radius) * window_size +
                                   (win_col + window_radius)] *
          bilateral_color_table_[std::abs(ref_center_q - ref_q)];

      const float bilateral_weight_src = bilateral_weight * src_color;

      src_color_sum += bilateral_weight_src;
      src_color_squared_sum += bilateral_weight_src * src_color;
      src_ref_color_sum += bilateral_weight_src * ref_color;
      bilateral_weight_sum += bilateral_weight;

      // Accumulate warped source coordinates per row to reduce numerical
      // errors. Note that this is necessary since coordinates usually are in
      // the order of 1000s as opposed to the color values which are
      // normalized to the range [0, 1].
      col_src += tform_step[0];
      row_src += tform_step[3];
      z += tform_step[6];
    }

    base_col_src += tform_step[1];
    base_row_src += tform_step[4];
    base_z += tform_step[7];

    col_src = base_col_src;
    row_src = base_row_src;
    z = base_z;
  }

  const float inv_bilateral_weight_sum = 1.0f / bilateral_weight_sum;
  src_color_sum *= inv_bilateral_weight_sum;
  src_color_squared_sum *= inv_bilateral_weight_sum;
  src_ref_color_sum *= inv_bilateral_weight_sum;

  const float ref_color_var =
      ref_color_squared_sum - ref_color_sum * ref_color_sum;
  const float src_color_var =
      src_color_squared_sum - src_color_sum * src_color_sum;

  // Based on Jensen's Inequality for convex functions, the variance
  // should always be larger than 0. Do not make this threshold smaller.
  if (ref_color_var < kMinVar || src_color_var < kMinVar) {
    return kMaxPhotoCost;
  } else {
    const float src_ref_color_covar =
        src_ref_color_sum - ref_color_sum * src_color_sum;
    const float src_ref_color_var =
        std::sqrt(ref_color_var * src_color_var);
    return std::max(
        0.0f,
        std::min(kMaxPhotoCost, 1.0f - src_ref_color_covar / src_ref_color_var));
  }
}

inline float PatchMatchCpu::ComputeGeomConsistencyCost(
    const float row,
    const float col,
    const float depth,
    const int image_idx,
    const float max_cost) const {
  const float* pose =
      poses_[rotation_in_half_pi_].data() + image_idx * kNumTformParams;
  // Extract projection matrices for source image.
  const float* P = pose + 19;
  const float* inv_P = pose + 31;

  // Project point in reference image to world.
  float forward_point[3];
  ComputePointAtDepth(row, col, ref_inv_K_, depth, forward_point);

  // Project world point to source image.
  const float inv_forward_z =
      1.0f / (P[8] * forward_point[0] + P[9] * forward_point[1] +
              P[10] * forward_point[2] + P[11]);
  float src_col =
      inv_forward_z * (P[0] * forward_point[0] + P[1] * forward_point[1] +
                       P[2] * forward_point[2] + P[3]);
  float src_row =
      inv_forward_z * (P[4] * forward_point[0] + P[5] * forward_point[1] +
                       P[6] * forward_point[2] + P[7]);

  // Extract depth in source image.
  const float src_depth = SampleSrcDepthNearest(image_idx, src_col, src_row);

  // Projection outside of source image.
  if (src_depth == 0.0f) {
    return max_cost;
  }

  // Project point in source image to world.
  src_col *= src_depth;
  src_row *= src_depth;
  const float backward_point_x =
      inv_P[0] * src_col + inv_P[1] * src_row + inv_P[2] * src_depth + inv_P[3];
  const float backward_point_y =
      inv_P[4] * src_col + inv_P[5] * src_row + inv_P[6] * src_depth + inv_P[7];
  const float backward_point_z = inv_P[8] * src_col + inv_P[9] * src_row +
                                 inv_P[10] * src_depth + inv_P[11];
  const float inv_backward_point_z = 1.0f / backward_point_z;

  // Project world point back to reference image.
  const float backward_col =
      inv_backward_point_z *
      (ref_K_[0] * backward_point_x + ref_K_[1] * backward_point_z);
  const float backward_row =
      inv_backward_point_z *
      (ref_K_[2] * backward_point_y + ref_K_[3] * backward_point_z);

  // Return truncated reprojection error between original observation and
  // the forward-backward projected observation.
  const float diff_col = col - backward_col;
  const float diff_row = row - backward_row;
  return std::min(max_cost,
                  std::sqrt(diff_col * diff_col + diff_row * diff_row));
}

////////////////////////////////////////////////////////////////////////////////
// Initial cost computation (mirror of the ComputeInitialCost kernel).
////////////////////////////////////////////////////////////////////////////////

void PatchMatchCpu::ComputeInitialCost() {
  const int width = static_cast<int>(cost_map_.GetWidth());
  const int height = static_cast<int>(cost_map_.GetHeight());
  const int num_src = static_cast<int>(cost_map_.GetDepth());

#pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
  for (int row = 0; row < height; ++row) {
    float normal[3];
    for (int col = 0; col < width; ++col) {
      const float depth = depth_map_.Get(row, col);
      normal_map_.GetSlice(row, col, normal);
      const float ref_sum = ref_sum_image_.Get(row, col);
      const float ref_squared_sum = ref_squared_sum_image_.Get(row, col);
      for (int image_idx = 0; image_idx < num_src; ++image_idx) {
        cost_map_.Set(row,
                      col,
                      image_idx,
                      ComputePhotoConsistencyCost(row,
                                                  col,
                                                  depth,
                                                  normal,
                                                  image_idx,
                                                  ref_sum,
                                                  ref_squared_sum));
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Sweep from top to bottom (mirror of the SweepFromTopToBottom kernel).
// Columns are independent and processed in parallel; rows are sequential
// within each column due to the propagation.
////////////////////////////////////////////////////////////////////////////////

void PatchMatchCpu::SweepFromTopToBottom(const SweepOptions& sweep_options,
                                         const bool geom_consistency_term,
                                         const bool filter_photo_consistency,
                                         const bool filter_geom_consistency) {
  const int width = static_cast<int>(cost_map_.GetWidth());
  const int height = static_cast<int>(cost_map_.GetHeight());
  const int num_src = static_cast<int>(cost_map_.GetDepth());

  const LikelihoodComputer likelihood_computer(
      sweep_options.ncc_sigma,
      sweep_options.min_triangulation_angle,
      sweep_options.incident_angle_sigma);

  const int window_size = 2 * options_.window_radius + 1;
  const bool filter = filter_photo_consistency || filter_geom_consistency;

#pragma omp parallel for schedule(dynamic) num_threads(num_threads_)
  for (int col = 0; col < width; ++col) {
    std::vector<float> forward_message(num_src);
    std::vector<float> sampling_probs(num_src);

    ////////////////////////////////////////////////////////////////////////////
    // Compute backward message for all rows. Note that the backward messages
    // are temporarily stored in the sel_prob_map and replaced row by row as
    // the updated forward messages are computed further below.
    ////////////////////////////////////////////////////////////////////////////

    for (int image_idx = 0; image_idx < num_src; ++image_idx) {
      float beta = kUniformProb;
      for (int row = height - 1; row >= 0; --row) {
        const float cost = cost_map_.Get(row, col, image_idx);
        beta = likelihood_computer.ComputeBackwardMessage(cost, beta);
        sel_prob_map_.Set(row, col, image_idx, beta);
      }
      forward_message[image_idx] = kUniformProb;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Estimate parameters for remaining rows and compute selection
    // probabilities.
    ////////////////////////////////////////////////////////////////////////////

    struct ParamState {
      float depth = 0.0f;
      float normal[3] = {0};
    };

    // Parameters of previous pixel in column.
    ParamState prev_param_state;
    // Parameters of current pixel in column.
    ParamState curr_param_state;
    // Randomly sampled parameters.
    ParamState rand_param_state;

    // PRNG state for the current column (row 0, like the CUDA kernel).
    Pcg32State rand_state = rand_state_map_[col];

    // Parameters for first row in column.
    prev_param_state.depth = depth_map_.Get(0, col);
    normal_map_.GetSlice(0, col, prev_param_state.normal);

    for (int row = 0; row < height; ++row) {
      const float ref_sum = ref_sum_image_.Get(row, col);
      const float ref_squared_sum = ref_squared_sum_image_.Get(row, col);

      // Propagate the depth at which the current ray intersects with the
      // plane of the normal of the previous ray. This helps to better
      // estimate the depth of very oblique structures, i.e. pixels whose
      // normal direction is significantly different from their viewing
      // direction.
      prev_param_state.depth = PropagateDepth(ref_inv_K_,
                                              prev_param_state.depth,
                                              prev_param_state.normal,
                                              row - 1,
                                              row);

      // Read parameters for current pixel from previous sweep.
      curr_param_state.depth = depth_map_.Get(row, col);
      normal_map_.GetSlice(row, col, curr_param_state.normal);

      // Generate random parameters.
      rand_param_state.depth = PerturbDepth(
          sweep_options.perturbation, curr_param_state.depth, &rand_state);
      PerturbNormal(row,
                    col,
                    sweep_options.perturbation * kPi,
                    ref_inv_K_,
                    curr_param_state.normal,
                    &rand_state,
                    rand_param_state.normal);

      // Read in the backward message, compute selection probabilities and
      // modulate selection probabilities with priors.

      float point[3];
      ComputePointAtDepth(
          row, col, ref_inv_K_, curr_param_state.depth, point);

      for (int image_idx = 0; image_idx < num_src; ++image_idx) {
        const float cost = cost_map_.Get(row, col, image_idx);
        const float alpha = likelihood_computer.ComputeForwardMessage(
            cost, forward_message[image_idx]);
        const float beta = sel_prob_map_.Get(row, col, image_idx);
        const float prev_prob = prev_sel_prob_map_.Get(row, col, image_idx);
        const float sel_prob = likelihood_computer.ComputeSelProb(
            alpha, beta, prev_prob, sweep_options.prev_sel_prob_weight);

        float cos_triangulation_angle;
        float cos_incident_angle;
        ComputeViewingAngles(point,
                             curr_param_state.normal,
                             image_idx,
                             &cos_triangulation_angle,
                             &cos_incident_angle);
        const float tri_prob =
            likelihood_computer.ComputeTriProb(cos_triangulation_angle);
        const float inc_prob =
            likelihood_computer.ComputeIncProb(cos_incident_angle);

        float H[9];
        ComposeHomography(image_idx,
                          row,
                          col,
                          curr_param_state.depth,
                          curr_param_state.normal,
                          H);
        const float res_prob = likelihood_computer.ComputeResolutionProb(
            H, row, col, window_size);

        sampling_probs[image_idx] = sel_prob * tri_prob * inc_prob * res_prob;
      }

      TransformPDFToCDF(sampling_probs.data(), num_src);

      // Compute matching cost using Monte Carlo sampling of source images.
      // Images with higher selection probability are more likely to be
      // sampled. Hence, if only very few source images see the reference
      // image pixel, the same source image is likely to be sampled many
      // times. Instead of taking the best K probabilities, this sampling
      // scheme has the advantage of being adaptive to any distribution of
      // selection probabilities.

      float costs[kNumCosts] = {0};
      const float depths[kNumCosts] = {curr_param_state.depth,
                                       prev_param_state.depth,
                                       rand_param_state.depth,
                                       curr_param_state.depth,
                                       rand_param_state.depth};
      const float* normals[kNumCosts] = {curr_param_state.normal,
                                         prev_param_state.normal,
                                         rand_param_state.normal,
                                         rand_param_state.normal,
                                         curr_param_state.normal};

      for (int sample = 0; sample < sweep_options.num_samples; ++sample) {
        const float rand_prob = RandUniform(&rand_state) - FLT_EPSILON;

        int sampled_image_idx = -1;
        for (int image_idx = 0; image_idx < num_src; ++image_idx) {
          const float prob = sampling_probs[image_idx];
          if (prob > rand_prob) {
            sampled_image_idx = image_idx;
            break;
          }
        }

        if (sampled_image_idx == -1) {
          continue;
        }

        costs[0] += cost_map_.Get(row, col, sampled_image_idx);
        if (geom_consistency_term) {
          costs[0] += sweep_options.geom_consistency_regularizer *
                      ComputeGeomConsistencyCost(
                          row,
                          col,
                          depths[0],
                          sampled_image_idx,
                          sweep_options.geom_consistency_max_cost);
        }

        for (int i = 1; i < kNumCosts; ++i) {
          costs[i] += ComputePhotoConsistencyCost(row,
                                                  col,
                                                  depths[i],
                                                  normals[i],
                                                  sampled_image_idx,
                                                  ref_sum,
                                                  ref_squared_sum);
          if (geom_consistency_term) {
            costs[i] += sweep_options.geom_consistency_regularizer *
                        ComputeGeomConsistencyCost(
                            row,
                            col,
                            depths[i],
                            sampled_image_idx,
                            sweep_options.geom_consistency_max_cost);
          }
        }
      }

      // Find the parameters of the minimum cost.
      const int min_cost_idx = FindMinCost(costs);
      const float best_depth = depths[min_cost_idx];
      const float* best_normal = normals[min_cost_idx];

      // Save best new parameters.
      depth_map_.Set(row, col, best_depth);
      SetSlice3(normal_map_, row, col, best_normal);

      // Use the new cost to recompute the updated forward message and
      // the selection probability.
      for (int image_idx = 0; image_idx < num_src; ++image_idx) {
        // Determine the cost for best depth.
        float cost;
        if (min_cost_idx == 0) {
          cost = cost_map_.Get(row, col, image_idx);
        } else {
          cost = ComputePhotoConsistencyCost(row,
                                             col,
                                             best_depth,
                                             best_normal,
                                             image_idx,
                                             ref_sum,
                                             ref_squared_sum);
          cost_map_.Set(row, col, image_idx, cost);
        }

        const float alpha = likelihood_computer.ComputeForwardMessage(
            cost, forward_message[image_idx]);
        const float beta = sel_prob_map_.Get(row, col, image_idx);
        const float prev_prob = prev_sel_prob_map_.Get(row, col, image_idx);
        const float prob = likelihood_computer.ComputeSelProb(
            alpha, beta, prev_prob, sweep_options.prev_sel_prob_weight);
        forward_message[image_idx] = alpha;
        sel_prob_map_.Set(row, col, image_idx, prob);
      }

      if (filter) {
        int num_consistent = 0;

        float best_point[3];
        ComputePointAtDepth(row, col, ref_inv_K_, best_depth, best_point);

        const float min_ncc_prob = likelihood_computer.ComputeNCCProb(
            1.0f - sweep_options.filter_min_ncc);
        const float cos_min_triangulation_angle =
            std::cos(sweep_options.filter_min_triangulation_angle);

        for (int image_idx = 0; image_idx < num_src; ++image_idx) {
          float cos_triangulation_angle;
          float cos_incident_angle;
          ComputeViewingAngles(best_point,
                               best_normal,
                               image_idx,
                               &cos_triangulation_angle,
                               &cos_incident_angle);
          if (cos_triangulation_angle > cos_min_triangulation_angle ||
              cos_incident_angle <= 0.0f) {
            continue;
          }

          if (!filter_geom_consistency) {
            if (sel_prob_map_.Get(row, col, image_idx) >= min_ncc_prob) {
              consistency_mask_.Set(row, col, image_idx, 1);
              num_consistent += 1;
            }
          } else if (!filter_photo_consistency) {
            if (ComputeGeomConsistencyCost(
                    row,
                    col,
                    best_depth,
                    image_idx,
                    sweep_options.geom_consistency_max_cost) <=
                sweep_options.filter_geom_consistency_max_cost) {
              consistency_mask_.Set(row, col, image_idx, 1);
              num_consistent += 1;
            }
          } else {
            if (sel_prob_map_.Get(row, col, image_idx) >= min_ncc_prob &&
                ComputeGeomConsistencyCost(
                    row,
                    col,
                    best_depth,
                    image_idx,
                    sweep_options.geom_consistency_max_cost) <=
                    sweep_options.filter_geom_consistency_max_cost) {
              consistency_mask_.Set(row, col, image_idx, 1);
              num_consistent += 1;
            }
          }
        }

        if (num_consistent < sweep_options.filter_min_num_consistent) {
          depth_map_.Set(row, col, 0.0f);
          normal_map_.Set(row, col, 0, 0.0f);
          normal_map_.Set(row, col, 1, 0.0f);
          normal_map_.Set(row, col, 2, 0.0f);
          for (int image_idx = 0; image_idx < num_src; ++image_idx) {
            consistency_mask_.Set(row, col, image_idx, 0);
          }
        }
      }

      // Update previous depth for next row.
      prev_param_state.depth = best_depth;
      for (int i = 0; i < 3; ++i) {
        prev_param_state.normal[i] = best_normal[i];
      }
    }

    rand_state_map_[col] = rand_state;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Rotation of all maps by 90 degrees counter-clockwise (mirror of
// PatchMatchCuda::Rotate).
////////////////////////////////////////////////////////////////////////////////

void PatchMatchCpu::Rotate() {
  rotation_in_half_pi_ = (rotation_in_half_pi_ + 1) % 4;

  size_t width;
  size_t height;
  if (rotation_in_half_pi_ % 2 == 0) {
    width = ref_width_;
    height = ref_height_;
  } else {
    width = ref_height_;
    height = ref_width_;
  }

  const size_t num_src = problem_.src_image_idxs.size();

  // Rotate random map.
  rand_state_map_ = RotateVecCCW(rand_state_map_, width_, height_);

  // Rotate depth map.
  depth_map_ = RotateMatCCW(depth_map_);

  // Rotate normal map: first rotate the normal vector components by 90deg
  // around the z-axis in counter-clockwise direction, then rotate spatially.
  {
    const size_t old_height = normal_map_.GetHeight();
    const size_t old_width = normal_map_.GetWidth();
    for (size_t row = 0; row < old_height; ++row) {
      for (size_t col = 0; col < old_width; ++col) {
        float normal[3];
        normal_map_.GetSlice(row, col, normal);
        const float rotated_normal[3] = {normal[1], -normal[0], normal[2]};
        SetSlice3(normal_map_, row, col, rotated_normal);
      }
    }
    normal_map_ = RotateMatCCW(normal_map_);
  }

  // Rotate reference image (intensities and precomputed local sums).
  ref_image_ = RotateMatCCW(ref_image_);
  ref_sum_image_ = RotateMatCCW(ref_sum_image_);
  ref_squared_sum_image_ = RotateMatCCW(ref_squared_sum_image_);

  // Rotate selection probability map: the rotated current selection
  // probabilities become the previous ones, and a fresh map is allocated
  // for the next sweep.
  prev_sel_prob_map_ = RotateMatCCW(sel_prob_map_);
  sel_prob_map_ = Mat<float>(width, height, num_src);

  // Rotate cost map.
  cost_map_ = RotateMatCCW(cost_map_);

  // Rotate calibration.
  ref_K_ = ref_K_host_[rotation_in_half_pi_];
  ref_inv_K_ = ref_inv_K_host_[rotation_in_half_pi_];

  // Update current dimensions.
  width_ = width;
  height_ = height;
}

////////////////////////////////////////////////////////////////////////////////
// Run.
////////////////////////////////////////////////////////////////////////////////

void PatchMatchCpu::Run() {
  Timer total_timer;
  total_timer.Start();

  LOG(INFO) << "Running PatchMatch stereo on the CPU with " << num_threads_
            << " threads...";

  Timer init_timer;
  init_timer.Start();
  ComputeInitialCost();
  LOG(INFO) << "Initialization: " << init_timer.ElapsedSeconds() << " [s]";

  const float total_num_steps = options_.num_iterations * 4;

  SweepOptions sweep_options;
  sweep_options.depth_min = options_.depth_min;
  sweep_options.depth_max = options_.depth_max;
  sweep_options.sigma_spatial = options_.sigma_spatial;
  sweep_options.sigma_color = options_.sigma_color;
  sweep_options.num_samples = options_.num_samples;
  sweep_options.ncc_sigma = options_.ncc_sigma;
  sweep_options.min_triangulation_angle =
      static_cast<float>(options_.min_triangulation_angle) * kDegToRad;
  sweep_options.incident_angle_sigma = options_.incident_angle_sigma;
  sweep_options.geom_consistency_regularizer =
      options_.geom_consistency_regularizer;
  sweep_options.geom_consistency_max_cost = options_.geom_consistency_max_cost;
  sweep_options.filter_min_ncc = options_.filter_min_ncc;
  sweep_options.filter_min_triangulation_angle =
      static_cast<float>(options_.filter_min_triangulation_angle) * kDegToRad;
  sweep_options.filter_min_num_consistent = options_.filter_min_num_consistent;
  sweep_options.filter_geom_consistency_max_cost =
      options_.filter_geom_consistency_max_cost;

  for (int iter = 0; iter < options_.num_iterations; ++iter) {
    Timer iter_timer;
    iter_timer.Start();

    for (int sweep = 0; sweep < 4; ++sweep) {
      Timer sweep_timer;
      sweep_timer.Start();

      // Exponentially reduce amount of perturbation during the optimization.
      sweep_options.perturbation = 1.0f / std::pow(2.0f, iter + sweep / 4.0f);

      // Linearly increase the influence of previous selection probabilities.
      sweep_options.prev_sel_prob_weight =
          static_cast<float>(iter * 4 + sweep) / total_num_steps;

      const bool last_sweep =
          iter == options_.num_iterations - 1 && sweep == 3;

      // Mirror the CUDA template dispatch: filtering only happens in the
      // last sweep; with geometric consistency, both the photometric and the
      // geometric filters are active, otherwise only the photometric one.
      bool filter_photo_consistency = false;
      bool filter_geom_consistency = false;
      if (last_sweep && options_.filter) {
        consistency_mask_ = Mat<uint8_t>(cost_map_.GetWidth(),
                                         cost_map_.GetHeight(),
                                         cost_map_.GetDepth());
        consistency_mask_.Fill(0);
        filter_photo_consistency = true;
        filter_geom_consistency = options_.geom_consistency;
      }

      SweepFromTopToBottom(sweep_options,
                           options_.geom_consistency,
                           filter_photo_consistency,
                           filter_geom_consistency);

      Rotate();

      // Rotate the consistency mask alongside the other maps.
      if (last_sweep && options_.filter) {
        consistency_mask_ = RotateMatCCW(consistency_mask_);
      }

      LOG(INFO) << " Sweep " << sweep + 1 << ": "
                << sweep_timer.ElapsedSeconds() << " [s]";
    }

    LOG(INFO) << "Iteration " << iter + 1 << ": "
              << iter_timer.ElapsedSeconds() << " [s]";
  }

  LOG(INFO) << "Total: " << total_timer.ElapsedSeconds() << " [s]";
}

////////////////////////////////////////////////////////////////////////////////
// Result extraction.
////////////////////////////////////////////////////////////////////////////////

DepthMap PatchMatchCpu::GetDepthMap() const {
  return DepthMap(depth_map_, options_.depth_min, options_.depth_max);
}

NormalMap PatchMatchCpu::GetNormalMap() const {
  return NormalMap(normal_map_);
}

Mat<float> PatchMatchCpu::GetSelProbMap() const { return prev_sel_prob_map_; }

std::vector<int> PatchMatchCpu::GetConsistentImageIdxs() const {
  const Mat<uint8_t>& mask = consistency_mask_;
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
      if (pixel_consistent_image_idxs.size() > 0) {
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
