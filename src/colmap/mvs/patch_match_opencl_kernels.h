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

namespace colmap {
namespace mvs {

// OpenCL C source for the PatchMatch photometric backend (Phase 2b). This is a
// faithful port of the validated CPU implementation in patch_match_cpu.cc:
// identical PCG32 PRNG, identical cost function (bilaterally weighted NCC),
// identical sweeping scheme (one work-item per column, sequential rows). The
// geometric-consistency term is NOT included here (Phase 2c); only the
// photometric path + photometric filter are implemented.
//
// Layout conventions (must match the host and Mat<T>):
//   - reference-frame maps are slice-major: idx(r,c,s) = s*W*H + r*W + c.
//   - source images / depths are layer-major (zero-padded to src_max dims):
//     idx = layer*src_max_w*src_max_h + y*src_max_w + x.
//   - poses are [rotation][image][43] with stride kNumTformParams=43.
//
// Source images are NOT rotated (only the reference-frame maps are); the
// per-rotation relative poses account for the reference-frame rotation.
inline constexpr const char* kPatchMatchOpenCLKernelSource = R"CLC(
// ---------------------------------------------------------------------------
// Constants (mirror patch_match_cpu.cc / patch_match_cuda.cu).
// ---------------------------------------------------------------------------
#define KPI 3.14159265358979323846f
#define KUNIFORM_PROB 0.5f
#define KMAX_PHOTO_COST 2.0f
#define KMIN_VAR 1e-5f
#define KNUM_COSTS 5
#define KMAX_PERTURB_NORMAL_TRIALS 3
#define KINV_MAX_INTENSITY (1.0f / 255.0f)
// Upper bound on the number of source images (sizes private per-column arrays).
#define MAX_SRC 32
// Number of transform parameters per source image: K(4) R(9) T(3) C(3) P(12) invP(12).
#define KNUM_TFORM 43

// ---------------------------------------------------------------------------
// PCG32 PRNG (O'Neill, 2014); RandUniform returns a float in (0, 1].
// ---------------------------------------------------------------------------
uint pcg32_next(private ulong* state, ulong inc) {
  ulong old_state = *state;
  *state = old_state * 6364136223846793005UL + inc;
  uint xorshifted = (uint)(((old_state >> 18u) ^ old_state) >> 27u);
  uint rot = (uint)(old_state >> 59u);
  return (xorshifted >> rot) | (xorshifted << ((0u - rot) & 31u));
}

float rand_uniform(private ulong* state, ulong inc) {
  return ((pcg32_next(state, inc) >> 8) + 1) * 0x1p-24f;
}

// ---------------------------------------------------------------------------
// Small math helpers (verbatim ports from patch_match_cpu.cc).
// ---------------------------------------------------------------------------
void mat33_dot_vec3(const float* mat, const float* vec, float* result) {
  result[0] = mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2];
  result[1] = mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2];
  result[2] = mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2];
}

void mat33_dot_vec3_homogeneous(const float* mat,
                                const float* vec,
                                float* result) {
  const float inv_z = 1.0f / (mat[6] * vec[0] + mat[7] * vec[1] + mat[8]);
  result[0] = inv_z * (mat[0] * vec[0] + mat[1] * vec[1] + mat[2]);
  result[1] = inv_z * (mat[3] * vec[0] + mat[4] * vec[1] + mat[5]);
}

float dot3(const float* a, const float* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

float generate_random_depth(float depth_min,
                            float depth_max,
                            private ulong* state,
                            ulong inc) {
  return rand_uniform(state, inc) * (depth_max - depth_min) + depth_min;
}

void generate_random_normal(int row,
                            int col,
                            const float* ref_inv_K,
                            private ulong* state,
                            ulong inc,
                            float* normal) {
  float v1 = 0.0f;
  float v2 = 0.0f;
  float s = 2.0f;
  while (s >= 1.0f) {
    v1 = 2.0f * rand_uniform(state, inc) - 1.0f;
    v2 = 2.0f * rand_uniform(state, inc) - 1.0f;
    s = v1 * v1 + v2 * v2;
  }
  const float s_norm = sqrt(1.0f - s);
  normal[0] = 2.0f * v1 * s_norm;
  normal[1] = 2.0f * v2 * s_norm;
  normal[2] = 1.0f - 2.0f * s;

  const float view_ray[3] = {ref_inv_K[0] * col + ref_inv_K[1],
                             ref_inv_K[2] * row + ref_inv_K[3],
                             1.0f};
  if (dot3(normal, view_ray) > 0.0f) {
    normal[0] = -normal[0];
    normal[1] = -normal[1];
    normal[2] = -normal[2];
  }
}

float perturb_depth(float perturbation, float depth, private ulong* state, ulong inc) {
  const float depth_min = (1.0f - perturbation) * depth;
  const float depth_max = (1.0f + perturbation) * depth;
  return generate_random_depth(depth_min, depth_max, state, inc);
}

// Iterative replacement for the recursive CPU PerturbNormal (OpenCL C forbids
// recursion). Halves the perturbation up to kMaxPerturbNormalTrials times if
// the perturbed normal flips to face the camera.
void perturb_normal(int row,
                    int col,
                    float perturbation,
                    const float* ref_inv_K,
                    const float* normal,
                    private ulong* state,
                    ulong inc,
                    float* perturbed_normal) {
  const float view_ray[3] = {ref_inv_K[0] * col + ref_inv_K[1],
                             ref_inv_K[2] * row + ref_inv_K[3],
                             1.0f};
  float pert = perturbation;
  for (int trial = 0; trial <= KMAX_PERTURB_NORMAL_TRIALS; ++trial) {
    const float a1 = (rand_uniform(state, inc) - 0.5f) * pert;
    const float a2 = (rand_uniform(state, inc) - 0.5f) * pert;
    const float a3 = (rand_uniform(state, inc) - 0.5f) * pert;
    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);

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

    mat33_dot_vec3(R, normal, perturbed_normal);

    if (dot3(perturbed_normal, view_ray) < 0.0f) {
      const float inv_norm = 1.0f / sqrt(dot3(perturbed_normal, perturbed_normal));
      perturbed_normal[0] *= inv_norm;
      perturbed_normal[1] *= inv_norm;
      perturbed_normal[2] *= inv_norm;
      return;
    }
    if (trial == KMAX_PERTURB_NORMAL_TRIALS) {
      perturbed_normal[0] = normal[0];
      perturbed_normal[1] = normal[1];
      perturbed_normal[2] = normal[2];
      return;
    }
    pert *= 0.5f;
  }
}

void compute_point_at_depth(float row,
                            float col,
                            const float* ref_inv_K,
                            float depth,
                            float* point) {
  point[0] = depth * (ref_inv_K[0] * col + ref_inv_K[1]);
  point[1] = depth * (ref_inv_K[2] * row + ref_inv_K[3]);
  point[2] = depth;
}

float propagate_depth(const float* ref_inv_K,
                      float depth1,
                      const float* normal1,
                      float row1,
                      float row2) {
  const float x1 = depth1 * (ref_inv_K[2] * row1 + ref_inv_K[3]);
  const float y1 = depth1;
  const float x2 = x1 + normal1[2];
  const float y2 = y1 - normal1[1];
  const float x4 = ref_inv_K[2] * row2 + ref_inv_K[3];
  const float denom = x2 - x1 + x4 * (y1 - y2);
  const float kEps = 1e-5f;
  if (fabs(denom) < kEps) {
    return depth1;
  }
  const float nom = y1 * x2 - x1 * y2;
  return nom / denom;
}

int find_min_cost(const float* costs) {
  float min_cost = costs[0];
  int min_cost_idx = 0;
  for (int idx = 1; idx < KNUM_COSTS; ++idx) {
    if (costs[idx] <= min_cost) {
      min_cost = costs[idx];
      min_cost_idx = idx;
    }
  }
  return min_cost_idx;
}

void transform_pdf_to_cdf(private float* probs, int num_probs) {
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

// ---------------------------------------------------------------------------
// Likelihood computer (constants precomputed on the host and passed in).
// ---------------------------------------------------------------------------
float compute_ncc_prob(float cost, float inv_ncc_sigma_sq, float ncc_norm_factor) {
  return exp(cost * cost * inv_ncc_sigma_sq) * ncc_norm_factor;
}

float compute_message(float cost,
                      float prev,
                      int forward,
                      float inv_ncc_sigma_sq,
                      float ncc_norm_factor) {
  const float kNoChangeProb = 0.99999f;
  const float kChangeProb = 1.0f - kNoChangeProb;
  const float emission = compute_ncc_prob(cost, inv_ncc_sigma_sq, ncc_norm_factor);
  float zn0;
  float zn1;
  if (forward) {
    zn0 = (prev * kChangeProb + (1.0f - prev) * kNoChangeProb) * KUNIFORM_PROB;
    zn1 = (prev * kNoChangeProb + (1.0f - prev) * kChangeProb) * emission;
  } else {
    zn0 = prev * emission * kChangeProb + (1.0f - prev) * KUNIFORM_PROB * kNoChangeProb;
    zn1 = prev * emission * kNoChangeProb + (1.0f - prev) * KUNIFORM_PROB * kChangeProb;
  }
  return zn1 / (zn0 + zn1);
}

float compute_sel_prob(float alpha, float beta, float prev, float prev_weight) {
  const float zn0 = (1.0f - alpha) * (1.0f - beta);
  const float zn1 = alpha * beta;
  const float curr = zn1 / (zn0 + zn1);
  return prev_weight * prev + (1.0f - prev_weight) * curr;
}

float compute_tri_prob(float cos_triangulation_angle, float cos_min_triangulation_angle) {
  if (cos_triangulation_angle > cos_min_triangulation_angle) {
    const float scaled = 1.0f - (1.0f - cos_triangulation_angle) /
                                    (1.0f - cos_min_triangulation_angle);
    const float likelihood = 1.0f - scaled * scaled;
    return fmin(1.0f, fmax(0.0f, likelihood));
  }
  return 1.0f;
}

float compute_inc_prob(float cos_incident_angle, float inv_incident_angle_sigma_sq) {
  const float x = 1.0f - fmax(0.0f, cos_incident_angle);
  return exp(x * x * inv_incident_angle_sigma_sq);
}

float compute_resolution_prob(const float* H, float row, float col, int window_size) {
  const int window_radius = window_size / 2;
  float src1[2];
  const float ref1[2] = {col - window_radius, row - window_radius};
  mat33_dot_vec3_homogeneous(H, ref1, src1);
  float src2[2];
  const float ref2[2] = {col - window_radius, row + window_radius};
  mat33_dot_vec3_homogeneous(H, ref2, src2);
  float src3[2];
  const float ref3[2] = {col + window_radius, row + window_radius};
  mat33_dot_vec3_homogeneous(H, ref3, src3);
  float src4[2];
  const float ref4[2] = {col + window_radius, row - window_radius};
  mat33_dot_vec3_homogeneous(H, ref4, src4);

  const float ref_area = window_size * window_size;
  const float src_area = fabs(0.5f * (src1[0] * src2[1] - src2[0] * src1[1] -
                                      src1[0] * src4[1] + src2[0] * src3[1] -
                                      src3[0] * src2[1] + src4[0] * src1[1] +
                                      src3[0] * src4[1] - src4[0] * src3[1]));
  if (ref_area > src_area) {
    return src_area / ref_area;
  }
  return ref_area / src_area;
}

// ---------------------------------------------------------------------------
// Sampling helpers (manual float bilinear/nearest for parity with the CPU).
// ---------------------------------------------------------------------------
float sample_src_image_bilinear(__global const uchar* src_images,
                                int image_idx,
                                int src_max_w,
                                int src_max_h,
                                float x,
                                float y) {
  const float x0f = floor(x);
  const float y0f = floor(y);
  const int x0 = (int)x0f;
  const int y0 = (int)y0f;
  const float fx = x - x0f;
  const float fy = y - y0f;
  __global const uchar* layer =
      src_images + (size_t)image_idx * src_max_w * src_max_h;

  float v00 = 0.0f, v10 = 0.0f, v01 = 0.0f, v11 = 0.0f;
  if (x0 >= 0 && y0 >= 0 && x0 < src_max_w && y0 < src_max_h)
    v00 = layer[(size_t)y0 * src_max_w + x0] * KINV_MAX_INTENSITY;
  if (x0 + 1 >= 0 && y0 >= 0 && x0 + 1 < src_max_w && y0 < src_max_h)
    v10 = layer[(size_t)y0 * src_max_w + (x0 + 1)] * KINV_MAX_INTENSITY;
  if (x0 >= 0 && y0 + 1 >= 0 && x0 < src_max_w && y0 + 1 < src_max_h)
    v01 = layer[(size_t)(y0 + 1) * src_max_w + x0] * KINV_MAX_INTENSITY;
  if (x0 + 1 >= 0 && y0 + 1 >= 0 && x0 + 1 < src_max_w && y0 + 1 < src_max_h)
    v11 = layer[(size_t)(y0 + 1) * src_max_w + (x0 + 1)] * KINV_MAX_INTENSITY;

  return (1.0f - fy) * ((1.0f - fx) * v00 + fx * v10) +
         fy * ((1.0f - fx) * v01 + fx * v11);
}

// ---------------------------------------------------------------------------
// Homography H = K * (R - T * n' / d) * Kref^-1 (verbatim port).
// pose layout: K(4) at 0, R(9) at 4, T(3) at 13, C(3) at 16, P/invP after.
// ---------------------------------------------------------------------------
void compose_homography(__global const float* pose,
                        const float* ref_inv_K,
                        int row,
                        int col,
                        float depth,
                        const float* normal,
                        float* H) {
  const float K0 = pose[0], K1 = pose[1], K2 = pose[2], K3 = pose[3];
  __global const float* R = pose + 4;
  __global const float* T = pose + 13;

  const float dist =
      depth * (normal[0] * (ref_inv_K[0] * col + ref_inv_K[1]) +
               normal[1] * (ref_inv_K[2] * row + ref_inv_K[3]) + normal[2]);
  const float inv_dist = 1.0f / dist;
  const float inv_dist_N0 = inv_dist * normal[0];
  const float inv_dist_N1 = inv_dist * normal[1];
  const float inv_dist_N2 = inv_dist * normal[2];

  H[0] = ref_inv_K[0] * (K0 * (R[0] + inv_dist_N0 * T[0]) +
                         K1 * (R[6] + inv_dist_N0 * T[2]));
  H[1] = ref_inv_K[2] * (K0 * (R[1] + inv_dist_N1 * T[0]) +
                         K1 * (R[7] + inv_dist_N1 * T[2]));
  H[2] = K0 * (R[2] + inv_dist_N2 * T[0]) +
         K1 * (R[8] + inv_dist_N2 * T[2]) +
         ref_inv_K[1] * (K0 * (R[0] + inv_dist_N0 * T[0]) +
                         K1 * (R[6] + inv_dist_N0 * T[2])) +
         ref_inv_K[3] * (K0 * (R[1] + inv_dist_N1 * T[0]) +
                         K1 * (R[7] + inv_dist_N1 * T[2]));
  H[3] = ref_inv_K[0] * (K2 * (R[3] + inv_dist_N0 * T[1]) +
                         K3 * (R[6] + inv_dist_N0 * T[2]));
  H[4] = ref_inv_K[2] * (K2 * (R[4] + inv_dist_N1 * T[1]) +
                         K3 * (R[7] + inv_dist_N1 * T[2]));
  H[5] = K2 * (R[5] + inv_dist_N2 * T[1]) +
         K3 * (R[8] + inv_dist_N2 * T[2]) +
         ref_inv_K[1] * (K2 * (R[3] + inv_dist_N0 * T[1]) +
                         K3 * (R[6] + inv_dist_N0 * T[2])) +
         ref_inv_K[3] * (K2 * (R[4] + inv_dist_N1 * T[1]) +
                         K3 * (R[7] + inv_dist_N1 * T[2]));
  H[6] = ref_inv_K[0] * (R[6] + inv_dist_N0 * T[2]);
  H[7] = ref_inv_K[2] * (R[7] + inv_dist_N1 * T[2]);
  H[8] = R[8] + ref_inv_K[1] * (R[6] + inv_dist_N0 * T[2]) +
         ref_inv_K[3] * (R[7] + inv_dist_N1 * T[2]) + inv_dist_N2 * T[2];
}

void compute_viewing_angles(__global const float* pose,
                            const float* point,
                            const float* normal,
                            float* cos_triangulation_angle,
                            float* cos_incident_angle) {
  __global const float* C = pose + 16;
  const float SX[3] = {C[0] - point[0], C[1] - point[1], C[2] - point[2]};
  const float RX_inv_norm = 1.0f / sqrt(dot3(point, point));
  const float SX_inv_norm = 1.0f / sqrt(dot3(SX, SX));
  *cos_incident_angle = dot3(SX, normal) * SX_inv_norm;
  *cos_triangulation_angle = -dot3(SX, point) * RX_inv_norm * SX_inv_norm;
}

// Bilaterally weighted NCC photo-consistency cost (verbatim port, with the
// incremental warp accumulation to limit numerical error).
float compute_photo_consistency_cost(__global const uchar* ref_image,
                                     __global const uchar* src_images,
                                     __global const float* pose,
                                     const float* ref_inv_K,
                                     __global const float* bilateral_spatial,
                                     __global const float* bilateral_color,
                                     int width,
                                     int height,
                                     int src_max_w,
                                     int src_max_h,
                                     int window_radius,
                                     int window_step,
                                     int row,
                                     int col,
                                     float depth,
                                     const float* normal,
                                     int src_image_idx,
                                     float ref_sum,
                                     float ref_squared_sum) {
  const int window_size = 2 * window_radius + 1;

  float tform[9];
  compose_homography(pose, ref_inv_K, row, col, depth, normal, tform);

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
      (row >= 0 && row < height && col >= 0 && col < width)
          ? ref_image[(size_t)row * width + col]
          : 0;
  float src_color_sum = 0.0f;
  float src_color_squared_sum = 0.0f;
  float src_ref_color_sum = 0.0f;
  float bilateral_weight_sum = 0.0f;

  for (int win_row = -window_radius; win_row <= window_radius;
       win_row += window_step) {
    const int ref_row = row + win_row;
    const int row_in_bounds = (ref_row >= 0 && ref_row < height);
    for (int win_col = -window_radius; win_col <= window_radius;
         win_col += window_step) {
      const float inv_z = 1.0f / z;
      const float norm_col_src = inv_z * col_src;
      const float norm_row_src = inv_z * row_src;

      const int ref_col = col + win_col;
      const int ref_q =
          (row_in_bounds && ref_col >= 0 && ref_col < width)
              ? ref_image[(size_t)ref_row * width + ref_col]
              : 0;
      const float ref_color = ref_q * KINV_MAX_INTENSITY;
      const float src_color = sample_src_image_bilinear(
          src_images, src_image_idx, src_max_w, src_max_h, norm_col_src,
          norm_row_src);

      const float bilateral_weight =
          bilateral_spatial[(win_row + window_radius) * window_size +
                            (win_col + window_radius)] *
          bilateral_color[abs(ref_center_q - ref_q)];
      const float bilateral_weight_src = bilateral_weight * src_color;

      src_color_sum += bilateral_weight_src;
      src_color_squared_sum += bilateral_weight_src * src_color;
      src_ref_color_sum += bilateral_weight_src * ref_color;
      bilateral_weight_sum += bilateral_weight;

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

  const float ref_color_var = ref_squared_sum - ref_sum * ref_sum;
  const float src_color_var =
      src_color_squared_sum - src_color_sum * src_color_sum;
  if (ref_color_var < KMIN_VAR || src_color_var < KMIN_VAR) {
    return KMAX_PHOTO_COST;
  }
  const float src_ref_color_covar = src_ref_color_sum - ref_sum * src_color_sum;
  const float src_ref_color_var = sqrt(ref_color_var * src_color_var);
  return fmax(0.0f, fmin(KMAX_PHOTO_COST,
                         1.0f - src_ref_color_covar / src_ref_color_var));
}

// ===========================================================================
// Kernel: initial cost (one work-item per pixel; loops over source images).
// ===========================================================================
__kernel void compute_initial_cost(__global const float* depth_map,
                                    __global const float* normal_map,
                                    __global const uchar* ref_image,
                                    __global const float* ref_sum_image,
                                    __global const float* ref_squared_sum_image,
                                    __global const uchar* src_images,
                                    __global const float* poses,
                                    __global const float* ref_inv_K_all,
                                    __global const float* bilateral_spatial,
                                    __global const float* bilateral_color,
                                    __global float* cost_map,
                                    int width,
                                    int height,
                                    int num_src,
                                    int src_max_w,
                                    int src_max_h,
                                    int window_radius,
                                    int window_step,
                                    int rotation) {
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if (col >= width || row >= height) {
    return;
  }
  float ref_inv_K[4];
  ref_inv_K[0] = ref_inv_K_all[rotation * 4 + 0];
  ref_inv_K[1] = ref_inv_K_all[rotation * 4 + 1];
  ref_inv_K[2] = ref_inv_K_all[rotation * 4 + 2];
  ref_inv_K[3] = ref_inv_K_all[rotation * 4 + 3];

  const size_t pix = (size_t)row * width + col;
  const float depth = depth_map[pix];
  float normal[3];
  normal[0] = normal_map[0 * (size_t)width * height + pix];
  normal[1] = normal_map[1 * (size_t)width * height + pix];
  normal[2] = normal_map[2 * (size_t)width * height + pix];
  const float ref_sum = ref_sum_image[pix];
  const float ref_squared_sum = ref_squared_sum_image[pix];

  for (int image_idx = 0; image_idx < num_src; ++image_idx) {
    __global const float* pose =
        poses + ((size_t)rotation * num_src + image_idx) * KNUM_TFORM;
    const float cost = compute_photo_consistency_cost(
        ref_image, src_images, pose, ref_inv_K, bilateral_spatial,
        bilateral_color, width, height, src_max_w, src_max_h, window_radius,
        window_step, row, col, depth, normal, image_idx, ref_sum,
        ref_squared_sum);
    cost_map[(size_t)image_idx * width * height + pix] = cost;
  }
}

// ===========================================================================
// Kernel: sweep from top to bottom (one work-item per column).
// Photometric only (no geometric-consistency term). The photometric filter
// runs when filter_photo != 0 (last sweep).
// ===========================================================================
__kernel void sweep(__global float* depth_map,
                    __global float* normal_map,
                    __global const uchar* ref_image,
                    __global const float* ref_sum_image,
                    __global const float* ref_squared_sum_image,
                    __global const uchar* src_images,
                    __global const float* poses,
                    __global const float* ref_inv_K_all,
                    __global const float* bilateral_spatial,
                    __global const float* bilateral_color,
                    __global ulong2* rand_state,
                    __global float* sel_prob_map,
                    __global const float* prev_sel_prob_map,
                    __global float* cost_map,
                    __global uchar* consistency_mask,
                    int width,
                    int height,
                    int num_src,
                    int src_max_w,
                    int src_max_h,
                    int window_radius,
                    int window_step,
                    int rotation,
                    int num_samples,
                    float perturbation,
                    float prev_sel_prob_weight,
                    float inv_ncc_sigma_sq,
                    float ncc_norm_factor,
                    float cos_min_triangulation_angle,
                    float inv_incident_angle_sigma_sq,
                    int filter_photo,
                    float filter_min_ncc_prob,
                    float filter_cos_min_triangulation_angle,
                    int filter_min_num_consistent) {
  const int col = get_global_id(0);
  if (col >= width) {
    return;
  }
  const int window_size = 2 * window_radius + 1;
  const size_t plane = (size_t)width * height;

  float ref_inv_K[4];
  ref_inv_K[0] = ref_inv_K_all[rotation * 4 + 0];
  ref_inv_K[1] = ref_inv_K_all[rotation * 4 + 1];
  ref_inv_K[2] = ref_inv_K_all[rotation * 4 + 2];
  ref_inv_K[3] = ref_inv_K_all[rotation * 4 + 3];

  float forward_message[MAX_SRC];
  float sampling_probs[MAX_SRC];

  // Backward pass: store backward messages temporarily in sel_prob_map.
  for (int image_idx = 0; image_idx < num_src; ++image_idx) {
    float beta = KUNIFORM_PROB;
    for (int row = height - 1; row >= 0; --row) {
      const float cost = cost_map[(size_t)image_idx * plane + (size_t)row * width + col];
      beta = compute_message(cost, beta, 0, inv_ncc_sigma_sq, ncc_norm_factor);
      sel_prob_map[(size_t)image_idx * plane + (size_t)row * width + col] = beta;
    }
    forward_message[image_idx] = KUNIFORM_PROB;
  }

  // PRNG state for this column (row 0, like the CUDA/CPU kernels).
  ulong rs_state = rand_state[col].x;
  const ulong rs_inc = rand_state[col].y;

  float prev_depth = depth_map[(size_t)0 * width + col];
  float prev_normal[3];
  prev_normal[0] = normal_map[0 * plane + (size_t)0 * width + col];
  prev_normal[1] = normal_map[1 * plane + (size_t)0 * width + col];
  prev_normal[2] = normal_map[2 * plane + (size_t)0 * width + col];

  for (int row = 0; row < height; ++row) {
    const size_t pix = (size_t)row * width + col;
    const float ref_sum = ref_sum_image[pix];
    const float ref_squared_sum = ref_squared_sum_image[pix];

    prev_depth = propagate_depth(ref_inv_K, prev_depth, prev_normal,
                                 (float)(row - 1), (float)row);

    float curr_depth = depth_map[pix];
    float curr_normal[3];
    curr_normal[0] = normal_map[0 * plane + pix];
    curr_normal[1] = normal_map[1 * plane + pix];
    curr_normal[2] = normal_map[2 * plane + pix];

    float rand_depth = perturb_depth(perturbation, curr_depth, &rs_state, rs_inc);
    float rand_normal[3];
    perturb_normal(row, col, perturbation * KPI, ref_inv_K, curr_normal,
                   &rs_state, rs_inc, rand_normal);

    float point[3];
    compute_point_at_depth((float)row, (float)col, ref_inv_K, curr_depth, point);

    for (int image_idx = 0; image_idx < num_src; ++image_idx) {
      __global const float* pose =
          poses + ((size_t)rotation * num_src + image_idx) * KNUM_TFORM;
      const float cost = cost_map[(size_t)image_idx * plane + pix];
      const float alpha = compute_message(cost, forward_message[image_idx], 1,
                                          inv_ncc_sigma_sq, ncc_norm_factor);
      const float beta = sel_prob_map[(size_t)image_idx * plane + pix];
      const float prev_prob = prev_sel_prob_map[(size_t)image_idx * plane + pix];
      const float sel_prob = compute_sel_prob(alpha, beta, prev_prob, prev_sel_prob_weight);

      float cos_tri, cos_inc;
      compute_viewing_angles(pose, point, curr_normal, &cos_tri, &cos_inc);
      const float tri_prob = compute_tri_prob(cos_tri, cos_min_triangulation_angle);
      const float inc_prob = compute_inc_prob(cos_inc, inv_incident_angle_sigma_sq);

      float H[9];
      compose_homography(pose, ref_inv_K, row, col, curr_depth, curr_normal, H);
      const float res_prob = compute_resolution_prob(H, (float)row, (float)col, window_size);

      sampling_probs[image_idx] = sel_prob * tri_prob * inc_prob * res_prob;
    }

    transform_pdf_to_cdf(sampling_probs, num_src);

    float costs[KNUM_COSTS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const float depths[KNUM_COSTS] = {curr_depth, prev_depth, rand_depth, curr_depth, rand_depth};
    // normals indices: 0=curr 1=prev 2=rand 3=rand 4=curr
    for (int sample = 0; sample < num_samples; ++sample) {
      const float rand_prob = rand_uniform(&rs_state, rs_inc) - FLT_EPSILON;
      int sampled_image_idx = -1;
      for (int image_idx = 0; image_idx < num_src; ++image_idx) {
        if (sampling_probs[image_idx] > rand_prob) {
          sampled_image_idx = image_idx;
          break;
        }
      }
      if (sampled_image_idx == -1) {
        continue;
      }
      __global const float* pose =
          poses + ((size_t)rotation * num_src + sampled_image_idx) * KNUM_TFORM;

      costs[0] += cost_map[(size_t)sampled_image_idx * plane + pix];
      // hypothesis 1: prev_depth, prev_normal
      costs[1] += compute_photo_consistency_cost(
          ref_image, src_images, pose, ref_inv_K, bilateral_spatial,
          bilateral_color, width, height, src_max_w, src_max_h, window_radius,
          window_step, row, col, prev_depth, prev_normal, sampled_image_idx,
          ref_sum, ref_squared_sum);
      // hypothesis 2: rand_depth, rand_normal
      costs[2] += compute_photo_consistency_cost(
          ref_image, src_images, pose, ref_inv_K, bilateral_spatial,
          bilateral_color, width, height, src_max_w, src_max_h, window_radius,
          window_step, row, col, rand_depth, rand_normal, sampled_image_idx,
          ref_sum, ref_squared_sum);
      // hypothesis 3: depths[3]=curr_depth, normals[3]=rand_normal
      costs[3] += compute_photo_consistency_cost(
          ref_image, src_images, pose, ref_inv_K, bilateral_spatial,
          bilateral_color, width, height, src_max_w, src_max_h, window_radius,
          window_step, row, col, curr_depth, rand_normal, sampled_image_idx,
          ref_sum, ref_squared_sum);
      // hypothesis 4: depths[4]=rand_depth, normals[4]=curr_normal
      costs[4] += compute_photo_consistency_cost(
          ref_image, src_images, pose, ref_inv_K, bilateral_spatial,
          bilateral_color, width, height, src_max_w, src_max_h, window_radius,
          window_step, row, col, rand_depth, curr_normal, sampled_image_idx,
          ref_sum, ref_squared_sum);
    }

    const int min_cost_idx = find_min_cost(costs);
    // depths  = {curr, prev, rand, curr, rand}
    // normals = {curr, prev, rand, rand, curr}
    float best_depth = depths[min_cost_idx];
    float best_normal[3];
    if (min_cost_idx == 0 || min_cost_idx == 4) {
      best_normal[0] = curr_normal[0]; best_normal[1] = curr_normal[1]; best_normal[2] = curr_normal[2];
    } else if (min_cost_idx == 1) {
      best_normal[0] = prev_normal[0]; best_normal[1] = prev_normal[1]; best_normal[2] = prev_normal[2];
    } else { // 2 or 3
      best_normal[0] = rand_normal[0]; best_normal[1] = rand_normal[1]; best_normal[2] = rand_normal[2];
    }

    depth_map[pix] = best_depth;
    normal_map[0 * plane + pix] = best_normal[0];
    normal_map[1 * plane + pix] = best_normal[1];
    normal_map[2 * plane + pix] = best_normal[2];

    // Recompute forward message + selection probability with the best params.
    for (int image_idx = 0; image_idx < num_src; ++image_idx) {
      __global const float* pose =
          poses + ((size_t)rotation * num_src + image_idx) * KNUM_TFORM;
      float cost;
      if (min_cost_idx == 0) {
        cost = cost_map[(size_t)image_idx * plane + pix];
      } else {
        cost = compute_photo_consistency_cost(
            ref_image, src_images, pose, ref_inv_K, bilateral_spatial,
            bilateral_color, width, height, src_max_w, src_max_h, window_radius,
            window_step, row, col, best_depth, best_normal, image_idx, ref_sum,
            ref_squared_sum);
        cost_map[(size_t)image_idx * plane + pix] = cost;
      }
      const float alpha = compute_message(cost, forward_message[image_idx], 1,
                                          inv_ncc_sigma_sq, ncc_norm_factor);
      const float beta = sel_prob_map[(size_t)image_idx * plane + pix];
      const float prev_prob = prev_sel_prob_map[(size_t)image_idx * plane + pix];
      const float prob = compute_sel_prob(alpha, beta, prev_prob, prev_sel_prob_weight);
      forward_message[image_idx] = alpha;
      sel_prob_map[(size_t)image_idx * plane + pix] = prob;
    }

    if (filter_photo) {
      int num_consistent = 0;
      float best_point[3];
      compute_point_at_depth((float)row, (float)col, ref_inv_K, best_depth, best_point);
      for (int image_idx = 0; image_idx < num_src; ++image_idx) {
        __global const float* pose =
            poses + ((size_t)rotation * num_src + image_idx) * KNUM_TFORM;
        float cos_tri, cos_inc;
        compute_viewing_angles(pose, best_point, best_normal, &cos_tri, &cos_inc);
        if (cos_tri > filter_cos_min_triangulation_angle || cos_inc <= 0.0f) {
          continue;
        }
        if (sel_prob_map[(size_t)image_idx * plane + pix] >= filter_min_ncc_prob) {
          consistency_mask[(size_t)image_idx * plane + pix] = 1;
          num_consistent += 1;
        }
      }
      if (num_consistent < filter_min_num_consistent) {
        depth_map[pix] = 0.0f;
        normal_map[0 * plane + pix] = 0.0f;
        normal_map[1 * plane + pix] = 0.0f;
        normal_map[2 * plane + pix] = 0.0f;
        for (int image_idx = 0; image_idx < num_src; ++image_idx) {
          consistency_mask[(size_t)image_idx * plane + pix] = 0;
        }
      }
    }

    prev_depth = best_depth;
    prev_normal[0] = best_normal[0];
    prev_normal[1] = best_normal[1];
    prev_normal[2] = best_normal[2];
  }

  rand_state[col] = (ulong2)(rs_state, rs_inc);
}

// ===========================================================================
// Rotation kernels (CCW): output(width-1-col, row) = input(row, col), with
// swapped output dimensions. Run over the INPUT (in_width x in_height) grid.
// ===========================================================================
__kernel void rotate_ccw_f(__global const float* in,
                           __global float* out,
                           int in_width,
                           int in_height,
                           int channels) {
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if (col >= in_width || row >= in_height) {
    return;
  }
  const int out_width = in_height;
  const int out_row = in_width - 1 - col;
  const int out_col = row;
  const size_t in_plane = (size_t)in_width * in_height;
  const size_t out_plane = in_plane;  // total elements per channel unchanged
  for (int s = 0; s < channels; ++s) {
    out[(size_t)s * out_plane + (size_t)out_row * out_width + out_col] =
        in[(size_t)s * in_plane + (size_t)row * in_width + col];
  }
}

__kernel void rotate_ccw_u8(__global const uchar* in,
                            __global uchar* out,
                            int in_width,
                            int in_height,
                            int channels) {
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if (col >= in_width || row >= in_height) {
    return;
  }
  const int out_width = in_height;
  const int out_row = in_width - 1 - col;
  const int out_col = row;
  const size_t plane = (size_t)in_width * in_height;
  for (int s = 0; s < channels; ++s) {
    out[(size_t)s * plane + (size_t)out_row * out_width + out_col] =
        in[(size_t)s * plane + (size_t)row * in_width + col];
  }
}

__kernel void rotate_ccw_rand(__global const ulong2* in,
                              __global ulong2* out,
                              int in_width,
                              int in_height) {
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if (col >= in_width || row >= in_height) {
    return;
  }
  const int out_width = in_height;
  const int out_row = in_width - 1 - col;
  const int out_col = row;
  out[(size_t)out_row * out_width + out_col] = in[(size_t)row * in_width + col];
}

// Normal map: rotate the normal vector by 90deg CCW around z
// (n' = {n.y, -n.x, n.z}) and then rotate spatially.
__kernel void rotate_ccw_normal(__global const float* in,
                                __global float* out,
                                int in_width,
                                int in_height) {
  const int col = get_global_id(0);
  const int row = get_global_id(1);
  if (col >= in_width || row >= in_height) {
    return;
  }
  const int out_width = in_height;
  const int out_row = in_width - 1 - col;
  const int out_col = row;
  const size_t in_plane = (size_t)in_width * in_height;
  const size_t out_plane = in_plane;
  const float nx = in[0 * in_plane + (size_t)row * in_width + col];
  const float ny = in[1 * in_plane + (size_t)row * in_width + col];
  const float nz = in[2 * in_plane + (size_t)row * in_width + col];
  out[0 * out_plane + (size_t)out_row * out_width + out_col] = ny;
  out[1 * out_plane + (size_t)out_row * out_width + out_col] = -nx;
  out[2 * out_plane + (size_t)out_row * out_width + out_col] = nz;
}
)CLC";

}  // namespace mvs
}  // namespace colmap
