// Copyright Chen Jun 2023. Licensed under the MIT License.
// Copyright xinyang 2021.
// Additional modifications and features by Chengfu Zou, Labor. Licensed under
// Apache License 2.0. Copyright (C) FYT Vision Group. All rights reserved.
/// Copyright 2025 Zhenghua Nie
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once

#include "angles.h"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <ceres/jet.h>
#include <concepts>
#include <functional>
#include <ranges>

namespace kalman_lib {

template <typename T>
concept MatrixType =
    std::same_as<T, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>>;

// Concepts for functors
template <typename F, typename StateType>
concept PredictFunctor =
    std::invocable<F, StateType *, StateType *> &&
    std::same_as<StateType, Eigen::Matrix<double, Eigen::Dynamic, 1>>;

template <typename F, typename StateType, typename MeasureType>
concept MeasureFunctor =
    std::invocable<F, StateType *, MeasureType *> &&
    std::same_as<StateType, Eigen::Matrix<double, Eigen::Dynamic, 1>>;

template <typename F, typename CovType>
concept UpdateQFunctor =
    std::invocable<F> && std::same_as<std::invoke_result_t<F>, CovType>;

template <typename F, typename MeasureType, typename CovType>
concept UpdateRFunctor =
    std::invocable<F, MeasureType> &&
    std::same_as<std::invoke_result_t<F, MeasureType>, CovType>;

/**
 * @brief Extended Kalman Filter (EKF) implementation using Ceres Jet for
 * automatic differentiation.
 *
 * Optimized for RoboMaster auto-aim: fixed-size matrices, compile-time noise,
 * minimal allocations.
 *
 * @tparam N_X          Dimension of the state vector (e.g., 6 for
 * position+velocity).
 * @tparam N_Z          Dimension of the measurement vector (e.g., 2 for pixel
 * coordinates).
 * @tparam PredicFunc   Functor for process model: x_{k+1} = f(x_k).
 * @tparam MeasureFunc  Functor for measurement model: z_k = h(x_k).
 * @tparam SmallNoise   Small noise term for numerical stability (default 1e-6).
 */
template <int N_X, int N_Z, class PredicFunc, class MeasureFunc,
          double SmallNoise = 1e-6>
  requires PredictFunctor<PredicFunc, Eigen::Matrix<double, N_X, 1>> &&
           MeasureFunctor<MeasureFunc, Eigen::Matrix<double, N_X, 1>,
                          Eigen::Matrix<double, N_Z, 1>>
class ExtendedKalmanFilter {
public:
  using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
  using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
  using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
  using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
  using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
  using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

  using UpdateQFunc = std::function<MatrixXX()>;
  using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1 &)>;

  /**
   * @brief Constructor initializing models, noise updaters, and prior
   * covariance.
   */
  explicit ExtendedKalmanFilter(const PredicFunc &f, const MeasureFunc &h,
                                const UpdateQFunc &u_q, const UpdateRFunc &u_r,
                                const MatrixXX &P0) noexcept
      : f(f), h(h), update_Q(u_q), update_R(u_r), P_post(P0) {
    std::ranges::sort(angle_dims_); // Optimize cache access
  }

  void setState(const MatrixX1 &x0) noexcept { x_post = x0; }
  void setPredictFunc(const PredicFunc &f) noexcept { this->f = f; }
  void setMeasureFunc(const MeasureFunc &h) noexcept { this->h = h; }
  void setIterationNum(int num) noexcept { iteration_num_ = std::max(1, num); }
  void setAngleDims(std::vector<int> dims) {
    angle_dims_ = std::move(dims);
    std::ranges::sort(angle_dims_);
  }

  [[nodiscard]] const MatrixXX &getPriorCovariance() const noexcept {
    return P_pri;
  }
  [[nodiscard]] const MatrixXX &getPosteriorCovariance() const noexcept {
    return P_post;
  }
  [[nodiscard]] double getResidualNorm() const noexcept {
    return last_residual_.norm();
  }

  [[nodiscard]] MatrixX1 predict() noexcept {
    Q = update_Q();

    // Ceres Jet for auto-diff
    std::array<ceres::Jet<double, N_X>, N_X> x_e_jet;
    for (const auto i : std::views::iota(0, N_X)) {
      x_e_jet[i].a = x_post[i];
      x_e_jet[i].v.setZero();
      x_e_jet[i].v[i] = 1.0;
    }

    std::array<ceres::Jet<double, N_X>, N_X> x_p_jet;
    f(x_e_jet.data(), x_p_jet.data());

    x_pri.setZero();
    F.setZero();
    for (const auto i : std::views::iota(0, N_X)) {
      x_pri[i] = std::isfinite(x_p_jet[i].a) ? x_p_jet[i].a : 0.0;
      F.row(i) = x_p_jet[i].v.transpose();
    }

    P_pri.noalias() = F * P_post * F.transpose() + Q;
    P_pri = 0.5 * (P_pri + P_pri.transpose());

    x_post = x_pri;
    return x_pri;
  }

  [[nodiscard]] MatrixX1 update(const MatrixZ1 &z) noexcept {
    R = update_R(z);
    MatrixX1 x_iter = x_post;

    for ([[maybe_unused]] const auto iter :
         std::views::iota(0, iteration_num_)) {
      std::array<ceres::Jet<double, N_X>, N_X> x_p_jet;
      for (const auto i : std::views::iota(0, N_X)) {
        x_p_jet[i].a = x_iter[i];
        x_p_jet[i].v.setZero();
        x_p_jet[i].v[i] = 1.0;
      }

      std::array<ceres::Jet<double, N_X>, N_Z> z_p_jet;
      h(x_p_jet.data(), z_p_jet.data());

      MatrixZ1 z_pri = MatrixZ1::Zero();
      H.setZero();
      for (const auto i : std::views::iota(0, N_Z)) {
        z_pri[i] = std::isfinite(z_p_jet[i].a) ? z_p_jet[i].a : 0.0;
        H.row(i) = z_p_jet[i].v.transpose();
      }

      MatrixZZ S =
          H * P_pri * H.transpose() + R + SmallNoise * MatrixZZ::Identity();
      K.noalias() = P_pri * H.transpose() * S.inverse();

      auto residual = z - z_pri;
      for (const int idx : angle_dims_) {
        residual[idx] = angles::shortest_angular_distance(z_pri[idx], z[idx]);
      }
      for (Eigen::Index i = 0; i < N_Z; ++i) {
        if (!std::isfinite(residual[i]))
          residual[i] = 0.0;
        residual[i] = std::clamp(residual[i], -1e2, 1e2);
      }

      auto x_new = x_iter + K * residual;
      for (Eigen::Index i = 0; i < N_X; ++i) {
        if (!std::isfinite(x_new[i]))
          x_new[i] = x_iter[i];
      }
      x_iter = x_new;
      last_residual_ = residual;
    }

    x_post = x_iter;
    for (Eigen::Index i = 0; i < N_X; ++i) {
      if (!std::isfinite(x_post[i]))
        x_post[i] = 0.0;
    }

    P_post.noalias() = (MatrixXX::Identity() - K * H) * P_pri;
    P_post = 0.5 * (P_post + P_post.transpose());

    return x_post;
  }

private:
  PredicFunc f;
  MeasureFunc h;
  UpdateQFunc update_Q;
  UpdateRFunc update_R;

  MatrixXX F = MatrixXX::Zero();
  MatrixZX H = MatrixZX::Zero();
  MatrixXX Q = MatrixXX::Zero();
  MatrixZZ R = MatrixZZ::Zero();
  MatrixXX P_pri = MatrixXX::Identity();
  MatrixXX P_post = MatrixXX::Identity();
  MatrixXZ K = MatrixXZ::Zero();
  MatrixX1 x_pri = MatrixX1::Zero();
  MatrixX1 x_post = MatrixX1::Zero();
  MatrixZ1 last_residual_ = MatrixZ1::Zero();

  std::vector<int> angle_dims_;
  int iteration_num_ = 1;
};

} // namespace kalman_lib