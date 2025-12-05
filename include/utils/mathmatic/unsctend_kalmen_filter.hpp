/**
************************************************************************
*
* @file unsctend_kalmen_filter.hpp
* @author Xlqmu (niezhenghua2004@gmail.com)
* @brief
*
* ************************************************************************
* @copyright Copyright (c) 2025 Xlqmu
* For study and research only, no reprinting
* ************************************************************************
*/

#pragma once

#include "angles.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <functional>
#include <ranges>
#include <vector>

#include <Eigen/Dense>

namespace kalman_lib {

template <typename T>
concept MatrixType = std::same_as<T, Eigen::Matrix<double, Eigen::Dynamic,
                                                   Eigen::Dynamic>>;  // Simplified, but for
                                                                      // specific sizes

// Concepts for functors
template <typename F, typename StateType, typename MeasureType>
concept PredictFunctor =
    std::invocable<F, StateType> && std::same_as<std::invoke_result_t<F, StateType>, StateType>;

template <typename F, typename StateType, typename MeasureType>
concept MeasureFunctor =
    std::invocable<F, StateType> && std::same_as<std::invoke_result_t<F, StateType>, MeasureType>;

template <typename F, typename CovType>
concept UpdateQFunctor = std::invocable<F> && std::same_as<std::invoke_result_t<F>, CovType>;

template <typename F, typename MeasureType, typename CovType>
concept UpdateRFunctor =
    std::invocable<F, MeasureType> && std::same_as<std::invoke_result_t<F, MeasureType>, CovType>;

/**
 * @brief Unscented Kalman Filter (UKF) implementation with optional
 * Gauss-Newton style iterative update.
 *
 * This UKF uses the Unscented Transform to handle nonlinear process and
 * measurement models. An optional iterative update can refine the Gauss
 * posterior state estimate via multiple measurement updates. Supports handling
 * of angular dimensions with shortest angular distance wrap-around.
 *
 * Optimized for performance using C++20 features: constexpr computations,
 * ranges for loops, concepts for type safety. Alpha, Beta, Kappa are template
 * parameters for compile-time optimization of weights.
 *
 * @tparam N_X          Dimension of the state vector (compile-time constant for
 * fixed-size matrices).
 * @tparam N_Z          Dimension of the measurement vector.
 * @tparam PredictFunc  Functor type for the process model: x_{k+1} = f(x_k).
 * @tparam MeasureFunc  Functor type for the measurement model: z_k = h(x_k).
 * @tparam Alpha        UKF scaling parameter (constexpr, default 1e-3).
 * @tparam Beta         UKF prior knowledge parameter (constexpr, default 2 for
 * Gaussian).
 * @tparam Kappa        UKF secondary scaling parameter (constexpr, default 0).
 */
template <int N_X, int N_Z, class PredictFunc, class MeasureFunc, double Alpha = 1e-3,
          double Beta = 2.0, double Kappa = 0.0>
    requires PredictFunctor<PredictFunc, Eigen::Matrix<double, N_X, 1>,
                            Eigen::Matrix<double, N_X, 1>> &&
             MeasureFunctor<MeasureFunc, Eigen::Matrix<double, N_X, 1>,
                            Eigen::Matrix<double, N_Z, 1>>
class UnscentedKalmanFilter {
public:
    static constexpr int SigmaPoints = 2 * N_X + 1;

    using MatrixXX = Eigen::Matrix<double, N_X, N_X>;
    using MatrixZX = Eigen::Matrix<double, N_Z, N_X>;
    using MatrixXZ = Eigen::Matrix<double, N_X, N_Z>;
    using MatrixZZ = Eigen::Matrix<double, N_Z, N_Z>;
    using MatrixX1 = Eigen::Matrix<double, N_X, 1>;
    using MatrixZ1 = Eigen::Matrix<double, N_Z, 1>;

    using UpdateQFunc = std::function<MatrixXX()>;
    using UpdateRFunc = std::function<MatrixZZ(const MatrixZ1&)>;

    /**
     * @brief Construct the Unscented Kalman Filter.
     *
     * Alpha, Beta, Kappa are template parameters for compile-time weight
     * computation.
     *
     * @param f         Process model functor.
     * @param h         Measurement model functor.
     * @param u_q       Function to compute process noise covariance Q.
     * @param u_r       Function to compute measurement noise covariance R given
     * measurement.
     * @param P0        Initial posterior covariance matrix.
     */
    explicit UnscentedKalmanFilter(const PredictFunc& f, const MeasureFunc& h,
                                   const UpdateQFunc& u_q, const UpdateRFunc& u_r,
                                   const MatrixXX& P0) noexcept
        : f(f), h(h), update_Q(u_q), update_R(u_r), P_post(P0) {
        // Sort angle_dims_ for potential cache-friendly access (minor optimization)
        std::ranges::sort(angle_dims_);
    }

    /**
     * @brief Set the initial state estimate.
     * @param x0 Initial state vector.
     */
    void setState(const MatrixX1& x0) noexcept { x_post = x0; }

    /**
     * @brief Specify which indices in state or measurement vector represent
     * angles. Those dimensions will be wrapped via shortest angular distance.
     * @param dims Vector of angle dimension indices.
     */
    void setAngleDims(std::vector<int> dims) {
        angle_dims_ = std::move(dims);
        std::ranges::sort(angle_dims_);  // Sort for optimized access in loops
    }

    /**
     * @brief Set the number of Gauss-Newton style iterations during update.
     *        Minimum is 1 (standard UKF update).
     * @param num Number of iterations.
     */
    void setIterationNum(int num) noexcept { iteration_num_ = std::max(1, num); }

    /**
     * @brief Get the predicted (prior) covariance matrix.
     * @return Prior covariance.
     */
    [[nodiscard]] const MatrixXX& getPriorCovariance() const noexcept { return P_pri; }

    /**
     * @brief Get the updated (posterior) covariance matrix.
     * @return Posterior covariance.
     */
    [[nodiscard]] const MatrixXX& getPosteriorCovariance() const noexcept { return P_post; }

    /**
     * @brief Perform the prediction step of UKF.
     * @return Predicted (prior) state vector.
     */
    [[nodiscard]] MatrixX1 predict() noexcept {
        Q = update_Q();

        generateSigmaPoints(x_post, P_post, Xsig);

        for (const auto i : std::views::iota(0, SigmaPoints)) {
            Xsig_pred.col(i) = f(Xsig.col(i));
        }

        x_pri.setZero();
        for (const auto i : std::views::iota(0, SigmaPoints)) {
            x_pri.noalias() += weights_mean[i] * Xsig_pred.col(i);
        }

        P_pri.setZero();
        for (const auto i : std::views::iota(0, SigmaPoints)) {
            const auto dx = Xsig_pred.col(i) - x_pri;
            P_pri.noalias() += weights_cov[i] * dx * dx.transpose();
        }
        P_pri.noalias() += Q;

        x_post = x_pri;
        return x_pri;
    }

    /**
     * @brief Perform the measurement update step with iterative refinement.
     *
     * The update iteratively refines the posterior state estimate by repeatedly
     * generating sigma points around the current estimate and applying the
     * measurement update.
     *
     * @param z Measurement vector.
     * @return Updated (posterior) state vector.
     */
    [[nodiscard]] MatrixX1 update(const MatrixZ1& z) noexcept {
        R = update_R(z);

        // Initialize iterative update state with prior mean
        MatrixX1 x_iter = x_pri;

        for ([[maybe_unused]] const auto iter : std::views::iota(0, iteration_num_)) {
            // Generate sigma points around current estimate
            generateSigmaPoints(x_iter, P_pri, Xsig);

            // Predict measurement sigma points
            Eigen::Matrix<double, N_Z, SigmaPoints> Zsig;
            for (const auto i : std::views::iota(0, SigmaPoints)) {
                Zsig.col(i) = h(Xsig.col(i));
            }

            // Calculate predicted measurement mean
            MatrixZ1 z_pred = MatrixZ1::Zero();
            for (const auto i : std::views::iota(0, SigmaPoints)) {
                z_pred.noalias() += weights_mean[i] * Zsig.col(i);
            }

            // Calculate innovation covariance matrix S
            MatrixZZ S = MatrixZZ::Zero();
            for (const auto i : std::views::iota(0, SigmaPoints)) {
                auto dz = Zsig.col(i) - z_pred;
                for (const int idx : angle_dims_) {
                    dz[idx] = angles::shortest_angular_distance(z_pred[idx], Zsig.col(i)[idx]);
                }
                S.noalias() += weights_cov[i] * dz * dz.transpose();
            }
            S.noalias() += R;

            // Calculate cross covariance matrix Tc
            MatrixXZ Tc = MatrixXZ::Zero();
            for (const auto i : std::views::iota(0, SigmaPoints)) {
                const auto dx = Xsig.col(i) - x_iter;
                auto dz = Zsig.col(i) - z_pred;
                for (const int idx : angle_dims_) {
                    dz[idx] = angles::shortest_angular_distance(z_pred[idx], Zsig.col(i)[idx]);
                }
                Tc.noalias() += weights_cov[i] * dx * dz.transpose();
            }

            // Calculate Kalman gain
            MatrixXZ K_iter = Tc * S.inverse();

            // Calculate residual (measurement innovation) with angle wrapping
            auto residual = z - z_pred;
            for (const int idx : angle_dims_) {
                residual[idx] = angles::shortest_angular_distance(z_pred[idx], z[idx]);
            }
            for (int i = 0; i < N_Z; ++i) {
                if (!std::isfinite(residual[i]))
                    residual[i] = 0.0;
                residual[i] = std::clamp(residual[i], -1e2, 1e2);
            }

            // Update state estimate
            auto x_new = x_iter + K_iter * residual;
            for (int i = 0; i < N_X; ++i) {
                if (!std::isfinite(x_new[i]))
                    x_new[i] = x_iter[i];
            }

            x_iter = x_new;
        }

        // Save final updated state
        x_post = x_iter;

        // Recompute final Kalman gain and covariance update
        generateSigmaPoints(x_post, P_pri, Xsig);
        Eigen::Matrix<double, N_Z, SigmaPoints> Zsig_final;
        for (const auto i : std::views::iota(0, SigmaPoints)) {
            Zsig_final.col(i) = h(Xsig.col(i));
        }

        MatrixZ1 z_pred_final = MatrixZ1::Zero();
        for (const auto i : std::views::iota(0, SigmaPoints)) {
            z_pred_final.noalias() += weights_mean[i] * Zsig_final.col(i);
        }

        MatrixZZ S_final = MatrixZZ::Zero();
        for (const auto i : std::views::iota(0, SigmaPoints)) {
            auto dz = Zsig_final.col(i) - z_pred_final;
            for (const int idx : angle_dims_) {
                dz[idx] =
                    angles::shortest_angular_distance(z_pred_final[idx], Zsig_final.col(i)[idx]);
            }
            S_final.noalias() += weights_cov[i] * dz * dz.transpose();
        }
        S_final.noalias() += R;

        MatrixXZ Tc_final = MatrixXZ::Zero();
        for (const auto i : std::views::iota(0, SigmaPoints)) {
            const auto dx = Xsig.col(i) - x_post;
            auto dz = Zsig_final.col(i) - z_pred_final;
            for (const int idx : angle_dims_) {
                dz[idx] =
                    angles::shortest_angular_distance(z_pred_final[idx], Zsig_final.col(i)[idx]);
            }
            Tc_final.noalias() += weights_cov[i] * dx * dz.transpose();
        }

        K = Tc_final * S_final.inverse();

        // Update covariance
        P_post = P_pri - K * S_final * K.transpose();
        // Symmetrize covariance matrix (ensure positive semi-definite)
        P_post = 0.5 * (P_post + P_post.transpose());

        return x_post;
    }

private:
    static constexpr double lambda_ = Alpha * Alpha * (N_X + Kappa) - N_X;
    static constexpr double gamma_ = std::sqrt(N_X + lambda_);

    const std::array<double, SigmaPoints> weights_mean = []() constexpr {
        std::array<double, SigmaPoints> w{};
        w[0] = lambda_ / (N_X + lambda_);
        for (int i = 1; i < SigmaPoints; ++i) {
            w[i] = 1.0 / (2 * (N_X + lambda_));
        }
        return w;
    }();

    const std::array<double, SigmaPoints> weights_cov = []() constexpr {
        std::array<double, SigmaPoints> w{};
        w[0] = lambda_ / (N_X + lambda_) + (1 - Alpha * Alpha + Beta);
        for (int i = 1; i < SigmaPoints; ++i) {
            w[i] = 1.0 / (2 * (N_X + lambda_));
        }
        return w;
    }();

    PredictFunc f;         ///< Process model function
    MeasureFunc h;         ///< Measurement model function
    UpdateQFunc update_Q;  ///< Process noise covariance updater
    UpdateRFunc update_R;  ///< Measurement noise covariance updater

    Eigen::Matrix<double, N_X, SigmaPoints> Xsig;       ///< Sigma points matrix
    Eigen::Matrix<double, N_X, SigmaPoints> Xsig_pred;  ///< Predicted sigma points matrix

    MatrixXX Q = MatrixXX::Zero();           ///< Process noise covariance
    MatrixXX P_pri = MatrixXX::Identity();   ///< Prior covariance
    MatrixXX P_post = MatrixXX::Identity();  ///< Posterior covariance

    MatrixZZ R = MatrixZZ::Zero();  ///< Measurement noise covariance
    MatrixXZ K = MatrixXZ::Zero();  ///< Kalman gain

    MatrixX1 x_pri = MatrixX1::Zero();   ///< Predicted (prior) state
    MatrixX1 x_post = MatrixX1::Zero();  ///< Updated (posterior) state

    std::vector<int> angle_dims_;  ///< Indices of angular dimensions to wrap
                                   ///< (sorted for optimization)

    int iteration_num_ = 1;  ///< Number of iterations during update (>=1)

    /**
     * @brief Generate sigma points from state and covariance.
     * @param x State vector.
     * @param P Covariance matrix.
     * @param Xsig_out Output sigma points matrix (size N_X x SigmaPoints).
     */
    void generateSigmaPoints(const MatrixX1& x, const MatrixXX& P,
                             Eigen::Matrix<double, N_X, SigmaPoints>& Xsig_out) noexcept {
        const auto A = P.llt().matrixL();  // Cholesky decomposition (optimized in Eigen)
        Xsig_out.col(0) = x;
        for (const auto i : std::views::iota(0, N_X)) {
            const auto scaled_col = gamma_ * A.col(i);
            Xsig_out.col(i + 1) = x + scaled_col;
            Xsig_out.col(i + 1 + N_X) = x - scaled_col;
        }
    }
};

}  // namespace kalman_lib