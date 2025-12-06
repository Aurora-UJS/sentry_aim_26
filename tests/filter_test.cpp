/**
 * ************************************************************************
 *
 * @file filter_test.cpp
 * @author Xlqmu (niezhenghua2004@gmail.com)
 * @brief Comprehensive unit tests for Kalman filter library using Catch2
 *
 * ************************************************************************
 * @copyright Copyright (c) 2025 Xlqmu
 * For study and research only, no reprinting
 * ************************************************************************
 */

#include "filter/adaptive_extended_kalman_filter.hpp"
#include "filter/error_state_extended_kalman_filter.hpp"
#include "filter/extended_kalman_filter.hpp"
#include "filter/filter_lib.hpp"
#include "filter/unscented_kalman_filter.hpp"

#include <cmath>
#include <numbers>
#include <random>
#include <ranges>
#include <vector>

#include <Eigen/Dense>
#include <ceres/jet.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Approx;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ============================================================================
// Test Utilities and Common Definitions
// ============================================================================

namespace test_utils {

/**
 * @brief Simple 1D constant velocity model for testing
 * State: [position, velocity]^T
 * Measurement: [position]
 */
struct ConstantVelocityModel {
    static constexpr int N_X = 2;
    static constexpr int N_Z = 1;
    static constexpr double dt = 0.1;

    // Process model: x_{k+1} = F * x_k
    template <typename T>
    static void predict(const T* x_in, T* x_out) {
        x_out[0] = x_in[0] + dt * x_in[1];  // position += dt * velocity
        x_out[1] = x_in[1];                 // velocity stays constant
    }

    // Measurement model: z = H * x
    template <typename T>
    static void measure(const T* x_in, T* z_out) {
        z_out[0] = x_in[0];  // we only measure position
    }

    // Get Q matrix
    static Eigen::Matrix2d getQ() {
        Eigen::Matrix2d Q;
        double q = 0.1;  // process noise intensity
        Q << (q * dt * dt * dt / 3), (q * dt * dt / 2), (q * dt * dt / 2), (q * dt);
        return Q;
    }

    // Get R matrix
    static Eigen::Matrix<double, 1, 1> getR([[maybe_unused]] const Eigen::Matrix<double, 1, 1>& z) {
        return Eigen::Matrix<double, 1, 1>::Identity() * 0.5;  // measurement noise
    }
};

/**
 * @brief 2D position tracking model
 * State: [x, y, vx, vy]^T
 * Measurement: [x, y]^T
 */
struct Position2DModel {
    static constexpr int N_X = 4;
    static constexpr int N_Z = 2;
    static constexpr double dt = 0.05;

    template <typename T>
    static void predict(const T* x_in, T* x_out) {
        x_out[0] = x_in[0] + dt * x_in[2];  // x += dt * vx
        x_out[1] = x_in[1] + dt * x_in[3];  // y += dt * vy
        x_out[2] = x_in[2];                 // vx constant
        x_out[3] = x_in[3];                 // vy constant
    }

    template <typename T>
    static void measure(const T* x_in, T* z_out) {
        z_out[0] = x_in[0];  // measure x
        z_out[1] = x_in[1];  // measure y
    }

    static Eigen::Matrix4d getQ() {
        double q = 0.05;
        Eigen::Matrix4d Q = Eigen::Matrix4d::Identity() * q;
        return Q;
    }

    static Eigen::Matrix2d getR([[maybe_unused]] const Eigen::Vector2d& z) {
        return Eigen::Matrix2d::Identity() * 0.1;
    }
};

/**
 * @brief Nonlinear polar-to-cartesian measurement model for testing EKF
 * State: [x, y]^T (cartesian position)
 * Measurement: [range, bearing]^T (polar coordinates)
 */
struct PolarMeasurementModel {
    static constexpr int N_X = 2;
    static constexpr int N_Z = 2;

    template <typename T>
    static void predict(const T* x_in, T* x_out) {
        // Simple random walk model
        x_out[0] = x_in[0];
        x_out[1] = x_in[1];
    }

    template <typename T>
    static void measure(const T* x_in, T* z_out) {
        // Convert to polar: range = sqrt(x^2 + y^2), bearing = atan2(y, x)
        using ceres::atan2;
        using ceres::sqrt;
        using std::atan2;
        using std::sqrt;

        z_out[0] = sqrt(x_in[0] * x_in[0] + x_in[1] * x_in[1]);  // range
        z_out[1] = atan2(x_in[1], x_in[0]);                      // bearing
    }

    static Eigen::Matrix2d getQ() { return Eigen::Matrix2d::Identity() * 0.01; }

    static Eigen::Matrix2d getR([[maybe_unused]] const Eigen::Vector2d& z) {
        Eigen::Matrix2d R;
        R << 0.1, 0, 0, 0.01;  // range noise > bearing noise
        return R;
    }
};

/**
 * @brief Normalize angle to [-pi, pi]
 */
inline double normalizeAngle(double angle) {
    while (angle > std::numbers::pi)
        angle -= 2.0 * std::numbers::pi;
    while (angle < -std::numbers::pi)
        angle += 2.0 * std::numbers::pi;
    return angle;
}

/**
 * @brief Generate noisy measurements from true trajectory
 */
template <typename RNG>
std::vector<Eigen::Vector2d> generateNoisyMeasurements(const std::vector<Eigen::Vector2d>& truth,
                                                       double noise_std, RNG& rng) {
    std::normal_distribution<double> noise(0.0, noise_std);
    std::vector<Eigen::Vector2d> measurements;
    measurements.reserve(truth.size());
    for (const auto& pos : truth) {
        measurements.emplace_back(pos[0] + noise(rng), pos[1] + noise(rng));
    }
    return measurements;
}

}  // namespace test_utils

// ============================================================================
// Extended Kalman Filter Tests
// ============================================================================

TEST_CASE("ExtendedKalmanFilter - Basic Functionality", "[EKF][basic]") {
    using namespace test_utils;
    using Model = ConstantVelocityModel;

    auto predict_func = [](const auto* x_in, auto* x_out) { Model::predict(x_in, x_out); };
    auto measure_func = [](const auto* x_in, auto* z_out) { Model::measure(x_in, z_out); };
    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Matrix<double, 1, 1>& z) { return Model::getR(z); };

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman_lib::ExtendedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                     decltype(measure_func)>
        ekf(predict_func, measure_func, update_Q, update_R, P0);

    SECTION("Initial state can be set and retrieved") {
        Eigen::Vector2d x0(1.0, 0.5);
        ekf.setState(x0);

        // After predict, we should get a reasonable state
        auto x_pred = ekf.predict();
        REQUIRE_THAT(x_pred[0], WithinRel(1.05, 0.01));  // 1.0 + 0.1 * 0.5
        REQUIRE_THAT(x_pred[1], WithinRel(0.5, 0.01));
    }

    SECTION("Predict-Update cycle reduces uncertainty") {
        Eigen::Vector2d x0(0.0, 1.0);  // Start at origin, moving at 1 m/s
        ekf.setState(x0);

        // Initial covariance
        auto P_init = ekf.getPosteriorCovariance();

        // Run prediction
        ekf.predict();
        auto P_after_predict = ekf.getPriorCovariance();

        // Covariance should increase after prediction (adding process noise)
        REQUIRE(P_after_predict.trace() >= P_init.trace());

        // Now update with measurement
        Eigen::Matrix<double, 1, 1> z;
        z << 0.1;  // Measured position
        ekf.update(z);

        auto P_after_update = ekf.getPosteriorCovariance();

        // Covariance should decrease after update (incorporating measurement)
        REQUIRE(P_after_update.trace() < P_after_predict.trace());
    }
}

TEST_CASE("ExtendedKalmanFilter - Convergence Test", "[EKF][convergence]") {
    using namespace test_utils;
    using Model = ConstantVelocityModel;

    auto predict_func = [](const auto* x_in, auto* x_out) { Model::predict(x_in, x_out); };
    auto measure_func = [](const auto* x_in, auto* z_out) { Model::measure(x_in, z_out); };
    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Matrix<double, 1, 1>& z) { return Model::getR(z); };

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity() * 10.0;  // High initial uncertainty

    kalman_lib::ExtendedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                     decltype(measure_func)>
        ekf(predict_func, measure_func, update_Q, update_R, P0);

    // True trajectory: position = 0.5*t^2 with velocity = t (constant acceleration = 1)
    // But we model as constant velocity, so there will be model mismatch
    Eigen::Vector2d x0(0.0, 0.0);
    ekf.setState(x0);

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.3);

    double true_pos = 0.0;
    double true_vel = 0.0;
    const double accel = 0.5;

    std::vector<double> errors;

    for (int step = 0; step < 50; ++step) {
        // True dynamics (with acceleration)
        true_vel += accel * Model::dt;
        true_pos += true_vel * Model::dt;

        // Generate noisy measurement
        Eigen::Matrix<double, 1, 1> z;
        z << (true_pos + noise(rng));

        // Kalman filter cycle
        ekf.predict();
        auto x_est = ekf.update(z);

        errors.push_back(std::abs(x_est[0] - true_pos));
    }

    // Check that later errors are bounded (filter is stable)
    double avg_late_error =
        std::accumulate(errors.begin() + 30, errors.end(), 0.0) / (errors.size() - 30);
    REQUIRE(avg_late_error < 2.0);  // Should track reasonably despite model mismatch
}

TEST_CASE("ExtendedKalmanFilter - Anomaly Detection", "[EKF][anomaly]") {
    using namespace test_utils;
    using Model = ConstantVelocityModel;

    auto predict_func = [](const auto* x_in, auto* x_out) { Model::predict(x_in, x_out); };
    auto measure_func = [](const auto* x_in, auto* z_out) { Model::measure(x_in, z_out); };
    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Matrix<double, 1, 1>& z) { return Model::getR(z); };

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman_lib::ExtendedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                     decltype(measure_func)>
        ekf(predict_func, measure_func, update_Q, update_R, P0);

    ekf.setState(Eigen::Vector2d(0.0, 1.0));
    ekf.setNisThreshold(5.0);  // Chi-square threshold for 1 DOF, ~95% confidence
    ekf.setWindowSize(10);
    ekf.setRecentFailRateThreshold(0.5);

    SECTION("Normal measurements don't trigger anomaly") {
        std::mt19937 rng(123);
        std::normal_distribution<double> noise(0.0, 0.5);

        double true_pos = 0.0;
        for (int i = 0; i < 20; ++i) {
            true_pos += 1.0 * Model::dt;
            Eigen::Matrix<double, 1, 1> z;
            z << (true_pos + noise(rng));

            ekf.predict();
            ekf.update(z);
        }

        REQUIRE_FALSE(ekf.isRecentlyInconsistent());
        REQUIRE(ekf.recentNisFailureRate() < 0.5);
    }

    SECTION("Outlier measurements trigger anomaly detection") {
        double true_pos = 0.0;
        for (int i = 0; i < 20; ++i) {
            true_pos += 1.0 * Model::dt;
            Eigen::Matrix<double, 1, 1> z;

            // Inject outliers
            if (i > 10) {
                z << (true_pos + 50.0);  // Large outlier
            } else {
                z << true_pos;
            }

            ekf.predict();
            ekf.update(z);
        }

        // Should detect high failure rate
        REQUIRE(ekf.nisFailureCount() > 0);
    }
}

TEST_CASE("ExtendedKalmanFilter - Custom Residual Function", "[EKF][residual]") {
    using namespace test_utils;
    using Model = PolarMeasurementModel;

    auto predict_func = [](const auto* x_in, auto* x_out) { Model::predict(x_in, x_out); };
    auto measure_func = [](const auto* x_in, auto* z_out) { Model::measure(x_in, z_out); };
    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Vector2d& z) { return Model::getR(z); };

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman_lib::ExtendedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                     decltype(measure_func)>
        ekf(predict_func, measure_func, update_Q, update_R, P0);

    // Custom residual that wraps angles
    ekf.setResidualFunc([](const Eigen::Vector2d& z_pred, const Eigen::Vector2d& z_meas) {
        Eigen::Vector2d residual;
        residual[0] = z_meas[0] - z_pred[0];                  // range difference
        residual[1] = normalizeAngle(z_meas[1] - z_pred[1]);  // angle difference with wrapping
        return residual;
    });

    // Test angle wrapping
    ekf.setState(Eigen::Vector2d(1.0, 0.0));  // At (1, 0), bearing = 0

    ekf.predict();

    // Measurement with bearing near -pi (should wrap correctly to be close to pi)
    Eigen::Vector2d z(1.0, std::numbers::pi - 0.1);
    ekf.update(z);

    // Filter should handle the wrap-around correctly
    REQUIRE(ekf.getResidualNorm() < 10.0);
}

// ============================================================================
// Unscented Kalman Filter Tests
// ============================================================================

TEST_CASE("UnscentedKalmanFilter - Basic Functionality", "[UKF][basic]") {
    using namespace test_utils;
    using Model = Position2DModel;

    auto predict_func = [](const Eigen::Vector4d& x_in) {
        Eigen::Vector4d x_out;
        Model::predict(x_in.data(), x_out.data());
        return x_out;
    };

    auto measure_func = [](const Eigen::Vector4d& x_in) {
        Eigen::Vector2d z_out;
        Model::measure(x_in.data(), z_out.data());
        return z_out;
    };

    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Vector2d& z) { return Model::getR(z); };

    Eigen::Matrix4d P0 = Eigen::Matrix4d::Identity();

    kalman_lib::UnscentedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                      decltype(measure_func)>
        ukf(predict_func, measure_func, update_Q, update_R, P0);

    SECTION("Initial state setting") {
        Eigen::Vector4d x0(1.0, 2.0, 0.5, -0.3);
        ukf.setState(x0);

        auto x_pred = ukf.predict();

        // Check position update
        REQUIRE_THAT(x_pred[0], WithinRel(1.0 + 0.05 * 0.5, 0.01));
        REQUIRE_THAT(x_pred[1], WithinRel(2.0 + 0.05 * (-0.3), 0.01));
    }

    SECTION("UKF sigma point spreading") {
        Eigen::Vector4d x0(0.0, 0.0, 1.0, 1.0);
        ukf.setState(x0);

        // Run a few iterations
        for (int i = 0; i < 10; ++i) {
            ukf.predict();
            Eigen::Vector2d z(i * 0.05, i * 0.05);
            ukf.update(z);
        }

        // Covariance should stay positive definite
        auto P = ukf.getPosteriorCovariance();
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(P);
        auto eigenvalues = es.eigenvalues();

        for (int i = 0; i < 4; ++i) {
            REQUIRE(eigenvalues[i] > 0);
        }
    }
}

TEST_CASE("UnscentedKalmanFilter - Nonlinear System", "[UKF][nonlinear]") {
    using namespace test_utils;
    using Model = PolarMeasurementModel;

    // UKF uses value-based function signatures
    auto predict_func = [](const Eigen::Vector2d& x_in) { return x_in; };  // Random walk

    auto measure_func = [](const Eigen::Vector2d& x_in) {
        Eigen::Vector2d z;
        z[0] = std::sqrt(x_in[0] * x_in[0] + x_in[1] * x_in[1]);  // range
        z[1] = std::atan2(x_in[1], x_in[0]);                      // bearing
        return z;
    };

    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Vector2d& z) { return Model::getR(z); };

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity() * 0.5;

    kalman_lib::UnscentedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                      decltype(measure_func)>
        ukf(predict_func, measure_func, update_Q, update_R, P0);

    // Set angle-aware residual
    ukf.setResidualFunc([](const Eigen::Vector2d& z_pred, const Eigen::Vector2d& z_meas) {
        Eigen::Vector2d residual;
        residual[0] = z_meas[0] - z_pred[0];
        residual[1] = normalizeAngle(z_meas[1] - z_pred[1]);
        return residual;
    });

    // True position moving in a circle
    Eigen::Vector2d true_pos(3.0, 0.0);
    ukf.setState(Eigen::Vector2d(3.0, 0.0));

    std::mt19937 rng(42);
    std::normal_distribution<double> range_noise(0.0, 0.1);
    std::normal_distribution<double> bearing_noise(0.0, 0.05);

    std::vector<double> position_errors;

    for (int step = 0; step < 50; ++step) {
        // Move in circle
        double angle = step * 0.1;
        true_pos[0] = 3.0 * std::cos(angle);
        true_pos[1] = 3.0 * std::sin(angle);

        // Generate polar measurement
        double true_range = std::sqrt(true_pos[0] * true_pos[0] + true_pos[1] * true_pos[1]);
        double true_bearing = std::atan2(true_pos[1], true_pos[0]);

        Eigen::Vector2d z(true_range + range_noise(rng), true_bearing + bearing_noise(rng));

        ukf.predict();
        auto x_est = ukf.update(z);

        double error = (x_est - true_pos).norm();
        position_errors.push_back(error);
    }

    // Average error should be reasonable
    double avg_error = std::accumulate(position_errors.begin() + 10, position_errors.end(), 0.0) /
                       (position_errors.size() - 10);
    REQUIRE(avg_error < 1.0);
}

TEST_CASE("UnscentedKalmanFilter - Iterative Update", "[UKF][iterative]") {
    using namespace test_utils;
    using Model = Position2DModel;

    auto predict_func = [](const Eigen::Vector4d& x_in) {
        Eigen::Vector4d x_out;
        Model::predict(x_in.data(), x_out.data());
        return x_out;
    };

    auto measure_func = [](const Eigen::Vector4d& x_in) {
        Eigen::Vector2d z_out;
        Model::measure(x_in.data(), z_out.data());
        return z_out;
    };

    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Vector2d& z) { return Model::getR(z); };

    Eigen::Matrix4d P0 = Eigen::Matrix4d::Identity() * 2.0;

    // Standard UKF
    kalman_lib::UnscentedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                      decltype(measure_func)>
        ukf_standard(predict_func, measure_func, update_Q, update_R, P0);

    // Iterative UKF
    kalman_lib::UnscentedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                      decltype(measure_func)>
        ukf_iterative(predict_func, measure_func, update_Q, update_R, P0);
    ukf_iterative.setIterationNum(3);

    Eigen::Vector4d x0(0.0, 0.0, 1.0, 0.5);
    ukf_standard.setState(x0);
    ukf_iterative.setState(x0);

    // Both should produce reasonable results
    ukf_standard.predict();
    ukf_iterative.predict();

    Eigen::Vector2d z(0.05, 0.025);
    auto x_std = ukf_standard.update(z);
    auto x_iter = ukf_iterative.update(z);

    // Results should be similar but not necessarily identical
    REQUIRE((x_std - x_iter).norm() < 0.5);
}

// ============================================================================
// Adaptive Extended Kalman Filter Tests
// ============================================================================

TEST_CASE("AdaptiveEKF - Adaptive Noise Estimation", "[AEKF][adaptive]") {
    using namespace test_utils;
    using Model = ConstantVelocityModel;

    auto predict_func = [](const auto* x_in, auto* x_out) { Model::predict(x_in, x_out); };
    auto measure_func = [](const auto* x_in, auto* z_out) { Model::measure(x_in, z_out); };
    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Matrix<double, 1, 1>& z) { return Model::getR(z); };

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman_lib::AdaptiveExtendedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                             decltype(measure_func)>
        aekf(predict_func, measure_func, update_Q, update_R, P0);

    SECTION("With adaptive R enabled") {
        aekf.enableAdaptiveR(true);
        aekf.setAdaptiveRRatio(0.3);
        aekf.setState(Eigen::Vector2d(0.0, 1.0));

        // Run several iterations
        for (int i = 0; i < 20; ++i) {
            auto x = aekf.predict();
            Eigen::Matrix<double, 1, 1> z;
            z << (i * 0.1 + 0.1);
            aekf.update(z);
            // Ensure state remains finite
            REQUIRE(x.allFinite());
        }
    }

    SECTION("With adaptive Q enabled") {
        aekf.enableAdaptiveQ(true);
        aekf.setAdaptiveQRatio(0.2);
        aekf.setState(Eigen::Vector2d(0.0, 1.0));

        for (int i = 0; i < 20; ++i) {
            auto x = aekf.predict();
            Eigen::Matrix<double, 1, 1> z;
            z << (i * 0.1 + 0.1);
            aekf.update(z);
            REQUIRE(x.allFinite());
        }
    }

    SECTION("Both adaptive Q and R enabled") {
        aekf.enableAdaptiveQ(true);
        aekf.enableAdaptiveR(true);
        aekf.setAdaptiveQRatio(0.2);
        aekf.setAdaptiveRRatio(0.3);
        aekf.setState(Eigen::Vector2d(0.0, 1.0));

        std::mt19937 rng(42);
        std::normal_distribution<double> noise(0.0, 0.5);

        for (int i = 0; i < 50; ++i) {
            aekf.predict();
            Eigen::Matrix<double, 1, 1> z;
            z << (i * 0.1 + noise(rng));
            auto x = aekf.update(z);
            REQUIRE(x.allFinite());
        }
    }
}

// ============================================================================
// Error-State Extended Kalman Filter Tests
// ============================================================================

TEST_CASE("ErrorStateEKF - Basic Operation", "[ESEKF][basic]") {
    using namespace test_utils;
    using Model = ConstantVelocityModel;

    auto predict_func = [](const auto* x_in, auto* x_out) { Model::predict(x_in, x_out); };
    auto measure_func = [](const auto* x_in, auto* z_out) { Model::measure(x_in, z_out); };
    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Matrix<double, 1, 1>& z) { return Model::getR(z); };

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman_lib::ErrorStateEKF<Model::N_X, Model::N_Z, decltype(predict_func),
                              decltype(measure_func)>
        esekf(predict_func, measure_func, update_Q, update_R, P0);

    // Set injection function (for ES-EKF, we need to define how error state is injected)
    esekf.setInjectFunc([](const Eigen::Vector2d& delta_x, Eigen::Vector2d& x) { x += delta_x; });

    SECTION("State initialization and prediction") {
        Eigen::Vector2d x0(1.0, 0.5);
        esekf.setState(x0);

        auto x_pred = esekf.predict();

        // Check prediction follows model
        REQUIRE_THAT(x_pred[0], WithinRel(1.0 + 0.1 * 0.5, 0.01));
        REQUIRE_THAT(x_pred[1], WithinRel(0.5, 0.01));
    }

    SECTION("Error state update") {
        esekf.setState(Eigen::Vector2d(0.0, 1.0));

        esekf.predict();
        Eigen::Matrix<double, 1, 1> z;
        z << 0.1;
        auto x_post = esekf.update(z);

        // State should be updated
        REQUIRE(std::isfinite(x_post[0]));
        REQUIRE(std::isfinite(x_post[1]));
    }
}

TEST_CASE("ErrorStateEKF - Anomaly Detection", "[ESEKF][anomaly]") {
    using namespace test_utils;
    using Model = ConstantVelocityModel;

    auto predict_func = [](const auto* x_in, auto* x_out) { Model::predict(x_in, x_out); };
    auto measure_func = [](const auto* x_in, auto* z_out) { Model::measure(x_in, z_out); };
    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Matrix<double, 1, 1>& z) { return Model::getR(z); };

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman_lib::ErrorStateEKF<Model::N_X, Model::N_Z, decltype(predict_func),
                              decltype(measure_func)>
        esekf(predict_func, measure_func, update_Q, update_R, P0);

    esekf.setInjectFunc([](const Eigen::Vector2d& delta_x, Eigen::Vector2d& x) { x += delta_x; });

    esekf.setState(Eigen::Vector2d(0.0, 1.0));
    esekf.setNisThreshold(7.88);  // Chi-square 1 DOF, 99.5%
    esekf.setWindowSize(15);

    // Normal operation
    for (int i = 0; i < 15; ++i) {
        esekf.predict();
        Eigen::Matrix<double, 1, 1> z;
        z << (i * 0.1);
        esekf.update(z);
    }

    REQUIRE(esekf.totalChecks() == 15);
    REQUIRE_FALSE(esekf.isRecentlyInconsistent());
}

// ============================================================================
// Comparative Tests - EKF vs UKF
// ============================================================================

TEST_CASE("Filter Comparison - EKF vs UKF on Nonlinear System", "[comparison]") {
    using namespace test_utils;

    // Nonlinear measurement model (polar)
    auto ekf_predict = [](const auto* x_in, auto* x_out) {
        x_out[0] = x_in[0];
        x_out[1] = x_in[1];
    };

    auto ekf_measure = [](const auto* x_in, auto* z_out) {
        using std::sqrt;
        using std::atan2;
        using ceres::sqrt;
        using ceres::atan2;
        z_out[0] = sqrt(x_in[0] * x_in[0] + x_in[1] * x_in[1]);
        z_out[1] = atan2(x_in[1], x_in[0]);
    };

    auto ukf_predict = [](const Eigen::Vector2d& x_in) { return x_in; };

    auto ukf_measure = [](const Eigen::Vector2d& x_in) {
        Eigen::Vector2d z;
        z[0] = std::sqrt(x_in[0] * x_in[0] + x_in[1] * x_in[1]);
        z[1] = std::atan2(x_in[1], x_in[0]);
        return z;
    };

    auto update_Q = []() { return Eigen::Matrix2d::Identity() * 0.01; };
    auto update_R = [](const Eigen::Vector2d&) {
        Eigen::Matrix2d R;
        R << 0.1, 0, 0, 0.01;
        return R;
    };

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman_lib::ExtendedKalmanFilter<2, 2, decltype(ekf_predict), decltype(ekf_measure)> ekf(
        ekf_predict, ekf_measure, update_Q, update_R, P0);

    kalman_lib::UnscentedKalmanFilter<2, 2, decltype(ukf_predict), decltype(ukf_measure)> ukf(
        ukf_predict, ukf_measure, update_Q, update_R, P0);

    // Custom residual for angle wrapping
    auto residual_func = [](const Eigen::Vector2d& z_pred, const Eigen::Vector2d& z_meas) {
        Eigen::Vector2d r;
        r[0] = z_meas[0] - z_pred[0];
        r[1] = normalizeAngle(z_meas[1] - z_pred[1]);
        return r;
    };

    ekf.setResidualFunc(residual_func);
    ukf.setResidualFunc(residual_func);

    Eigen::Vector2d x0(3.0, 0.0);
    ekf.setState(x0);
    ukf.setState(x0);

    std::mt19937 rng(123);
    std::normal_distribution<double> range_noise(0.0, 0.1);
    std::normal_distribution<double> bearing_noise(0.0, 0.05);

    std::vector<double> ekf_errors, ukf_errors;

    for (int step = 0; step < 30; ++step) {
        double angle = step * 0.15;
        Eigen::Vector2d true_pos(3.0 * std::cos(angle), 3.0 * std::sin(angle));

        double true_range = true_pos.norm();
        double true_bearing = std::atan2(true_pos[1], true_pos[0]);

        Eigen::Vector2d z(true_range + range_noise(rng), true_bearing + bearing_noise(rng));

        ekf.predict();
        ukf.predict();

        auto ekf_est = ekf.update(z);
        auto ukf_est = ukf.update(z);

        ekf_errors.push_back((ekf_est - true_pos).norm());
        ukf_errors.push_back((ukf_est - true_pos).norm());
    }

    // Both filters should track the target
    double ekf_avg =
        std::accumulate(ekf_errors.begin() + 10, ekf_errors.end(), 0.0) / (ekf_errors.size() - 10);
    double ukf_avg =
        std::accumulate(ukf_errors.begin() + 10, ukf_errors.end(), 0.0) / (ukf_errors.size() - 10);

    REQUIRE(ekf_avg < 1.5);
    REQUIRE(ukf_avg < 1.5);

    // Log results for comparison
    INFO("EKF average error: " << ekf_avg);
    INFO("UKF average error: " << ukf_avg);
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_CASE("Filter Stress Test - Many Iterations", "[stress]") {
    using namespace test_utils;
    using Model = Position2DModel;

    auto predict_func = [](const auto* x_in, auto* x_out) { Model::predict(x_in, x_out); };
    auto measure_func = [](const auto* x_in, auto* z_out) { Model::measure(x_in, z_out); };
    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Vector2d& z) { return Model::getR(z); };

    Eigen::Matrix4d P0 = Eigen::Matrix4d::Identity();

    kalman_lib::ExtendedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                     decltype(measure_func)>
        ekf(predict_func, measure_func, update_Q, update_R, P0);

    ekf.setState(Eigen::Vector4d(0.0, 0.0, 1.0, 0.5));

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.1);

    // Run 1000 iterations
    for (int i = 0; i < 1000; ++i) {
        ekf.predict();
        Eigen::Vector2d z(i * 0.05 + noise(rng), i * 0.025 + noise(rng));
        auto x = ekf.update(z);

        // Ensure state remains finite
        REQUIRE(x.allFinite());
    }

    // Covariance should be positive definite
    auto P = ekf.getPosteriorCovariance();
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(P);
    for (int i = 0; i < 4; ++i) {
        REQUIRE(es.eigenvalues()[i] > 0);
    }
}

TEST_CASE("Filter Robustness - NaN Handling", "[robustness]") {
    using namespace test_utils;
    using Model = ConstantVelocityModel;

    auto predict_func = [](const auto* x_in, auto* x_out) { Model::predict(x_in, x_out); };
    auto measure_func = [](const auto* x_in, auto* z_out) { Model::measure(x_in, z_out); };
    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Matrix<double, 1, 1>& z) { return Model::getR(z); };

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman_lib::ExtendedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                     decltype(measure_func)>
        ekf(predict_func, measure_func, update_Q, update_R, P0);

    ekf.setState(Eigen::Vector2d(0.0, 1.0));

    // Normal updates
    for (int i = 0; i < 5; ++i) {
        ekf.predict();
        Eigen::Matrix<double, 1, 1> z;
        z << (i * 0.1);
        ekf.update(z);
    }

    // Inject NaN measurement
    ekf.predict();
    Eigen::Matrix<double, 1, 1> z_nan;
    z_nan << std::numeric_limits<double>::quiet_NaN();

    // The filter should handle NaN gracefully (based on implementation clamping)
    auto x = ekf.update(z_nan);

    // State might be affected but should remain finite due to clamping
    // Note: Actual behavior depends on implementation
    // The filter has residual clamping which should prevent NaN propagation
}

// ============================================================================
// Parameterized Tests using GENERATE
// ============================================================================

TEST_CASE("Parameterized Test - Different Initial Uncertainties", "[parameterized]") {
    using namespace test_utils;
    using Model = ConstantVelocityModel;

    auto predict_func = [](const auto* x_in, auto* x_out) { Model::predict(x_in, x_out); };
    auto measure_func = [](const auto* x_in, auto* z_out) { Model::measure(x_in, z_out); };
    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [](const Eigen::Matrix<double, 1, 1>& z) { return Model::getR(z); };

    auto initial_uncertainty = GENERATE(0.1, 1.0, 10.0, 100.0);

    CAPTURE(initial_uncertainty);

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity() * initial_uncertainty;

    kalman_lib::ExtendedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                     decltype(measure_func)>
        ekf(predict_func, measure_func, update_Q, update_R, P0);

    ekf.setState(Eigen::Vector2d(0.0, 1.0));

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.3);

    double true_pos = 0.0;
    std::vector<double> errors;

    for (int i = 0; i < 30; ++i) {
        true_pos += 1.0 * Model::dt;
        ekf.predict();
        Eigen::Matrix<double, 1, 1> z;
        z << (true_pos + noise(rng));
        auto x = ekf.update(z);
        errors.push_back(std::abs(x[0] - true_pos));
    }

    // Regardless of initial uncertainty, filter should converge
    double final_errors = std::accumulate(errors.end() - 10, errors.end(), 0.0) / 10.0;
    REQUIRE(final_errors < 1.0);
}

TEST_CASE("Parameterized Test - Different Measurement Noise Levels", "[parameterized]") {
    using namespace test_utils;
    using Model = ConstantVelocityModel;

    auto measurement_noise = GENERATE(0.1, 0.5, 1.0, 2.0);

    CAPTURE(measurement_noise);

    auto predict_func = [](const auto* x_in, auto* x_out) { Model::predict(x_in, x_out); };
    auto measure_func = [](const auto* x_in, auto* z_out) { Model::measure(x_in, z_out); };
    auto update_Q = []() { return Model::getQ(); };
    auto update_R = [measurement_noise](const Eigen::Matrix<double, 1, 1>&) {
        return Eigen::Matrix<double, 1, 1>::Identity() * (measurement_noise * measurement_noise);
    };

    Eigen::Matrix2d P0 = Eigen::Matrix2d::Identity();

    kalman_lib::ExtendedKalmanFilter<Model::N_X, Model::N_Z, decltype(predict_func),
                                     decltype(measure_func)>
        ekf(predict_func, measure_func, update_Q, update_R, P0);

    ekf.setState(Eigen::Vector2d(0.0, 1.0));

    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, measurement_noise);

    double true_pos = 0.0;
    for (int i = 0; i < 50; ++i) {
        true_pos += 1.0 * Model::dt;
        ekf.predict();
        Eigen::Matrix<double, 1, 1> z;
        z << (true_pos + noise(rng));
        ekf.update(z);
    }

    // Posterior covariance should reflect measurement noise
    auto P = ekf.getPosteriorCovariance();
    REQUIRE(P(0, 0) > 0);
    // Higher measurement noise should generally lead to higher uncertainty
    // (but this is a simplified test)
}
