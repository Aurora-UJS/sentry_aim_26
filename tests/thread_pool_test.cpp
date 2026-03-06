/**
 * ************************************************************************
 *
 * @file thread_pool_test.cpp
 * @author Xlqmu (niezhenghua2004@gmail.com)
 * @brief
 *
 * ************************************************************************
 * @copyright Copyright (c) 2025 Xlqmu
 * For study and research only, no reprinting
 * ************************************************************************
 */

#include "concurrency/thread_pool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

/**
 * @brief Test basic task submission and execution.
 */
TEST_CASE("SubmitBasicTask", "[ThreadPool]") {
    ThreadPool pool(4, 100);
    auto future = pool.submit([]() { return 42; });
    REQUIRE(future.get() == 42);
}

/**
 * @brief Test high-priority task execution.
 */
TEST_CASE("HighPriorityTask", "[ThreadPool]") {
    ThreadPool pool(4, 100);
    std::atomic<int> counter{0};
    std::vector<std::future<void>> futures;

    // Submit normal-priority tasks
    for (int i = 0; i < 10; ++i) {
        futures.push_back(pool.submit([&, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter.fetch_add(1);
        }));
    }

    // Submit a high-priority task
    auto high_future = pool.submit_high([&]() { counter.fetch_add(100); });

    // Wait for all tasks
    high_future.get();
    for (auto& f : futures)
        f.get();

    // High-priority task should execute first, contributing 100 to counter
    REQUIRE(counter.load() >= 100);
    REQUIRE(counter.load() <= 110);  // 10 normal tasks + 1 high-priority
}

/**
 * @brief Test ordered frame processing.
 */
TEST_CASE("OrderedFrameProcessing", "[ThreadPool]") {
    ThreadPool pool(4, 100);
    std::vector<int> processed_ids;
    std::mutex mtx;
    std::vector<std::future<std::vector<int>>> futures;

    // Submit frames in random order
    std::vector<int> ids = {1, 3, 2, 5, 4};
    for (int id : ids) {
        Frame frame{id};
        futures.push_back(pool.submit_frame(frame, [&mtx, &processed_ids](const Frame& f) {
            std::lock_guard<std::mutex> lock(mtx);
            processed_ids.push_back(f.id);
            return std::vector<int>{f.id};
        }));
    }

    // Wait for all frames and collect results
    std::vector<int> results;
    std::vector<bool> done(futures.size(), false);
    size_t finished = 0;
    while (finished < futures.size()) {
        for (size_t i = 0; i < futures.size(); ++i) {
            if (done[i])
                continue;
            if (futures[i].wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                auto r = futures[i].get();
                if (!r.empty())
                    results.push_back(r[0]);
                done[i] = true;
                ++finished;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::sort(results.begin(), results.end());
    std::vector<int> expected = {1, 2, 3, 4, 5};
    REQUIRE(processed_ids == expected);
    REQUIRE(results == expected);
}

/**
 * @brief Test dropping of outdated frames.
 */
TEST_CASE("DropOutdatedFrames", "[ThreadPool]") {
    auto drop_pred = [](const Frame& f) {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - f.t).count() > 500;
    };
    ThreadPool pool(4, 100, drop_pred);
    std::vector<int> processed_ids;
    std::mutex mtx;

    // Submit frames: 3, 1 (outdated), 2
    std::vector<Frame> frames = {
        {3, std::chrono::steady_clock::now()},
        {1, std::chrono::steady_clock::now() - std::chrono::milliseconds(1000)},
        {2, std::chrono::steady_clock::now()}};
    std::vector<std::future<std::vector<int>>> futures;

    for (const auto& frame : frames) {
        futures.push_back(pool.submit_frame(frame, [&mtx, &processed_ids](const Frame& f) {
            std::lock_guard<std::mutex> lock(mtx);
            processed_ids.push_back(f.id);
            return std::vector<int>{f.id};
        }));
    }

    // Wait for completion
    for (auto& f : futures) {
        try {
            f.get();
        } catch (...) {
        }  // Ignore exceptions for dropped frames
    }

    // Frame 1 should be dropped, only 2 and 3 processed
    std::vector<int> expected = {2, 3};
    REQUIRE(processed_ids == expected);
}

/**
 * @brief Test concurrent frame submission.
 */
TEST_CASE("ConcurrentFrameSubmission", "[ThreadPool]") {
    ThreadPool pool(4, 100);
    std::vector<int> processed_ids;
    std::mutex mtx, futures_mtx;
    std::vector<std::future<std::vector<int>>> futures;
    const int num_frames = 50;

    // Submit frames from multiple threads
    std::vector<std::thread> submitters;
    for (int t = 0; t < 4; ++t) {
        submitters.emplace_back(
            [&pool, t, num_frames, &mtx, &processed_ids, &futures, &futures_mtx]() {
                const int start = (num_frames * t) / 4;
                const int end = (num_frames * (t + 1)) / 4;
                for (int i = start; i < end; ++i) {
                    Frame frame{i + 1};
                    auto future = pool.submit_frame(frame, [&mtx, &processed_ids](const Frame& f) {
                        std::lock_guard<std::mutex> lock(mtx);
                        processed_ids.push_back(f.id);
                        return std::vector<int>{f.id};
                    });
                    std::lock_guard<std::mutex> lock(futures_mtx);
                    futures.push_back(std::move(future));
                }
            });
    }

    // Wait for submission threads
    for (auto& t : submitters)
        t.join();
    for (auto& f : futures)
        f.get();

    // Verify ordered processing
    std::vector<int> expected(num_frames);
    std::iota(expected.begin(), expected.end(), 1);
    REQUIRE(processed_ids == expected);
}

/**
 * @brief Test exception handling.
 */
TEST_CASE("ExceptionHandling", "[ThreadPool]") {
    ThreadPool pool(4, 100);
    auto future = pool.submit([]() -> int {
        throw std::runtime_error("Test exception");
        return 0;
    });

    REQUIRE_THROWS_AS(future.get(), std::runtime_error);
}

/**
 * @brief Test queue full behavior.
 */
TEST_CASE("QueueFull", "[ThreadPool]") {
    // Create pool with small queue capacity
    ThreadPool small_pool(2, 2);
    std::vector<std::future<std::vector<int>>> futures;

    // Submit frames to fill queue
    for (int i = 0; i < 2; ++i) {
        Frame frame{i + 1};
        futures.push_back(small_pool.submit_frame(frame, [](const Frame& f) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return std::vector<int>{f.id};
        }));
    }

    // Try to submit one more frame (completion future should hold queue-full exception)
    Frame frame{3};
    auto overflow_future =
        small_pool.submit_frame(frame, [](const Frame& f) { return std::vector<int>{f.id}; });
    REQUIRE_THROWS_AS(overflow_future.get(), std::runtime_error);

    // Wait for tasks to complete
    for (auto& f : futures)
        f.get();
}

/**
 * @brief Test zero threads fallback.
 */
TEST_CASE("ZeroThreads", "[ThreadPool]") {
    ThreadPool single_thread_pool(0);
    auto future = single_thread_pool.submit([]() { return 42; });
    REQUIRE(future.get() == 42);
}

/**
 * @brief Test pending tasks counter.
 */
TEST_CASE("PendingTasks", "[ThreadPool]") {
    ThreadPool pool(4, 100);
    std::vector<std::future<void>> futures;
    for (int i = 0; i < 5; ++i) {
        futures.push_back(
            pool.submit([]() { std::this_thread::sleep_for(std::chrono::milliseconds(50)); }));
    }
    REQUIRE(pool.pending_tasks() >= 0);
    for (auto& f : futures)
        f.get();
    REQUIRE(pool.pending_tasks() == 0);
}

/**
 * @brief Performance test (not a strict unit test).
 */
TEST_CASE("Performance", "[ThreadPool]") {
    ThreadPool pool(4, 100);
    const int num_tasks = 1000;
    std::vector<std::future<int>> futures;
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < num_tasks; ++i) {
        futures.push_back(pool.submit([i]() {
            // Simulate YOLO-like computation
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            return i;
        }));
    }

    for (auto& f : futures)
        f.get();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Processed " << num_tasks << " tasks in " << duration << " ms" << std::endl;
}
