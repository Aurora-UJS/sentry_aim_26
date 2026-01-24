// Copyright 2025 Zhenghua Nie
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

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <semaphore>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/lockfree/queue.hpp>

/**
 * @class MoveOnlyFunction
 * @brief A minimal move-only function wrapper for C++20, replacing
 * std::function. Supports move-only callables, avoiding the
 * copy-constructible requirement.
 */
class MoveOnlyFunction {
public:
    MoveOnlyFunction() = default;

    template <typename F>
    MoveOnlyFunction(F&& f) {
        using Functor = std::decay_t<F>;
        impl_ = std::make_unique<Impl<Functor>>(std::forward<F>(f));
    }

    MoveOnlyFunction(MoveOnlyFunction&&) = default;
    MoveOnlyFunction& operator=(MoveOnlyFunction&&) = default;

    MoveOnlyFunction(const MoveOnlyFunction&) = delete;
    MoveOnlyFunction& operator=(const MoveOnlyFunction&) = delete;

    void operator()() {
        if (impl_)
            impl_->call();
    }

private:
    struct Concept {
        virtual ~Concept() = default;
        virtual void call() = 0;
    };

    template <typename F>
    struct Impl : Concept {
        Impl(F&& f) : func_(std::move(f)) {}
        void call() override { func_(); }
        F func_;
    };

    std::unique_ptr<Concept> impl_;
};

/**
 * @struct Frame
 * @brief Represents a frame for task processing.
 * Contains a unique identifier and timestamp, extensible for specific use
 * cases.
 */
struct Frame {
    int id;  // Unique frame identifier
    std::chrono::steady_clock::time_point t =
        std::chrono::steady_clock::now();  // Timestamp for ordering or dropping
};

/**
 * @brief Type alias for a predicate to determine if a frame should be dropped.
 */
using DropPredicate = std::function<bool(const Frame&)>;

/**
 * @class OrderedQueue
 * @brief A thread-safe queue that ensures frames are dequeued in ID order.
 * Supports configurable dropping of outdated frames via a predicate.
 */
class OrderedQueue {
public:
    /**
     * @brief Constructs the OrderedQueue with a capacity and optional drop
     * predicate.
     * @param max_size Maximum number of frames in the queue and buffer.
     * @param drop_pred Predicate to determine if a frame should be dropped (e.g.,
     * outdated).
     */
    OrderedQueue(
        size_t max_size = 1024, DropPredicate drop_pred = [](const Frame&) { return false; })
        : current_id_(1), max_size_(max_size), drop_pred_(std::move(drop_pred)) {}

    /**
     * @brief Destructor that clears the queue and buffer.
     */
    ~OrderedQueue() {
        std::lock_guard<std::mutex> lock(mutex_);
        main_queue_ = std::queue<Frame>();
        buffer_.clear();
        current_id_ = 0;
        std::cerr << "OrderedQueue destroyed, queue and buffer cleared." << std::endl;
    }

    /**
     * @brief Enqueues a frame, maintaining order based on frame ID.
     * Drops frames that fail the predicate or have IDs less than the current ID.
     * @param item The frame to enqueue.
     * @throws std::runtime_error if the queue is full.
     */

    /**
    有bug，测试不过，好像是死锁
    */
    bool enqueue(const Frame& item) {
        if (drop_pred_(item)) {
            std::cerr << "OrderedQueue: Dropped frame with id: " << item.id << std::endl;
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        if (main_queue_.size() + buffer_.size() >= max_size_) {
            std::cerr << "OrderedQueue: Dropped frame due to full queue: " << item.id << std::endl;
            throw std::runtime_error("OrderedQueue is full");
        }
        if (item.id < current_id_) {
            std::cerr << "OrderedQueue: Dropped frame with small id: " << item.id << std::endl;
            return false;
        }
        if (item.id == current_id_) {
            main_queue_.push(item);
            current_id_++;
            auto it = buffer_.find(current_id_);
            while (it != buffer_.end()) {
                if (!drop_pred_(it->second)) {
                    main_queue_.push(it->second);
                    buffer_.erase(it);
                    current_id_++;
                    it = buffer_.find(current_id_);
                } else {
                    buffer_.erase(it);
                    it = buffer_.find(current_id_);
                }
            }
            if (main_queue_.size() >= 1) {
                cond_var_.notify_all();
            }
            return true;
        } else {
            buffer_[item.id] = item;
            return true;
        }
    }

    /**
     * @brief Dequeues the next frame in order, blocking until available.
     * @return The next frame in sequence.
     */
    Frame dequeue() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this]() { return !main_queue_.empty(); });
        Frame item = main_queue_.front();
        main_queue_.pop();
        return item;
    }

    /**
     * @brief Blocks until the front frame id equals target_id, then pops and returns it.
     */
    Frame dequeue_until(int target_id) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_var_.wait(lock, [this, target_id]() {
            return !main_queue_.empty() && main_queue_.front().id == target_id;
        });
        Frame item = main_queue_.front();
        main_queue_.pop();
        return item;
    }

private:
    std::queue<Frame> main_queue_;           // Main queue for ordered frames
    std::unordered_map<int, Frame> buffer_;  // Buffer for out-of-order frames
    int current_id_;                         // Next expected frame ID
    size_t max_size_;                        // Maximum queue capacity
    std::mutex mutex_;                       // Mutex for thread safety
    std::condition_variable cond_var_;       // Condition variable for blocking
    DropPredicate drop_pred_;                // Predicate for dropping frames
};

/**
 * @class ThreadPool
 * @brief A C++20-based thread pool implementation for task processing.
 * Supports task submission with priorities, exception propagation via
 * std::future, and ordered frame processing via OrderedQueue. Uses lock-free
 * queues for tasks.
 */
class ThreadPool {
public:
    /**
     * @brief A type-erased, move-only callable object wrapper for tasks.
     * Uses MoveOnlyFunction to support move-only callables in C++20.
     */
    using Task = MoveOnlyFunction;

    /**
     * @brief Constructs the ThreadPool and spawns worker threads.
     * @param num_threads The number of worker threads. If 0, defaults to
     *        std::thread::hardware_concurrency().
     * @param queue_capacity The fixed capacity for the underlying lock-free
     *        queues and OrderedQueue.
     * @param drop_pred Predicate to determine if frames should be dropped.
     */
    explicit ThreadPool(
        size_t num_threads = 0, size_t queue_capacity = 1024,
        DropPredicate drop_pred = [](const Frame&) { return false; })
        : m_high_priority_queue_(queue_capacity),
          m_normal_priority_queue_(queue_capacity),
          m_stop_(false),
          m_drop_pred_(std::move(drop_pred)),
          m_capacity_(queue_capacity) {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
            if (num_threads == 0) {  // Fallback if hardware_concurrency unavailable
                num_threads = 1;
            }
        }
        m_threads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            m_threads_.emplace_back([this] { worker_loop(); });
        }
    }

    /**
     * @brief Destructor that orchestrates a graceful shutdown of the thread pool.
     * Ensures all threads are joined and unexecuted tasks are cleaned up.
     */
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            m_stop_ = true;
        }
        m_tasks_available_sem_.release(m_threads_.size());
        for (auto& thread : m_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        Task* task_ptr;
        while (m_high_priority_queue_.pop(task_ptr)) {
            delete task_ptr;
        }
        while (m_normal_priority_queue_.pop(task_ptr)) {
            delete task_ptr;
        }
    }

    // Non-copyable and non-movable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    /**
     * @brief Submits a task for execution at normal priority.
     * @tparam F The type of the callable object.
     * @tparam Args The types of the arguments to the callable.
     * @param f The callable object.
     * @param args The arguments to pass to the callable.
     * @return A std::future that will hold the result of the task's execution.
     */
    template <typename F, typename... Args>
    auto submit(F&& f,
                Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        return submit_impl(false, std::forward<F>(f), std::forward<Args>(args)...);
    }

    /**
     * @brief Submits a task for execution at high priority.
     * High-priority tasks are processed before normal-priority tasks.
     * @return A std::future for the task's result.
     */
    template <typename F, typename... Args>
    auto submit_high(F&& f,
                     Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        return submit_impl(true, std::forward<F>(f), std::forward<Args>(args)...);
    }

    /**
     * @brief Submits a frame for ordered processing.
     * @param frame The frame to process.
     * @param processor A callable that processes the frame.
     * @return A std::future for the processing result.
     */
    template <typename F>
    auto submit_frame(const Frame& frame,
                      F&& processor) -> std::future<typename std::invoke_result<F, Frame>::type> {
        using ReturnType = typename std::invoke_result<F, Frame>::type;
        auto promise_ptr = std::make_shared<std::promise<ReturnType>>();
        std::future<ReturnType> future = promise_ptr->get_future();

        bool dropped = m_drop_pred_(frame);
        if (dropped) {
            promise_ptr->set_exception(
                std::make_exception_ptr(std::runtime_error("Frame dropped")));
        }

        auto chain_logic = [this]() {
            std::lock_guard<std::mutex> lock(mutex_);
            m_next_ordered_id_++;
            auto it = m_ordered_tasks_buffer_.find(m_next_ordered_id_);
            if (it != m_ordered_tasks_buffer_.end()) {
                BufferedTask& buffered = it->second;
                Task next_task;
                if (m_drop_pred_(buffered.frame)) {
                    next_task = std::move(buffered.cancel);
                } else {
                    next_task = std::move(buffered.task);
                }
                m_ordered_tasks_buffer_.erase(it);
                enqueue_task(std::move(next_task), true);
            }
        };

        Task task;
        if (dropped) {
            task = Task([chain_logic]() mutable { chain_logic(); });
        } else {
            task = Task(
                [promise_ptr, frame, proc = std::forward<F>(processor), chain_logic]() mutable {
                    try {
                        if constexpr (std::is_void_v<ReturnType>) {
                            std::invoke(std::move(proc), frame);
                            promise_ptr->set_value();
                        } else {
                            promise_ptr->set_value(std::invoke(std::move(proc), frame));
                        }
                    } catch (...) {
                        promise_ptr->set_exception(std::current_exception());
                    }
                    chain_logic();
                });
        }

        Task cancel([promise_ptr, chain_logic]() mutable {
            try {
                promise_ptr->set_exception(
                    std::make_exception_ptr(std::runtime_error("Frame dropped")));
            } catch (...) {
            }
            chain_logic();
        });

        std::lock_guard<std::mutex> lock(mutex_);

        if (frame.id < m_next_ordered_id_) {
            if (!dropped) {
                promise_ptr->set_exception(
                    std::make_exception_ptr(std::runtime_error("Frame dropped (outdated)")));
            }
            return future;
        }

        if (frame.id == m_next_ordered_id_) {
            enqueue_task(std::move(task), true);
        } else {
            std::cerr << "Check size: " << m_ordered_tasks_buffer_.size()
                      << " >= " << m_capacity_ - 1 << std::endl;
            if (m_ordered_tasks_buffer_.size() >= m_capacity_ - 1) {
                std::cerr << "Throwing!" << std::endl;
                promise_ptr->set_exception(
                    std::make_exception_ptr(std::runtime_error("OrderedQueue is full")));
                return future;
            }
            m_ordered_tasks_buffer_.emplace(
                frame.id, BufferedTask{frame, std::move(task), std::move(cancel)});
        }
        return future;
    }

    /**
     * @brief Returns the approximate number of tasks pending execution.
     * @return The number of tasks in the queues.
     */
    size_t pending_tasks() const { return m_pending_tasks_.load(std::memory_order_relaxed); }

private:
    /**
     * @brief Internal implementation for task submission.
     * Packages the callable and arguments into a task and enqueues it.
     */
    template <typename F, typename... Args>
    auto submit_impl(bool high_priority, F&& f,
                     Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using ReturnType = typename std::invoke_result<F, Args...>::type;
        std::packaged_task<ReturnType()> task(
            [func = std::forward<F>(f), ... largs = std::forward<Args>(args)]() mutable {
                return std::invoke(std::move(func), std::move(largs)...);
            });
        std::future<ReturnType> future = task.get_future();
        enqueue_task([task = std::move(task)]() mutable { task(); }, high_priority);
        return future;
    }

    /**
     * @brief The main loop for each worker thread.
     * Processes tasks until the pool is stopped.
     */
    void worker_loop() {
        while (true) {
            m_tasks_available_sem_.acquire();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (m_stop_) {
                    break;
                }
            }
            Task* task_ptr = nullptr;
            if (m_high_priority_queue_.pop(task_ptr) || m_normal_priority_queue_.pop(task_ptr)) {
                std::unique_ptr<Task> guard(task_ptr);
                try {
                    (*task_ptr)();
                } catch (const std::exception& e) {
                    std::cerr << "ThreadPool: Task threw an exception: " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "ThreadPool: Task threw an unknown exception." << std::endl;
                }
                m_pending_tasks_.fetch_sub(1, std::memory_order_relaxed);
            }
        }
    }

    /**
     * @brief Allocates a task on the heap and pushes it to the appropriate queue.
     */
    void enqueue_task(Task&& task, bool high_priority) {
        m_pending_tasks_.fetch_add(1, std::memory_order_relaxed);
        Task* task_ptr = new Task(std::move(task));
        auto& queue = high_priority ? m_high_priority_queue_ : m_normal_priority_queue_;
        if (!queue.push(task_ptr)) {
            delete task_ptr;
            m_pending_tasks_.fetch_sub(1, std::memory_order_relaxed);
            throw std::runtime_error("ThreadPool queue is full.");
        }
        m_tasks_available_sem_.release();
    }

    // --- Core Members ---
    std::vector<std::thread> m_threads_;                     // Worker threads
    boost::lockfree::queue<Task*> m_high_priority_queue_;    // High-priority task queue
    boost::lockfree::queue<Task*> m_normal_priority_queue_;  // Normal-priority task queue
    std::counting_semaphore<> m_tasks_available_sem_{0};     // Semaphore for task availability
    alignas(64) std::atomic<size_t> m_pending_tasks_{0};     // Pending task counter
    std::mutex mutex_;                                       // Mutex for stop flag
    bool m_stop_;                                            // Stop flag for shutdown

    struct BufferedTask {
        Frame frame;
        Task task;
        Task cancel;
    };
    std::unordered_map<int, BufferedTask> m_ordered_tasks_buffer_;
    int m_next_ordered_id_ = 1;
    DropPredicate m_drop_pred_;
    size_t m_capacity_;
};