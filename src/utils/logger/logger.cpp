#include "utils/logger/logger.hpp"

#include <chrono>
#include <filesystem>

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <fmt/chrono.h>

namespace utils {
std::shared_ptr<spdlog::logger> logger_ = nullptr;

void set_logger() {
    // 确保日志目录存在
    std::filesystem::create_directories("logs");

    // 生成日志文件名（带时间戳）
    auto file_name = fmt::format("logs/{:%Y-%m-%d_%H-%M-%S}.log", std::chrono::system_clock::now());
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(file_name, true);
    file_sink->set_level(spdlog::level::debug);
    // 文件日志格式：[时间] [级别] [线程ID] 消息
    file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");

    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);  // 控制台只显示 info 及以上
    // 控制台格式：[时间] [级别] 消息（更简洁）
    console_sink->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");

    logger_ = std::make_shared<spdlog::logger>("aurora",
                                               spdlog::sinks_init_list{file_sink, console_sink});
    logger_->set_level(spdlog::level::debug);  // 总体级别：debug（文件会记录所有）
    logger_->flush_on(spdlog::level::warn);    // warn 及以上立即刷新到磁盘
}

std::shared_ptr<spdlog::logger> logger() {
    if (!logger_) {
        set_logger();
    }
    return logger_;
}

}  // namespace utils
