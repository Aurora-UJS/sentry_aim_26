#include "utils/logger/logger.hpp"

int main() {
    // 测试各级别日志
    utils::logger()->trace("这是 trace 日志（最详细，通常不显示）");
    utils::logger()->debug("这是 debug 日志（开发调试用）");
    utils::logger()->info("这是 info 日志（正常信息）");
    utils::logger()->warn("这是 warn 日志（警告信息）");
    utils::logger()->error("这是 error 日志（错误信息）");
    utils::logger()->critical("这是 critical 日志（严重错误）");

    // 测试格式化输出
    int target_id = 42;
    double distance = 3.14;
    utils::logger()->info("检测到目标 ID={}, 距离={:.2f}m", target_id, distance);

    // 测试多次调用（验证单例）
    for (int i = 0; i < 3; ++i) {
        utils::logger()->debug("循环测试 {}", i);
    }

    utils::logger()->info("日志模块测试完成");
    return 0;
}
