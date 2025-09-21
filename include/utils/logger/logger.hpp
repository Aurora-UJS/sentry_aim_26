#ifndef UTILS__LOGGER_HPP
#define UTILS__LOGGER_HPP

#include <spdlog/spdlog.h>

namespace utils {
std::shared_ptr<spdlog::logger> logger();

} // namespace utils

#endif // UTILS__LOGGER_HPP