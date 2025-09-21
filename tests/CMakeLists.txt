cmake_minimum_required(VERSION 3.28)

# 单元测试可执行文件
add_executable(logger_module_test
    logger_module_test.cpp
)

target_link_libraries(logger_module_test
    PRIVATE
        utils
        fmt::fmt
        Ceres::ceres
)

# 可选：启用 ctest
# enable_testing()
# add_test(NAME logger_module_test COMMAND logger_module_test)
