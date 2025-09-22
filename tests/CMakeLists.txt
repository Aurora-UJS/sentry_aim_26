cmake_minimum_required(VERSION 3.28)

# Unit test executables
add_executable(logger_module_test
    logger_module_test.cpp
)

add_executable(thread_pool_test
    thread_pool_test.cpp
)

# Include directories for tests
target_include_directories(logger_module_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(thread_pool_test PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Link dependencies for logger_module_test
target_link_libraries(logger_module_test
    PRIVATE
        utils
        fmt::fmt
        Ceres::ceres
        Catch2::Catch2WithMain
)

# Link dependencies for thread_pool_test
target_link_libraries(thread_pool_test
    PRIVATE
        utils
        Catch2::Catch2WithMain
        Boost::boost
        pthread
)

# Enable ctest
enable_testing()
add_test(NAME logger_module_test COMMAND logger_module_test)
add_test(NAME thread_pool_test COMMAND thread_pool_test)