cmake_minimum_required(VERSION 3.22.1)

# Unit test executables
add_executable(logger_module_test
    logger_module_test.cpp
)

add_executable(thread_pool_test
    thread_pool_test.cpp
)

add_executable(filter_test
    filter_test.cpp
)

add_executable(armor_detector_test
    armor_detector_test.cpp
)

# Include directories for tests
target_include_directories(logger_module_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(thread_pool_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(filter_test PRIVATE ${CMAKE_SOURCE_DIR}/base/math)
target_include_directories(armor_detector_test PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tasks)

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

# Link dependencies for filter_test
target_link_libraries(filter_test
    PRIVATE
        Eigen3::Eigen
        Ceres::ceres
        Catch2::Catch2WithMain
)

# Link dependencies for armor_detector_test
target_link_libraries(armor_detector_test
    PRIVATE
        tasks
        utils
        ${OpenCV_LIBS}
)

# Enable ctest
enable_testing()
add_test(NAME logger_module_test COMMAND logger_module_test)
add_test(NAME thread_pool_test COMMAND thread_pool_test)
add_test(NAME armor_detector_test COMMAND armor_detector_test)
add_test(NAME filter_test COMMAND filter_test)