cmake_minimum_required(VERSION 3.28)

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

add_executable(armor_solver_test
    armor_solver_test.cpp
)

add_executable(pnp_solver_unit_test
    pnp_solver_unit_test.cpp
)

# 完整流水线测试：检测 + 求解 + 跟踪
add_executable(auto_aim_pipeline_test
    auto_aim_pipeline_test.cpp
)

# Include directories for tests
target_include_directories(logger_module_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(thread_pool_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_include_directories(filter_test PRIVATE ${CMAKE_SOURCE_DIR}/base/math)
target_include_directories(armor_detector_test PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tasks)
target_include_directories(armor_solver_test PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tasks)
target_include_directories(pnp_solver_unit_test PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tasks)
target_include_directories(auto_aim_pipeline_test PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/tasks)

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

# Link dependencies for armor_solver_test
target_link_libraries(armor_solver_test
    PRIVATE
        tasks
        utils
        ${OpenCV_LIBS}
        Eigen3::Eigen
)

# Link dependencies for pnp_solver_unit_test
target_link_libraries(pnp_solver_unit_test
    PRIVATE
        tasks
        utils
        ${OpenCV_LIBS}
        Eigen3::Eigen
        Catch2::Catch2WithMain
)

# Link dependencies for auto_aim_pipeline_test
target_link_libraries(auto_aim_pipeline_test
    PRIVATE
        tasks
        utils
        ${OpenCV_LIBS}
        Eigen3::Eigen
        yaml-cpp
)

# Enable ctest
enable_testing()
add_test(NAME logger_module_test COMMAND logger_module_test)
add_test(NAME thread_pool_test COMMAND thread_pool_test)
add_test(NAME armor_detector_test COMMAND armor_detector_test)
add_test(NAME armor_solver_test COMMAND armor_solver_test)
add_test(NAME pnp_solver_unit_test COMMAND pnp_solver_unit_test)
add_test(NAME filter_test COMMAND filter_test)