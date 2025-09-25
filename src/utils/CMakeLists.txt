add_library(utils
    logger/logger.cpp

)
target_include_directories(utils
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../include>
)
target_link_libraries(utils PUBLIC spdlog::spdlog fmt::fmt Eigen3::Eigen)
target_link_libraries(sp_vision_refactoring PUBLIC utils)